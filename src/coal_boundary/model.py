from __future__ import annotations

from pathlib import Path
import logging
import os

import h5py
import numpy as np


def _load_tensorflow():
    # TensorFlow 2.21 emits native Windows CPU/GPU capability notices before
    # its Python logger is configured.  They are not extraction errors and
    # otherwise obscure the stable stage/cache log consumed by the GUI.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("GLOG_minloglevel", "2")
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    return tf


class LegacyKerasSegmenter:
    """Loads the legacy Keras 2.2.4 weight-only HDF5 without private TF APIs."""

    def __init__(self, weights_path: str | Path, window_size: int = 1024):
        self.tf = _load_tensorflow()
        self.window_size = int(window_size)
        self.model = self._build_model()
        self._load_legacy_weights(Path(weights_path))

    def _build_model(self):
        tf = self.tf
        layers = tf.keras.layers
        inputs = layers.Input((self.window_size, 4), name="input_1")
        x = layers.Conv1D(32, 10, activation="relu", padding="same", name="conv1d")(inputs)
        for i in range(2):
            a = layers.Conv1D(128, 10, activation="relu", padding="same", name=f"conv1d_{1 + i * 3}")(x)
            b = layers.Conv1D(128, 10, activation="relu", padding="same", name=f"conv1d_{2 + i * 3}")(a)
            x = layers.Add(name="add" if i == 0 else f"add_{i}")([a, b])
            x = layers.Conv1D(128, 10, activation="relu", padding="same", name=f"conv1d_{3 + i * 3}")(x)
            x = layers.BatchNormalization(name="batch_normalization_v1" if i == 0 else f"batch_normalization_v1_{i}")(x)
        a = layers.Conv1D(32, 10, activation="relu", padding="same", name="conv1d_7")(x)
        b = layers.Conv1D(32, 10, activation="relu", padding="same", name="conv1d_8")(a)
        x = layers.Add(name="add_2")([a, b])
        x = layers.Conv1D(2, 10, activation="relu", padding="same", name="conv1d_9")(x)
        x = layers.BatchNormalization(name="batch_normalization_v1_2")(x)
        outputs = layers.Conv1D(1, 1, activation="sigmoid", padding="valid", name="conv1d_10")(x)
        return tf.keras.Model(inputs=inputs, outputs=outputs, name="legacy_coal_segmenter")

    def _load_legacy_weights(self, path: Path) -> None:
        with h5py.File(path, "r") as h5:
            for layer in self.model.layers:
                if layer.name not in h5 or not layer.weights:
                    continue
                group = h5[layer.name]
                nested = group[layer.name] if layer.name in group else group
                if layer.name.startswith("conv1d"):
                    weights = [np.asarray(nested["kernel:0"]), np.asarray(nested["bias:0"])]
                elif layer.name.startswith("batch_normalization"):
                    weights = [
                        np.asarray(nested["gamma:0"]), np.asarray(nested["beta:0"]),
                        np.asarray(nested["moving_mean:0"]), np.asarray(nested["moving_variance:0"]),
                    ]
                else:
                    continue
                layer.set_weights(weights)

    def predict_points(self, features: np.ndarray, batch_size: int = 64) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        n = len(features)
        if n == 0:
            return np.empty(0, dtype=np.float32)
        padded_n = int(np.ceil(n / self.window_size)) * self.window_size
        padded = np.zeros((padded_n, 4), dtype=np.float32)
        padded[:n] = features
        windows = padded.reshape(-1, self.window_size, 4)
        prediction = self.model.predict(windows, batch_size=batch_size, verbose=0)
        return np.asarray(prediction, dtype=np.float32).reshape(-1)[:n]


def infer_probability_grid(
    segmenter: LegacyKerasSegmenter,
    mean_z: np.ndarray,
    mean_rgb: np.ndarray,
    valid_mask: np.ndarray,
    cell_size_m: float,
    voxel_size_m: float,
    batch_size: int,
) -> np.ndarray:
    stride = max(1, int(round(voxel_size_m / cell_size_m)))
    rr = np.arange(0, mean_z.shape[0], stride)
    cc = np.arange(0, mean_z.shape[1], stride)
    row_grid, col_grid = np.meshgrid(rr, cc, indexing="ij")
    valid = valid_mask[row_grid, col_grid]
    rows, cols = row_grid[valid], col_grid[valid]
    features = np.column_stack([
        mean_z[rows, cols] / 20.0,
        mean_rgb[rows, cols, 0] / 255.0,
        mean_rgb[rows, cols, 1] / 255.0,
        mean_rgb[rows, cols, 2] / 255.0,
    ]).astype(np.float32)
    probability = segmenter.predict_points(features, batch_size=batch_size)
    coarse = np.full(row_grid.shape, np.nan, dtype=np.float32)
    coarse[valid] = probability
    expanded = np.repeat(np.repeat(coarse, stride, axis=0), stride, axis=1)
    return expanded[: mean_z.shape[0], : mean_z.shape[1]]


class CellClassifier:
    """Calibrated classifier for coal surface versus yard objects."""

    def __init__(self, model_path: str | Path):
        tf = _load_tensorflow()
        self.model = tf.keras.models.load_model(Path(model_path), compile=False)

    def predict_grid(self, features: np.ndarray, valid_mask: np.ndarray, batch_size: int = 8192) -> np.ndarray:
        probability = np.zeros(valid_mask.shape, dtype=np.float32)
        indices = np.flatnonzero(valid_mask)
        if len(indices) == 0:
            return probability
        flat_features = features.reshape(-1, features.shape[2])[indices]
        prediction = self.model.predict(flat_features, batch_size=batch_size, verbose=0).reshape(-1)
        probability.reshape(-1)[indices] = np.asarray(prediction, dtype=np.float32)
        return probability
