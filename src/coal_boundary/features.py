from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


FEATURE_NAMES = (
    "relative_height", "red", "green", "blue", "red_ratio", "green_ratio",
    "blue_ratio", "log_point_density", "vertical_range", "surface_thickness",
    "height_gradient", "local_height_std", "legacy_probability",
)


def build_cell_features(
    relative_height: np.ndarray,
    mean_z: np.ndarray,
    min_z: np.ndarray,
    max_z: np.ndarray,
    mean_rgb: np.ndarray,
    point_count: np.ndarray,
    legacy_probability: np.ndarray | None,
) -> np.ndarray:
    height = np.nan_to_num(relative_height, nan=0.0).astype(np.float32, copy=False)
    gradient_x = ndi.sobel(height, axis=1, mode="nearest") / 8.0
    gradient_y = ndi.sobel(height, axis=0, mode="nearest") / 8.0
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    local_mean = ndi.uniform_filter(height, size=5, mode="nearest")
    local_square_mean = ndi.uniform_filter(height * height, size=5, mode="nearest")
    local_std = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    vertical_range = np.maximum(np.nan_to_num(max_z - min_z, nan=0.0), 0.0)
    surface_thickness = np.maximum(np.nan_to_num(mean_z - min_z, nan=0.0), 0.0)
    rgb = np.nan_to_num(mean_rgb, nan=0.0).astype(np.float32, copy=False)
    rgb_total = np.maximum(np.sum(rgb, axis=2), 1.0)
    legacy = np.zeros_like(height) if legacy_probability is None else np.nan_to_num(legacy_probability, nan=0.0)
    return np.stack(
        [
            np.clip(height / 15.0, 0.0, 2.0),
            np.clip(rgb[:, :, 0] / 255.0, 0.0, 1.0),
            np.clip(rgb[:, :, 1] / 255.0, 0.0, 1.0),
            np.clip(rgb[:, :, 2] / 255.0, 0.0, 1.0),
            np.clip(rgb[:, :, 0] / rgb_total, 0.0, 1.0),
            np.clip(rgb[:, :, 1] / rgb_total, 0.0, 1.0),
            np.clip(rgb[:, :, 2] / rgb_total, 0.0, 1.0),
            np.clip(np.log1p(point_count.astype(np.float32)) / 8.0, 0.0, 2.0),
            np.clip(vertical_range / 10.0, 0.0, 2.0),
            np.clip(surface_thickness / 5.0, 0.0, 2.0),
            np.clip(gradient / 5.0, 0.0, 2.0),
            np.clip(local_std / 5.0, 0.0, 2.0),
            np.clip(legacy, 0.0, 1.0),
        ], axis=2,
    ).astype(np.float32)
