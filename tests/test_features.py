import numpy as np

from coal_boundary.features import FEATURE_NAMES, build_cell_features


def test_cell_features_are_finite_and_stable_shape():
    shape = (6, 8)
    height = np.ones(shape, dtype=np.float32)
    rgb = np.full((*shape, 3), 64.0, dtype=np.float32)
    features = build_cell_features(
        height, height + 10, height + 9, height + 11,
        rgb, np.full(shape, 5, dtype=np.uint32), None,
    )
    assert features.shape == (*shape, len(FEATURE_NAMES))
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
