"""Optional model test; model files are never distributed here."""
import os
from pathlib import Path
import numpy as np
import pytest
from coal_boundary.model import LegacyKerasSegmenter


def test_optional_weights_load_and_predict():
    value = os.environ.get('COAL_TEST_MODEL_PATH')
    if not value:
        pytest.skip('Set COAL_TEST_MODEL_PATH to a locally authorized model')
    path = Path(value)
    assert path.is_file()
    output = LegacyKerasSegmenter(path).predict_points(
        np.zeros((1024, 4), dtype=np.float32), batch_size=1)
    assert output.shape == (1024,)
    assert np.all(np.isfinite(output))
    assert np.all((output >= 0) & (output <= 1))
