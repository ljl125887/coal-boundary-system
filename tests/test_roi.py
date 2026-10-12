import json

import numpy as np

from coal_boundary.roi import load_projected_roi, rasterize_geometry


def test_roi_crs_validation_and_rasterization(tmp_path):
    path = tmp_path / "roi.geojson"
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:4549"}},
        "features": [{
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]]},
        }],
    }), encoding="utf-8")
    geometry = load_projected_roi(path)
    mask = rasterize_geometry(geometry, 6, 6, 0, 0, 1)
    assert mask.dtype == np.bool_
    assert int(mask.sum()) == 16
