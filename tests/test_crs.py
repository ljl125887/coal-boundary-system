import numpy as np
from types import SimpleNamespace

from coal_boundary.crs import lonlat_to_projected, projected_to_lonlat, validate_crs_config_and_bounds


def test_cgcs2000_round_trip():
    lon = np.array([120.0, 120.01])
    lat = np.array([30.0, 30.01])
    x, y = lonlat_to_projected(lon, lat)
    lon2, lat2 = projected_to_lonlat(x, y)
    assert np.max(np.abs(lon2 - lon)) < 1e-8
    assert np.max(np.abs(lat2 - lat)) < 1e-8


def test_central_meridian_has_false_easting():
    x, y = lonlat_to_projected(120.0, 30.0)
    assert abs(float(x) - 500_000) < 1e-6
    assert 3_000_000 < float(y) < 4_000_000


def test_project_bounds_crs_validation():
    config = SimpleNamespace(
        source_epsg=4549, output_epsg=4326, central_meridian_deg=120.0,
        false_easting_m=500000.0, false_northing_m=0.0,
    )
    result = validate_crs_config_and_bounds(config, [500000, 3300000, 0], [500100, 3300100, 30])
    assert result["validated"] is True
