from types import SimpleNamespace

from shapely.geometry import Polygon

from coal_boundary.export import make_service_fence_document
from coal_boundary.geometry import PileResult


def test_service_fence_document_matches_volume3d_shape():
    result = PileResult(
        pile_id=7,
        polygon=Polygon([
            (500000.0, 3300000.0),
            (500010.0, 3300000.0),
            (500010.0, 3300010.0),
            (500000.0, 3300000.0),
        ]),
        volume_m3=100.0,
        area_m2=50.0,
        max_relative_height_m=5.0,
        mean_relative_height_m=2.0,
        quality_score=0.9,
        quality_grade="A",
        boundary_confidence=0.9,
        pixel_count=50,
    )
    crs = SimpleNamespace(
        central_meridian_deg=120.0,
        false_easting_m=500000.0,
        false_northing_m=0.0,
    )

    document = make_service_fence_document([result], crs)

    assert isinstance(document, list)
    assert set(document[0]) == {"num", "volume_m3", "maxlon", "minlon", "maxlat", "minlat", "points"}
    assert document[0]["num"] == 7
    assert document[0]["volume_m3"] == 100.0
    assert len(document[0]["points"]) == 3
    assert all(set(point) == {"lon", "lat"} for point in document[0]["points"])
    assert document[0]["minlon"] <= document[0]["maxlon"]
    assert document[0]["minlat"] <= document[0]["maxlat"]
