from __future__ import annotations

import json

from shapely.geometry import Polygon

from coal_boundary.config import CrsConfig
from coal_boundary.display_cache import (
    DISPLAY_GEOMETRY_METHOD,
    DISPLAY_GEOMETRY_PARAMETERS,
    display_geometry_cache_key,
    load_display_geometry_cache,
    load_display_geometry_from_result,
    save_display_geometry_cache,
)
from coal_boundary.export import _polygon_to_lonlat_mapping
from coal_boundary.geometry import PileResult


def _result(polygon: Polygon, pile_id: int = 1) -> PileResult:
    return PileResult(
        pile_id=pile_id,
        polygon=polygon,
        volume_m3=100.0,
        area_m2=polygon.area,
        max_relative_height_m=5.0,
        mean_relative_height_m=2.5,
        quality_score=0.9,
        quality_grade="A",
        boundary_confidence=0.8,
        pixel_count=100,
    )


def test_display_geometry_cache_round_trip_and_invalidation(tmp_path):
    source = Polygon([
        (500_000.0, 4_000_000.0),
        (500_010.0, 4_000_000.0),
        (500_010.0, 4_000_010.0),
        (500_000.0, 4_000_010.0),
    ])
    original = _result(source)
    original.display_polygon = source.buffer(0.2, join_style="round")
    original.display_geometry_metrics = {"area_delta_percent": 0.1}
    cache_path = tmp_path / "display_geometry_full.json"
    save_display_geometry_cache(cache_path, [original], "dataset-1", "config-1")

    restored = _result(source)
    loaded = load_display_geometry_cache(
        cache_path, [restored], "dataset-1", "config-1",
    )
    assert loaded.hit
    assert loaded.source == "display-cache"
    assert restored.display_polygon is not None
    assert restored.display_polygon.equals_exact(original.display_polygon, 1e-9)
    assert restored.display_geometry_metrics == original.display_geometry_metrics

    changed = _result(Polygon([
        (500_000.0, 4_000_000.0),
        (500_011.0, 4_000_000.0),
        (500_011.0, 4_000_010.0),
        (500_000.0, 4_000_010.0),
    ]))
    missed = load_display_geometry_cache(
        cache_path, [changed], "dataset-1", "config-1",
    )
    assert not missed.hit
    assert "计算边界已变化" in missed.reason
    assert changed.display_polygon is None
    assert display_geometry_cache_key([original]) != display_geometry_cache_key([changed])


def test_previous_result_bootstraps_display_cache_without_las_scan(tmp_path):
    crs = CrsConfig()
    source = Polygon([
        (500_000.0, 4_000_000.0),
        (500_010.0, 4_000_000.0),
        (500_010.0, 4_000_010.0),
        (500_000.0, 4_000_010.0),
    ])
    display = source.buffer(0.2, join_style="round")
    result_path = tmp_path / "coal_piles_full.json"
    result_path.write_text(json.dumps({
        "metadata": {
            "dataset_fingerprint": "dataset-1",
            "config_hash": "config-1",
            "display_geometry": {"method": DISPLAY_GEOMETRY_METHOD, **DISPLAY_GEOMETRY_PARAMETERS},
        },
        "piles": [{
            "pile_id": 1,
            "geometry": _polygon_to_lonlat_mapping(source, 120.0, 500_000.0, 0.0),
            "display_geometry": _polygon_to_lonlat_mapping(display, 120.0, 500_000.0, 0.0),
            "display_geometry_metrics": {"display_vertices": len(display.exterior.coords)},
        }],
    }), encoding="utf-8")

    restored = _result(source)
    loaded = load_display_geometry_from_result(
        result_path, [restored], "dataset-1", "config-1", crs,
    )
    assert loaded.hit
    assert loaded.source == "previous-result"
    assert restored.display_polygon is not None
    assert restored.display_polygon.boundary.hausdorff_distance(display.boundary) < 0.05


def test_previous_result_with_stale_parameters_is_not_reused(tmp_path):
    """调整展示曲线参数（如改用贴地优先）后，旧成果里的旧曲线不能被当成有效缓存。"""
    crs = CrsConfig()
    source = Polygon([
        (500_000.0, 4_000_000.0),
        (500_010.0, 4_000_000.0),
        (500_010.0, 4_000_010.0),
        (500_000.0, 4_000_010.0),
    ])
    display = source.buffer(0.2, join_style="round")
    result_path = tmp_path / "coal_piles_full.json"
    stale = {**DISPLAY_GEOMETRY_PARAMETERS, "selection_strategy": "area_first"}
    result_path.write_text(json.dumps({
        "metadata": {
            "dataset_fingerprint": "dataset-1",
            "config_hash": "config-1",
            "display_geometry": {"method": DISPLAY_GEOMETRY_METHOD, **stale},
        },
        "piles": [{
            "pile_id": 1,
            "geometry": _polygon_to_lonlat_mapping(source, 120.0, 500_000.0, 0.0),
            "display_geometry": _polygon_to_lonlat_mapping(display, 120.0, 500_000.0, 0.0),
            "display_geometry_metrics": {"display_vertices": len(display.exterior.coords)},
        }],
    }), encoding="utf-8")

    restored = _result(source)
    loaded = load_display_geometry_from_result(
        result_path, [restored], "dataset-1", "config-1", crs,
    )
    assert not loaded.hit
    assert "参数" in loaded.reason
    assert restored.display_polygon is None
