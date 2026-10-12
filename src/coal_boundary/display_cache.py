from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import Polygon, mapping, shape

from .crs import lonlat_to_projected
from .export import atomic_write_json
from .geometry import PileResult


DISPLAY_GEOMETRY_METHOD = "raw-point-cloud-toe-adaptive-spline-v1"
DISPLAY_GEOMETRY_PARAMETERS = {
    "fine_cell_size_m": 0.20,
    "search_margin_m": 3.0,
    "height_threshold_m": 0.35,
    "field_smoothing_m": 0.50,
    "max_gap_fill_m": 0.60,
    "toe_height_floor_m": 0.05,
    "area_tolerance_percent": 15.0,
    "selection_strategy": "toe_first",
    "corridor_buffer_m": 2.8,
    "ground_offset_inner_m": 1.0,
    "ground_offset_outer_m": 3.0,
}
DISPLAY_CACHE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class DisplayGeometryLoadResult:
    hit: bool
    reason: str
    source: str | None = None
    pile_count: int = 0


def calculation_geometry_fingerprint(results: list[PileResult]) -> str:
    """Hash calculation polygons after millimetre precision normalization."""
    digest = hashlib.sha256()
    for result in sorted(results, key=lambda item: item.pile_id):
        polygon = shapely.set_precision(result.polygon, grid_size=0.001)
        polygon = shapely.normalize(polygon)
        digest.update(str(result.pile_id).encode("ascii"))
        digest.update(shapely.to_wkb(polygon, byte_order=1))
    return digest.hexdigest()


def display_geometry_cache_key(results: list[PileResult], parameters: dict | None = None) -> str:
    raw = json.dumps({
        "method": DISPLAY_GEOMETRY_METHOD,
        "parameters": dict(parameters or DISPLAY_GEOMETRY_PARAMETERS),
        "calculation_geometry_fingerprint": calculation_geometry_fingerprint(results),
    }, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _valid_polygon(payload: dict) -> Polygon:
    value = shape(payload)
    if not isinstance(value, Polygon) or value.is_empty or not value.is_valid:
        raise ValueError("缓存中的展示边界不是有效 Polygon")
    return value


def _apply_entries(results: list[PileResult], entries: list[dict]) -> None:
    by_id = {int(entry["pile_id"]): entry for entry in entries}
    expected_ids = {result.pile_id for result in results}
    if set(by_id) != expected_ids:
        raise ValueError("缓存煤堆编号与本次提取结果不一致")
    loaded: dict[int, tuple[Polygon, dict | None]] = {}
    for pile_id in expected_ids:
        entry = by_id[pile_id]
        loaded[pile_id] = (
            _valid_polygon(entry["display_geometry_projected"]),
            entry.get("display_geometry_metrics"),
        )
    # Only mutate results after every cache entry has passed validation.
    for result in results:
        result.display_polygon, result.display_geometry_metrics = loaded[result.pile_id]


def save_display_geometry_cache(
    path: str | Path,
    results: list[PileResult],
    dataset_fingerprint: str,
    config_hash: str,
    parameters: dict | None = None,
) -> None:
    missing = [result.pile_id for result in results if result.display_polygon is None]
    if missing:
        raise ValueError(f"展示边界尚未生成: {missing}")
    document = {
        "schema_version": DISPLAY_CACHE_SCHEMA_VERSION,
        "dataset_fingerprint": dataset_fingerprint,
        "config_hash": config_hash,
        "calculation_geometry_fingerprint": calculation_geometry_fingerprint(results),
        "algorithm": {
            "method": DISPLAY_GEOMETRY_METHOD,
            "parameters": dict(parameters or DISPLAY_GEOMETRY_PARAMETERS),
        },
        "piles": [
            {
                "pile_id": result.pile_id,
                "display_geometry_projected": mapping(result.display_polygon),
                "display_geometry_metrics": result.display_geometry_metrics,
            }
            for result in results
        ],
    }
    atomic_write_json(path, document)


def load_display_geometry_cache(
    path: str | Path,
    results: list[PileResult],
    dataset_fingerprint: str,
    config_hash: str,
    parameters: dict | None = None,
) -> DisplayGeometryLoadResult:
    cache_path = Path(path)
    if not cache_path.exists():
        return DisplayGeometryLoadResult(False, "缓存文件不存在")
    try:
        document = json.loads(cache_path.read_text(encoding="utf-8"))
        if document.get("schema_version") != DISPLAY_CACHE_SCHEMA_VERSION:
            return DisplayGeometryLoadResult(False, "缓存版本不匹配")
        if document.get("dataset_fingerprint") != dataset_fingerprint:
            return DisplayGeometryLoadResult(False, "LAS 数据集已变化")
        if document.get("config_hash") != config_hash:
            return DisplayGeometryLoadResult(False, "算法配置已变化")
        algorithm = document.get("algorithm") or {}
        if algorithm.get("method") != DISPLAY_GEOMETRY_METHOD:
            return DisplayGeometryLoadResult(False, "展示曲线算法版本已变化")
        if algorithm.get("parameters") != dict(parameters or DISPLAY_GEOMETRY_PARAMETERS):
            return DisplayGeometryLoadResult(False, "展示曲线参数已变化")
        if document.get("calculation_geometry_fingerprint") != calculation_geometry_fingerprint(results):
            return DisplayGeometryLoadResult(False, "本次计算边界已变化")
        _apply_entries(results, document.get("piles") or [])
        return DisplayGeometryLoadResult(True, "有效缓存", "display-cache", len(results))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return DisplayGeometryLoadResult(False, f"缓存无效: {type(exc).__name__}: {exc}")


def _project_lonlat_polygon(payload: dict, crs_config) -> Polygon:
    source = _valid_polygon(payload)
    rings = []
    for ring in [source.exterior, *source.interiors]:
        coordinates = np.asarray(ring.coords, dtype=np.float64)
        x, y = lonlat_to_projected(
            coordinates[:, 0], coordinates[:, 1],
            crs_config.central_meridian_deg,
            crs_config.false_easting_m,
            crs_config.false_northing_m,
        )
        rings.append(np.column_stack([x, y]))
    polygon = Polygon(rings[0], rings[1:])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if not isinstance(polygon, Polygon) or polygon.is_empty or not polygon.is_valid:
        raise ValueError("经纬度边界无法还原为有效投影 Polygon")
    return polygon


def _same_calculation_polygon(current: Polygon, previous: Polygon) -> bool:
    area_delta = abs(current.area - previous.area) / max(current.area, 1e-9)
    return area_delta <= 0.0001 and current.boundary.hausdorff_distance(previous.boundary) <= 0.05


def load_display_geometry_from_result(
    path: str | Path,
    results: list[PileResult],
    dataset_fingerprint: str,
    config_hash: str,
    crs_config,
    parameters: dict | None = None,
) -> DisplayGeometryLoadResult:
    """Migrate display curves from a compatible pre-cache result document."""
    result_path = Path(path)
    if not result_path.exists():
        return DisplayGeometryLoadResult(False, "历史成果不存在")
    try:
        document = json.loads(result_path.read_text(encoding="utf-8"))
        metadata = document.get("metadata") or {}
        if metadata.get("dataset_fingerprint") != dataset_fingerprint:
            return DisplayGeometryLoadResult(False, "历史成果对应的 LAS 数据集已变化")
        # Older local results may carry a legacy config hash that included the
        # per-run output path.  Exact dataset, method, pile IDs and projected
        # calculation polygons below are the authoritative migration checks.
        display_metadata = metadata.get("display_geometry") or {}
        if display_metadata.get("method") != DISPLAY_GEOMETRY_METHOD:
            return DisplayGeometryLoadResult(False, "历史成果不含当前版本展示曲线")
        # 参数（含贴地策略）必须一致，否则历史成果里的旧曲线会被当成有效缓存，
        # 导致调整参数后看不到变化。
        expected_parameters = dict(parameters or DISPLAY_GEOMETRY_PARAMETERS)
        if any(display_metadata.get(key) != value for key, value in expected_parameters.items()):
            return DisplayGeometryLoadResult(False, "历史成果的展示曲线参数与当前配置不一致")
        entries = {int(item["pile_id"]): item for item in document.get("piles") or []}
        if set(entries) != {result.pile_id for result in results}:
            return DisplayGeometryLoadResult(False, "历史成果煤堆编号不一致")
        loaded: dict[int, tuple[Polygon, dict | None]] = {}
        for result in results:
            item = entries[result.pile_id]
            previous = _project_lonlat_polygon(item["geometry"], crs_config)
            if not _same_calculation_polygon(result.polygon, previous):
                return DisplayGeometryLoadResult(False, "历史成果计算边界已变化")
            loaded[result.pile_id] = (
                _project_lonlat_polygon(item["display_geometry"], crs_config),
                item.get("display_geometry_metrics"),
            )
        for result in results:
            result.display_polygon, result.display_geometry_metrics = loaded[result.pile_id]
        return DisplayGeometryLoadResult(True, "已迁移历史成果", "previous-result", len(results))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        return DisplayGeometryLoadResult(False, f"历史成果无效: {type(exc).__name__}: {exc}")
