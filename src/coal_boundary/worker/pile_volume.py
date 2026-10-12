from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon, shape

from ..config import AppConfig
from ..crs import lonlat_to_projected
from ..export import atomic_write_json
from ..ground import estimate_ground_surface, fill_nearest
from ..integration.models import sha256_json, utc_now
from ..las import dataset_fingerprint, read_las_header
from ..pile_volume_validation import validate_pile_volume_result
from ..raster import SurfaceGrid, build_surface_grid_for_bounds


@dataclass(frozen=True)
class RecalculationOutput:
    result: dict
    path: Path
    cache_reused: bool


class PileRecalculationError(ValueError):
    """Stable failure code for a rejected single-pile recalculation."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def validate_recalculation_source(context: dict, source_result_path: Path, fingerprint: str, config: AppConfig) -> None:
    required = {
        "recalculation_id", "source_job_id", "source_run_id", "source_pile_id",
        "source_result_sha256", "dataset_fingerprint", "selection_mode", "geometry", "source_metrics",
    }
    missing = sorted(required - set(context))
    if missing:
        raise PileRecalculationError("INVALID_RECALCULATION_CONTEXT", f"重算上下文缺少字段: {missing}")
    if str(context["dataset_fingerprint"]).lower() != fingerprint.lower():
        raise PileRecalculationError("SOURCE_VERSION_CONFLICT", "LAS 数据指纹与源任务不一致")
    if str(context.get("geometry_crs", "EPSG:4326")).upper() != "EPSG:4326":
        raise PileRecalculationError("INVALID_SELECTION_GEOMETRY", "重算边界必须使用 EPSG:4326")
    if context["selection_mode"] not in {"EXISTING_BOUNDARY", "USER_DRAWN_BOUNDARY"}:
        raise PileRecalculationError("INVALID_RECALCULATION_CONTEXT", "未知 selection_mode")
    try:
        polygon = _project_polygon(context["geometry"], config)
        if not np.all(np.isfinite(polygon.bounds)):
            raise ValueError("边界坐标不是有限数值")
    except (ValueError, TypeError, KeyError) as exc:
        raise PileRecalculationError("INVALID_SELECTION_GEOMETRY", str(exc)) from exc
    if not source_result_path.is_file():
        raise PileRecalculationError("SOURCE_VERSION_CONFLICT", "本机源结果文件不存在")
    try:
        source_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PileRecalculationError("SOURCE_VERSION_CONFLICT", "本机源结果文件不可读取") from exc
    if sha256_json(source_result).lower() != str(context["source_result_sha256"]).lower():
        raise PileRecalculationError("SOURCE_VERSION_CONFLICT", "源结果 SHA256 与本机文件不一致")
    if str(source_result.get("metadata", {}).get("dataset_fingerprint", "")).lower() != fingerprint.lower():
        raise PileRecalculationError("SOURCE_VERSION_CONFLICT", "源结果的数据指纹与当前 LAS 不一致")
    source_pile = next((item for item in source_result.get("piles", [])
                        if item.get("pile_id") == int(context["source_pile_id"])), None)
    if source_pile is None:
        raise PileRecalculationError("SOURCE_VERSION_CONFLICT", "源结果中不存在指定煤堆")
    if context["selection_mode"] == "EXISTING_BOUNDARY":
        try:
            same_boundary = shape(source_pile["geometry"]).equals(shape(context["geometry"]))
        except (ValueError, TypeError, KeyError) as exc:
            raise PileRecalculationError("INVALID_SELECTION_GEOMETRY", "源煤堆边界无效") from exc
        if not same_boundary:
            raise PileRecalculationError("SOURCE_VERSION_CONFLICT", "复用原边界时，任务边界与源结果不一致")


def _project_polygon(geometry: dict, config: AppConfig) -> Polygon:
    try:
        source = shape(geometry)
    except (ValueError, TypeError, KeyError) as exc:
        raise PileRecalculationError("INVALID_SELECTION_GEOMETRY", "重算边界不是有效 GeoJSON") from exc
    if not isinstance(source, Polygon) or source.is_empty or not source.is_valid:
        raise PileRecalculationError("INVALID_SELECTION_GEOMETRY", "重算 geometry 必须是有效的非空 GeoJSON Polygon")
    rings = []
    for ring in [source.exterior, *source.interiors]:
        coords = np.asarray(ring.coords, dtype=np.float64)
        x, y = lonlat_to_projected(
            coords[:, 0], coords[:, 1], config.crs.central_meridian_deg,
            config.crs.false_easting_m, config.crs.false_northing_m,
        )
        rings.append(list(zip(x.tolist(), y.tolist())))
    projected = Polygon(rings[0], rings[1:])
    if not projected.is_valid or projected.area <= 0:
        raise PileRecalculationError("INVALID_SELECTION_GEOMETRY", "边界投影后无效，请检查坐标轴序和 CRS")
    return projected


def _crop_slices(grid: SurfaceGrid, bounds: tuple[float, float, float, float]) -> tuple[slice, slice]:
    min_x, min_y, max_x, max_y = bounds
    c0 = max(0, int(np.floor((min_x - grid.x_min) / grid.cell_size)))
    c1 = min(grid.cols, int(np.ceil((max_x - grid.x_min) / grid.cell_size)) + 1)
    r0 = max(0, int(np.floor((min_y - grid.y_min) / grid.cell_size)))
    r1 = min(grid.rows, int(np.ceil((max_y - grid.y_min) / grid.cell_size)) + 1)
    if r0 >= r1 or c0 >= c1:
        raise PileRecalculationError("SELECTION_OUTSIDE_DATASET", "目标边界不在源表面栅格范围内")
    return slice(r0, r1), slice(c0, c1)


def _crop_grid(grid: SurfaceGrid, bounds: tuple[float, float, float, float]) -> SurfaceGrid:
    row_slice, col_slice = _crop_slices(grid, bounds)
    r0, r1 = row_slice.start, row_slice.stop
    c0, c1 = col_slice.start, col_slice.stop
    return SurfaceGrid(
        x_min=grid.x_min + c0 * grid.cell_size, y_min=grid.y_min + r0 * grid.cell_size,
        cell_size=grid.cell_size, rows=r1 - r0, cols=c1 - c0,
        point_count=grid.point_count[r0:r1, c0:c1], mean_z=grid.mean_z[r0:r1, c0:c1],
        min_z=grid.min_z[r0:r1, c0:c1], max_z=grid.max_z[r0:r1, c0:c1],
        mean_rgb=grid.mean_rgb[r0:r1, c0:c1],
    )


def recalculate_pile_volume(
    config: AppConfig,
    *,
    las_files: tuple[Path, ...],
    output_dir: Path,
    context: dict,
    source_cache_path: Path | None = None,
    source_diagnostics_path: Path | None = None,
    reporter=None,
) -> RecalculationOutput:
    started = time.time()
    required = {
        "recalculation_id", "source_job_id", "source_run_id", "source_pile_id",
        "source_result_sha256", "dataset_fingerprint", "selection_mode", "geometry", "source_metrics",
    }
    missing = sorted(required - set(context))
    if missing:
        raise PileRecalculationError("INVALID_RECALCULATION_CONTEXT", f"重算上下文缺少字段: {missing}")
    headers = [read_las_header(path) for path in las_files]
    fingerprint = dataset_fingerprint(headers)
    if fingerprint.lower() != str(context["dataset_fingerprint"]).lower():
        raise PileRecalculationError("SOURCE_VERSION_CONFLICT", "LAS 数据指纹与源任务不一致")

    geometry = context["geometry"]
    polygon = _project_polygon(geometry, config)
    buffer_m = max(config.ground.opening_radius_m, 15.0) + config.ground.smoothing_sigma_m * 3.0
    bounds = polygon.buffer(buffer_m).bounds
    cache_reused = False
    ground_cache_reused = False
    cached_ground = None
    grid = None
    if source_cache_path and source_cache_path.is_file():
        candidate, metadata = SurfaceGrid.load(source_cache_path)
        if (
            metadata.get("dataset_fingerprint") == fingerprint
            and float(metadata.get("cell_size_m", -1)) == float(config.raster.cell_size_m)
        ):
            grid = _crop_grid(candidate, bounds)
            cache_reused = True
            if source_diagnostics_path and source_diagnostics_path.is_file():
                with np.load(source_diagnostics_path, allow_pickle=False) as diagnostic:
                    matches = (
                        str(diagnostic.get("dataset_fingerprint", "")) == fingerprint
                        and float(diagnostic.get("cell_size", -1)) == float(config.raster.cell_size_m)
                        and float(diagnostic.get("ground_opening_radius_m", -1)) == float(config.ground.opening_radius_m)
                        and float(diagnostic.get("ground_smoothing_sigma_m", -1)) == float(config.ground.smoothing_sigma_m)
                    )
                    if matches and "ground" in diagnostic:
                        row_slice, col_slice = _crop_slices(candidate, bounds)
                        cached_ground = diagnostic["ground"][row_slice, col_slice].astype(np.float32, copy=True)
                        ground_cache_reused = cached_ground.shape == grid.mean_z.shape
    if grid is None:
        try:
            grid = build_surface_grid_for_bounds(
                headers, bounds, config.raster.cell_size_m, config.raster.chunk_points,
                config.raster.minimum_z_m, config.raster.maximum_z_m,
            )
        except ValueError as exc:
            raise PileRecalculationError("SELECTION_OUTSIDE_DATASET", str(exc)) from exc
    if reporter:
        reporter.progress("RECALC_LOAD_LOCAL", 35.0, message="已读取目标边界及周边 LAS", context={"cache_reused": cache_reused})

    valid = grid.valid_mask
    mean_surface = fill_nearest(grid.mean_z, valid)
    ground = cached_ground if ground_cache_reused else estimate_ground_surface(
        grid.min_z, valid, grid.cell_size,
        config.ground.opening_radius_m, config.ground.smoothing_sigma_m,
    )
    relative = np.maximum(mean_surface - ground, 0.0)
    rr, cc = np.indices((grid.rows, grid.cols))
    xs, ys = grid.xy_from_rc(rr, cc)
    inside = contains_xy(polygon, xs, ys)
    measured_inside = inside & valid
    inside_count = int(np.count_nonzero(inside))
    valid_count = int(np.count_nonzero(measured_inside))
    if valid_count == 0:
        raise PileRecalculationError("SELECTION_OUTSIDE_DATASET", "目标边界内没有有效点云栅格")
    heights = relative[measured_inside]
    volume = float(np.sum(heights, dtype=np.float64) * grid.cell_size**2)
    point_count = int(np.sum(grid.point_count[measured_inside], dtype=np.int64))
    coverage = min(1.0, valid_count / max(inside_count, 1))
    ring = polygon.buffer(max(5.0, config.ground.smoothing_sigma_m * 2.0)).difference(polygon)
    ring_mask = contains_xy(ring, xs, ys) & valid
    ring_cells = int(np.count_nonzero(ring_mask))
    ground_quality = min(1.0, ring_cells / max(30.0, polygon.length / max(grid.cell_size, 0.1)))
    quality_score = float(np.clip(0.65 * coverage + 0.35 * ground_quality, 0.0, 1.0))
    warnings = []
    if coverage < 0.9:
        warnings.append("边界内点云栅格覆盖率低于 90%")
    if ground_quality < 0.7:
        warnings.append("边界周围可用于坡地曲面估计的裸露地面样本偏少")
    grade = "A" if quality_score >= 0.8 else "B" if quality_score >= 0.6 else "C"
    source_area = float(context["source_metrics"]["area_m2"])
    source_volume = float(context["source_metrics"]["volume_m3"])
    area = float(polygon.area)
    volume_delta = volume - source_volume
    result = {
        "result_kind": "PILE_VOLUME_RESULT",
        "schema_version": "1.0",
        "created_at": utc_now(),
        "recalculation_id": str(context["recalculation_id"]),
        "source_job_id": str(context["source_job_id"]),
        "source_run_id": str(context["source_run_id"]),
        "source_pile_id": int(context["source_pile_id"]),
        # 后端 P1 接口说明（2026-09-17）按扁平字段读取指标与质量分，这里与嵌套结构并存，
        # 保证 GET /pile-recalculations/{id} 能回显面积、体积、质量分、等级和像素数。
        "pile_id": int(context["source_pile_id"]),
        "quality_score": quality_score,
        "quality_grade": grade,
        "pixel_count": valid_count,
        "source_result_sha256": str(context["source_result_sha256"]),
        "dataset_fingerprint": fingerprint,
        "selection_mode": str(context["selection_mode"]),
        "crs": {"source_epsg": config.crs.source_epsg, "output_epsg": 4326},
        "geometry": geometry,
        "metrics": {
            "area_m2": area, "volume_m3": volume,
            "max_relative_height_m": float(np.max(heights)),
            "mean_relative_height_m": float(np.mean(heights)),
            "point_count": point_count, "valid_cell_count": valid_count,
        },
        "ground_surface": {
            "method": "source_run_ground_surface" if ground_cache_reused else "local_sloped_surface",
            "cache_reused": cache_reused, "quality_score": float(ground_quality),
            "surface_cache_reused": cache_reused, "ground_cache_reused": ground_cache_reused,
            "buffer_m": float(buffer_m), "ground_ring_valid_cells": ring_cells,
        },
        "quality": {"score": quality_score, "grade": grade, "warnings": warnings},
        "comparison": {
            "source_area_m2": source_area, "source_volume_m3": source_volume,
            "area_delta_m2": area - source_area, "volume_delta_m3": volume_delta,
            "volume_delta_percent": None if source_volume == 0 else volume_delta / source_volume * 100.0,
        },
        "runtime_seconds": float(time.time() - started),
    }
    validate_pile_volume_result(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"pile_{int(context['source_pile_id'])}_volume_recalculation.json"
    atomic_write_json(path, result)
    return RecalculationOutput(result=result, path=path, cache_reused=cache_reused)
