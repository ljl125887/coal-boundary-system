from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable

import numpy as np
import shapely
from scipy import ndimage as ndi
from scipy.interpolate import splprep, splev
from shapely.geometry import MultiPolygon, Polygon
from skimage.measure import find_contours

from .las import LasHeader, iter_las_xyz, iter_las_xyz_in_bounds


@dataclass(frozen=True)
class RefinedBoundaryDiagnostics:
    selected_las_files: int
    selected_points: int
    fine_cell_size_m: float
    height_threshold_m: float
    source_area_m2: float
    refined_area_m2: float
    area_delta_percent: float
    source_vertices: int
    evidence_vertices: int
    display_vertices: int
    evidence_to_display_hausdorff_m: float
    source_to_display_hausdorff_m: float


@dataclass(frozen=True)
class FineSurfaceEvidence:
    x_min: float
    y_min: float
    cell_size: float
    relative_height: np.ndarray
    observed: np.ndarray

    @property
    def extent(self) -> tuple[float, float, float, float]:
        rows, cols = self.relative_height.shape
        return (
            self.x_min,
            self.x_min + cols * self.cell_size,
            self.y_min,
            self.y_min + rows * self.cell_size,
        )


@dataclass
class _FineAccumulator:
    x_min: float
    y_min: float
    rows: int
    cols: int
    cell_size: float
    count: np.ndarray
    sum_z: np.ndarray
    selected_las_files: int = 0
    selected_points: int = 0


def _largest_polygon(value) -> Polygon | None:
    if isinstance(value, Polygon):
        return value
    if isinstance(value, MultiPolygon) and value.geoms:
        return max(value.geoms, key=lambda item: item.area)
    return None


def _sample_ground(
    ground: np.ndarray,
    ground_x_min: float,
    ground_y_min: float,
    ground_cell_size: float,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    # Surface-grid values represent cell centres.
    col = (x - ground_x_min) / ground_cell_size - 0.5
    row = (y - ground_y_min) / ground_cell_size - 0.5
    return ndi.map_coordinates(ground, [row, col], order=1, mode="nearest")


def build_fine_surface_evidence(
    source: Polygon,
    headers: list[LasHeader],
    ground: np.ndarray,
    ground_x_min: float,
    ground_y_min: float,
    ground_cell_size: float,
    *,
    fine_cell_size_m: float = 0.20,
    search_margin_m: float = 3.0,
    minimum_z_m: float = -20.0,
    maximum_z_m: float = 60.0,
    chunk_points: int = 1_500_000,
) -> tuple[FineSurfaceEvidence, int, int]:
    min_x, min_y, max_x, max_y = source.buffer(search_margin_m).bounds
    x_min = math.floor(min_x / fine_cell_size_m) * fine_cell_size_m
    y_min = math.floor(min_y / fine_cell_size_m) * fine_cell_size_m
    cols = int(math.ceil((max_x - x_min) / fine_cell_size_m)) + 1
    rows = int(math.ceil((max_y - y_min) / fine_cell_size_m)) + 1
    n_cells = rows * cols
    count = np.zeros(n_cells, dtype=np.uint32)
    sum_z = np.zeros(n_cells, dtype=np.float64)
    selected_headers = [
        header for header in headers
        if header.maxs[0] >= min_x and header.mins[0] <= max_x
        and header.maxs[1] >= min_y and header.mins[1] <= max_y
    ]
    selected_points = 0
    bounds = (min_x, min_y, max_x, max_y)
    for header in selected_headers:
        for x, y, z in iter_las_xyz_in_bounds(header, bounds, chunk_points):
            valid = np.isfinite(z) & (z >= minimum_z_m) & (z <= maximum_z_m)
            if not np.any(valid):
                continue
            x, y, z = x[valid], y[valid], z[valid]
            col = np.floor((x - x_min) / fine_cell_size_m).astype(np.int64)
            row = np.floor((y - y_min) / fine_cell_size_m).astype(np.int64)
            inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
            if not np.any(inside):
                continue
            flat = row[inside] * cols + col[inside]
            values = z[inside].astype(np.float64)
            count += np.bincount(flat, minlength=n_cells).astype(np.uint32)
            sum_z += np.bincount(flat, weights=values, minlength=n_cells)
            selected_points += int(values.size)
    observed = count.reshape(rows, cols) > 0
    if not np.any(observed):
        raise ValueError("煤堆坡脚窄带内没有有效原始点云")
    mean_z = np.full(n_cells, np.nan, dtype=np.float64)
    present = count > 0
    mean_z[present] = sum_z[present] / count[present]
    mean_z = mean_z.reshape(rows, cols)
    x_centres = x_min + (np.arange(cols, dtype=np.float64) + 0.5) * fine_cell_size_m
    y_centres = y_min + (np.arange(rows, dtype=np.float64) + 0.5) * fine_cell_size_m
    xx, yy = np.meshgrid(x_centres, y_centres)
    local_ground = _sample_ground(
        ground, ground_x_min, ground_y_min, ground_cell_size, xx, yy,
    )
    relative_height = mean_z - local_ground
    return (
        FineSurfaceEvidence(x_min, y_min, fine_cell_size_m, relative_height, observed),
        len(selected_headers),
        selected_points,
    )


def _polygon_from_contour(
    contour: np.ndarray,
    x_min: float,
    y_min: float,
    cell_size: float,
) -> Polygon | None:
    if len(contour) < 4:
        return None
    x = x_min + (contour[:, 1] + 0.5) * cell_size
    y = y_min + (contour[:, 0] + 0.5) * cell_size
    polygon = Polygon(np.column_stack([x, y]))
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    polygon = _largest_polygon(polygon)
    return polygon if polygon is not None and not polygon.is_empty else None


def _select_evidence_polygon(
    contours: list[np.ndarray],
    source: Polygon,
    evidence: FineSurfaceEvidence,
    *,
    corridor: Polygon | None = None,
    mask_area_ratio: float = 0.97,
) -> Polygon:
    candidates: list[tuple[float, Polygon]] = []
    source_corridor = corridor if corridor is not None else source.buffer(2.0, join_style="round")
    for contour in contours:
        candidate = _polygon_from_contour(
            contour, evidence.x_min, evidence.y_min, evidence.cell_size,
        )
        if candidate is None or candidate.area < max(20.0, source.area * 0.20):
            continue
        # 纯掩膜边界假轮廓：裁剪掩膜外填了 -1000，若某 level 低于掩膜边缘处的场值，
        # 等高线会沿掩膜边界闭合成一圈“假坡脚”（半径≈掩膜半径，与 level 无关，
        # 面积接近整条走廊）。这类轮廓会把“贴地优先”引到错误结果，必须剔除。
        if candidate.area >= source_corridor.area * mask_area_ratio:
            continue
        overlap = candidate.intersection(source).area / max(candidate.union(source).area, 1e-9)
        corridor_fraction = candidate.intersection(source_corridor).area / max(candidate.area, 1e-9)
        if corridor_fraction < 0.96:
            continue
        score = overlap + (0.15 if candidate.contains(source.representative_point()) else 0.0)
        candidates.append((score, candidate))
    if not candidates:
        raise ValueError("细粒度坡脚等高线未形成与原煤堆匹配的闭合边界")
    return max(candidates, key=lambda item: item[0])[1]


def _continuous_display_curve(
    evidence_polygon: Polygon,
    *,
    smoothing_m: float = 0.06,
    output_spacing_m: float = 0.15,
) -> Polygon:
    points = np.asarray(evidence_polygon.exterior.coords[:-1], dtype=np.float64)
    if len(points) < 8:
        return evidence_polygon
    # Parameterize from the first vertex along the ring.  Using the closing edge
    # as the first increment leaves u[0] > 0 and can make FITPACK reject an
    # otherwise valid periodic curve.
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    u = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    u = u / max(u[-1], 1e-9)
    try:
        spline, _ = splprep(
            [points[:, 0], points[:, 1]], u=u, per=True, k=3,
            s=len(points) * smoothing_m**2,
        )
        count = max(32, int(math.ceil(evidence_polygon.length / output_spacing_m)))
        sampled = np.linspace(0.0, 1.0, count, endpoint=False)
        x, y = splev(sampled, spline)
        candidate = Polygon(np.column_stack([x, y]))
        if not candidate.is_valid:
            candidate = candidate.buffer(0)
        candidate = _largest_polygon(candidate)
        if candidate is None or candidate.is_empty:
            return evidence_polygon
        if candidate.boundary.hausdorff_distance(evidence_polygon.boundary) > 0.35:
            return evidence_polygon
        return candidate
    except (TypeError, ValueError):
        return evidence_polygon


def refine_polygon_from_point_cloud(
    source: Polygon,
    headers: list[LasHeader],
    ground: np.ndarray,
    ground_x_min: float,
    ground_y_min: float,
    ground_cell_size: float,
    *,
    fine_cell_size_m: float = 0.20,
    height_threshold_m: float = 0.35,
    field_smoothing_m: float = 0.50,
    max_gap_fill_m: float = 0.60,
) -> tuple[Polygon, FineSurfaceEvidence, RefinedBoundaryDiagnostics]:
    evidence, las_count, point_count = build_fine_surface_evidence(
        source, headers, ground, ground_x_min, ground_y_min, ground_cell_size,
        fine_cell_size_m=fine_cell_size_m,
    )
    return _refine_evidence(
        source, evidence, las_count, point_count,
        height_threshold_m=height_threshold_m,
        field_smoothing_m=field_smoothing_m,
        max_gap_fill_m=max_gap_fill_m,
        toe_height_floor_m=toe_height_floor_m,
        area_tolerance_percent=area_tolerance_percent,
        selection_strategy=selection_strategy,
        corridor_buffer_m=corridor_buffer_m,
        ground_offset_inner_m=ground_offset_inner_m,
        ground_offset_outer_m=ground_offset_outer_m,
    )


def _fine_grid_coordinates(evidence: FineSurfaceEvidence) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = evidence.relative_height.shape
    return np.meshgrid(
        evidence.x_min + (np.arange(cols) + 0.5) * evidence.cell_size,
        evidence.y_min + (np.arange(rows) + 0.5) * evidence.cell_size,
    )


def estimate_ground_offset(
    evidence: FineSurfaceEvidence,
    source: Polygon,
    *,
    inner_m: float = 1.0,
    outer_m: float = 3.0,
) -> float:
    """估算“地面零点”偏置（用 source 外侧环带相对高度的中位数）。

    地面估计算法（大半径形态学开运算）会把煤堆外围地面压低，使“煤堆外的地面”
    相对高度约为 +0.3 m 而不是 0。不校正时找不到 0 m 等高线，只能靠抬升 level
    才能圈住边界，于是展示边界浮在煤堆侧面（看起来不贴地面）。
    """
    xx, yy = _fine_grid_coordinates(evidence)
    band = source.buffer(outer_m).difference(source.buffer(inner_m))
    selected = shapely.contains_xy(band, xx, yy) & evidence.observed
    values = evidence.relative_height[selected]
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else 0.0


def _refine_evidence(
    source: Polygon,
    evidence: FineSurfaceEvidence,
    las_count: int,
    point_count: int,
    *,
    height_threshold_m: float = 0.35,
    field_smoothing_m: float = 0.50,
    max_gap_fill_m: float = 0.60,
    toe_height_floor_m: float = 0.05,
    area_tolerance_percent: float = 15.0,
    selection_strategy: str = "toe_first",
    corridor_buffer_m: float = 2.8,
    ground_offset_inner_m: float = 1.0,
    ground_offset_outer_m: float = 3.0,
) -> tuple[Polygon, FineSurfaceEvidence, RefinedBoundaryDiagnostics]:
    fine_cell_size_m = evidence.cell_size
    observed = evidence.observed
    # 先把“地面”归零：否则煤堆外围地面本身就有约 +0.3 m 相对高度，
    # 低 level 无法形成真实坡脚轮廓。
    ground_offset = estimate_ground_offset(
        evidence, source, inner_m=ground_offset_inner_m, outer_m=ground_offset_outer_m,
    )
    values = np.where(observed, evidence.relative_height - ground_offset, 0.0)
    sigma = field_smoothing_m / fine_cell_size_m
    weights = ndi.gaussian_filter(observed.astype(np.float64), sigma=sigma, mode="nearest")
    smooth = ndi.gaussian_filter(values, sigma=sigma, mode="nearest") / np.maximum(weights, 1e-9)
    supported = ndi.distance_transform_edt(~observed) * fine_cell_size_m <= max_gap_fill_m
    rows, cols = smooth.shape
    xx, yy = np.meshgrid(
        evidence.x_min + (np.arange(cols) + 0.5) * fine_cell_size_m,
        evidence.y_min + (np.arange(rows) + 0.5) * fine_cell_size_m,
    )
    corridor = source.buffer(corridor_buffer_m, join_style="round")
    allowed = shapely.contains_xy(corridor, xx, yy) & supported
    contour_field = np.where(allowed, smooth, -1_000.0)
    # The toe (contact line between the pile slope and the surrounding ground)
    # is what the displayed outline should follow.  A contour at height h sits
    # h/tan(slope) horizontally away from that contact line: at h=0.5 m on a 15°
    # slope that is already ~1.9 m, which reads as "the outline floats above the
    # ground".  So the search range must start near the ground (toe_height_floor_m)
    # and the selector must not trade height for area:
    #
    # * toe_first (default) - among contours whose area stays within
    #   area_tolerance_percent of the approved pile identity, take the lowest
    #   (closest to the ground) one.  Area/volume accounting is unaffected
    #   because it always uses the audited `geometry`, never this outline.
    # * area_first - legacy behaviour: minimise the area difference only, which
    #   is what pushed the outline onto a 0.25-0.90 m high contour.
    levels = np.unique(np.concatenate((
        np.linspace(float(toe_height_floor_m), 0.90, 45, dtype=np.float64),
        np.asarray([height_threshold_m], dtype=np.float64),
    )))
    candidates: list[tuple[float, float, float, Polygon]] = []
    for level in levels:
        try:
            polygon = _select_evidence_polygon(
                find_contours(contour_field, float(level)), source, evidence,
                corridor=corridor,
            )
        except ValueError:
            continue
        area_delta = abs(polygon.area - source.area) / source.area
        overlap = polygon.intersection(source).area / max(polygon.union(source).area, 1e-9)
        candidates.append((area_delta, -overlap, float(level), polygon))
    if not candidates:
        raise ValueError("自适应坡脚高度范围内没有形成有效闭合边界")
    if selection_strategy == "area_first":
        _, _, selected_level, evidence_polygon = min(candidates, key=lambda item: item[:2])
    else:
        tolerance = max(float(area_tolerance_percent), 0.0) / 100.0
        acceptable = [item for item in candidates if item[0] <= tolerance]
        _, _, selected_level, evidence_polygon = min(
            acceptable or candidates, key=lambda item: (item[2], item[0]),
        )
    display = _continuous_display_curve(evidence_polygon)
    diagnostics = RefinedBoundaryDiagnostics(
        selected_las_files=las_count,
        selected_points=point_count,
        fine_cell_size_m=fine_cell_size_m,
        height_threshold_m=selected_level,
        source_area_m2=float(source.area),
        refined_area_m2=float(display.area),
        area_delta_percent=float((display.area - source.area) / source.area * 100.0),
        source_vertices=len(source.exterior.coords),
        evidence_vertices=len(evidence_polygon.exterior.coords),
        display_vertices=len(display.exterior.coords),
        evidence_to_display_hausdorff_m=float(
            evidence_polygon.boundary.hausdorff_distance(display.boundary)
        ),
        source_to_display_hausdorff_m=float(source.boundary.hausdorff_distance(display.boundary)),
    )
    return display, FineSurfaceEvidence(
        evidence.x_min, evidence.y_min, evidence.cell_size, smooth, observed,
    ), diagnostics


def _new_accumulator(
    source: Polygon,
    fine_cell_size_m: float,
    search_margin_m: float,
) -> _FineAccumulator:
    min_x, min_y, max_x, max_y = source.buffer(search_margin_m).bounds
    x_min = math.floor(min_x / fine_cell_size_m) * fine_cell_size_m
    y_min = math.floor(min_y / fine_cell_size_m) * fine_cell_size_m
    cols = int(math.ceil((max_x - x_min) / fine_cell_size_m)) + 1
    rows = int(math.ceil((max_y - y_min) / fine_cell_size_m)) + 1
    return _FineAccumulator(
        x_min=x_min,
        y_min=y_min,
        rows=rows,
        cols=cols,
        cell_size=fine_cell_size_m,
        count=np.zeros(rows * cols, dtype=np.uint32),
        sum_z=np.zeros(rows * cols, dtype=np.float64),
    )


def _evidence_from_accumulator(
    accumulator: _FineAccumulator,
    ground: np.ndarray,
    ground_x_min: float,
    ground_y_min: float,
    ground_cell_size: float,
) -> FineSurfaceEvidence:
    observed = accumulator.count.reshape(accumulator.rows, accumulator.cols) > 0
    if not np.any(observed):
        raise ValueError("煤堆坡脚窄带内没有有效原始点云")
    mean_z = np.full(accumulator.count.size, np.nan, dtype=np.float64)
    present = accumulator.count > 0
    mean_z[present] = accumulator.sum_z[present] / accumulator.count[present]
    mean_z = mean_z.reshape(accumulator.rows, accumulator.cols)
    x_centres = accumulator.x_min + (
        np.arange(accumulator.cols, dtype=np.float64) + 0.5
    ) * accumulator.cell_size
    y_centres = accumulator.y_min + (
        np.arange(accumulator.rows, dtype=np.float64) + 0.5
    ) * accumulator.cell_size
    xx, yy = np.meshgrid(x_centres, y_centres)
    local_ground = _sample_ground(
        ground, ground_x_min, ground_y_min, ground_cell_size, xx, yy,
    )
    return FineSurfaceEvidence(
        accumulator.x_min,
        accumulator.y_min,
        accumulator.cell_size,
        mean_z - local_ground,
        observed,
    )


def refine_polygons_from_point_cloud(
    sources: dict[int, Polygon],
    headers: list[LasHeader],
    ground: np.ndarray,
    labels: np.ndarray,
    grid_x_min: float,
    grid_y_min: float,
    grid_cell_size: float,
    *,
    fine_cell_size_m: float = 0.20,
    search_margin_m: float = 3.0,
    minimum_z_m: float = -20.0,
    maximum_z_m: float = 60.0,
    chunk_points: int = 1_500_000,
    progress: Callable[[int, int, Path], None] | None = None,
    height_threshold_m: float = 0.35,
    field_smoothing_m: float = 0.50,
    max_gap_fill_m: float = 0.60,
    toe_height_floor_m: float = 0.05,
    area_tolerance_percent: float = 15.0,
    selection_strategy: str = "toe_first",
    corridor_buffer_m: float = 2.8,
    ground_offset_inner_m: float = 1.0,
    ground_offset_outer_m: float = 3.0,
) -> dict[int, tuple[Polygon, FineSurfaceEvidence, RefinedBoundaryDiagnostics]]:
    """Refine many pile boundaries with one sequential LAS pass.

    The existing label raster is expanded by ``search_margin_m`` using a
    nearest-label Voronoi assignment.  It routes each raw point to one current
    pile, so close piles remain separate and every LAS file is decoded once.
    """
    if not sources:
        return {}
    accumulators = {
        pile_id: _new_accumulator(source, fine_cell_size_m, search_margin_m)
        for pile_id, source in sources.items()
    }
    source_ids = np.zeros(int(labels.max()) + 1, dtype=bool)
    for pile_id in sources:
        if 0 < pile_id < len(source_ids):
            source_ids[pile_id] = True
    outside = labels == 0
    distance, nearest = ndi.distance_transform_edt(
        outside, sampling=grid_cell_size, return_indices=True,
    )
    owner = labels[nearest[0], nearest[1]].copy()
    owner[(distance > search_margin_m + grid_cell_size * math.sqrt(2.0))] = 0
    owner[(owner < 0) | (owner >= len(source_ids))] = 0
    owner[~source_ids[owner]] = 0

    for pile_id, source in sources.items():
        bounds = source.buffer(search_margin_m).bounds
        accumulators[pile_id].selected_las_files = sum(
            header.maxs[0] >= bounds[0] and header.mins[0] <= bounds[2]
            and header.maxs[1] >= bounds[1] and header.mins[1] <= bounds[3]
            for header in headers
        )

    rows, cols = labels.shape
    for file_index, header in enumerate(headers, 1):
        for x, y, z in iter_las_xyz(header, chunk_points):
            valid = np.isfinite(z) & (z >= minimum_z_m) & (z <= maximum_z_m)
            col = np.floor((x - grid_x_min) / grid_cell_size).astype(np.int64)
            row = np.floor((y - grid_y_min) / grid_cell_size).astype(np.int64)
            valid &= (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
            if not np.any(valid):
                continue
            x, y, z, row, col = x[valid], y[valid], z[valid], row[valid], col[valid]
            point_owner = owner[row, col]
            keep = point_owner > 0
            if not np.any(keep):
                continue
            x, y, z, point_owner = x[keep], y[keep], z[keep], point_owner[keep]
            for raw_pile_id in np.unique(point_owner):
                pile_id = int(raw_pile_id)
                accumulator = accumulators.get(pile_id)
                if accumulator is None:
                    continue
                selected = point_owner == pile_id
                local_col = np.floor(
                    (x[selected] - accumulator.x_min) / accumulator.cell_size
                ).astype(np.int64)
                local_row = np.floor(
                    (y[selected] - accumulator.y_min) / accumulator.cell_size
                ).astype(np.int64)
                inside = (
                    (local_row >= 0) & (local_row < accumulator.rows)
                    & (local_col >= 0) & (local_col < accumulator.cols)
                )
                if not np.any(inside):
                    continue
                flat = local_row[inside] * accumulator.cols + local_col[inside]
                local_z = z[selected][inside].astype(np.float64)
                accumulator.count += np.bincount(
                    flat, minlength=accumulator.count.size,
                ).astype(np.uint32)
                accumulator.sum_z += np.bincount(
                    flat, weights=local_z, minlength=accumulator.count.size,
                )
                accumulator.selected_points += int(local_z.size)
        if progress:
            progress(file_index, len(headers), header.path)

    refined = {}
    for pile_id, source in sources.items():
        accumulator = accumulators[pile_id]
        evidence = _evidence_from_accumulator(
            accumulator, ground, grid_x_min, grid_y_min, grid_cell_size,
        )
        refined[pile_id] = _refine_evidence(
            source,
            evidence,
            accumulator.selected_las_files,
            accumulator.selected_points,
            height_threshold_m=height_threshold_m,
            field_smoothing_m=field_smoothing_m,
            max_gap_fill_m=max_gap_fill_m,
            toe_height_floor_m=toe_height_floor_m,
            area_tolerance_percent=area_tolerance_percent,
            selection_strategy=selection_strategy,
            corridor_buffer_m=corridor_buffer_m,
            ground_offset_inner_m=ground_offset_inner_m,
            ground_offset_outer_m=ground_offset_outer_m,
        )
    return refined
