from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from scipy.ndimage import gaussian_filter1d


@dataclass(frozen=True)
class DisplayGeometryMetrics:
    area_delta_percent: float
    hausdorff_m: float
    source_vertices: int
    display_vertices: int
    smoothing_sigma_m: float
    fallback_used: bool


def _largest_polygon(value) -> Polygon | None:
    if isinstance(value, Polygon):
        return value
    if isinstance(value, MultiPolygon) and value.geoms:
        return max(value.geoms, key=lambda item: item.area)
    return None


def _resample_ring(coords, spacing_m: float) -> np.ndarray:
    points = np.asarray(coords, dtype=np.float64)
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    following = np.roll(points, -1, axis=0)
    vectors = following - points
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths > 1e-9
    points, vectors, lengths = points[keep], vectors[keep], lengths[keep]
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = float(cumulative[-1])
    count = max(4, int(math.ceil(total / max(spacing_m, 1e-6))))
    distances = np.arange(count, dtype=np.float64) * (total / count)
    indices = np.searchsorted(cumulative[1:], distances, side="right")
    fraction = (distances - cumulative[indices]) / lengths[indices]
    return points[indices] + vectors[indices] * fraction[:, None]


def _smooth_ring(coords, sample_spacing_m: float, sigma_m: float, output_spacing_m: float) -> np.ndarray:
    sampled = _resample_ring(coords, sample_spacing_m)
    sigma_samples = sigma_m / sample_spacing_m
    smoothed = np.column_stack([
        gaussian_filter1d(sampled[:, axis], sigma_samples, mode="wrap")
        for axis in range(2)
    ])
    return _resample_ring(np.vstack([smoothed, smoothed[0]]), output_spacing_m)


def _smooth_candidate(
    source: Polygon,
    sample_spacing_m: float,
    sigma_m: float,
    output_spacing_m: float,
) -> Polygon | None:
    exterior = _smooth_ring(source.exterior.coords, sample_spacing_m, sigma_m, output_spacing_m)
    holes = [_smooth_ring(interior.coords, sample_spacing_m, sigma_m, output_spacing_m)
             for interior in source.interiors]
    candidate = Polygon(exterior, holes)
    if not candidate.is_valid:
        candidate = candidate.buffer(0)
    candidate = _largest_polygon(candidate)
    if candidate is None or candidate.is_empty or candidate.area <= 0:
        return None
    # Corner cutting changes area slightly. Recover the source area before applying
    # the displacement/area safety gates so presentation never changes accounting.
    scale = math.sqrt(source.area / candidate.area)
    center = source.representative_point()
    candidate = affinity.scale(candidate, xfact=scale, yfact=scale, origin=(center.x, center.y))
    if not candidate.is_valid:
        candidate = candidate.buffer(0)
    return _largest_polygon(candidate)


def smooth_display_polygon(
    source: Polygon,
    *,
    max_offset_m: float = 0.5,
    max_area_delta_ratio: float = 0.005,
    sample_spacing_m: float = 0.25,
    output_spacing_m: float = 0.4,
) -> tuple[Polygon, DisplayGeometryMetrics]:
    """Create a dense visual boundary while preserving the audited source polygon.

    Candidates are tried from visually smoother to more conservative. A candidate
    is accepted only when it remains topologically valid, stays inside a symmetric
    max-offset corridor, and keeps area within the configured tolerance.
    """
    if source.is_empty or not source.is_valid or source.area <= 0:
        raise ValueError("source polygon must be valid and non-empty")
    source_vertices = len(source.exterior.coords)
    trials = (2.0, 1.5, 1.2, 1.0, 0.75, 0.5)
    corridor = source.buffer(max_offset_m, join_style="round")
    for sigma_m in trials:
        candidate = _smooth_candidate(source, sample_spacing_m, sigma_m, output_spacing_m)
        if candidate is None or not candidate.within(corridor):
            continue
        hausdorff = float(source.boundary.hausdorff_distance(candidate.boundary))
        area_delta = abs(candidate.area - source.area) / source.area
        if hausdorff <= max_offset_m + 1e-9 and area_delta <= max_area_delta_ratio + 1e-12:
            return candidate, DisplayGeometryMetrics(
                area_delta_percent=(candidate.area - source.area) / source.area * 100.0,
                hausdorff_m=hausdorff,
                source_vertices=source_vertices,
                display_vertices=len(candidate.exterior.coords),
                smoothing_sigma_m=sigma_m,
                fallback_used=False,
            )
    return source, DisplayGeometryMetrics(
        area_delta_percent=0.0,
        hausdorff_m=0.0,
        source_vertices=source_vertices,
        display_vertices=source_vertices,
        smoothing_sigma_m=0.0,
        fallback_used=True,
    )
