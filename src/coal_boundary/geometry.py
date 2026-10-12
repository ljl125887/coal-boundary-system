from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from scipy import ndimage as ndi
from shapely.geometry import Polygon, MultiPolygon


@dataclass
class PileResult:
    pile_id: int
    polygon: Polygon
    volume_m3: float
    area_m2: float
    max_relative_height_m: float
    mean_relative_height_m: float
    quality_score: float
    quality_grade: str
    boundary_confidence: float
    pixel_count: int
    display_polygon: Polygon | None = None
    display_geometry_metrics: dict | None = None


def _edge_segments(mask: np.ndarray, x_min: float, y_min: float, cell: float) -> np.ndarray:
    segments = []
    left = mask & ~np.pad(mask[:, :-1], ((0, 0), (1, 0)), constant_values=False)
    right = mask & ~np.pad(mask[:, 1:], ((0, 0), (0, 1)), constant_values=False)
    bottom = mask & ~np.pad(mask[:-1, :], ((1, 0), (0, 0)), constant_values=False)
    top = mask & ~np.pad(mask[1:, :], ((0, 1), (0, 0)), constant_values=False)
    for edge, kind in ((left, "left"), (right, "right"), (bottom, "bottom"), (top, "top")):
        rows, cols = np.nonzero(edge)
        x0 = x_min + cols * cell
        y0 = y_min + rows * cell
        if kind == "left": coords = np.stack([np.column_stack([x0, y0]), np.column_stack([x0, y0 + cell])], axis=1)
        elif kind == "right": coords = np.stack([np.column_stack([x0 + cell, y0]), np.column_stack([x0 + cell, y0 + cell])], axis=1)
        elif kind == "bottom": coords = np.stack([np.column_stack([x0, y0]), np.column_stack([x0 + cell, y0])], axis=1)
        else: coords = np.stack([np.column_stack([x0, y0 + cell]), np.column_stack([x0 + cell, y0 + cell])], axis=1)
        if len(coords): segments.append(coords)
    return np.concatenate(segments, axis=0) if segments else np.empty((0, 2, 2), dtype=np.float64)


def polygon_from_label(
    labels: np.ndarray,
    label_id: int,
    x_min: float,
    y_min: float,
    cell_size: float,
    smoothing_m: float,
    simplify_m: float,
) -> Polygon | None:
    segments = _edge_segments(labels == label_id, x_min, y_min, cell_size)
    if len(segments) == 0:
        return None
    lines = shapely.linestrings(segments)
    collection = shapely.polygonize(lines)
    polygons = [geom for geom in collection.geoms if geom.area > 0]
    if not polygons:
        return None
    polygon = max(polygons, key=lambda p: p.area)
    if smoothing_m > 0:
        polygon = polygon.buffer(smoothing_m, join_style="round").buffer(-smoothing_m, join_style="round")
    if simplify_m > 0:
        polygon = polygon.simplify(simplify_m, preserve_topology=True)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if isinstance(polygon, MultiPolygon):
        polygon = max(polygon.geoms, key=lambda p: p.area)
    return polygon if not polygon.is_empty else None


def build_pile_results(
    labels: np.ndarray,
    relative_height: np.ndarray,
    probability: np.ndarray | None,
    x_min: float,
    y_min: float,
    cell_size: float,
    smoothing_m: float,
    simplify_m: float,
) -> list[PileResult]:
    results = []
    for label_id in range(1, int(labels.max()) + 1):
        mask = labels == label_id
        heights = np.maximum(relative_height[mask], 0.0)
        if heights.size == 0:
            continue
        polygon = polygon_from_label(labels, label_id, x_min, y_min, cell_size, smoothing_m, simplify_m)
        if polygon is None:
            continue
        volume = float(np.sum(heights, dtype=np.float64) * cell_size**2)
        support = 0.75
        boundary_support = 0.75
        if probability is not None:
            support = float(np.nanmean(probability[mask]))
            if not np.isfinite(support): support = 0.0
            boundary = mask & ~ndi.binary_erosion(mask, structure=np.ones((3, 3), dtype=np.uint8))
            boundary_support = float(np.nanmean(probability[boundary])) if np.any(boundary) else support
            if not np.isfinite(boundary_support): boundary_support = 0.0
        validity = 1.0 if polygon.is_valid else 0.0
        compactness = min(1.0, 4 * np.pi * polygon.area / max(polygon.length**2, 1e-6) * 2.5)
        size_confidence = min(1.0, heights.size / 500.0)
        score = float(np.clip(
            0.35 * support + 0.30 * boundary_support + 0.15 * compactness
            + 0.10 * size_confidence + 0.10 * validity,
            0, 1,
        ))
        grade = "A" if score >= 0.85 else "B" if score >= 0.70 else "C"
        results.append(PileResult(
            pile_id=len(results) + 1, polygon=polygon, volume_m3=volume,
            area_m2=float(polygon.area), max_relative_height_m=float(np.max(heights)),
            mean_relative_height_m=float(np.mean(heights)), quality_score=score,
            quality_grade=grade, boundary_confidence=boundary_support,
            pixel_count=int(heights.size),
        ))
    return results
