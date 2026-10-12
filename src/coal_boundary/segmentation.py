from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
import heapq
import shapely
from shapely.geometry import MultiPoint


def _disk(radius: int) -> np.ndarray:
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy) <= radius * radius


def _remove_small_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(labels.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    return keep[labels]


def _fill_small_holes(mask: np.ndarray, max_hole_pixels: int) -> np.ndarray:
    all_filled = ndi.binary_fill_holes(mask)
    holes = all_filled & ~mask
    hole_labels, _ = ndi.label(holes, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(hole_labels.ravel())
    small = counts <= max_hole_pixels
    small[0] = False
    return mask | small[hole_labels]


def _terrain_continuity(relative_height: np.ndarray, cell_size_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return slope magnitude and local relief used for geometry-only decisions."""
    sigma = max(0.8, 1.0 / cell_size_m)
    smooth = ndi.gaussian_filter(np.nan_to_num(relative_height, nan=0.0), sigma=sigma, mode="nearest")
    gradient_x = ndi.sobel(smooth, axis=1, mode="nearest") / (8.0 * cell_size_m)
    gradient_y = ndi.sobel(smooth, axis=0, mode="nearest") / (8.0 * cell_size_m)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    window = max(3, int(round(5.0 / cell_size_m)))
    if window % 2 == 0:
        window += 1
    local_mean = ndi.uniform_filter(relative_height, size=window, mode="nearest")
    local_square_mean = ndi.uniform_filter(relative_height * relative_height, size=window, mode="nearest")
    local_relief = np.sqrt(np.maximum(local_square_mean - local_mean * local_mean, 0.0))
    return gradient.astype(np.float32), local_relief.astype(np.float32)


def candidate_mask(
    relative_height: np.ndarray,
    valid_mask: np.ndarray,
    cell_size_m: float,
    minimum_height_m: float,
    core_height_m: float,
    minimum_area_m2: float,
    maximum_hole_area_m2: float,
    closing_radius_m: float,
    probability: np.ndarray | None = None,
    probability_low: float = 0.35,
    probability_high: float | None = None,
    closing_min_gradient: float = 0.45,
    closing_min_local_relief_m: float = 0.8,
    closing_min_vertical_relief_m: float = 0.5,
) -> np.ndarray:
    height_mask = valid_mask & np.isfinite(relative_height) & (relative_height >= minimum_height_m)
    mask = height_mask.copy()
    if probability is not None:
        semantic_probability = np.nan_to_num(probability, nan=0.0)
        support = height_mask & (semantic_probability >= probability_low)
        if probability_high is None or probability_high <= probability_low:
            mask = support
        else:
            seeds = (
                valid_mask
                & np.isfinite(relative_height)
                & (relative_height >= core_height_m)
                & (semantic_probability >= probability_high)
            )
            mask = ndi.binary_propagation(
                seeds, structure=np.ones((3, 3), dtype=np.uint8), mask=support,
            )
    radius = max(0, int(round(closing_radius_m / cell_size_m)))
    if radius > 0:
        closed = ndi.binary_closing(mask, structure=_disk(radius))
        if probability is None:
            mask = closed & height_mask
        else:
            # A free binary closing can turn narrow railway ridges into large
            # foreground spurs. Only accept newly closed cells when the height
            # field itself shows a continuous coal-pile slope.
            gradient, local_relief = _terrain_continuity(relative_height, cell_size_m)
            geometric_support = (
                height_mask
                & (gradient >= closing_min_gradient)
                & (local_relief >= closing_min_local_relief_m)
            )
            additions = closed & ~mask & geometric_support
            addition_labels, addition_count = ndi.label(
                additions, structure=np.ones((3, 3), dtype=np.uint8),
            )
            accepted_additions = np.zeros(additions.shape, dtype=bool)
            for addition_id in range(1, addition_count + 1):
                component = addition_labels == addition_id
                values = relative_height[component]
                if values.size and float(np.max(values) - np.min(values)) >= closing_min_vertical_relief_m:
                    accepted_additions |= component
            mask = mask | accepted_additions
    min_pixels = max(1, int(round(minimum_area_m2 / (cell_size_m**2))))
    hole_pixels = max(1, int(round(maximum_hole_area_m2 / (cell_size_m**2))))
    mask = _remove_small_components(mask, min_pixels)
    mask = _fill_small_holes(mask, hole_pixels)
    return np.asarray(mask, dtype=bool)


def _select_peaks(smooth: np.ndarray, mask: np.ndarray, min_distance: int, core_height: float) -> list[tuple[int, int]]:
    local_max = smooth == ndi.maximum_filter(smooth, size=2 * min_distance + 1, mode="nearest")
    positions = np.argwhere(local_max & mask & (smooth >= core_height))
    ranked = sorted(((float(smooth[r, c]), int(r), int(c)) for r, c in positions), reverse=True)
    accepted: list[tuple[int, int]] = []
    for _, r, c in ranked:
        if all((r - ar) ** 2 + (c - ac) ** 2 >= min_distance**2 for ar, ac in accepted):
            accepted.append((r, c))
    return accepted


def _merge_shallow_basins(
    labels: np.ndarray,
    height: np.ndarray,
    valley_depth: float,
    minimum_valley_depth_ratio: float = 0.0,
) -> np.ndarray:
    n = int(labels.max())
    if n <= 1:
        return labels
    peaks = np.zeros(n + 1, dtype=np.float32)
    for label_id in range(1, n + 1):
        values = height[labels == label_id]
        peaks[label_id] = float(np.max(values)) if values.size else 0.0
    saddles: dict[tuple[int, int], float] = {}
    for first, second, h1, h2 in (
        (labels[:, :-1], labels[:, 1:], height[:, :-1], height[:, 1:]),
        (labels[:-1, :], labels[1:, :], height[:-1, :], height[1:, :]),
    ):
        valid = (first > 0) & (second > 0) & (first != second)
        a, b = first[valid], second[valid]
        values = np.minimum(h1[valid], h2[valid])
        for x, y, value in zip(a.tolist(), b.tolist(), values.tolist()):
            key = (x, y) if x < y else (y, x)
            saddles[key] = max(saddles.get(key, -np.inf), value)
    parent = np.arange(n + 1)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for (a, b), saddle in sorted(saddles.items(), key=lambda item: item[1], reverse=True):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        lower_peak = max(float(min(peaks[ra], peaks[rb])), 1e-6)
        depth = lower_peak - float(saddle)
        relative_depth = depth / lower_peak
        if depth < valley_depth or relative_depth < minimum_valley_depth_ratio:
            if peaks[ra] >= peaks[rb]: parent[rb] = ra
            else: parent[ra] = rb
    mapping = np.zeros(n + 1, dtype=np.int32)
    roots = {}
    for label_id in range(1, n + 1):
        root = find(label_id)
        roots.setdefault(root, len(roots) + 1)
        mapping[label_id] = roots[root]
    return mapping[labels]


def _priority_watershed(height: np.ndarray, mask: np.ndarray, markers: np.ndarray) -> np.ndarray:
    """Maximin multi-source watershed whose boundaries settle on height saddles."""
    rows, cols = height.shape
    labels = np.zeros((rows, cols), dtype=np.int32)
    best = np.full((rows, cols), -np.inf, dtype=np.float32)
    heap: list[tuple[float, int, int, int]] = []
    seed_rows, seed_cols = np.nonzero(markers)
    for r, c in zip(seed_rows.tolist(), seed_cols.tolist()):
        label_id = int(markers[r, c])
        strength = float(height[r, c])
        best[r, c] = strength
        labels[r, c] = label_id
        heapq.heappush(heap, (-strength, label_id, r, c))
    neighbors = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
    while heap:
        negative_strength, label_id, r, c = heapq.heappop(heap)
        strength = -negative_strength
        if strength + 1e-7 < float(best[r, c]) or labels[r, c] != label_id:
            continue
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols or not mask[nr, nc]:
                continue
            candidate = min(strength, float(height[nr, nc]))
            if candidate > float(best[nr, nc]) + 1e-7:
                best[nr, nc] = candidate
                labels[nr, nc] = label_id
                heapq.heappush(heap, (-candidate, label_id, nr, nc))
    return labels


def split_by_height_valleys(
    relative_height: np.ndarray,
    mask: np.ndarray,
    cell_size_m: float,
    valley_depth_m: float,
    minimum_peak_distance_m: float,
    minimum_area_m2: float,
    core_height_m: float = 0.8,
    peak_smoothing_m: float = 4.0,
    minimum_valley_depth_ratio: float = 0.0,
) -> np.ndarray:
    smooth_sigma = max(0.8, peak_smoothing_m / cell_size_m)
    smooth = ndi.gaussian_filter(np.where(mask, relative_height, 0.0), sigma=smooth_sigma)
    minimum_distance = max(1, int(round(minimum_peak_distance_m / cell_size_m)))
    # Most coal piles are already disconnected by exposed ground. Process each
    # connected component independently and only run the Python priority queue
    # on the few components that actually contain multiple peaks.
    components, component_count = ndi.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    local_max = smooth == ndi.maximum_filter(smooth, size=2 * minimum_distance + 1, mode="nearest")
    peak_positions = np.argwhere(local_max & mask & (smooth >= core_height_m))
    ranked_peaks = sorted(
        ((float(smooth[r, c]), int(r), int(c), int(components[r, c])) for r, c in peak_positions),
        reverse=True,
    )
    component_peaks: dict[int, list[tuple[int, int]]] = {}
    for _, r, c, component_id in ranked_peaks:
        accepted = component_peaks.setdefault(component_id, [])
        if all((r - ar) ** 2 + (c - ac) ** 2 >= minimum_distance**2 for ar, ac in accepted):
            accepted.append((r, c))
    labels = np.zeros(mask.shape, dtype=np.int32)
    next_label = 1
    for component_id, component_slice in enumerate(ndi.find_objects(components), 1):
        if component_slice is None:
            continue
        submask = components[component_slice] == component_id
        subsmooth = smooth[component_slice]
        row_offset = component_slice[0].start or 0
        col_offset = component_slice[1].start or 0
        peaks = [
            (r - row_offset, c - col_offset)
            for r, c in component_peaks.get(component_id, [])
        ]
        if len(peaks) <= 1:
            target = labels[component_slice]
            target[submask] = next_label
            next_label += 1
            continue
        markers = np.zeros(submask.shape, dtype=np.int32)
        for marker_id, (r, c) in enumerate(peaks, 1):
            markers[r, c] = marker_id
        local_labels = _priority_watershed(subsmooth, submask, markers)
        local_labels = _merge_shallow_basins(
            local_labels, subsmooth, valley_depth_m, minimum_valley_depth_ratio,
        )
        local_unique = np.unique(local_labels)
        local_unique = local_unique[local_unique > 0]
        local_mapping = np.zeros(int(local_labels.max()) + 1, dtype=np.int32)
        local_mapping[local_unique] = np.arange(next_label, next_label + len(local_unique))
        target = labels[component_slice]
        target[submask] = local_mapping[local_labels[submask]]
        next_label += len(local_unique)
    min_pixels = max(1, int(round(minimum_area_m2 / (cell_size_m**2))))
    counts = np.bincount(labels.ravel())
    keep = counts >= min_pixels
    keep[0] = False
    labels = np.where(keep[labels], labels, 0)
    unique = np.unique(labels)
    unique = unique[unique > 0]
    mapping = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    mapping[unique] = np.arange(1, len(unique) + 1)
    return mapping[labels]


def recover_concave_slopes(
    labels: np.ndarray,
    relative_height: np.ndarray,
    valid_mask: np.ndarray,
    cell_size_m: float,
    minimum_height_m: float,
    buffer_m: float,
    probability: np.ndarray | None = None,
    probability_floor: float = 0.4,
    minimum_gradient: float = 0.45,
    minimum_local_relief_m: float = 0.8,
    minimum_component_area_m2: float = 25.0,
    minimum_peak_height_m: float = 3.0,
    minimum_vertical_relief_m: float = 2.0,
) -> np.ndarray:
    """Recover deep, slope-continuous concavities without expanding flat spurs.

    Model confidence may be low on a dark or shadowed coal face. Candidate
    pixels are therefore accepted from RGB/model evidence *or* from geometric
    slope continuity. A recovered patch must also contain meaningful vertical
    relief; this rejects railway beds and other low, nearly level structures.
    """
    if buffer_m < 0 or not np.any(labels > 0):
        return labels.copy()
    gradient, local_relief = _terrain_continuity(relative_height, cell_size_m)
    allowed = (
        valid_mask
        & np.isfinite(relative_height)
        & (relative_height >= minimum_height_m)
        & (gradient >= minimum_gradient)
        & (local_relief >= minimum_local_relief_m)
    )
    if probability is not None:
        allowed |= (
            valid_mask
            & np.isfinite(relative_height)
            & (relative_height >= minimum_height_m)
            & (np.nan_to_num(probability, nan=0.0) >= probability_floor)
        )

    rows, cols = labels.shape
    buffer_cells = buffer_m / cell_size_m
    minimum_pixels = max(1, int(round(minimum_component_area_m2 / (cell_size_m**2))))
    owner = np.zeros(labels.shape, dtype=np.int32)
    best_distance = np.full(labels.shape, np.inf, dtype=np.float32)

    for label_id in np.unique(labels):
        if label_id <= 0:
            continue
        row_indices, col_indices = np.where(labels == label_id)
        if row_indices.size < 3:
            continue
        points = np.column_stack([col_indices + 0.5, row_indices + 0.5])
        recovery_region = MultiPoint(points).convex_hull.buffer(buffer_cells)
        min_col, min_row, max_col, max_row = recovery_region.bounds
        col0 = max(0, int(np.floor(min_col)))
        col1 = min(cols, int(np.ceil(max_col)))
        row0 = max(0, int(np.floor(min_row)))
        row1 = min(rows, int(np.ceil(max_row)))
        if row1 <= row0 or col1 <= col0:
            continue
        yy, xx = np.mgrid[row0:row1, col0:col1]
        inside = shapely.contains_xy(recovery_region, xx + 0.5, yy + 0.5)
        local_labels = labels[row0:row1, col0:col1]
        candidates = inside & allowed[row0:row1, col0:col1] & (local_labels == 0)
        candidate_labels, candidate_count = ndi.label(
            candidates, structure=np.ones((3, 3), dtype=np.uint8),
        )
        accepted = np.zeros(candidates.shape, dtype=bool)
        local_height = relative_height[row0:row1, col0:col1]
        for candidate_id in range(1, candidate_count + 1):
            component = candidate_labels == candidate_id
            if int(np.count_nonzero(component)) < minimum_pixels:
                continue
            values = local_height[component]
            peak = float(np.max(values))
            vertical_relief = peak - float(np.min(values))
            if peak >= minimum_peak_height_m and vertical_relief >= minimum_vertical_relief_m:
                accepted |= component
        if not np.any(accepted):
            continue

        source = local_labels == label_id
        distance = ndi.distance_transform_edt(~source).astype(np.float32)
        local_best = best_distance[row0:row1, col0:col1]
        local_owner = owner[row0:row1, col0:col1]
        update = accepted & (distance < local_best)
        local_best[update] = distance[update]
        local_owner[update] = int(label_id)

    recovered = labels.copy()
    recovered[owner > 0] = owner[owner > 0]
    return recovered


def merge_nearby_toe_fragments(
    labels: np.ndarray,
    relative_height: np.ndarray,
    valid_mask: np.ndarray,
    cell_size_m: float,
    maximum_gap_m: float,
    maximum_fragment_area_m2: float,
    maximum_area_ratio: float,
    maximum_peak_height_m: float,
    probability: np.ndarray | None = None,
    minimum_probability: float = 0.75,
    peak_smoothing_m: float = 4.0,
) -> np.ndarray:
    """Reconnect small, low toe fragments cut off by the semantic mask.

    The rule is deliberately asymmetric: only a small and low fragment can be
    attached to a substantially larger neighboring pile. Two normal-sized or
    two high piles remain separate even when they are close. When RGB/model
    probability is available, the fragment must also retain strong coal
    evidence. The short raster bridge keeps the exported polygon connected.
    """
    if maximum_gap_m <= 0 or maximum_fragment_area_m2 <= 0:
        return labels.copy()
    result = labels.copy()
    counts = np.bincount(result.ravel())
    gap_cells = max(1, int(np.ceil(maximum_gap_m / cell_size_m)))
    maximum_fragment_pixels = max(1, int(round(maximum_fragment_area_m2 / (cell_size_m**2))))
    smooth_sigma = max(0.8, peak_smoothing_m / cell_size_m)
    smooth = ndi.gaussian_filter(np.where(result > 0, relative_height, 0.0), sigma=smooth_sigma)
    probability_grid = None if probability is None else np.nan_to_num(probability, nan=0.0)
    merged_any = False

    fragment_ids = [
        label_id for label_id in range(1, len(counts))
        if 0 < int(counts[label_id]) <= maximum_fragment_pixels
    ]
    fragment_ids.sort(key=lambda label_id: int(counts[label_id]))
    for fragment_id in fragment_ids:
        fragment = result == fragment_id
        if not np.any(fragment):
            continue
        fragment_peak = max(
            float(np.max(smooth[fragment])), float(np.max(relative_height[fragment])),
        )
        if fragment_peak > maximum_peak_height_m:
            continue
        if probability_grid is not None and float(np.mean(probability_grid[fragment])) < minimum_probability:
            continue
        nearby_region = ndi.binary_dilation(fragment, structure=_disk(gap_cells))
        nearby_ids = np.unique(result[nearby_region])
        nearby_ids = nearby_ids[(nearby_ids > 0) & (nearby_ids != fragment_id)]
        if nearby_ids.size == 0:
            continue

        fragment_points = np.argwhere(fragment)
        best: tuple[float, int, np.ndarray, np.ndarray] | None = None
        for neighbor_id in nearby_ids.tolist():
            neighbor_count = int(np.count_nonzero(result == neighbor_id))
            fragment_count = int(fragment_points.shape[0])
            if neighbor_count <= 0 or fragment_count / neighbor_count > maximum_area_ratio:
                continue
            neighbor_points = np.argwhere(result == neighbor_id)
            distances, indices = cKDTree(neighbor_points).query(fragment_points, k=1)
            closest_index = int(np.argmin(distances))
            distance_m = float(distances[closest_index] * cell_size_m)
            if distance_m > maximum_gap_m + cell_size_m:
                continue
            fragment_point = fragment_points[closest_index]
            neighbor_point = neighbor_points[int(indices[closest_index])]
            candidate = (distance_m, int(neighbor_id), fragment_point, neighbor_point)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            continue

        _, neighbor_id, fragment_point, neighbor_point = best
        result[fragment] = neighbor_id
        steps = int(max(abs(int(fragment_point[0]) - int(neighbor_point[0])), abs(int(fragment_point[1]) - int(neighbor_point[1])))) + 1
        line_rows = np.rint(np.linspace(fragment_point[0], neighbor_point[0], steps)).astype(int)
        line_cols = np.rint(np.linspace(fragment_point[1], neighbor_point[1], steps)).astype(int)
        bridge_ok = valid_mask[line_rows, line_cols] & (
            (result[line_rows, line_cols] == 0) | (result[line_rows, line_cols] == neighbor_id)
        )
        result[line_rows[bridge_ok], line_cols[bridge_ok]] = neighbor_id
        merged_any = True

    if not merged_any:
        return result
    unique = np.unique(result)
    unique = unique[unique > 0]
    mapping = np.zeros(int(result.max()) + 1, dtype=np.int32)
    mapping[unique] = np.arange(1, len(unique) + 1)
    return mapping[result]
