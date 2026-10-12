from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def fill_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if not np.any(valid):
        raise ValueError("点云栅格没有有效高程")
    if np.all(valid):
        return values.astype(np.float32, copy=True)
    indices = ndi.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[tuple(indices)].astype(np.float32, copy=False)


def estimate_ground_surface(
    min_surface: np.ndarray,
    valid_mask: np.ndarray,
    cell_size_m: float,
    opening_radius_m: float,
    smoothing_sigma_m: float,
) -> np.ndarray:
    filled = fill_nearest(min_surface, valid_mask)
    radius_cells = max(1, int(round(opening_radius_m / cell_size_m)))
    size = 2 * radius_cells + 1
    ground = ndi.grey_opening(filled, size=(size, size), mode="nearest")
    sigma = max(0.0, smoothing_sigma_m / cell_size_m)
    if sigma > 0:
        ground = ndi.gaussian_filter(ground, sigma=sigma, mode="nearest")
    # Ground must not rise above the observed low envelope at measured cells by a large amount.
    upper = filled + 0.15
    ground = np.minimum(ground, upper)
    return ground.astype(np.float32)
