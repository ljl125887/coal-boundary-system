import numpy as np

from coal_boundary.segmentation import (
    _merge_shallow_basins,
    candidate_mask,
    merge_nearby_toe_fragments,
    recover_concave_slopes,
    split_by_height_valleys,
)


def test_height_valley_splits_connected_piles():
    yy, xx = np.mgrid[:100, :160]
    first = 8.0 * np.exp(-((xx - 55) ** 2 + (yy - 50) ** 2) / (2 * 18**2))
    second = 7.0 * np.exp(-((xx - 105) ** 2 + (yy - 50) ** 2) / (2 * 18**2))
    height = np.maximum(first, second).astype(np.float32)
    valid = np.ones_like(height, dtype=bool)
    mask = candidate_mask(height, valid, 1.0, 0.3, 0.8, 20, 10, 1)
    labels = split_by_height_valleys(height, mask, 1.0, 0.8, 15, 20, 0.8)
    assert labels.max() == 2


def test_shallow_ripple_does_not_oversegment():
    yy, xx = np.mgrid[:80, :100]
    height = 6.0 * np.exp(-((xx - 50) ** 2 + (yy - 40) ** 2) / (2 * 20**2))
    height += 0.15 * np.cos(xx / 5)
    mask = height > 0.4
    labels = split_by_height_valleys(height.astype(np.float32), mask, 1.0, 1.0, 12, 20, 0.8)
    assert labels.max() == 1


def test_relative_saddle_depth_merges_business_double_peak():
    labels = np.ones((30, 60), dtype=np.int32)
    labels[:, 30:] = 2
    height = np.full(labels.shape, 6.0, dtype=np.float32)
    height[10, 10] = 12.0
    height[10, 50] = 11.8

    absolute_only = _merge_shallow_basins(labels, height, valley_depth=2.0)
    business_rule = _merge_shallow_basins(
        labels, height, valley_depth=2.0, minimum_valley_depth_ratio=0.6,
    )

    assert absolute_only.max() == 2
    assert business_rule.max() == 1


def test_low_saddle_still_splits_two_piles():
    labels = np.ones((30, 60), dtype=np.int32)
    labels[:, 30:] = 2
    height = np.full(labels.shape, 1.0, dtype=np.float32)
    height[10, 10] = 12.0
    height[10, 50] = 11.8

    merged = _merge_shallow_basins(
        labels, height, valley_depth=2.0, minimum_valley_depth_ratio=0.6,
    )

    assert merged.max() == 2


def test_concave_continuous_slope_is_recovered():
    yy, xx = np.mgrid[:100, :100]
    radius = np.sqrt((xx - 50) ** 2 + (yy - 50) ** 2)
    height = np.maximum(8.0 * (1.0 - radius / 42.0), 0.0).astype(np.float32)
    pile = radius <= 40
    missing_wedge = pile & (xx < 50) & (np.abs(yy - 50) < (50 - xx) * 0.35)
    labels = np.where(pile & ~missing_wedge, 1, 0).astype(np.int32)

    recovered = recover_concave_slopes(
        labels, height, np.ones_like(pile), 1.0, 0.35, 0.0,
        probability=None, minimum_gradient=0.05, minimum_local_relief_m=0.05,
        minimum_component_area_m2=10.0, minimum_peak_height_m=1.0,
        minimum_vertical_relief_m=0.5,
    )

    assert np.count_nonzero((recovered == 1) & missing_wedge) > 50


def test_probability_closing_does_not_fill_flat_track_bridge():
    height = np.zeros((60, 100), dtype=np.float32)
    probability = np.zeros_like(height)
    height[15:45, 10:40] = 5.0
    probability[15:45, 10:40] = 0.95
    height[27:33, 40:58] = 1.8
    probability[27:33, 55:58] = 0.95

    mask = candidate_mask(
        height, np.ones_like(height, dtype=bool), 1.0,
        minimum_height_m=0.35, core_height_m=0.8,
        minimum_area_m2=20.0, maximum_hole_area_m2=10.0,
        closing_radius_m=10.0, probability=probability,
        probability_low=0.55, probability_high=0.55,
        closing_min_gradient=0.45, closing_min_local_relief_m=0.8,
        closing_min_vertical_relief_m=0.5,
    )

    assert not np.any(mask[27:33, 43:53])


def test_small_low_coal_toe_fragment_is_reconnected():
    labels = np.zeros((70, 100), dtype=np.int32)
    labels[15:55, 35:75] = 1
    labels[28:38, 29:33] = 2
    height = np.zeros(labels.shape, dtype=np.float32)
    height[labels == 1] = 8.0
    height[labels == 2] = 2.0
    probability = np.zeros(labels.shape, dtype=np.float32)
    probability[labels > 0] = 0.9

    merged = merge_nearby_toe_fragments(
        labels, height, np.ones_like(labels, dtype=bool), 1.0,
        maximum_gap_m=4.0, maximum_fragment_area_m2=100.0,
        maximum_area_ratio=0.3, maximum_peak_height_m=4.0,
        probability=probability, minimum_probability=0.75,
    )

    assert merged.max() == 1
    assert np.all(merged[28:38, 29:33] == 1)
    assert np.any(merged[:, 33:35] == 1)


def test_high_nearby_pile_is_not_treated_as_toe_fragment():
    labels = np.zeros((70, 100), dtype=np.int32)
    labels[15:55, 35:75] = 1
    labels[28:38, 29:33] = 2
    height = np.zeros(labels.shape, dtype=np.float32)
    height[labels == 1] = 8.0
    height[labels == 2] = 6.0

    merged = merge_nearby_toe_fragments(
        labels, height, np.ones_like(labels, dtype=bool), 1.0,
        maximum_gap_m=4.0, maximum_fragment_area_m2=100.0,
        maximum_area_ratio=0.3, maximum_peak_height_m=4.0,
        probability=None,
    )

    assert merged.max() == 2
