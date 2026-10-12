"""展示用连续坡脚轮廓的选择策略测试（合成高度场，不需要 LAS）。"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from coal_boundary.pointcloud_boundary import (
    FineSurfaceEvidence,
    _refine_evidence,
    estimate_ground_offset,
)

CELL = 0.2
CORRIDOR_M = 2.8


def _radial_evidence(*, cell: float = CELL, half: float = 25.0, lift_m: float = 0.0):
    """合成一座“内缓外陡”的煤堆。

    低处（0~0.30 m）等高线包住很大面积，越往上面积急剧收缩。这种形状复现了真实
    数据里的取舍：只按面积匹配会挑到离地一两米的轮廓，而贴地优先会挑到贴近地面
    的那条。``lift_m`` 用来整体抬高场地，模拟“地面被地面估计算法压低”的现场情况。
    """
    coords = np.arange(-half + cell / 2.0, half, cell)
    xx, yy = np.meshgrid(coords, coords)
    radius = np.hypot(xx, yy)
    height = np.where(
        radius <= 8.0,
        0.30 * radius / 8.0,
        np.where(radius <= 9.0, 0.30 + 2.70 * (radius - 8.0), 0.0),
    ).astype(np.float64) + lift_m
    evidence = FineSurfaceEvidence(
        x_min=float(coords[0] - cell / 2.0),
        y_min=float(coords[0] - cell / 2.0),
        cell_size=cell,
        relative_height=height,
        observed=np.ones_like(height, dtype=bool),
    )
    return evidence, Point(0.0, 0.0).buffer(8.55, quad_segs=16)


def test_toe_first_prefers_the_lowest_contour_within_area_tolerance():
    evidence, source = _radial_evidence()

    toe_polygon, _, toe = _refine_evidence(
        source, evidence, 1, 1, selection_strategy="toe_first", area_tolerance_percent=30.0,
    )
    _, _, area = _refine_evidence(
        source, evidence, 1, 1, selection_strategy="area_first",
    )

    # 贴地优先必须挑到更低的等高线（更贴近地面）
    assert toe.height_threshold_m < area.height_threshold_m
    assert toe.height_threshold_m <= 0.35
    # 同时守住面积容差：展示轮廓不会明显偏离原先认可的煤堆范围
    assert abs(toe.refined_area_m2 - source.area) / source.area <= 0.30 + 1e-9
    # 选中的轮廓不能是沿着裁剪掩膜闭合出来的“假坡脚”
    corridor = source.buffer(CORRIDOR_M, join_style="round")
    assert toe_polygon.boundary.distance(corridor.boundary) > 0.30


def test_toe_height_floor_bounds_the_search_range():
    evidence, source = _radial_evidence()

    _, _, deep = _refine_evidence(
        source, evidence, 1, 1, selection_strategy="toe_first",
        toe_height_floor_m=0.05, area_tolerance_percent=95.0,
    )
    _, _, shallow = _refine_evidence(
        source, evidence, 1, 1, selection_strategy="toe_first",
        toe_height_floor_m=0.25, area_tolerance_percent=95.0,
    )

    # 旧范围下限 0.25 m 时够不到贴地轮廓；下限放到 0.05 m 后才行
    assert shallow.height_threshold_m >= 0.25
    assert deep.height_threshold_m < 0.25


def test_ground_offset_recovers_the_lifted_ground_level():
    evidence, source = _radial_evidence(lift_m=0.33)
    offset = estimate_ground_offset(evidence, source)
    assert abs(offset - 0.33) < 0.06


def test_ground_offset_keeps_the_toe_height_stable_when_ground_is_lifted():
    """地面被低估 0.33 m 时，归零后应当仍能选到与未抬高场相同的贴地高度。"""
    flat, source = _radial_evidence()
    lifted, _ = _radial_evidence(lift_m=0.33)

    _, _, reference = _refine_evidence(source, flat, 1, 1, selection_strategy="toe_first")
    _, _, corrected = _refine_evidence(source, lifted, 1, 1, selection_strategy="toe_first")

    assert abs(corrected.height_threshold_m - reference.height_threshold_m) < 0.10
