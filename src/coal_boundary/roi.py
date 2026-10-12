from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union


def load_projected_roi(path: str | Path, expected_epsg: int = 4549):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    crs_name = str(document.get("crs", {}).get("properties", {}).get("name", ""))
    if str(expected_epsg) not in crs_name:
        raise ValueError(f"作业区 CRS 必须为 EPSG:{expected_epsg}，实际为 {crs_name or '未声明'}")
    geometries = [shape(feature["geometry"]) for feature in document.get("features", [])]
    if not geometries:
        raise ValueError("作业区 GeoJSON 不包含有效 geometry")
    geometry = unary_union(geometries)
    if geometry.is_empty or not geometry.is_valid:
        raise ValueError("作业区 geometry 为空或无效")
    return geometry


def rasterize_geometry(geometry, rows: int, cols: int, x_min: float, y_min: float, cell_size: float) -> np.ndarray:
    result = np.zeros((rows, cols), dtype=bool)
    minx, miny, maxx, maxy = geometry.bounds
    c0 = max(0, int(math.floor((minx - x_min) / cell_size)))
    c1 = min(cols, int(math.ceil((maxx - x_min) / cell_size)))
    r0 = max(0, int(math.floor((miny - y_min) / cell_size)))
    r1 = min(rows, int(math.ceil((maxy - y_min) / cell_size)))
    if r0 >= r1 or c0 >= c1:
        raise ValueError("作业区与 LAS 点云范围不相交，请检查 CRS")
    yy, xx = np.mgrid[r0:r1, c0:c1]
    xx = x_min + (xx + 0.5) * cell_size
    yy = y_min + (yy + 0.5) * cell_size
    result[r0:r1, c0:c1] = shapely.contains_xy(geometry, xx, yy)
    if not np.any(result):
        raise ValueError("作业区栅格为空，请检查范围和 CRS")
    return result
