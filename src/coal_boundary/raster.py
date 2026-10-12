from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .las import LasHeader, iter_las_points


@dataclass
class SurfaceGrid:
    x_min: float
    y_min: float
    cell_size: float
    rows: int
    cols: int
    point_count: np.ndarray
    mean_z: np.ndarray
    min_z: np.ndarray
    max_z: np.ndarray
    mean_rgb: np.ndarray

    @property
    def x_max(self) -> float:
        return self.x_min + self.cols * self.cell_size

    @property
    def y_max(self) -> float:
        return self.y_min + self.rows * self.cell_size

    @property
    def valid_mask(self) -> np.ndarray:
        return self.point_count > 0

    def xy_from_rc(self, rows, cols):
        return (
            self.x_min + (np.asarray(cols) + 0.5) * self.cell_size,
            self.y_min + (np.asarray(rows) + 0.5) * self.cell_size,
        )

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez_compressed(
            tmp,
            x_min=np.float64(self.x_min), y_min=np.float64(self.y_min),
            cell_size=np.float64(self.cell_size), rows=np.int64(self.rows), cols=np.int64(self.cols),
            point_count=self.point_count, mean_z=self.mean_z, min_z=self.min_z,
            max_z=self.max_z, mean_rgb=self.mean_rgb,
            metadata=np.asarray([metadata or {}], dtype=object),
        )
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> tuple["SurfaceGrid", dict]:
        with np.load(Path(path), allow_pickle=True) as z:
            grid = cls(
                float(z["x_min"]), float(z["y_min"]), float(z["cell_size"]),
                int(z["rows"]), int(z["cols"]), z["point_count"], z["mean_z"],
                z["min_z"], z["max_z"], z["mean_rgb"],
            )
            metadata = dict(z["metadata"][0]) if "metadata" in z else {}
        return grid, metadata


def build_surface_grid(
    headers: list[LasHeader],
    cell_size: float,
    chunk_points: int,
    minimum_z: float,
    maximum_z: float,
    progress: Callable[[int, int, Path], None] | None = None,
) -> SurfaceGrid:
    mins = np.min(np.asarray([h.mins for h in headers]), axis=0)
    maxs = np.max(np.asarray([h.maxs for h in headers]), axis=0)
    x_min = np.floor(mins[0] / cell_size) * cell_size
    y_min = np.floor(mins[1] / cell_size) * cell_size
    cols = int(np.ceil((maxs[0] - x_min) / cell_size)) + 1
    rows = int(np.ceil((maxs[1] - y_min) / cell_size)) + 1
    n_cells = rows * cols
    count = np.zeros(n_cells, dtype=np.uint32)
    sum_z = np.zeros(n_cells, dtype=np.float64)
    min_z = np.full(n_cells, np.inf, dtype=np.float32)
    max_z = np.full(n_cells, -np.inf, dtype=np.float32)
    sum_rgb = np.zeros((3, n_cells), dtype=np.float32)

    for file_index, header in enumerate(headers, 1):
        for chunk in iter_las_points(header, chunk_points):
            x, y, z = chunk["x"], chunk["y"], chunk["z"]
            valid = np.isfinite(z) & (z >= minimum_z) & (z <= maximum_z)
            if not np.any(valid):
                continue
            x, y, z = x[valid], y[valid], z[valid]
            col = np.floor((x - x_min) / cell_size).astype(np.int64)
            row = np.floor((y - y_min) / cell_size).astype(np.int64)
            inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
            if not np.all(inside):
                row, col, z = row[inside], col[inside], z[inside]
                valid_indices = np.flatnonzero(valid)[inside]
            else:
                valid_indices = np.flatnonzero(valid)
            flat = row * cols + col
            order = np.argsort(flat, kind="stable")
            flat_s = flat[order]
            z_s = z[order]
            starts = np.r_[0, np.flatnonzero(np.diff(flat_s)) + 1]
            unique = flat_s[starts]
            local_count = np.diff(np.r_[starts, flat_s.size]).astype(np.uint32)
            count[unique] += local_count
            sum_z[unique] += np.add.reduceat(z_s.astype(np.float64), starts)
            min_z[unique] = np.minimum(min_z[unique], np.minimum.reduceat(z_s, starts))
            max_z[unique] = np.maximum(max_z[unique], np.maximum.reduceat(z_s, starts))
            for channel_i, name in enumerate(("red", "green", "blue")):
                values = chunk[name][valid_indices][order].astype(np.float32)
                sum_rgb[channel_i, unique] += np.add.reduceat(values, starts)
        if progress:
            progress(file_index, len(headers), header.path)

    valid = count > 0
    mean_z = np.full(n_cells, np.nan, dtype=np.float32)
    mean_z[valid] = (sum_z[valid] / count[valid]).astype(np.float32)
    min_z[~valid] = np.nan
    max_z[~valid] = np.nan
    mean_rgb = np.zeros((3, n_cells), dtype=np.float32)
    mean_rgb[:, valid] = sum_rgb[:, valid] / count[valid]
    return SurfaceGrid(
        x_min=x_min, y_min=y_min, cell_size=cell_size, rows=rows, cols=cols,
        point_count=count.reshape(rows, cols), mean_z=mean_z.reshape(rows, cols),
        min_z=min_z.reshape(rows, cols), max_z=max_z.reshape(rows, cols),
        mean_rgb=np.moveaxis(mean_rgb.reshape(3, rows, cols), 0, -1),
    )


def build_surface_grid_for_bounds(
    headers: list[LasHeader], bounds: tuple[float, float, float, float], cell_size: float,
    chunk_points: int, minimum_z: float, maximum_z: float,
    progress: Callable[[int, int, Path], None] | None = None,
) -> SurfaceGrid:
    """仅栅格化与给定 XY 范围相交的 LAS 点，供单煤堆局部复算。"""
    min_x, min_y, max_x, max_y = bounds
    selected = [h for h in headers if h.maxs[0] >= min_x and h.mins[0] <= max_x and h.maxs[1] >= min_y and h.mins[1] <= max_y]
    if not selected:
        raise ValueError("目标边界范围内没有 LAS 文件")
    x_min = np.floor(min_x / cell_size) * cell_size
    y_min = np.floor(min_y / cell_size) * cell_size
    cols = max(1, int(np.ceil((max_x - x_min) / cell_size)) + 1)
    rows = max(1, int(np.ceil((max_y - y_min) / cell_size)) + 1)
    n_cells = rows * cols
    count = np.zeros(n_cells, dtype=np.uint32)
    sum_z = np.zeros(n_cells, dtype=np.float64)
    min_z_arr = np.full(n_cells, np.inf, dtype=np.float32)
    max_z_arr = np.full(n_cells, -np.inf, dtype=np.float32)
    sum_rgb = np.zeros((3, n_cells), dtype=np.float64)
    for file_index, header in enumerate(selected, 1):
        for chunk in iter_las_points(header, chunk_points):
            x, y, z = chunk["x"], chunk["y"], chunk["z"]
            valid = (
                np.isfinite(z) & (z >= minimum_z) & (z <= maximum_z)
                & (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
            )
            if not np.any(valid):
                continue
            indices = np.flatnonzero(valid)
            x, y, z = x[valid], y[valid], z[valid]
            col = np.floor((x - x_min) / cell_size).astype(np.int64)
            row = np.floor((y - y_min) / cell_size).astype(np.int64)
            inside = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
            row, col, z, indices = row[inside], col[inside], z[inside], indices[inside]
            if not z.size:
                continue
            flat = row * cols + col
            order = np.argsort(flat, kind="stable")
            flat_s, z_s = flat[order], z[order]
            starts = np.r_[0, np.flatnonzero(np.diff(flat_s)) + 1]
            unique = flat_s[starts]
            count[unique] += np.diff(np.r_[starts, flat_s.size]).astype(np.uint32)
            sum_z[unique] += np.add.reduceat(z_s.astype(np.float64), starts)
            min_z_arr[unique] = np.minimum(min_z_arr[unique], np.minimum.reduceat(z_s, starts))
            max_z_arr[unique] = np.maximum(max_z_arr[unique], np.maximum.reduceat(z_s, starts))
            for channel_i, name in enumerate(("red", "green", "blue")):
                values = chunk[name][indices][order].astype(np.float64)
                sum_rgb[channel_i, unique] += np.add.reduceat(values, starts)
        if progress:
            progress(file_index, len(selected), header.path)
    valid = count > 0
    if not np.any(valid):
        raise ValueError("目标边界范围内没有有效 LAS 点")
    mean_z = np.full(n_cells, np.nan, dtype=np.float32)
    mean_z[valid] = (sum_z[valid] / count[valid]).astype(np.float32)
    min_z_arr[~valid] = np.nan
    max_z_arr[~valid] = np.nan
    mean_rgb = np.zeros((3, n_cells), dtype=np.float32)
    mean_rgb[:, valid] = (sum_rgb[:, valid] / count[valid]).astype(np.float32)
    return SurfaceGrid(
        x_min=x_min, y_min=y_min, cell_size=cell_size, rows=rows, cols=cols,
        point_count=count.reshape(rows, cols), mean_z=mean_z.reshape(rows, cols),
        min_z=min_z_arr.reshape(rows, cols), max_z=max_z_arr.reshape(rows, cols),
        mean_rgb=np.moveaxis(mean_rgb.reshape(3, rows, cols), 0, -1),
    )
