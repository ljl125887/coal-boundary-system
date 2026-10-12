from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import struct
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class LasHeader:
    path: Path
    version_major: int
    version_minor: int
    point_format: int
    point_record_length: int
    point_count: int
    point_data_offset: int
    scales: tuple[float, float, float]
    offsets: tuple[float, float, float]
    mins: tuple[float, float, float]
    maxs: tuple[float, float, float]


def read_las_header(path: str | Path) -> LasHeader:
    path = Path(path)
    with path.open("rb") as f:
        data = f.read(375)
    if len(data) < 227 or data[:4] != b"LASF":
        raise ValueError(f"不是有效 LAS 文件: {path}")
    major, minor = data[24], data[25]
    point_offset = struct.unpack_from("<I", data, 96)[0]
    point_format = data[104] & 0x3F
    record_length = struct.unpack_from("<H", data, 105)[0]
    legacy_count = struct.unpack_from("<I", data, 107)[0]
    scales = struct.unpack_from("<ddd", data, 131)
    offsets = struct.unpack_from("<ddd", data, 155)
    max_x, min_x, max_y, min_y, max_z, min_z = struct.unpack_from("<dddddd", data, 179)
    point_count = legacy_count
    if major == 1 and minor >= 4 and len(data) >= 255:
        extended = struct.unpack_from("<Q", data, 247)[0]
        point_count = extended or legacy_count
    return LasHeader(
        path=path, version_major=major, version_minor=minor,
        point_format=point_format, point_record_length=record_length,
        point_count=point_count, point_data_offset=point_offset,
        scales=tuple(scales), offsets=tuple(offsets),
        mins=(min_x, min_y, min_z), maxs=(max_x, max_y, max_z),
    )


def _point_dtype(header: LasHeader) -> np.dtype:
    if header.point_format == 3:
        names = ["X", "Y", "Z", "intensity", "flags", "classification", "scan_angle", "user_data", "source", "gps_time", "red", "green", "blue"]
        formats = ["<i4", "<i4", "<i4", "<u2", "u1", "u1", "i1", "u1", "<u2", "<f8", "<u2", "<u2", "<u2"]
        offsets = [0, 4, 8, 12, 14, 15, 16, 17, 18, 20, 28, 30, 32]
    elif header.point_format == 2:
        names = ["X", "Y", "Z", "intensity", "flags", "classification", "scan_angle", "user_data", "source", "red", "green", "blue"]
        formats = ["<i4", "<i4", "<i4", "<u2", "u1", "u1", "i1", "u1", "<u2", "<u2", "<u2", "<u2"]
        offsets = [0, 4, 8, 12, 14, 15, 16, 17, 18, 20, 22, 24]
    else:
        raise NotImplementedError(f"当前仅支持带 RGB 的 LAS 点格式 2/3，收到格式 {header.point_format}: {header.path}")
    return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": header.point_record_length})


def iter_las_points(header: LasHeader, chunk_points: int = 1_500_000) -> Iterator[dict[str, np.ndarray]]:
    dtype = _point_dtype(header)
    points = np.memmap(
        header.path, mode="r", dtype=dtype,
        offset=header.point_data_offset, shape=(header.point_count,),
    )
    sx, sy, sz = header.scales
    ox, oy, oz = header.offsets
    for start in range(0, header.point_count, chunk_points):
        raw = points[start : min(start + chunk_points, header.point_count)]
        yield {
            "x": np.asarray(raw["X"], dtype=np.float64) * sx + ox,
            "y": np.asarray(raw["Y"], dtype=np.float64) * sy + oy,
            "z": np.asarray(raw["Z"], dtype=np.float32) * sz + oz,
            "red": np.asarray(raw["red"], dtype=np.uint16),
            "green": np.asarray(raw["green"], dtype=np.uint16),
            "blue": np.asarray(raw["blue"], dtype=np.uint16),
        }


def iter_las_xyz_in_bounds(
    header: LasHeader,
    bounds: tuple[float, float, float, float],
    chunk_points: int = 1_500_000,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read only XYZ points inside an XY window.

    The integer-coordinate bounds test happens before scale/offset conversion.  It
    is materially faster for narrow boundary bands than decoding RGB and every
    XYZ value from a multi-gigabyte LAS file.
    """
    min_x, min_y, max_x, max_y = bounds
    if (
        header.maxs[0] < min_x or header.mins[0] > max_x
        or header.maxs[1] < min_y or header.mins[1] > max_y
    ):
        return
    dtype = _point_dtype(header)
    points = np.memmap(
        header.path, mode="r", dtype=dtype,
        offset=header.point_data_offset, shape=(header.point_count,),
    )
    sx, sy, sz = header.scales
    ox, oy, oz = header.offsets
    raw_min_x = int(np.ceil((min_x - ox) / sx))
    raw_max_x = int(np.floor((max_x - ox) / sx))
    raw_min_y = int(np.ceil((min_y - oy) / sy))
    raw_max_y = int(np.floor((max_y - oy) / sy))
    for start in range(0, header.point_count, chunk_points):
        raw = points[start : min(start + chunk_points, header.point_count)]
        keep = (
            (raw["X"] >= raw_min_x) & (raw["X"] <= raw_max_x)
            & (raw["Y"] >= raw_min_y) & (raw["Y"] <= raw_max_y)
        )
        if not np.any(keep):
            continue
        selected = raw[keep]
        yield (
            np.asarray(selected["X"], dtype=np.float64) * sx + ox,
            np.asarray(selected["Y"], dtype=np.float64) * sy + oy,
            np.asarray(selected["Z"], dtype=np.float32) * sz + oz,
        )


def iter_las_xyz(
    header: LasHeader,
    chunk_points: int = 1_500_000,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stream XYZ only, avoiding RGB decoding for geometry refinement."""
    dtype = _point_dtype(header)
    points = np.memmap(
        header.path, mode="r", dtype=dtype,
        offset=header.point_data_offset, shape=(header.point_count,),
    )
    sx, sy, sz = header.scales
    ox, oy, oz = header.offsets
    for start in range(0, header.point_count, chunk_points):
        raw = points[start : min(start + chunk_points, header.point_count)]
        yield (
            np.asarray(raw["X"], dtype=np.float64) * sx + ox,
            np.asarray(raw["Y"], dtype=np.float64) * sy + oy,
            np.asarray(raw["Z"], dtype=np.float32) * sz + oz,
        )


def discover_las_files(directory: str | Path) -> list[Path]:
    files = sorted(Path(directory).glob("*.las")) + sorted(Path(directory).glob("*.laz"))
    if not files:
        raise FileNotFoundError(f"目录中没有 LAS/LAZ 文件: {directory}")
    if any(p.suffix.lower() == ".laz" for p in files):
        raise NotImplementedError("当前纯 NumPy 读取器暂不支持 LAZ；请先解压为 LAS")
    return files


def dataset_fingerprint(headers: list[LasHeader]) -> str:
    h = hashlib.sha256()
    for item in headers:
        stat = item.path.stat()
        h.update(str(item.path.resolve()).encode("utf-8"))
        h.update(str(stat.st_size).encode())
        h.update(str(stat.st_mtime_ns).encode())
        h.update(str(item.point_count).encode())
    return h.hexdigest()


def summarize_headers(headers: list[LasHeader]) -> dict:
    mins = np.min(np.asarray([h.mins for h in headers], dtype=np.float64), axis=0)
    maxs = np.max(np.asarray([h.maxs for h in headers], dtype=np.float64), axis=0)
    return {
        "file_count": len(headers),
        "point_count": int(sum(h.point_count for h in headers)),
        "bounds": {"min": mins.tolist(), "max": maxs.tolist()},
        "point_formats": sorted({h.point_format for h in headers}),
        "versions": sorted({f"{h.version_major}.{h.version_minor}" for h in headers}),
    }
