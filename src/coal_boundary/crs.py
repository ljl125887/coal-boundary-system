from __future__ import annotations

import numpy as np

# CGCS2000 / GRS80 parameters used by EPSG:4549.
_A = 6_378_137.0
_INV_F = 298.257222101
_F = 1.0 / _INV_F
_E2 = _F * (2.0 - _F)
_EP2 = _E2 / (1.0 - _E2)


def validate_crs_config_and_bounds(crs_config, bounds_min, bounds_max) -> dict:
    if crs_config.source_epsg != 4549 or crs_config.output_epsg != 4326:
        raise ValueError("当前纯数学坐标转换仅支持 EPSG:4549 -> EPSG:4326")
    minx, miny = float(bounds_min[0]), float(bounds_min[1])
    maxx, maxy = float(bounds_max[0]), float(bounds_max[1])
    if not (100_000.0 <= minx <= 900_000.0 and 100_000.0 <= maxx <= 900_000.0):
        raise ValueError("LAS X 坐标不符合高斯-克吕格投影米制范围，请检查 EPSG/中央经线")
    if not (0.0 <= miny <= 10_000_000.0 and 0.0 <= maxy <= 10_000_000.0):
        raise ValueError("LAS Y 坐标不符合北半球投影米制范围，请检查 EPSG")
    center_x, center_y = (minx + maxx) / 2.0, (miny + maxy) / 2.0
    lon, lat = projected_to_lonlat(
        center_x, center_y, crs_config.central_meridian_deg,
        crs_config.false_easting_m, crs_config.false_northing_m,
    )
    lon, lat = float(lon), float(lat)
    if not (70.0 <= lon <= 140.0 and 15.0 <= lat <= 60.0):
        raise ValueError(f"投影坐标反算位置异常: lon={lon:.6f}, lat={lat:.6f}")
    return {
        "validated": True,
        "source_epsg": crs_config.source_epsg,
        "output_epsg": crs_config.output_epsg,
        "center_lon": lon,
        "center_lat": lat,
    }


def _meridional_arc(lat: np.ndarray) -> np.ndarray:
    e2 = _E2
    return _A * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * lat)
        - (35 * e2**3 / 3072) * np.sin(6 * lat)
    )


def lonlat_to_projected(
    lon_deg,
    lat_deg,
    central_meridian_deg: float = 120.0,
    false_easting_m: float = 500_000.0,
    false_northing_m: float = 0.0,
):
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon0 = np.deg2rad(central_meridian_deg)
    sin_lat, cos_lat = np.sin(lat), np.cos(lat)
    tan_lat = np.tan(lat)
    n = _A / np.sqrt(1 - _E2 * sin_lat**2)
    t = tan_lat**2
    c = _EP2 * cos_lat**2
    aa = cos_lat * (lon - lon0)
    m = _meridional_arc(lat)
    x = false_easting_m + n * (
        aa + (1 - t + c) * aa**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * _EP2) * aa**5 / 120
    )
    y = false_northing_m + m + n * tan_lat * (
        aa**2 / 2
        + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
        + (61 - 58 * t + t**2 + 600 * c - 330 * _EP2) * aa**6 / 720
    )
    return x, y


def projected_to_lonlat(
    x_m,
    y_m,
    central_meridian_deg: float = 120.0,
    false_easting_m: float = 500_000.0,
    false_northing_m: float = 0.0,
):
    x = np.asarray(x_m, dtype=np.float64) - false_easting_m
    y = np.asarray(y_m, dtype=np.float64) - false_northing_m
    e2 = _E2
    mu = y / (_A * (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256))
    e1 = (1 - np.sqrt(1 - e2)) / (1 + np.sqrt(1 - e2))
    j1 = 3 * e1 / 2 - 27 * e1**3 / 32
    j2 = 21 * e1**2 / 16 - 55 * e1**4 / 32
    j3 = 151 * e1**3 / 96
    j4 = 1097 * e1**4 / 512
    fp = mu + j1 * np.sin(2 * mu) + j2 * np.sin(4 * mu) + j3 * np.sin(6 * mu) + j4 * np.sin(8 * mu)
    sin_fp, cos_fp = np.sin(fp), np.cos(fp)
    tan_fp = np.tan(fp)
    c1 = _EP2 * cos_fp**2
    t1 = tan_fp**2
    n1 = _A / np.sqrt(1 - e2 * sin_fp**2)
    r1 = _A * (1 - e2) / (1 - e2 * sin_fp**2) ** 1.5
    d = x / n1
    lat = fp - (n1 * tan_fp / r1) * (
        d**2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1**2 - 9 * _EP2) * d**4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1**2 - 252 * _EP2 - 3 * c1**2) * d**6 / 720
    )
    lon0 = np.deg2rad(central_meridian_deg)
    lon = lon0 + (
        d - (1 + 2 * t1 + c1) * d**3 / 6
        + (5 - 2 * c1 + 28 * t1 - 3 * c1**2 + 8 * _EP2 + 24 * t1**2) * d**5 / 120
    ) / cos_fp
    return np.rad2deg(lon), np.rad2deg(lat)
