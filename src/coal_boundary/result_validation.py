from __future__ import annotations

from functools import lru_cache
from importlib.resources import files
import json
import math
from numbers import Integral, Real
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from shapely.geometry import shape


SCHEMA_FILE = "coal-piles-result-1.0.schema.json"
GRADE_KEYS = ("A", "B", "C")
ABSOLUTE_TOLERANCE = 1e-6
RELATIVE_TOLERANCE = 1e-9


class ResultValidationError(ValueError):
    """Raised before an invalid boundary result reaches the spool or backend."""


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    resource = files("coal_boundary").joinpath("schemas", SCHEMA_FILE)
    schema = json.loads(resource.read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=FormatChecker())


def _path(parts: Any) -> str:
    return ".".join(str(part) for part in parts) or "result"


def _finite_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ResultValidationError(f"{path} 必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise ResultValidationError(f"{path} 不能是 NaN 或 Infinity")
    return number


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ResultValidationError(f"{path} 必须是整数")
    return int(value)


def _close_enough(declared: float, calculated: float) -> bool:
    tolerance = max(ABSOLUTE_TOLERANCE, abs(calculated) * RELATIVE_TOLERANCE)
    return abs(declared - calculated) <= tolerance


def _reject_non_finite(value: object, path: str = "result") -> None:
    if isinstance(value, Real) and not isinstance(value, (bool, Integral)):
        if not math.isfinite(float(value)):
            raise ResultValidationError(f"{path} 不能是 NaN 或 Infinity")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_non_finite(item, f"{path}[{index}]")


def _validate_polygon(geometry: dict, pile_index: int, field_name: str = "geometry") -> None:
    coordinates = geometry["coordinates"]
    prefix = f"piles[{pile_index}].{field_name}.coordinates"
    for ring_index, ring in enumerate(coordinates):
        ring_prefix = f"{prefix}[{ring_index}]"
        if ring[0] != ring[-1]:
            raise ResultValidationError(f"{ring_prefix} 首尾坐标必须闭合")
        for point_index, point in enumerate(ring):
            lon = _finite_number(point[0], f"{ring_prefix}[{point_index}][0]")
            lat = _finite_number(point[1], f"{ring_prefix}[{point_index}][1]")
            if not -180.0 <= lon <= 180.0:
                raise ResultValidationError(f"{ring_prefix}[{point_index}] 经度超出 [-180, 180]")
            if not -90.0 <= lat <= 90.0:
                raise ResultValidationError(f"{ring_prefix}[{point_index}] 纬度超出 [-90, 90]")
    polygon = shape(geometry)
    if polygon.is_empty or polygon.area <= 0:
        raise ResultValidationError(f"piles[{pile_index}].{field_name} 不能为空或零面积")
    if not polygon.is_valid:
        raise ResultValidationError(f"piles[{pile_index}].{field_name} 不是有效 Polygon")


def validate_boundary_result(result: dict) -> None:
    """Validate the version 1.0 boundary result before persistence or upload."""
    if not isinstance(result, dict):
        raise ResultValidationError("result 必须是 JSON 对象")

    # jsonschema's handling of IEEE special values varies by keyword. Reject
    # them explicitly before schema validation so errors stay deterministic.
    _reject_non_finite(result)
    errors = sorted(_validator().iter_errors(result), key=lambda error: _path(error.absolute_path))
    if errors:
        error = errors[0]
        raise ResultValidationError(f"{_path(error.absolute_path)}：{error.message}")

    summary = result["summary"]
    piles = result["piles"]
    declared_count = _integer(summary["pile_count"], "summary.pile_count")
    if declared_count != len(piles):
        raise ResultValidationError(
            f"summary.pile_count={declared_count} 与 piles 数量 {len(piles)} 不一致"
        )

    grade_summary = summary["quality_grades"]
    if set(grade_summary) != set(GRADE_KEYS):
        raise ResultValidationError("summary.quality_grades 必须完整包含 A、B、C，零计数也不能省略")

    seen_ids: set[int] = set()
    grade_counts = {grade: 0 for grade in GRADE_KEYS}
    areas: list[float] = []
    volumes: list[float] = []
    numeric_fields = (
        "area_m2",
        "volume_m3",
        "max_relative_height_m",
        "mean_relative_height_m",
        "quality_score",
        "boundary_confidence",
    )
    for index, pile in enumerate(piles):
        pile_id = _integer(pile["pile_id"], f"piles[{index}].pile_id")
        if pile_id in seen_ids:
            raise ResultValidationError(f"piles[{index}].pile_id={pile_id} 在当前任务中重复")
        seen_ids.add(pile_id)

        values = {
            field: _finite_number(pile[field], f"piles[{index}].{field}")
            for field in numeric_fields
        }
        areas.append(values["area_m2"])
        volumes.append(values["volume_m3"])
        _integer(pile["pixel_count"], f"piles[{index}].pixel_count")
        grade_counts[pile["quality_grade"]] += 1
        _validate_polygon(pile["geometry"], index)
        if pile.get("display_geometry") is not None:
            _validate_polygon(pile["display_geometry"], index, "display_geometry")

    for grade in GRADE_KEYS:
        declared_grade_count = _integer(grade_summary[grade], f"summary.quality_grades.{grade}")
        if declared_grade_count != grade_counts[grade]:
            raise ResultValidationError(
                f"summary.quality_grades.{grade}={declared_grade_count} "
                f"与 piles 重算数量 {grade_counts[grade]} 不一致"
            )

    declared_area = _finite_number(summary["total_area_m2"], "summary.total_area_m2")
    calculated_area = math.fsum(areas)
    if not _close_enough(declared_area, calculated_area):
        raise ResultValidationError(
            f"summary.total_area_m2={declared_area} 与 piles 重算值 {calculated_area} 不一致"
        )

    declared_volume = _finite_number(summary["total_volume_m3"], "summary.total_volume_m3")
    calculated_volume = math.fsum(volumes)
    if not _close_enough(declared_volume, calculated_volume):
        raise ResultValidationError(
            f"summary.total_volume_m3={declared_volume} 与 piles 重算值 {calculated_volume} 不一致"
        )
