from __future__ import annotations

import json
import math
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from shapely.geometry import shape


class PileVolumeValidationError(ValueError):
    pass


def _schema() -> dict:
    path = Path(__file__).with_name("schemas") / "pile-volume-result-1.0.schema.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_pile_volume_result(result: dict) -> None:
    errors = sorted(
        Draft202012Validator(_schema(), format_checker=FormatChecker()).iter_errors(result),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "$"
        raise PileVolumeValidationError(f"单煤堆体积结果不符合 Schema（{location}）：{error.message}")
    for ring in result["geometry"]["coordinates"]:
        if ring[0] != ring[-1]:
            raise PileVolumeValidationError("geometry 的每个线环必须首尾闭合")
    polygon = shape(result["geometry"])
    if polygon.is_empty or not polygon.is_valid or polygon.area <= 0:
        raise PileVolumeValidationError("geometry 必须是闭合且有效的非空 Polygon")
    for section in ("metrics", "comparison"):
        for name, value in result[section].items():
            if value is not None and isinstance(value, (int, float)) and not math.isfinite(float(value)):
                raise PileVolumeValidationError(f"{section}.{name} 不能是 NaN 或 Infinity")
    source = float(result["comparison"]["source_volume_m3"])
    current = float(result["metrics"]["volume_m3"])
    expected_delta = current - source
    if abs(float(result["comparison"]["volume_delta_m3"]) - expected_delta) > 1e-5:
        raise PileVolumeValidationError("comparison.volume_delta_m3 与体积差不一致")
    expected_percent = None if source == 0 else expected_delta / source * 100.0
    actual_percent = result["comparison"]["volume_delta_percent"]
    if expected_percent is None:
        if actual_percent is not None:
            raise PileVolumeValidationError("源体积为零时 volume_delta_percent 必须为 null")
    elif actual_percent is None or abs(float(actual_percent) - expected_percent) > 1e-5:
        raise PileVolumeValidationError("comparison.volume_delta_percent 与体积差不一致")
