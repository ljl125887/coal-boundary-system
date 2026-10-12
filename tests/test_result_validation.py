from __future__ import annotations

from copy import deepcopy
import math

import pytest

from coal_boundary.result_validation import ResultValidationError, validate_boundary_result


def _pile(pile_id: int = 1) -> dict:
    return {
        "pile_id": pile_id,
        "area_m2": 100.0,
        "volume_m3": 250.0,
        "max_relative_height_m": 5.0,
        "mean_relative_height_m": 2.5,
        "quality_score": 0.9,
        "quality_grade": "A",
        "boundary_confidence": 0.8,
        "pixel_count": 100,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [120.00100, 30.00100],
                [120.00110, 30.00100],
                [120.00110, 30.00110],
                [120.00100, 30.00110],
                [120.00100, 30.00100],
            ]],
        },
    }


def _result(piles: list[dict] | None = None) -> dict:
    piles = piles if piles is not None else [_pile()]
    grades = {"A": 0, "B": 0, "C": 0}
    for pile in piles:
        grades[pile["quality_grade"]] += 1
    return {
        "schema_version": "1.0",
        "created_at": "2026-01-01T00:00:00.000Z",
        "metadata": {"project_name": "synthetic-project"},
        "crs": {"source_epsg": 4549, "output_epsg": 4326},
        "summary": {
            "pile_count": len(piles),
            "total_area_m2": math.fsum(pile["area_m2"] for pile in piles),
            "total_volume_m3": math.fsum(pile["volume_m3"] for pile in piles),
            "quality_grades": grades,
        },
        "evaluation": None,
        "piles": piles,
    }


def test_simple_square_and_synthetic_pile_values_are_valid():
    validate_boundary_result(_result())


def test_zero_grade_and_empty_result_are_valid():
    validate_boundary_result(_result([]))


def test_summary_accepts_tiny_float_transport_difference():
    result = _result()
    result["summary"]["total_area_m2"] += 1e-10
    result["summary"]["total_volume_m3"] -= 1e-10
    validate_boundary_result(result)


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_pile_value_is_rejected(invalid):
    result = _result()
    result["piles"][0]["volume_m3"] = invalid
    with pytest.raises(ResultValidationError, match="NaN|Infinity"):
        validate_boundary_result(result)


def test_unclosed_polygon_is_rejected():
    result = _result()
    result["piles"][0]["geometry"]["coordinates"][0][-1] = [120.00099, 30.00100]
    with pytest.raises(ResultValidationError, match="闭合"):
        validate_boundary_result(result)


def test_duplicate_pile_id_is_rejected():
    first = _pile(1)
    second = deepcopy(first)
    second["geometry"]["coordinates"][0] = [
        [120.00200, 30.00200],
        [120.00210, 30.00200],
        [120.00210, 30.00210],
        [120.00200, 30.00210],
        [120.00200, 30.00200],
    ]
    result = _result([first, second])
    with pytest.raises(ResultValidationError, match="重复"):
        validate_boundary_result(result)


def test_swapped_synthetic_coordinate_is_rejected_as_invalid_latitude():
    result = _result()
    result["piles"][0]["geometry"]["coordinates"][0][1] = [30.00100, 120.00110]
    with pytest.raises(ResultValidationError, match="纬度超出"):
        validate_boundary_result(result)


def test_wrong_output_crs_is_rejected():
    result = _result()
    result["crs"]["output_epsg"] = 4549
    with pytest.raises(ResultValidationError, match="4326"):
        validate_boundary_result(result)


def test_missing_zero_grade_key_is_rejected():
    result = _result()
    del result["summary"]["quality_grades"]["C"]
    with pytest.raises(ResultValidationError, match="C"):
        validate_boundary_result(result)


def test_inconsistent_summary_is_rejected():
    result = _result()
    result["summary"]["total_volume_m3"] += 10.0
    with pytest.raises(ResultValidationError, match="total_volume_m3"):
        validate_boundary_result(result)
