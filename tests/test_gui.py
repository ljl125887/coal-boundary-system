import json
import os
import subprocess
import sys

import pytest

from coal_boundary.gui import (
    FIELD_SPECS,
    atomic_save_config,
    boundary_view_transform,
    build_multipart_payload,
    field_values_from_config,
    find_result_file,
    load_result_document,
    resolve_cli_python,
    update_raw_config,
    utf8_subprocess_environment,
)


def _complete_raw_config():
    raw = {"custom": {"keep_me": 42}}
    for spec in FIELD_SPECS:
        section = raw.setdefault(spec.section, {})
        if spec.optional:
            section[spec.key] = None
        elif spec.value_type is int:
            section[spec.key] = 10
        elif spec.value_type is float:
            section[spec.key] = 1.5
        else:
            section[spec.key] = "value"
    raw["project"].update({"name": "test", "las_dir": "las", "output_dir": "outputs"})
    raw["model"]["enabled"] = True
    return raw


def test_gui_config_update_preserves_unknown_sections_and_types(tmp_path):
    raw = _complete_raw_config()
    values = field_values_from_config(raw)
    values[("segmentation", "valley_depth_m")] = "2.25"
    values[("raster", "chunk_points")] = "1500000"
    values[("model", "enabled")] = False

    updated = update_raw_config(raw, values)

    assert updated["custom"] == {"keep_me": 42}
    assert updated["segmentation"]["valley_depth_m"] == 2.25
    assert updated["raster"]["chunk_points"] == 1_500_000
    assert updated["model"]["enabled"] is False

    path = tmp_path / "config.yaml"
    path.write_text("old: true\n", encoding="utf-8")
    atomic_save_config(path, updated)
    assert path.with_suffix(".yaml.gui-backup").read_text(encoding="utf-8") == "old: true\n"


@pytest.mark.parametrize("document_type", ["business", "geojson", "service"])
def test_result_loader_supports_all_upload_formats(tmp_path, document_type):
    ring = [[120.0, 30.0], [120.1, 30.0], [120.1, 30.1], [120.0, 30.0]]
    properties = {
        "pile_id": 3,
        "area_m2": 120.0,
        "volume_m3": 456.0,
        "max_relative_height_m": 7.0,
        "mean_relative_height_m": 3.0,
        "quality_score": 0.91,
        "quality_grade": "A",
        "boundary_confidence": 0.89,
    }
    if document_type == "business":
        document = {
            "summary": {"pile_count": 1, "total_volume_m3": 456.0},
            "evaluation": {"total_volume_error_percent": 1.2},
            "piles": [{**properties, "geometry": {"type": "Polygon", "coordinates": [ring]}}],
        }
    elif document_type == "geojson":
        document = {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "id": 3, "properties": properties, "geometry": {"type": "Polygon", "coordinates": [ring]}}],
        }
    else:
        document = [{"num": 3, "volume_m3": 456.0, "points": [{"lon": point[0], "lat": point[1]} for point in ring]}]
    path = tmp_path / f"result_{document_type}.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    piles, summary = load_result_document(path)

    assert len(piles) == 1
    assert piles[0].pile_id == "3"
    assert piles[0].volume_m3 == 456.0
    assert len(piles[0].coordinates) == 3
    assert summary["pile_count"] == 1


def test_find_result_and_multipart_payload(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("project: {}\n", encoding="utf-8")
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    result_path = output_dir / "coal_piles_full.json"
    result_path.write_text("[]", encoding="utf-8")
    raw = {"project": {"output_dir": "outputs"}}

    assert find_result_file(config_path, raw) == result_path
    payload, boundary = build_multipart_payload(result_path)
    assert boundary.encode("ascii") in payload
    assert b'filename="coal_piles_full.json"' in payload
    assert payload.endswith(f"--{boundary}--\r\n".encode("ascii"))


def test_result_loader_prefers_display_geometry(tmp_path):
    calculation_ring = [[120.0, 30.0], [120.1, 30.0], [120.1, 30.1], [120.0, 30.0]]
    display_ring = [[120.0, 40.0], [120.1, 40.0], [120.1, 40.1], [120.0, 40.0]]
    document = {
        "piles": [{
            "pile_id": 1,
            "geometry": {"type": "Polygon", "coordinates": [calculation_ring]},
            "display_geometry": {"type": "Polygon", "coordinates": [display_ring]},
        }],
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    piles, _ = load_result_document(path)

    assert piles[0].coordinates[0] == (120.0, 40.0)


def test_boundary_view_transform_accounts_for_wide_canvas_letterboxing():
    transform = boundary_view_transform(
        (0.0, 0.0, 100.0, 1000.0),
        (50.0, 500.0),
        1.0,
        1800,
        900,
    )
    min_x, min_y, max_x, max_y, _, _, _ = transform

    assert min_x < 0.0
    assert max_x > 100.0
    assert min_y == pytest.approx(0.0)
    assert max_y == pytest.approx(1000.0)
    assert (min_x + max_x) / 2 == pytest.approx(50.0)


def test_python_child_process_emits_utf8_chinese():
    environment = utf8_subprocess_environment(
        {**os.environ, "PYTHONIOENCODING": "gbk"},
    )

    completed = subprocess.run(
        [sys.executable, "-c", "print('读取并校验数据；路径：煤堆')"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="strict",
        env=environment,
        check=True,
    )

    assert completed.stdout.strip() == "读取并校验数据；路径：煤堆"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["PYTHONUTF8"] == "1"


def test_resolve_cli_python_honors_explicit_environment(monkeypatch, tmp_path):
    python_path = tmp_path / "python.exe"
    python_path.write_bytes(b"")
    monkeypatch.setenv("COAL_BOUNDARY_PYTHON", str(python_path))

    assert resolve_cli_python() == python_path.resolve()
