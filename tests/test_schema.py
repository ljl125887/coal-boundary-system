import json
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def test_result_schema_is_valid():
    schema = json.loads((ROOT / "schemas" / "coal-piles-result-1.0.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)


def test_packaged_result_schema_matches_project_schema():
    project_schema = (ROOT / "schemas" / "coal-piles-result-1.0.schema.json").read_text(encoding="utf-8")
    packaged_schema = files("coal_boundary").joinpath(
        "schemas", "coal-piles-result-1.0.schema.json"
    ).read_text(encoding="utf-8")
    assert json.loads(packaged_schema) == json.loads(project_schema)
