from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from unittest.mock import patch

from coal_boundary.integration.models import sha256_json
from coal_boundary.integration.client import ApiError
from coal_boundary.integration.reporter import BackendReporter, build_reporter
from coal_boundary.pile_volume_validation import PileVolumeValidationError, validate_pile_volume_result
from coal_boundary.worker.path_guard import InputValidationError, resolve_output_directory, resolve_source_artifact, resolve_source_result_directory, validate_local_input
from coal_boundary.integration.client import BackendApiClient
from coal_boundary.worker.client import Claim, WorkerApiClient
from coal_boundary.worker.pile_volume import PileRecalculationError, validate_recalculation_source
from coal_boundary.worker.service import _recalculation_context
from coal_boundary.config import load_config

from test_integration import _config, _mock_backend, _result, _run_context


def test_path_guard_accepts_relative_las_and_default_isolated_output(tmp_path):
    las_root = tmp_path / "las-root"
    las_root.mkdir()
    (las_root / "one.las").write_bytes(b"LASF")
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    validated = validate_local_input(
        {"root_id": "coal-las", "relative_path": ".", "las_subdirectory": ".", "file_pattern": "*.las"},
        {"coal-las": las_root}, stable_seconds=0,
    )
    root_id, output = resolve_output_directory(
        {"root_id": "coal-results"}, {"coal-results": output_root}, "job-123", "run-456",
    )
    assert validated.files == ((las_root / "one.las").resolve(),)
    assert root_id == "coal-results"
    assert output == (output_root / "job-123" / "run-456").resolve()


@pytest.mark.parametrize("relative", ["C:\\Windows", "\\\\server\\share", "../secret", "ok/../../secret", "/root"])
def test_path_guard_rejects_absolute_unc_and_parent_escape(tmp_path, relative):
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(InputValidationError, match="相对路径|不允许|越出"):
        validate_local_input(
            {"root_id": "coal-las", "relative_path": relative, "file_pattern": "*.las"},
            {"coal-las": root}, stable_seconds=0,
        )


def test_path_guard_detects_unstable_input(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    file = root / "one.las"
    file.write_bytes(b"LASF")
    def mutate(_seconds):
        file.write_bytes(b"LASF-more")
    with pytest.raises(InputValidationError) as raised:
        validate_local_input(
            {"root_id": "coal-las", "relative_path": ".", "file_pattern": "*.las"},
            {"coal-las": root}, stable_seconds=1, sleep=mutate,
        )
    assert raised.value.code == "INPUT_NOT_READY"


@pytest.mark.parametrize("source_id", ["../outside", "C:\\Windows\\Temp", "a/b", "short"])
def test_recalculation_source_ids_cannot_escape_output_root(tmp_path, source_id):
    with pytest.raises(InputValidationError) as raised:
        resolve_source_result_directory(tmp_path, source_id, "valid-source-run-0001")
    assert raised.value.code == "SOURCE_VERSION_CONFLICT"


def test_recalculation_source_sha_and_original_boundary_are_verified(tmp_path):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text('project:\n  las_dir: .\n  output_dir: output\nmodel:\n  enabled: false\n', encoding='utf-8')
    config = load_config(config_path)
    geometry = {"type": "Polygon", "coordinates": [[[120.0, 30.0], [120.001, 30.0],
        [120.001, 30.001], [120.0, 30.001], [120.0, 30.0]]]}
    fingerprint = "a" * 64
    source = {"metadata": {"dataset_fingerprint": fingerprint},
              "piles": [{"pile_id": 60, "geometry": geometry}]}
    source_path = tmp_path / "coal_piles_full.json"
    source_path.write_text(json.dumps(source), encoding="utf-8")
    context = {"recalculation_id": "recalc-0000000001", "source_job_id": "source-job-000001",
               "source_run_id": "source-run-000001", "source_pile_id": 60,
               "source_result_sha256": sha256_json(source), "dataset_fingerprint": fingerprint,
               "selection_mode": "EXISTING_BOUNDARY", "geometry_crs": "EPSG:4326",
               "geometry": geometry, "source_metrics": {"area_m2": 1, "volume_m3": 1}}
    validate_recalculation_source(context, source_path, fingerprint, config)
    context["source_result_sha256"] = "b" * 64
    with pytest.raises(PileRecalculationError) as raised:
        validate_recalculation_source(context, source_path, fingerprint, config)
    assert raised.value.code == "SOURCE_VERSION_CONFLICT"
    context["source_result_sha256"] = sha256_json(source)
    context["geometry"] = {"type": "Polygon", "coordinates": [[[120.0, 30.0], [120.002, 30.0],
        [120.002, 30.001], [120.0, 30.001], [120.0, 30.0]]]}
    with pytest.raises(PileRecalculationError) as raised:
        validate_recalculation_source(context, source_path, fingerprint, config)
    assert raised.value.code == "SOURCE_VERSION_CONFLICT"


def test_recalculation_source_artifact_rejects_external_symlink(tmp_path):
    source_dir = tmp_path / "source-job-000001" / "source-run-000001"
    source_dir.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    link = source_dir / "coal_piles_full.json"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("当前 Windows 环境不允许创建符号链接")
    with pytest.raises(InputValidationError) as raised:
        resolve_source_artifact(source_dir, link.name)
    assert raised.value.code == "SOURCE_VERSION_CONFLICT"


def test_recalculation_context_falls_back_to_job_id_when_backend_omits_recalculation_id():
    """后端 P1 说明中的 claim 上下文不含 recalculation_id，Worker 必须能继续执行。"""
    claim = Claim({
        "job_id": "11111111111111111111111111111111",
        "run_id": "22222222222222222222222222222222",
        "job_type": "PILE_VOLUME_RECALC",
        "attempt": 1,
        "lease_id": "lease-0000000001",
        "recalculation": {
            "source_job_id": "33333333333333333333333333333333",
            "source_run_id": "44444444444444444444444444444444",
            "source_pile_id": 62,
            "source_result_sha256": "a" * 64,
            "dataset_fingerprint": "b" * 64,
            "selection_mode": "EXISTING_BOUNDARY",
            "geometry": {"type": "Polygon", "coordinates": [[[120.0, 30.0], [120.001, 30.0],
                                                                 [120.001, 30.001], [120.0, 30.0]]]},
            "source_metrics": {"area_m2": 1.0, "volume_m3": 2.0},
        },
    })
    context = _recalculation_context(claim)
    assert context["recalculation_id"] == "11111111111111111111111111111111"
    # 后端自己带上的值优先，不被覆盖
    claim.payload["recalculation"]["recalculation_id"] = "55555555555555555555555555555555"
    assert _recalculation_context(claim)["recalculation_id"] == "55555555555555555555555555555555"
    # 完全缺少上下文时拒绝，不能退化成全场任务
    with pytest.raises(ValueError, match="recalculation"):
        _recalculation_context(Claim({"job_id": "job-000000000001"}))


def test_build_reporter_passes_configured_local_artifact_root(tmp_path):
    """手工运行（GUI/CLI）没有 claim 上下文，LOCAL_REGISTER 的登记根必须来自配置。"""
    cfg = _config(tmp_path, "http://backend.invalid")
    cfg.artifact_mode = "LOCAL_REGISTER"
    cfg.artifact_root_id = "coal-results"
    cfg.artifact_root = tmp_path / "results"
    reporter = build_reporter(cfg)
    assert reporter.artifact_root_id == "coal-results"
    assert Path(reporter.artifact_root) == (tmp_path / "results").resolve()
    reporter.close()
    # 未配置时保持可缺省，不阻塞手工运行
    cfg.artifact_root = None
    cfg.artifact_root_id = None
    fallback = build_reporter(cfg)
    assert fallback.artifact_root is None
    fallback.close()


def test_claimed_reporter_preserves_stable_recalculation_failure_code(tmp_path):
    cfg = _config(tmp_path, "http://backend.invalid")
    reporter = BackendReporter(cfg, claimed_job={"job_id": "job-000000000001",
        "run_id": "run-000000000001", "worker_id": "worker-01", "lease_id": "lease-0000000001"})
    reporter._sync_spool = lambda *_args, **_kwargs: None
    reporter.start_run(_run_context())
    reporter.fail(PileRecalculationError("SOURCE_VERSION_CONFLICT", "源结果已变化"), "RECALC_VALIDATE")
    terminal = reporter.spool.load_terminal()
    assert terminal["payload"]["error_code"] == "SOURCE_VERSION_CONFLICT"
    assert terminal["payload"]["stage_code"] == "RECALC_VALIDATE"
    reporter.close()


def test_claimed_reporter_reuses_run_and_completes_before_local_artifact_registration(tmp_path):
    with _mock_backend() as (base_url, state):
        cfg = _config(tmp_path, base_url)
        cfg.artifact_mode = "LOCAL_REGISTER"
        output_root = tmp_path / "results"
        output_dir = output_root / "job-000000000001" / "run-000000000001"
        output_dir.mkdir(parents=True)
        reporter = BackendReporter(
            cfg,
            claimed_job={"job_id": "job-000000000001", "run_id": "run-000000000001",
                         "worker_id": "worker-01", "lease_id": "lease-0000000001"},
            artifact_root_id="coal-results", artifact_root=output_root,
        )
        reporter.start_run(_run_context())
        result = _result()
        digest = reporter.submit_result("BOUNDARY_RESULT", result)
        artifact = output_dir / "coal_piles_full.json"
        artifact.write_text(json.dumps(result), encoding="utf-8")
        reporter.upload_artifact("BUSINESS_JSON", artifact)
        reporter.complete(result["summary"], 1.0, digest)
        reporter.close()
    paths = [(item["method"], item["path"]) for item in state.requests]
    assert ("POST", "/api/v1/coal-boundary/runs") not in paths
    complete_index = paths.index(("POST", "/api/v1/coal-boundary/runs/run-000000000001/complete"))
    artifact_index = paths.index(("POST", "/api/v1/coal-boundary/runs/run-000000000001/artifacts:register"))
    assert complete_index < artifact_index


def test_optional_artifact_failure_does_not_block_claimed_run_complete(tmp_path):
    cfg = _config(tmp_path, "http://backend.invalid")
    cfg.artifact_mode = "LOCAL_REGISTER"
    output_root = tmp_path / "results"
    output_root.mkdir()
    reporter = BackendReporter(
        cfg, claimed_job={"job_id": "job-000000000001", "run_id": "run-000000000001",
                          "worker_id": "worker-01", "lease_id": "lease-0000000001"},
        artifact_root_id="coal-results", artifact_root=output_root,
    )
    calls = []
    def request(method, path, payload=None, **kwargs):
        calls.append((method, path))
        if path.endswith("events:batch"):
            events = payload["events"]
            return {"accepted_through_seq": events[-1]["seq"]}
        if path.endswith("artifacts:register"):
            raise ApiError(500, "OBJECT_STORAGE_UNAVAILABLE", "附件暂不可用")
        return {"saved": True}
    reporter.client.request_json = request
    reporter.start_run(_run_context())
    digest = reporter.submit_result("BOUNDARY_RESULT", _result())
    artifact = output_root / "result.json"
    artifact.write_text("{}", encoding="utf-8")
    reporter.upload_artifact("BUSINESS_JSON", artifact)
    reporter.complete(_result()["summary"], 1.0, digest)
    assert ("POST", "/api/v1/coal-boundary/runs/run-000000000001/complete") in calls
    spool_dir = next((tmp_path / "spool").iterdir())
    assert json.loads((spool_dir / "run.json").read_text(encoding="utf-8"))["terminal_acknowledged"] is True
    assert (spool_dir / "artifacts.json").exists()
    reporter.close()


def _pile_result():
    result = {
        "result_kind": "PILE_VOLUME_RESULT",
        "schema_version": "1.0", "created_at": "2026-09-15T03:00:00Z",
        "recalculation_id": "recalc-0000000001", "source_job_id": "source-job-000001",
        "source_run_id": "source-run-000001", "source_pile_id": 60,
        "pile_id": 60, "quality_score": 0.9, "quality_grade": "A", "pixel_count": 100,
        "source_result_sha256": "a" * 64, "dataset_fingerprint": "b" * 64,
        "selection_mode": "USER_DRAWN_BOUNDARY", "crs": {"source_epsg": 4549, "output_epsg": 4326},
        "geometry": {"type": "Polygon", "coordinates": [[[120.0, 30.0], [120.001, 30.0], [120.001, 30.001], [120.0, 30.001], [120.0, 30.0]]]},
        "metrics": {"area_m2": 100.0, "volume_m3": 120.0, "max_relative_height_m": 3.0,
                    "mean_relative_height_m": 1.2, "point_count": 1000, "valid_cell_count": 100},
        "ground_surface": {"method": "local_sloped_surface", "cache_reused": False, "quality_score": 0.9},
        "quality": {"score": 0.9, "grade": "A", "warnings": []},
        "comparison": {"source_area_m2": 90.0, "source_volume_m3": 100.0,
                       "area_delta_m2": 10.0, "volume_delta_m3": 20.0, "volume_delta_percent": 20.0},
        "runtime_seconds": 0.2,
    }
    return result


def test_pile_volume_result_schema_and_comparison_validation():
    result = _pile_result()
    validate_pile_volume_result(result)
    result["comparison"]["volume_delta_m3"] = 19.0
    with pytest.raises(PileVolumeValidationError, match="不一致"):
        validate_pile_volume_result(result)


def test_pile_volume_result_schema_rejects_unknown_flat_metrics():
    """顶层仍禁止未知字段：扁平的 pile_id/quality_score 等必须与 schema 同步，不能随手新增。"""
    result = _pile_result()
    result["unsupported_metric"] = 1.0
    with pytest.raises(PileVolumeValidationError):
        validate_pile_volume_result(result)


def test_worker_claim_sends_worker_header_and_unwraps_claim():
    requests = []
    class Response:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self):
            return json.dumps({"code": "OK", "message": "", "request_id": "r", "data": {
                "job_id": "job-000000000001", "run_id": "run-000000000001", "job_type": "EXTRACTION",
                "lease_id": "lease-0000000001"
            }}).encode()
    def urlopen(request, **_kwargs):
        requests.append(request)
        return Response()
    api = BackendApiClient("http://backend.invalid", "token", timeout=1)
    worker = WorkerApiClient(api, "coal-worker-01")
    with patch("urllib.request.urlopen", side_effect=urlopen):
        claim = worker.claim(("EXTRACTION",), 90, idempotency_key="claim-test-0001")
    assert claim.job_id == "job-000000000001"
    assert requests[0].get_header("X-worker-id") == "coal-worker-01"
    assert requests[0].get_header("Idempotency-key") == "claim-test-0001"
