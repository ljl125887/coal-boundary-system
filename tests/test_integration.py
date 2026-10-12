from __future__ import annotations

from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from coal_boundary.config import IntegrationConfig, load_config
from coal_boundary.integration import BackendReporter, NoOpReporter, build_reporter
from coal_boundary.integration.client import ApiError, BackendApiClient
from coal_boundary.integration.models import sha256_json
from coal_boundary.result_validation import ResultValidationError


class _State:
    def __init__(self):
        self.requests: list[dict] = []


@contextmanager
def _mock_backend():
    state = _State()

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, data, status=200):
            body = json.dumps({"code": "OK", "message": "", "request_id": "test", "data": data}).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            state.requests.append({"method": "GET", "path": self.path})
            self._reply({"items": []})

        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            content_type = self.headers.get("Content-Type", "")
            payload = json.loads(raw.decode()) if raw and content_type.startswith("application/json") else None
            state.requests.append({"method": "POST", "path": self.path, "payload": payload, "raw": raw})
            if self.path == "/api/v1/coal-boundary/runs":
                self._reply({"run_id": "server-run-1"}, 201)
            elif self.path.endswith("/events:batch"):
                self._reply({"accepted_count": len(payload["events"]), "accepted_through_seq": payload["events"][-1]["seq"]})
            else:
                self._reply({"saved": True}, 201 if self.path.endswith("/artifacts") else 200)

        def do_PUT(self):
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode())
            state.requests.append({"method": "PUT", "path": self.path, "payload": payload})
            self._reply({"saved": True})

        def log_message(self, format, *args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        thread.join(timeout=3)
        server.server_close()


def _config(tmp_path: Path, base_url: str, enabled=True):
    return SimpleNamespace(
        enabled=enabled,
        base_url=base_url,
        token_env="COAL_BOUNDARY_TEST_TOKEN",
        connect_timeout_seconds=0.2,
        read_timeout_seconds=0.2,
        verify_tls=True,
        event_batch_size=2,
        flush_interval_seconds=60.0,
        heartbeat_interval_seconds=15.0,
        max_retries=0,
        spool_dir=tmp_path / "spool",
        upload_artifacts=("BUSINESS_JSON",),
        upload_large_diagnostics=False,
    )


def _run_context():
    return {
        "run_type": "EXTRACTION",
        "project_name": "中文煤场",
        "dataset": {"fingerprint": "a" * 64, "file_count": 1, "point_count": 100, "source_epsg": 4549},
        "algorithm": {"name": "coal-boundary-system", "version": "0.1.0", "config_hash": "b" * 64, "model_version": None},
    }


def _result():
    return {
        "schema_version": "1.0",
        "created_at": "2026-09-10T04:00:00Z",
        "metadata": {},
        "crs": {"source_epsg": 4549, "output_epsg": 4326},
        "summary": {
            "pile_count": 0,
            "total_area_m2": 0.0,
            "total_volume_m3": 0.0,
            "quality_grades": {"A": 0, "B": 0, "C": 0},
        },
        "evaluation": None,
        "piles": [],
    }


def test_http_200_business_error_is_not_treated_as_success():
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps({"code": 401, "msg": "认证失败"}, ensure_ascii=False).encode("utf-8")

    client = BackendApiClient("http://backend.invalid", None, timeout=1.0)
    with patch("urllib.request.urlopen", return_value=Response()):
        with pytest.raises(ApiError) as raised:
            client.request_json("GET", "/api/v1/coal-boundary/runs")

    assert raised.value.status == 401
    assert raised.value.code == "401"


def test_client_automatically_logs_in_and_reuses_in_memory_token():
    requests = []

    class Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")

    def urlopen(request, **_kwargs):
        requests.append(request)
        if request.full_url.endswith("/login"):
            assert request.get_header("Authorization") is None
            credentials = json.loads(request.data.decode("utf-8"))
            assert credentials == {"username": "operator", "password": "temporary-secret"}
            return Response({"code": 200, "msg": "操作成功", "token": "fresh-token"})
        assert request.get_header("Authorization") == "Bearer fresh-token"
        return Response({"code": "OK", "data": {"items": []}})

    client = BackendApiClient(
        "http://backend.invalid",
        None,
        timeout=1.0,
        username="operator",
        password="temporary-secret",
    )
    with patch("urllib.request.urlopen", side_effect=urlopen):
        client.request_json("GET", "/api/v1/coal-boundary/runs?page=1&size=1")
        client.request_json("GET", "/api/v1/coal-boundary/runs?page=1&size=1")

    assert [request.full_url for request in requests].count("http://backend.invalid/login") == 1
    assert client.token == "fresh-token"


def test_client_relogs_once_after_http_200_business_401():
    paths = []

    class Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload, ensure_ascii=False).encode("utf-8")

    def urlopen(request, **_kwargs):
        paths.append(request.full_url)
        if request.full_url.endswith("/login"):
            return Response({"code": 200, "token": "renewed-token"})
        if request.get_header("Authorization") == "Bearer expired-token":
            return Response({"code": 401, "msg": "认证失败"})
        assert request.get_header("Authorization") == "Bearer renewed-token"
        return Response({"code": "OK", "data": {"items": []}})

    client = BackendApiClient(
        "http://backend.invalid",
        "expired-token",
        timeout=1.0,
        username="operator",
        password="temporary-secret",
    )
    with patch("urllib.request.urlopen", side_effect=urlopen):
        client.request_json("GET", "/api/v1/coal-boundary/runs?page=1&size=1")

    assert paths == [
        "http://backend.invalid/api/v1/coal-boundary/runs?page=1&size=1",
        "http://backend.invalid/login",
        "http://backend.invalid/api/v1/coal-boundary/runs?page=1&size=1",
    ]
    assert client.token == "renewed-token"


def test_artifact_upload_uses_automatic_login(tmp_path):
    requests = []

    class Response:
        status = 200

        def __init__(self, payload):
            self.payload = payload

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(self.payload).encode("utf-8")

    def urlopen(request, **_kwargs):
        requests.append(request)
        if request.full_url.endswith("/login"):
            return Response({"code": 200, "token": "artifact-token"})
        assert request.get_header("Authorization") == "Bearer artifact-token"
        return Response({"code": "OK", "data": {"saved": True}})

    artifact_path = tmp_path / "evaluation.json"
    artifact_path.write_text("{}", encoding="utf-8")
    client = BackendApiClient(
        "http://backend.invalid",
        None,
        timeout=1.0,
        username="operator",
        password="temporary-secret",
    )
    with patch("urllib.request.urlopen", side_effect=urlopen):
        response = client.upload_artifact(
            "run-1",
            {"artifact_type": "EVALUATION_JSON", "file_name": "evaluation.json", "sha256": "a" * 64},
            artifact_path,
        )

    assert response == {"saved": True}
    assert len(requests) == 2


def test_disabled_integration_is_noop(tmp_path):
    reporter = build_reporter(_config(tmp_path, "http://127.0.0.1:1", enabled=False))
    assert isinstance(reporter, NoOpReporter)
    assert reporter.submit_result("BOUNDARY_RESULT", _result()) == sha256_json(_result())
    assert not (tmp_path / "spool").exists()


def test_backend_reporter_sends_run_events_result_artifact_and_complete(tmp_path, monkeypatch):
    monkeypatch.setenv("COAL_BOUNDARY_TEST_TOKEN", "secret-token-not-for-logs")
    with _mock_backend() as (base_url, state):
        reporter = BackendReporter(_config(tmp_path, base_url))
        reporter.start_run(_run_context())
        reporter.stage_started("SURFACE_GRID", "建立或加载表面栅格")
        reporter.progress("SURFACE_GRID", 20.0, current=1, total=2, message="处理中文文件")
        result = _result()
        result_hash = reporter.submit_result("BOUNDARY_RESULT", result)
        artifact = tmp_path / "coal_piles_full.json"
        artifact.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
        reporter.upload_artifact("BUSINESS_JSON", artifact)
        reporter.complete(result["summary"], 1.25, result_hash)
        reporter.close()

    paths = [(item["method"], item["path"]) for item in state.requests]
    assert ("POST", "/api/v1/coal-boundary/runs") in paths
    assert ("POST", "/api/v1/coal-boundary/runs/server-run-1/events:batch") in paths
    assert ("PUT", "/api/v1/coal-boundary/runs/server-run-1/result") in paths
    assert ("POST", "/api/v1/coal-boundary/runs/server-run-1/artifacts") in paths
    assert ("POST", "/api/v1/coal-boundary/runs/server-run-1/complete") in paths
    submitted = next(item["payload"] for item in state.requests if item["method"] == "PUT")
    assert submitted["result_kind"] == "BOUNDARY_RESULT"
    assert submitted["content_sha256"] == sha256_json(result)
    event_messages = [
        event["message"]
        for item in state.requests if item["path"].endswith("events:batch")
        for event in item["payload"]["events"]
    ]
    assert "处理中文文件" in event_messages
    assert not (tmp_path / "spool").exists()


def test_offline_backend_never_raises_and_leaves_replayable_spool(tmp_path):
    reporter = BackendReporter(_config(tmp_path, "http://127.0.0.1:1"))
    reporter.start_run(_run_context())
    reporter.stage_started("MODEL_INFERENCE", "执行模型推理")
    result_hash = reporter.submit_result("BOUNDARY_RESULT", _result())
    reporter.complete(_result()["summary"], 2.0, result_hash)
    reporter.close()

    run_dirs = list((tmp_path / "spool").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "run.json").exists()
    assert (run_dirs[0] / "events.jsonl").exists()
    assert (run_dirs[0] / "result.json").exists()
    assert (run_dirs[0] / "terminal.json").exists()


def test_invalid_boundary_result_is_rejected_before_spool_save(tmp_path):
    reporter = BackendReporter(_config(tmp_path, "http://127.0.0.1:1"))
    reporter.start_run(_run_context())
    invalid = _result()
    invalid["crs"]["output_epsg"] = 4549

    with pytest.raises(ResultValidationError, match="4326"):
        reporter.submit_result("BOUNDARY_RESULT", invalid)

    run_dirs = list((tmp_path / "spool").iterdir())
    assert len(run_dirs) == 1
    assert not (run_dirs[0] / "result.json").exists()
    reporter.close()


def test_load_config_resolves_integration_spool_and_defaults_disabled(tmp_path):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: test
  las_dir: las
  output_dir: output
model:
  enabled: false
integration:
  enabled: false
  spool_dir: pending-sync
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert isinstance(config.integration, IntegrationConfig)
    assert config.integration.enabled is False
    assert config.integration.spool_dir == (tmp_path / "pending-sync").resolve()
    assert config.integration.login_path == "/login"
    assert config.integration.username_env == "COAL_BOUNDARY_API_USERNAME"
    assert config.integration.password_env == "COAL_BOUNDARY_API_PASSWORD"
