from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import platform
import re
import socket
import sys
import threading
import time
import mimetypes
from typing import Protocol

from ..result_validation import validate_boundary_result
from .client import ApiError, BackendApiClient
from .models import new_id, sha256_file, sha256_json, utc_now
from .spool import SpoolRun, iter_spool_runs
from .stages import stage_name, stage_start


def _safe_message(value: object) -> str:
    text = str(value)
    user_home = str(Path.home())
    if user_home:
        text = text.replace(user_home, "<user-home>")
    return re.sub(r"(?i)\b[A-Z]:\\[^\r\n\t]+", "<local-path>", text)


class ProgressReporter(Protocol):
    def start_run(self, context: dict) -> str | None: ...
    def stage_started(self, stage_code: str, message: str | None = None) -> None: ...
    def progress(self, stage_code: str, percent: float, *, current: int | None = None, total: int | None = None, message: str = "", context: dict | None = None) -> None: ...
    def metric(self, name: str, value: float, context: dict | None = None) -> None: ...
    def log(self, level: str, message: str, context: dict | None = None) -> None: ...
    def submit_result(self, result_kind: str, result: dict) -> str: ...
    def upload_artifact(self, artifact_type: str, path: str | Path) -> None: ...
    def complete(self, summary: dict, runtime_seconds: float, result_sha256: str) -> None: ...
    def fail(self, exception: BaseException, stage_code: str | None = None) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class NoOpReporter:
    def start_run(self, context: dict) -> None: return None
    def stage_started(self, stage_code: str, message: str | None = None) -> None: return None
    def progress(self, stage_code: str, percent: float, **kwargs) -> None: return None
    def metric(self, name: str, value: float, context: dict | None = None) -> None: return None
    def log(self, level: str, message: str, context: dict | None = None) -> None: return None
    def submit_result(self, result_kind: str, result: dict) -> str: return sha256_json(result)
    def upload_artifact(self, artifact_type: str, path: str | Path) -> None: return None
    def complete(self, summary: dict, runtime_seconds: float, result_sha256: str) -> None: return None
    def fail(self, exception: BaseException, stage_code: str | None = None) -> None: return None
    def flush(self) -> None: return None
    def close(self) -> None: return None


@dataclass
class ReporterStatus:
    client_run_id: str | None = None
    backend_run_id: str | None = None
    pending_events: int = 0
    last_sync_at: str | None = None
    last_error: str | None = None
    auth_required: bool = False


class BackendReporter:
    def __init__(
        self,
        config,
        *,
        token: str | None = None,
        username: str | None = None,
        password: str | None = None,
        claimed_job: dict | None = None,
        artifact_root_id: str | None = None,
        artifact_root: str | Path | None = None,
    ):
        self.config = config
        token_env = getattr(config, "token_env", "COAL_BOUNDARY_API_TOKEN")
        username_env = getattr(config, "username_env", "COAL_BOUNDARY_API_USERNAME")
        password_env = getattr(config, "password_env", "COAL_BOUNDARY_API_PASSWORD")
        token = token or (os.environ.get(token_env) if token_env else None)
        username = username or (os.environ.get(username_env) if username_env else None)
        password = password or (os.environ.get(password_env) if password_env else None)
        timeout = max(float(config.connect_timeout_seconds), float(config.read_timeout_seconds))
        self.client = BackendApiClient(
            config.base_url,
            token,
            timeout,
            config.verify_tls,
            login_path=getattr(config, "login_path", "/login"),
            username=username,
            password=password,
        )
        self.spool_root = Path(config.spool_dir)
        self.spool_root.mkdir(parents=True, exist_ok=True)
        self.spool: SpoolRun | None = None
        self.claimed_job = claimed_job
        self.artifact_root_id = artifact_root_id
        self.artifact_root = Path(artifact_root).resolve() if artifact_root else None
        self.status = ReporterStatus()
        self.seq = 0
        self.current_stage: str | None = None
        self.last_progress = 0.0
        self.last_flush = 0.0
        self.result_sha256 = ""
        self._last_warning = ""
        self._retry_count: dict[str, int] = {}
        self._next_retry_at: dict[str, float] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self.replay_pending()

    def _warning(self, message: str) -> None:
        safe = _safe_message(message)
        self.status.last_error = safe
        if safe != self._last_warning:
            print(f"[integration] 后端同步暂不可用，已保存在本地待补传：{safe}", flush=True)
            self._last_warning = safe

    def _record_error(self, exc: BaseException) -> None:
        if isinstance(exc, ApiError) and (
            exc.status == 401 or exc.code.upper() in {"401", "UNAUTHORIZED", "AUTHENTICATION_FAILED"}
        ):
            self.status.auth_required = True
        self._warning(str(exc))

    def _retry_key(self, spool: SpoolRun) -> str:
        return str(spool.path)

    def _can_attempt(self, spool: SpoolRun) -> bool:
        key = self._retry_key(spool)
        retries = self._retry_count.get(key, 0)
        if retries > int(self.config.max_retries):
            return False
        return time.monotonic() >= self._next_retry_at.get(key, 0.0)

    def _schedule_retry(self, spool: SpoolRun) -> None:
        key = self._retry_key(spool)
        count = self._retry_count.get(key, 0) + 1
        self._retry_count[key] = count
        delays = (1.0, 2.0, 5.0, 10.0, 30.0)
        index = min(count, len(delays)) - 1
        self._next_retry_at[key] = time.monotonic() + delays[index]

    def _clear_retry(self, spool: SpoolRun) -> None:
        key = self._retry_key(spool)
        self._retry_count.pop(key, None)
        self._next_retry_at.pop(key, None)

    def start_run(self, context: dict) -> str:
        client_run_id = context.get("client_run_id") or (self.claimed_job or {}).get("run_id") or new_id()
        payload = {
            "client_run_id": client_run_id,
            "run_type": context.get("run_type", "EXTRACTION"),
            "project_name": context["project_name"],
            "dataset": context["dataset"],
            "algorithm": context["algorithm"],
            "machine": context.get("machine") or {
                "hostname": socket.gethostname(),
                "os": platform.platform(),
                "python_version": platform.python_version(),
            },
            "started_at": context.get("started_at", utc_now()),
            "attempt": int(context.get("attempt", 1)),
        }
        binding = None
        if self.claimed_job:
            binding = {
                "backend_run_id": str(self.claimed_job["run_id"]),
                "job_id": str(self.claimed_job["job_id"]),
                "worker_id": str(self.claimed_job["worker_id"]),
                "lease_id": str(self.claimed_job["lease_id"]),
                "claimed": True,
            }
        self.spool = SpoolRun.create(self.spool_root, client_run_id, payload, binding=binding)
        self.status.client_run_id = client_run_id
        self.status.backend_run_id = self._ensure_backend_run(self.spool)
        self._start_heartbeat()
        return client_run_id

    def _start_heartbeat(self) -> None:
        if self._heartbeat_thread is not None:
            return

        def heartbeat_worker():
            interval = max(1.0, float(self.config.heartbeat_interval_seconds))
            while not self._stop.wait(interval):
                if self.spool is not None and self.current_stage:
                    self._emit(
                        "HEARTBEAT", "INFO", "任务仍在运行",
                        stage_code=self.current_stage, percent=self.last_progress,
                    )

        self._heartbeat_thread = threading.Thread(
            target=heartbeat_worker, name="coal-boundary-heartbeat", daemon=True,
        )
        self._heartbeat_thread.start()

    def _ensure_backend_run(self, spool: SpoolRun) -> str | None:
        state = spool.load_run()
        if state.get("backend_run_id"):
            return str(state["backend_run_id"])
        if not self._can_attempt(spool):
            return None
        try:
            payload = state["create_payload"]
            data = self.client.request_json(
                "POST", "/api/v1/coal-boundary/runs", payload,
                idempotency_key=f"create:{payload['client_run_id']}",
            )
            run_id = str(data.get("run_id") or data.get("id") or "")
            if not run_id:
                raise ApiError(None, "INVALID_RESPONSE", "创建任务响应缺少 run_id")
            spool.set_backend_run_id(run_id)
            if spool is self.spool:
                self.status.backend_run_id = run_id
            self._clear_retry(spool)
            print(f"[integration] 后端任务 ID：{run_id}", flush=True)
            return run_id
        except Exception as exc:
            self._schedule_retry(spool)
            self._record_error(exc)
            return None

    @staticmethod
    def _binding_headers(spool: SpoolRun) -> dict[str, str] | None:
        state = spool.load_run()
        worker_id, lease_id = state.get("worker_id"), state.get("lease_id")
        if not worker_id:
            return None
        headers = {"X-Worker-Id": str(worker_id)}
        if lease_id:
            headers["X-Lease-Id"] = str(lease_id)
        return headers

    def _emit(self, event_type: str, level: str, message: str, *, stage_code: str | None = None, percent: float | None = None, current: int | None = None, total: int | None = None, context: dict | None = None) -> None:
        with self._lock:
            if self.spool is None:
                return
            self.seq += 1
            if percent is not None:
                percent = max(self.last_progress, min(100.0, max(0.0, float(percent))))
                self.last_progress = percent
            event = {
                "event_id": new_id(), "seq": self.seq, "occurred_at": utc_now(),
                "event_type": event_type, "level": level,
                "stage_code": stage_code, "stage_name": stage_name(stage_code) if stage_code else None,
                "progress_percent": percent, "current": current, "total": total,
                "message": _safe_message(message), "context": context or {},
            }
            try:
                self.spool.append_event(event)
                if self.seq % self.config.event_batch_size == 0 or time.monotonic() - self.last_flush >= self.config.flush_interval_seconds:
                    self.flush()
            except Exception as exc:
                self._warning(f"本地事件队列写入失败：{type(exc).__name__}")

    def stage_started(self, stage_code: str, message: str | None = None) -> None:
        self.current_stage = stage_code
        self._emit("STAGE_STARTED", "INFO", message or stage_name(stage_code), stage_code=stage_code, percent=stage_start(stage_code))

    def progress(self, stage_code: str, percent: float, *, current: int | None = None, total: int | None = None, message: str = "", context: dict | None = None) -> None:
        self._emit("PROGRESS", "INFO", message or stage_name(stage_code), stage_code=stage_code, percent=percent, current=current, total=total, context=context)

    def metric(self, name: str, value: float, context: dict | None = None) -> None:
        payload = {"metric_name": name, "metric_value": float(value), **(context or {})}
        self._emit("METRIC", "INFO", f"{name}={value}", stage_code=self.current_stage, percent=self.last_progress, context=payload)

    def log(self, level: str, message: str, context: dict | None = None) -> None:
        self._emit("LOG", level.upper(), message, stage_code=self.current_stage, percent=self.last_progress, context=context)

    def submit_result(self, result_kind: str, result: dict) -> str:
        if result_kind == "BOUNDARY_RESULT":
            validate_boundary_result(result)
        elif result_kind == "PILE_VOLUME_RESULT":
            from ..pile_volume_validation import validate_pile_volume_result
            validate_pile_volume_result(result)
        else:
            raise ValueError(f"未知结果类型: {result_kind}")
        self.result_sha256 = sha256_json(result)
        if self.spool is not None:
            self.spool.save_result({
                "result_kind": result_kind,
                "result_schema_version": str(result.get("schema_version", "1.0")),
                "content_sha256": self.result_sha256,
                "result": result,
            })
            self._sync_spool(self.spool)
        return self.result_sha256

    def upload_artifact(self, artifact_type: str, path: str | Path) -> None:
        source = Path(path)
        if self.spool is None or not source.exists():
            return
        try:
            mode = str(getattr(self.config, "artifact_mode", "OBJECT_STORAGE_UPLOAD")).upper()
            if mode == "DISABLED":
                return
            digest = sha256_file(source)
            if mode == "LOCAL_REGISTER":
                if not self.artifact_root or not self.artifact_root_id:
                    raise ValueError("LOCAL_REGISTER 缺少 artifact_root_id/artifact_root")
                actual = source.resolve(strict=True)
                relative = actual.relative_to(self.artifact_root).as_posix()
                content_type = (
                    "application/geo+json" if source.suffix.lower() == ".geojson"
                    else mimetypes.guess_type(source.name)[0] or "application/octet-stream"
                )
                self.spool.add_local_artifact(
                    artifact_type, actual, digest, root_id=self.artifact_root_id,
                    relative_path=relative, content_type=content_type,
                )
            else:
                self.spool.add_artifact(artifact_type, source, digest)
            # 统一在 complete/fail 同步中处理。这样附件服务或对象存储故障
            # 永远不会先于结构化结果与任务终态，不能阻塞算法任务成功。
        except Exception as exc:
            self._warning(str(exc))

    def complete(self, summary: dict, runtime_seconds: float, result_sha256: str) -> None:
        self.stage_started("COMPLETED", "任务计算完成")
        if self.spool is None:
            return
        self.spool.save_terminal("complete", {
            "completed_at": utc_now(), "runtime_seconds": float(runtime_seconds),
            "result_sha256": result_sha256, "summary": summary,
        })
        state = self.spool.load_run()
        if state.get("job_id"):
            terminal = self.spool.load_terminal()
            terminal["payload"]["job_id"] = state["job_id"]
            self.spool.save_terminal(terminal["kind"], terminal["payload"])
        self._sync_spool(self.spool)

    def fail(self, exception: BaseException, stage_code: str | None = None) -> None:
        if self.spool is None:
            return
        self._emit("LOG", "ERROR", f"{type(exception).__name__}: {exception}", stage_code=stage_code or self.current_stage, percent=self.last_progress)
        self.spool.save_terminal("fail", {
            "failed_at": utc_now(), "stage_code": stage_code or self.current_stage or "INITIALIZING",
            "error_code": str(getattr(exception, "code", type(exception).__name__.upper())),
            "error_message": _safe_message(exception)[:4000], "retryable": False,
        })
        state = self.spool.load_run()
        if state.get("job_id"):
            terminal = self.spool.load_terminal()
            terminal["payload"]["job_id"] = state["job_id"]
            self.spool.save_terminal(terminal["kind"], terminal["payload"])
        self._sync_spool(self.spool)

    def flush(self) -> None:
        if self.spool is not None:
            self._sync_spool(self.spool, events_only=True)
        self.last_flush = time.monotonic()

    def _sync_spool(self, spool: SpoolRun, *, events_only: bool = False) -> None:
        with self._lock:
            self._sync_spool_locked(spool, events_only=events_only)

    def _sync_spool_locked(self, spool: SpoolRun, *, events_only: bool = False) -> None:
        if not spool.run_path.exists():
            return
        if not self._can_attempt(spool):
            return
        run_id = self._ensure_backend_run(spool)
        if not run_id:
            return
        try:
            headers = self._binding_headers(spool)
            while True:
                events = spool.read_events(self.config.event_batch_size)
                if not events:
                    break
                data = self.client.request_json(
                    "POST", f"/api/v1/coal-boundary/runs/{run_id}/events:batch", {"events": events},
                    idempotency_key=f"events:{run_id}:{events[0]['seq']}:{events[-1]['seq']}",
                    extra_headers=headers,
                )
                accepted = int(data.get("accepted_through_seq", events[-1]["seq"]))
                spool.acknowledge_events(accepted)
            if events_only:
                self.status.last_sync_at = utc_now()
                return
            result = spool.load_result()
            if result is not None:
                if result.get("result_kind") == "BOUNDARY_RESULT":
                    validate_boundary_result(result.get("result"))
                elif result.get("result_kind") == "PILE_VOLUME_RESULT":
                    from ..pile_volume_validation import validate_pile_volume_result
                    validate_pile_volume_result(result.get("result"))
                self.client.request_json(
                    "PUT", f"/api/v1/coal-boundary/runs/{run_id}/result", result,
                    idempotency_key=f"result:{run_id}:{result['content_sha256']}",
                    extra_headers=headers,
                )
                spool.acknowledge_result()
            terminal = spool.load_terminal()
            terminal_sent = False
            if terminal is not None:
                self.client.request_json(
                    "POST", f"/api/v1/coal-boundary/runs/{run_id}/{terminal['kind']}", terminal["payload"],
                    idempotency_key=f"terminal:{run_id}:{terminal['kind']}",
                    extra_headers=headers,
                )
                spool.acknowledge_terminal()
                spool.mark_terminal_acknowledged()
                terminal_sent = True
            # 附件是派生物：结构化结果和终态优先。附件失败只进入补偿队列，不能阻止任务成功。
            # DISABLED 时保留已有 spool 文件，但不再请求已终态的旧 Run，待建桶或切换
            # LOCAL_REGISTER 后再补偿；这样不会丢文件，也不会每次启动都产生冲突日志。
            artifact_mode = str(getattr(self.config, "artifact_mode", "OBJECT_STORAGE_UPLOAD")).upper()
            all_artifacts = list(spool.load_artifacts())
            if artifact_mode == "DISABLED":
                pending_artifacts = []
            elif artifact_mode == "LOCAL_REGISTER":
                # 历史 multipart spool 没有 root_id/relative_path，不能伪造成同机登记。
                # 保留原文件，只有新格式 LOCAL_REGISTER 记录才在 Worker 中补登。
                pending_artifacts = [item for item in all_artifacts if item.get("delivery_mode") == "LOCAL_REGISTER"]
            else:
                pending_artifacts = all_artifacts
            for artifact in pending_artifacts:
                if artifact.get("delivery_mode") == "LOCAL_REGISTER":
                    self.client.request_json(
                        "POST", f"/api/v1/coal-boundary/runs/{run_id}/artifacts:register",
                        {"artifacts": [{k: v for k, v in artifact.items() if k != "delivery_mode"}]},
                        idempotency_key=f"artifact-register:{run_id}:{artifact['artifact_type']}:{artifact['sha256']}",
                        extra_headers=headers,
                    )
                else:
                    self.client.upload_artifact(
                        run_id, artifact, spool.path / "artifacts" / artifact["spool_path"],
                        extra_headers=headers,
                    )
                spool.acknowledge_artifact(artifact)
            self.status.last_sync_at = utc_now()
            self.status.last_error = None
            self.status.auth_required = False
            self._clear_retry(spool)
            # Keep an active run spool even if its current events/results have
            # already synced; it is only complete after a terminal call.
            terminal_done = bool(spool.load_run().get("terminal_acknowledged"))
            cleaned = terminal_done and spool.cleanup_if_drained()
            if cleaned and spool is self.spool:
                self.spool = None
            if cleaned:
                try:
                    self.spool_root.rmdir()
                except OSError:
                    pass
        except Exception as exc:
            self._schedule_retry(spool)
            self._record_error(exc)

    def replay_pending(self) -> None:
        for spool in list(iter_spool_runs(self.spool_root)):
            self._sync_spool(spool)
            if self.status.auth_required:
                break

    def close(self) -> None:
        self._stop.set()
        if self._heartbeat_thread is not None and self._heartbeat_thread is not threading.current_thread():
            self._heartbeat_thread.join(timeout=1.0)
        self.flush()


def build_reporter(
    config,
    *,
    token: str | None = None,
    username: str | None = None,
    password: str | None = None,
) -> ProgressReporter:
    if not config.enabled:
        return NoOpReporter()
    # 手工运行（GUI/CLI）没有后端 claim 上下文，本地成果登记根只能来自配置；
    # Worker 任务不走这里，而是由 service 传入 output_roots 解析出的根。
    return BackendReporter(
        config, token=token, username=username, password=password,
        artifact_root_id=getattr(config, "artifact_root_id", None),
        artifact_root=getattr(config, "artifact_root", None),
    )
