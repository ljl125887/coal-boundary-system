from __future__ import annotations

from dataclasses import fields, replace
import json
import os
from pathlib import Path
import platform
import re
import socket
import threading
import time

from ..config import AppConfig
from ..integration.client import ApiError, BackendApiClient
from ..integration.models import sha256_json, utc_now
from ..integration.reporter import BackendReporter
from ..las import dataset_fingerprint, read_las_header, summarize_headers
from ..pipeline import _config_hash, _package_version, inspect_project, run_pipeline
from .client import Claim, WorkerApiClient
from .path_guard import InputValidationError, resolve_output_directory, resolve_source_artifact, resolve_source_result_directory, validate_local_input
from .pile_volume import PileRecalculationError, recalculate_pile_volume, validate_recalculation_source


class WorkerCancelled(RuntimeError):
    pass


class LeaseLost(RuntimeError):
    pass


def _safe_log_value(value):
    if isinstance(value, str):
        home = str(Path.home())
        text = value.replace(home, "<user-home>") if home else value
        return re.sub(r"(?i)\b[A-Z]:\\[^\r\n\t]+", "<local-path>", text)
    if isinstance(value, dict):
        return {str(k): _safe_log_value(v) for k, v in value.items() if str(k).lower() not in {"token", "password", "geometry", "coordinates"}}
    if isinstance(value, (list, tuple)):
        return [_safe_log_value(item) for item in value]
    return value


def _recalculation_context(claim: Claim) -> dict:
    """取出 PILE_VOLUME_RECALC 的不可变上下文。

    后端 P1 接口说明（2026-09-17）下发的上下文只有
    source_job_id/source_run_id/source_pile_id/source_result_sha256/dataset_fingerprint/
    selection_mode/geometry/source_metrics，不含 recalculation_id；这里用本次重算 Job ID
    兜底，保证结果可追溯且不依赖后端是否补齐该字段。
    """
    context = claim.payload.get("recalculation")
    if not isinstance(context, dict):
        raise ValueError("PILE_VOLUME_RECALC 缺少 recalculation 上下文")
    context = dict(context)
    context.setdefault("recalculation_id", str(claim.job_id))
    return context


def _apply_job_algorithm_config(config: AppConfig, job_config: dict, accepted_profiles: tuple[str, ...]) -> AppConfig:
    profile = str(job_config.get("profile", "default"))
    if profile not in accepted_profiles:
        raise ValueError(f"CONFIG_PROFILE_NOT_ALLOWED: {profile}")
    overrides = job_config.get("overrides") or {}
    if not isinstance(overrides, dict):
        raise ValueError("config.overrides 必须是对象")
    allowed_sections = {"raster", "ground", "segmentation", "model", "evaluation"}
    forbidden_model_fields = {"weights_path", "cell_classifier_path"}
    updated = config
    for section, values in overrides.items():
        if section not in allowed_sections or not isinstance(values, dict):
            raise ValueError(f"CONFIG_OVERRIDE_NOT_ALLOWED: {section}")
        current = getattr(updated, section)
        allowed_fields = {item.name for item in fields(current)}
        unknown = set(values) - allowed_fields
        if section == "model":
            unknown |= set(values) & forbidden_model_fields
        if unknown:
            raise ValueError(f"CONFIG_OVERRIDE_NOT_ALLOWED: {section}.{sorted(unknown)}")
        updated = replace(updated, **{section: replace(current, **values)})
    return updated


class _LeaseMonitor:
    def __init__(self, client: WorkerApiClient, claim: Claim, interval: float, lease_seconds: int):
        self.client, self.claim, self.interval, self.lease_seconds = client, claim, interval, lease_seconds
        self.stage_code, self.progress = "INITIALIZING", 0.0
        self.cancel_requested = False
        self.lease_lost = False
        self.last_success = time.monotonic()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="coal-boundary-lease", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not threading.current_thread():
            self.thread.join(timeout=2.0)

    def update(self, stage: str, progress: float):
        self.stage_code, self.progress = stage, progress

    def check(self):
        if self.lease_lost:
            raise LeaseLost("任务租约已丢失，停止产生新结果")
        if self.cancel_requested:
            raise WorkerCancelled("后端已请求取消任务")

    def _run(self):
        while not self.stop_event.wait(self.interval):
            try:
                data = self.client.renew(self.claim, self.stage_code, self.progress)
                self.last_success = time.monotonic()
                if bool(data.get("cancel_requested", False)):
                    self.cancel_requested = True
            except ApiError as exc:
                if exc.status == 409 or exc.code.upper() == "LEASE_LOST":
                    self.lease_lost = True
                    return
                if time.monotonic() - self.last_success > self.lease_seconds:
                    self.lease_lost = True
                    return


class _GuardedReporter:
    def __init__(self, reporter: BackendReporter, monitor: _LeaseMonitor):
        self.reporter, self.monitor = reporter, monitor

    def __getattr__(self, name):
        return getattr(self.reporter, name)

    def stage_started(self, stage_code, message=None):
        self.monitor.check(); self.monitor.update(stage_code, self.reporter.last_progress)
        return self.reporter.stage_started(stage_code, message)

    def progress(self, stage_code, percent, **kwargs):
        self.monitor.check(); self.monitor.update(stage_code, float(percent))
        return self.reporter.progress(stage_code, percent, **kwargs)

    def submit_result(self, result_kind, result):
        self.monitor.check()
        return self.reporter.submit_result(result_kind, result)

    def complete(self, summary, runtime_seconds, result_sha256):
        self.monitor.check()
        return self.reporter.complete(summary, runtime_seconds, result_sha256)

    def fail(self, exception, stage_code=None):
        if isinstance(exception, (WorkerCancelled, LeaseLost)):
            return None
        return self.reporter.fail(exception, stage_code)


class CoalBoundaryWorker:
    def __init__(self, config: AppConfig):
        self.config = config
        if not config.worker.enabled:
            raise ValueError("worker.enabled=false，不能启动 Worker")
        integration = config.integration
        token = os.environ.get(integration.token_env) if integration.token_env else None
        username = os.environ.get(integration.username_env) if integration.username_env else None
        password = os.environ.get(integration.password_env) if integration.password_env else None
        api = BackendApiClient(
            integration.base_url, token,
            max(integration.connect_timeout_seconds, integration.read_timeout_seconds),
            integration.verify_tls, login_path=integration.login_path,
            username=username, password=password,
        )
        self.client = WorkerApiClient(api, config.worker.worker_id)
        self.stop_event = threading.Event()
        config.worker.local_log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = config.worker.local_log_dir / "worker.jsonl"

    def _log(self, level: str, message: str, **context):
        record = _safe_log_value({"timestamp": utc_now(), "level": level, "service": "coal-boundary-worker",
                  "worker_id": self.config.worker.worker_id, "message": message, **context})
        with self.log_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"[worker] {message}", flush=True)

    def _capabilities(self) -> dict:
        return {"run_types": list(self.config.worker.accepted_job_types), "gpu": None,
                "tensorflow_gpu": False, "max_concurrent_jobs": 1}

    def run_forever(self):
        self._log("INFO", "Worker 已启动")
        while not self.stop_event.is_set():
            try:
                self.run_once()
            except Exception as exc:
                self._log("ERROR", "Worker 轮询异常", error_type=type(exc).__name__, error=str(exc))
            self.stop_event.wait(self.config.worker.poll_interval_seconds)

    def stop(self):
        self.stop_event.set()

    def run_once(self) -> bool:
        self.client.heartbeat("IDLE", self._capabilities())
        claim = self.client.claim(self.config.worker.accepted_job_types, self.config.worker.lease_seconds)
        if claim is None:
            return False
        claim.payload["worker_id"] = self.config.worker.worker_id
        self.client.heartbeat("BUSY", self._capabilities(), claim.job_id)
        self._log("INFO", "已领取任务", job_id=claim.job_id, run_id=claim.run_id, job_type=claim.job_type)
        try:
            self._execute(claim)
        except Exception:
            try:
                self.client.heartbeat("DEGRADED", self._capabilities())
            except Exception:
                pass
            raise
        else:
            try:
                self.client.heartbeat("IDLE", self._capabilities())
            except Exception as exc:
                self._log("WARNING", "任务完成，但空闲状态心跳上报失败", error_type=type(exc).__name__)
        return True

    def _execute(self, claim: Claim):
        worker = self.config.worker
        monitor = _LeaseMonitor(self.client, claim, worker.lease_renew_interval_seconds, worker.lease_seconds)
        monitor.start()
        reporter = None
        try:
            validated = validate_local_input(
                claim.input, worker.input_roots,
                allowed_extensions=worker.allowed_extensions,
                require_ready_marker=worker.require_ready_marker,
                default_ready_marker=worker.ready_marker_name,
                stable_seconds=worker.stable_file_seconds,
                max_files=worker.max_files, max_total_bytes=worker.max_total_bytes,
            )
            output_root_id, output_dir = resolve_output_directory(
                claim.output, worker.output_roots, claim.job_id, claim.run_id,
            )
            headers = [read_las_header(path) for path in validated.files]
            summary = summarize_headers(headers)
            fingerprint = dataset_fingerprint(headers)
            job_config = _apply_job_algorithm_config(
                replace(
                    self.config,
                project=replace(self.config.project, name=str(claim.project_name), las_dir=validated.directory, output_dir=output_dir),
                integration=replace(self.config.integration, artifact_mode="LOCAL_REGISTER"),
                ),
                dict(claim.payload.get("config") or {}), worker.accepted_profiles,
            )
            inspection = inspect_project(job_config)
            dataset = {"fingerprint": fingerprint, "file_count": summary["file_count"],
                       "point_count": summary["point_count"], "source_epsg": job_config.crs.source_epsg}
            algorithm = {"name": "coal-boundary-system", "version": _package_version(),
                         "config_hash": _config_hash(job_config),
                         "model_version": job_config.model.cell_classifier_path.name if job_config.model.cell_classifier_path else None}
            monitor.check()
            self.client.start(claim, dataset, algorithm)
            artifact_root = worker.output_roots[output_root_id]
            reporter_impl = BackendReporter(
                job_config.integration, claimed_job=claim.payload,
                artifact_root_id=output_root_id, artifact_root=artifact_root,
            )
            reporter = _GuardedReporter(reporter_impl, monitor)
            if claim.job_type == "EXTRACTION":
                shared_cache_root = worker.cache_root or (artifact_root / "_algorithm_cache")
                run_pipeline(job_config, reporter=reporter, cache_root=shared_cache_root)
            elif claim.job_type == "PILE_VOLUME_RECALC":
                context = _recalculation_context(claim)
                reporter.start_run({"run_type": claim.job_type, "project_name": claim.project_name,
                    "dataset": dataset, "algorithm": algorithm, "attempt": claim.payload.get("attempt", 1)})
                reporter.stage_started("RECALC_VALIDATE", "校验源结果与目标边界")
                source_dir = resolve_source_result_directory(artifact_root, context.get("source_job_id"), context.get("source_run_id"))
                source_result_path = resolve_source_artifact(source_dir, "coal_piles_full.json")
                source_cache = resolve_source_artifact(source_dir, "surface_full.npz")
                source_diagnostics = resolve_source_artifact(source_dir, "diagnostics_full.npz")
                validate_recalculation_source(context, source_result_path, fingerprint, job_config)
                reporter.stage_started("RECALC_LOAD_LOCAL", "读取目标边界周边点云")
                calculated = recalculate_pile_volume(
                    job_config, las_files=validated.files, output_dir=output_dir,
                    context=context, source_cache_path=source_cache,
                    source_diagnostics_path=source_diagnostics, reporter=reporter,
                )
                reporter.stage_started("RECALC_VOLUME", "计算单煤堆坡地挖方体积")
                result_hash = reporter.submit_result("PILE_VOLUME_RESULT", calculated.result)
                # 后端已把 PILE_VOLUME_JSON 加入 artifacts:register 白名单（2026-09-17 复测 200），
                # 重算成果恢复使用专用类型登记。
                reporter.upload_artifact("PILE_VOLUME_JSON", calculated.path)
                reporter.complete({"source_pile_id": context["source_pile_id"], **calculated.result["metrics"]}, calculated.result["runtime_seconds"], result_hash)
                reporter.close()
            else:
                raise ValueError(f"UNSUPPORTED_JOB_TYPE: {claim.job_type}")
            self._log("INFO", "任务处理完成", job_id=claim.job_id, run_id=claim.run_id)
        except WorkerCancelled as exc:
            self.client.cancelled(claim, str(exc))
            self._log("WARNING", "任务已取消", job_id=claim.job_id, run_id=claim.run_id)
        except LeaseLost:
            self._log("ERROR", "租约丢失，已停止提交", job_id=claim.job_id, run_id=claim.run_id)
        except Exception as exc:
            code = exc.code if isinstance(exc, (InputValidationError, PileRecalculationError)) else type(exc).__name__.upper()
            if reporter is None:
                try:
                    self.client.fail(claim, code, str(exc), retryable=isinstance(exc, (OSError, TimeoutError)))
                except Exception:
                    pass
            else:
                reporter.fail(exc)
                reporter.close()
            self._log("ERROR", "任务执行失败", job_id=claim.job_id, run_id=claim.run_id,
                      error_code=code, error=str(exc))
            raise
        finally:
            monitor.stop()
