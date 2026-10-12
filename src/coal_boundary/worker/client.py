from __future__ import annotations

from dataclasses import dataclass

from ..integration.client import BackendApiClient
from ..integration.models import new_id, utc_now


@dataclass(frozen=True)
class Claim:
    payload: dict

    def __getattr__(self, name: str):
        try:
            return self.payload[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class WorkerApiClient:
    def __init__(self, client: BackendApiClient, worker_id: str):
        self.client = client
        self.worker_id = worker_id

    def _headers(self, lease_id: str | None = None) -> dict[str, str]:
        headers = {"X-Worker-Id": self.worker_id}
        if lease_id:
            headers["X-Lease-Id"] = lease_id
        return headers

    def heartbeat(self, status: str, capabilities: dict, active_job_id: str | None = None) -> dict:
        return self.client.request_json(
            "POST", "/api/v1/coal-boundary/workers/heartbeat",
            {"worker_id": self.worker_id, "version": "0.1.0", "status": status,
             "active_job_id": active_job_id, "capabilities": capabilities, "occurred_at": utc_now()},
            extra_headers=self._headers(),
        )

    def claim(self, accepted_job_types: tuple[str, ...], lease_seconds: int, *, idempotency_key: str | None = None) -> Claim | None:
        data = self.client.request_json(
            "POST", "/api/v1/coal-boundary/worker/jobs:claim",
            {"worker_id": self.worker_id, "accepted_job_types": list(accepted_job_types), "lease_seconds": int(lease_seconds)},
            idempotency_key=idempotency_key or f"claim:{self.worker_id}:{new_id()}",
            extra_headers=self._headers(),
        )
        return Claim(data) if data else None

    def start(self, claim: Claim, dataset: dict, algorithm: dict) -> dict:
        return self.client.request_json(
            "POST", f"/api/v1/coal-boundary/jobs/{claim.job_id}/start",
            {"run_id": claim.run_id, "started_at": utc_now(), "dataset": dataset, "algorithm": algorithm},
            idempotency_key=f"start:{claim.job_id}:{claim.run_id}",
            extra_headers=self._headers(claim.lease_id),
        )

    def renew(self, claim: Claim, stage_code: str, progress_percent: float) -> dict:
        return self.client.request_json(
            "POST", f"/api/v1/coal-boundary/jobs/{claim.job_id}/lease:renew",
            {"run_id": claim.run_id, "lease_id": claim.lease_id, "stage_code": stage_code,
             "progress_percent": float(progress_percent), "renewed_at": utc_now()},
            extra_headers=self._headers(claim.lease_id),
        )

    def fail(self, claim: Claim, error_code: str, message: str, *, stage_code: str = "INITIALIZING", retryable: bool = False) -> dict:
        return self.client.request_json(
            "POST", f"/api/v1/coal-boundary/runs/{claim.run_id}/fail",
            {"job_id": claim.job_id, "failed_at": utc_now(), "stage_code": stage_code,
             "error_code": error_code, "error_message": str(message)[:4000], "retryable": bool(retryable)},
            idempotency_key=f"fail:{claim.run_id}:{error_code}", extra_headers=self._headers(claim.lease_id),
        )

    def cancelled(self, claim: Claim, message: str = "任务已按后端请求安全取消") -> dict:
        return self.client.request_json(
            "POST", f"/api/v1/coal-boundary/runs/{claim.run_id}/cancelled",
            {"job_id": claim.job_id, "cancelled_at": utc_now(), "message": message},
            idempotency_key=f"cancelled:{claim.run_id}", extra_headers=self._headers(claim.lease_id),
        )
