from __future__ import annotations

import json
import mimetypes
from pathlib import Path
import ssl
import threading
import urllib.error
import urllib.request
import uuid


class ApiError(RuntimeError):
    def __init__(self, status: int | None, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.status = status
        self.code = code
        self.retryable = status is None or status in {408, 429} or (status is not None and status >= 500)


class BackendApiClient:
    def __init__(
        self,
        base_url: str,
        token: str | None,
        timeout: float,
        verify_tls: bool = True,
        *,
        login_path: str = "/login",
        username: str | None = None,
        password: str | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token.strip() if token else None
        self.timeout = timeout
        self.ssl_context = None if verify_tls else ssl._create_unverified_context()
        self.login_path = "/" + login_path.strip().lstrip("/")
        self.username = username.strip() if username else None
        self.password = password if password else None
        self._login_lock = threading.RLock()

    @property
    def can_auto_login(self) -> bool:
        return bool(self.username and self.password)

    def _headers(self, idempotency_key: str | None = None, *, include_auth: bool = True, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "X-Request-Id": uuid.uuid4().hex,
        }
        if include_auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if extra_headers:
            headers.update({str(k): str(v) for k, v in extra_headers.items()})
        return headers

    @staticmethod
    def _error_from_http(exc: urllib.error.HTTPError) -> ApiError:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error = json.loads(body)
        except json.JSONDecodeError:
            error = {}
        code = error.get("code", "HTTP_ERROR") if isinstance(error, dict) else "HTTP_ERROR"
        if isinstance(error, dict):
            message = error.get("message") or error.get("msg") or body
        else:
            message = body
        return ApiError(exc.code, str(code), str(message))

    @staticmethod
    def _decode_json_response(response_status: int, body: str) -> dict:
        if not body:
            return {}
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ApiError(None, "INVALID_RESPONSE", "后端没有返回合法 JSON") from exc
        if isinstance(decoded, dict):
            code = decoded.get("code")
            if code not in {None, "OK", 200, "200"}:
                effective_status = response_status
                if isinstance(code, int) and 400 <= code <= 599:
                    effective_status = code
                elif isinstance(code, str) and code.isdigit() and 400 <= int(code) <= 599:
                    effective_status = int(code)
                raise ApiError(
                    effective_status,
                    str(code),
                    str(decoded.get("message") or decoded.get("msg") or "后端返回业务错误"),
                )
            return decoded.get("data", decoded)
        raise ApiError(None, "INVALID_RESPONSE", "后端 JSON 响应必须是对象")

    def _request_json_once(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        *,
        idempotency_key: str | None = None,
        include_auth: bool = True,
        extra_headers: dict[str, str] | None = None,
    ) -> dict:
        data = None if payload is None else json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path, data=data, method=method,
            headers=self._headers(idempotency_key, include_auth=include_auth, extra_headers=extra_headers),
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
                response_status = int(response.status)
                body = response.read().decode("utf-8", errors="strict")
        except urllib.error.HTTPError as exc:
            raise self._error_from_http(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiError(None, "NETWORK_ERROR", str(exc)) from exc
        return self._decode_json_response(response_status, body)

    @staticmethod
    def _is_auth_error(exc: ApiError) -> bool:
        return exc.status == 401 or exc.code.upper() in {"401", "UNAUTHORIZED", "AUTHENTICATION_FAILED"}

    def login(self, *, force: bool = False) -> str:
        if not self.can_auto_login:
            raise ApiError(401, "LOGIN_CREDENTIALS_MISSING", "未提供自动登录用户名和密码")
        with self._login_lock:
            if self.token and not force:
                return self.token
            data = self._request_json_once(
                "POST",
                self.login_path,
                {"username": self.username, "password": self.password},
                include_auth=False,
            )
            token = data.get("token") if isinstance(data, dict) else None
            if not isinstance(token, str) or not token.strip():
                raise ApiError(None, "INVALID_LOGIN_RESPONSE", "登录成功响应中缺少 token")
            self.token = token.strip()
            return self.token

    def _ensure_authentication(self) -> None:
        if not self.token and self.can_auto_login:
            self.login()

    def request_json(self, method: str, path: str, payload: dict | None = None, *, idempotency_key: str | None = None, extra_headers: dict[str, str] | None = None) -> dict:
        is_login_request = path.rstrip("/") == self.login_path.rstrip("/")
        if not is_login_request:
            self._ensure_authentication()
        try:
            return self._request_json_once(method, path, payload, idempotency_key=idempotency_key, extra_headers=extra_headers)
        except ApiError as exc:
            if is_login_request or not self.can_auto_login or not self._is_auth_error(exc):
                raise
            self.token = None
            self.login(force=True)
            return self._request_json_once(method, path, payload, idempotency_key=idempotency_key, extra_headers=extra_headers)

    def _upload_artifact_once(self, run_id: str, artifact: dict, path: Path, extra_headers: dict[str, str] | None = None) -> dict:
        boundary = "----CoalBoundary" + uuid.uuid4().hex
        parts: list[bytes] = []
        for name, value in (
            ("artifact_type", artifact["artifact_type"]),
            ("sha256", artifact["sha256"]),
        ):
            parts.append(
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode("utf-8")
            )
        mime = mimetypes.guess_type(artifact["file_name"])[0] or "application/octet-stream"
        parts.append(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{artifact['file_name']}\"\r\nContent-Type: {mime}\r\n\r\n".encode("utf-8")
        )
        parts.append(path.read_bytes())
        parts.append(f"\r\n--{boundary}--\r\n".encode("ascii"))
        headers = self._headers(f"artifact:{run_id}:{artifact['artifact_type']}:{artifact['sha256']}", extra_headers=extra_headers)
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        request = urllib.request.Request(
            self.base_url + f"/api/v1/coal-boundary/runs/{run_id}/artifacts",
            data=b"".join(parts), method="POST", headers=headers,
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout, context=self.ssl_context) as response:
                response_status = int(response.status)
                body = response.read().decode("utf-8", errors="strict")
        except urllib.error.HTTPError as exc:
            raise self._error_from_http(exc) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ApiError(None, "NETWORK_ERROR", str(exc)) from exc
        return self._decode_json_response(response_status, body)

    def upload_artifact(self, run_id: str, artifact: dict, path: Path, *, extra_headers: dict[str, str] | None = None) -> dict:
        self._ensure_authentication()
        try:
            return self._upload_artifact_once(run_id, artifact, path, extra_headers)
        except ApiError as exc:
            if not self.can_auto_login or not self._is_auth_error(exc):
                raise
            self.token = None
            self.login(force=True)
            return self._upload_artifact_once(run_id, artifact, path, extra_headers)
