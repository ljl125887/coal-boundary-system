from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import time
import uuid


class CacheLockTimeout(TimeoutError):
    pass


@contextmanager
def cache_file_lock(
    target: str | Path,
    *,
    timeout_seconds: float = 86_400.0,
    stale_seconds: float = 86_400.0,
    poll_seconds: float = 0.25,
):
    """Serialize generation of one cache artifact across worker processes.

    The lock is a sibling ``.lock`` file created with O_EXCL.  A unique owner
    token prevents one process from deleting a lock that another process has
    replaced after stale-lock recovery.
    """
    target_path = Path(target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target_path.with_name(target_path.name + ".lock")
    token = uuid.uuid4().hex
    deadline = time.monotonic() + timeout_seconds
    payload = json.dumps({
        "token": token,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "created_at": time.time(),
    }, ensure_ascii=False).encode("utf-8")

    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                os.write(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > stale_seconds:
                    lock_path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() >= deadline:
                raise CacheLockTimeout(f"等待缓存锁超时: {lock_path.name}")
            time.sleep(poll_seconds)

    try:
        yield lock_path
    finally:
        try:
            current = json.loads(lock_path.read_text(encoding="utf-8"))
            if current.get("token") == token:
                lock_path.unlink(missing_ok=True)
        except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
            pass
