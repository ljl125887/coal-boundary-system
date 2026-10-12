from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Iterator


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class SpoolRun:
    def __init__(self, path: Path):
        self.path = path

    @classmethod
    def create(cls, root: Path, client_run_id: str, create_payload: dict, *, binding: dict | None = None) -> "SpoolRun":
        run = cls(root / client_run_id)
        run.path.mkdir(parents=True, exist_ok=True)
        if not run.run_path.exists():
            state = {"create_payload": create_payload, "backend_run_id": None, "terminal_acknowledged": False}
            if binding:
                state.update(binding)
            _atomic_json(run.run_path, state)
        return run

    @property
    def run_path(self) -> Path:
        return self.path / "run.json"

    @property
    def events_path(self) -> Path:
        return self.path / "events.jsonl"

    def load_run(self) -> dict:
        return json.loads(self.run_path.read_text(encoding="utf-8"))

    def set_backend_run_id(self, run_id: str) -> None:
        state = self.load_run()
        state["backend_run_id"] = run_id
        _atomic_json(self.run_path, state)

    def mark_terminal_acknowledged(self) -> None:
        state = self.load_run()
        state["terminal_acknowledged"] = True
        _atomic_json(self.run_path, state)

    def append_event(self, event: dict) -> None:
        with self.events_path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")

    def read_events(self, limit: int) -> list[dict]:
        if not self.events_path.exists():
            return []
        events: list[dict] = []
        with self.events_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    events.append(json.loads(line))
                if len(events) >= limit:
                    break
        return events

    def acknowledge_events(self, through_seq: int) -> None:
        if not self.events_path.exists():
            return
        remaining = []
        with self.events_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip() and int(json.loads(line)["seq"]) > through_seq:
                    remaining.append(line.rstrip("\n"))
        if remaining:
            temporary = self.events_path.with_suffix(".jsonl.tmp")
            temporary.write_text("\n".join(remaining) + "\n", encoding="utf-8")
            temporary.replace(self.events_path)
        else:
            self.events_path.unlink(missing_ok=True)

    def save_result(self, payload: dict) -> None:
        _atomic_json(self.path / "result.json", payload)

    def load_result(self) -> dict | None:
        path = self.path / "result.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def acknowledge_result(self) -> None:
        (self.path / "result.json").unlink(missing_ok=True)

    def add_artifact(self, artifact_type: str, source: Path, sha256: str) -> None:
        artifact_dir = self.path / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        target = artifact_dir / f"{artifact_type.lower()}_{source.name}"
        shutil.copy2(source, target)
        manifest = self.load_artifacts()
        record = {
            "artifact_type": artifact_type,
            "file_name": source.name,
            "spool_path": target.name,
            "sha256": sha256,
        }
        manifest = [item for item in manifest if item["artifact_type"] != artifact_type]
        manifest.append(record)
        _atomic_json(self.path / "artifacts.json", manifest)

    def add_local_artifact(self, artifact_type: str, source: Path, sha256: str, *, root_id: str, relative_path: str, content_type: str) -> None:
        manifest = self.load_artifacts()
        record = {
            "artifact_type": artifact_type,
            "delivery_mode": "LOCAL_REGISTER",
            "root_id": root_id,
            "relative_path": relative_path.replace("\\", "/"),
            "file_name": source.name,
            "content_type": content_type,
            "size_bytes": source.stat().st_size,
            "sha256": sha256,
        }
        manifest = [item for item in manifest if item["artifact_type"] != artifact_type]
        manifest.append(record)
        _atomic_json(self.path / "artifacts.json", manifest)

    def load_artifacts(self) -> list[dict]:
        path = self.path / "artifacts.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []

    def acknowledge_artifact(self, artifact: dict) -> None:
        manifest = [item for item in self.load_artifacts() if item != artifact]
        if artifact.get("spool_path"):
            (self.path / "artifacts" / artifact["spool_path"]).unlink(missing_ok=True)
        if manifest:
            _atomic_json(self.path / "artifacts.json", manifest)
        else:
            (self.path / "artifacts.json").unlink(missing_ok=True)
            artifact_dir = self.path / "artifacts"
            if artifact_dir.exists() and not any(artifact_dir.iterdir()):
                artifact_dir.rmdir()

    def save_terminal(self, kind: str, payload: dict) -> None:
        _atomic_json(self.path / "terminal.json", {"kind": kind, "payload": payload})

    def load_terminal(self) -> dict | None:
        path = self.path / "terminal.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def acknowledge_terminal(self) -> None:
        (self.path / "terminal.json").unlink(missing_ok=True)

    def is_drained(self) -> bool:
        return not any(
            (self.path / name).exists()
            for name in ("events.jsonl", "result.json", "artifacts.json", "terminal.json")
        )

    def cleanup_if_drained(self) -> bool:
        if not self.is_drained():
            return False
        self.run_path.unlink(missing_ok=True)
        try:
            self.path.rmdir()
        except OSError:
            return False
        return True


def iter_spool_runs(root: Path) -> Iterator[SpoolRun]:
    if not root.exists():
        return
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "run.json").exists():
            yield SpoolRun(path)
