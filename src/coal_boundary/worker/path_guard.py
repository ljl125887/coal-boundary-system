from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
import re
import time
from typing import Callable, Mapping


class InputValidationError(ValueError):
    """带稳定错误码的本地输入校验错误。"""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ValidatedInput:
    root_id: str
    directory: Path
    files: tuple[Path, ...]
    total_bytes: int


def _safe_relative(value: object, field_name: str, *, allow_dot: bool = True) -> Path:
    text = str(value if value is not None else ".").strip().replace("/", "\\")
    win = PureWindowsPath(text)
    if not text or win.is_absolute() or win.drive or text.startswith("\\"):
        raise InputValidationError("INPUT_PATH_INVALID", f"{field_name} 必须是相对路径")
    if any(part == ".." for part in win.parts):
        raise InputValidationError("INPUT_PATH_INVALID", f"{field_name} 不允许包含 ..")
    if any(part in {"", "."} for part in win.parts if not (allow_dot and part == ".")):
        raise InputValidationError("INPUT_PATH_INVALID", f"{field_name} 包含无效目录段")
    return Path(*[part for part in win.parts if part != "."])


def resolve_under_root(roots: Mapping[str, Path], root_id: object, relative_path: object = ".", subdirectory: object = ".") -> Path:
    key = str(root_id or "")
    if key not in roots:
        raise InputValidationError("INPUT_PATH_INVALID", f"未知 root_id: {key}")
    root = Path(roots[key]).resolve(strict=True)
    relative = _safe_relative(relative_path, "relative_path")
    subdir = _safe_relative(subdirectory, "las_subdirectory")
    candidate = (root / relative / subdir).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise InputValidationError("INPUT_PATH_INVALID", "任务路径越出配置根目录") from exc
    # resolve() 会展开 Windows Junction/符号链接；存在时再做一次严格校验。
    if candidate.exists():
        actual = candidate.resolve(strict=True)
        try:
            actual.relative_to(root)
        except ValueError as exc:
            raise InputValidationError("INPUT_PATH_INVALID", "链接目标越出配置根目录") from exc
        candidate = actual
    return candidate


def validate_local_input(
    input_ref: Mapping[str, object],
    roots: Mapping[str, Path],
    *,
    allowed_extensions: tuple[str, ...] = (".las",),
    require_ready_marker: bool = False,
    default_ready_marker: str = "_READY",
    stable_seconds: float = 60.0,
    max_files: int = 1000,
    max_total_bytes: int = 2_000_000_000_000,
    sleep: Callable[[float], None] = time.sleep,
) -> ValidatedInput:
    directory = resolve_under_root(
        roots,
        input_ref.get("root_id"),
        input_ref.get("relative_path", "."),
        input_ref.get("las_subdirectory", "."),
    )
    if not directory.is_dir():
        raise InputValidationError("INPUT_PATH_INVALID", "LAS 输入目录不存在或不是目录")
    pattern = str(input_ref.get("file_pattern", "*.las"))
    if pattern.lower() != "*.las":
        raise InputValidationError("INPUT_PATH_INVALID", "当前只允许 file_pattern=*.las")
    ready_marker = str(input_ref.get("ready_marker") or default_ready_marker)
    if "/" in ready_marker or "\\" in ready_marker or ready_marker in {"", ".", ".."}:
        raise InputValidationError("INPUT_PATH_INVALID", "ready_marker 必须是单个文件名")
    if require_ready_marker and not (directory / ready_marker).is_file():
        raise InputValidationError("INPUT_NOT_READY", f"缺少就绪标记 {ready_marker}")

    allowed = {item.lower() for item in allowed_extensions}

    def snapshot() -> dict[Path, tuple[int, int]]:
        files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in allowed)
        if not files:
            raise InputValidationError("LAS_NOT_FOUND", "目录中没有允许的 LAS 文件")
        if len(files) > max_files:
            raise InputValidationError("INPUT_LIMIT_EXCEEDED", f"LAS 文件数超过限制 {max_files}")
        result = {path.resolve(strict=True): (path.stat().st_size, path.stat().st_mtime_ns) for path in files}
        total = sum(item[0] for item in result.values())
        if total > max_total_bytes:
            raise InputValidationError("INPUT_LIMIT_EXCEEDED", f"LAS 总大小超过限制 {max_total_bytes}")
        for path in result:
            try:
                path.relative_to(directory)
            except ValueError as exc:
                raise InputValidationError("INPUT_PATH_INVALID", "LAS 链接目标越出输入目录") from exc
        return result

    before = snapshot()
    if stable_seconds > 0:
        sleep(stable_seconds)
        after = snapshot()
        if before != after:
            raise InputValidationError("INPUT_NOT_READY", "LAS 文件仍在写入或目录内容发生变化")
        before = after
    expected = input_ref.get("expected_file_count")
    if expected is not None and int(expected) != len(before):
        raise InputValidationError("INPUT_NOT_READY", f"LAS 文件数 {len(before)} 与 expected_file_count={expected} 不一致")
    return ValidatedInput(str(input_ref["root_id"]), directory, tuple(before), sum(v[0] for v in before.values()))


def resolve_output_directory(output_ref: Mapping[str, object], roots: Mapping[str, Path], job_id: str, run_id: str) -> tuple[str, Path]:
    root_id = str(output_ref.get("root_id") or "")
    relative = output_ref.get("relative_path")
    if relative in {None, "", "."}:
        relative = f"{job_id}/{run_id}"
    target = resolve_under_root(roots, root_id, relative, ".")
    target.mkdir(parents=True, exist_ok=True)
    return root_id, target


def resolve_source_result_directory(output_root: Path, source_job_id: object, source_run_id: object) -> Path:
    """Resolve a backend-provided source identity without allowing path traversal."""
    ids = (str(source_job_id), str(source_run_id))
    if any(not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", value) for value in ids):
        raise InputValidationError("SOURCE_VERSION_CONFLICT", "源任务 ID 格式无效")
    root = Path(output_root).resolve(strict=True)
    target = (root / ids[0] / ids[1]).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise InputValidationError("SOURCE_VERSION_CONFLICT", "源任务路径越出结果根目录") from exc
    return target


def resolve_source_artifact(source_directory: Path, file_name: str) -> Path:
    if file_name not in {"coal_piles_full.json", "surface_full.npz", "diagnostics_full.npz"}:
        raise InputValidationError("SOURCE_VERSION_CONFLICT", "源结果文件名无效")
    directory = Path(source_directory).resolve(strict=False)
    path = directory / file_name
    if path.exists():
        try:
            path.resolve(strict=True).relative_to(directory)
        except ValueError as exc:
            raise InputValidationError("SOURCE_VERSION_CONFLICT", "源结果文件链接越出任务目录") from exc
    return path
