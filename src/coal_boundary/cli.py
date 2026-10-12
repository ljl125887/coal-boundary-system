from __future__ import annotations

import argparse
import json

from .config import load_config
from .integration import build_reporter
from .pipeline import inspect_project, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="煤堆点云独立边界与体积后台")
    parser.add_argument("--config", required=True, help="YAML 配置文件")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_cmd = sub.add_parser("inspect", help="只检查数据和配置")
    inspect_cmd.add_argument("--max-files", type=int)
    run_cmd = sub.add_parser("run", help="执行完整处理")
    run_cmd.add_argument("--max-files", type=int)
    run_cmd.add_argument("--no-cache", action="store_true")
    sub.add_parser("sync", help="补传后端待同步任务，不执行算法")
    sub.add_parser("worker", help="常驻轮询后端并执行同服务器任务")
    sub.add_parser("worker-once", help="轮询并最多执行一个任务")
    validate_cmd = sub.add_parser("validate-job", help="只校验任务 JSON 中的本地输入与输出路径")
    validate_cmd.add_argument("job_json", help="后端 Claim JSON 文件")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "inspect":
        result = inspect_project(config, args.max_files)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    elif args.command == "run":
        reporter = build_reporter(config.integration)
        result = run_pipeline(config, args.max_files, not args.no_cache, reporter=reporter)
        concise = {
            "summary": result.get("summary"),
            "evaluation": {k: v for k, v in (result.get("evaluation") or {}).items() if k != "matches"},
            "output_dir": str(config.project.output_dir),
        }
        print(json.dumps(concise, ensure_ascii=False, indent=2, default=str))
    elif args.command == "sync":
        if not config.integration.enabled:
            print("后端同步未启用；请先设置 integration.enabled=true")
            return
        reporter = build_reporter(config.integration)
        reporter.close()
        print("后端待同步队列已执行一次补传。")
    elif args.command in {"worker", "worker-once"}:
        from .worker import CoalBoundaryWorker
        worker = CoalBoundaryWorker(config)
        if args.command == "worker-once":
            processed = worker.run_once()
            print("已执行一个任务。" if processed else "当前没有待领取任务。")
        else:
            try:
                worker.run_forever()
            except KeyboardInterrupt:
                worker.stop()
                print("Worker 已停止。")
    else:
        from pathlib import Path
        from .worker.path_guard import resolve_output_directory, validate_local_input
        job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
        validated = validate_local_input(
            job["input"], config.worker.input_roots,
            allowed_extensions=config.worker.allowed_extensions,
            require_ready_marker=config.worker.require_ready_marker,
            default_ready_marker=config.worker.ready_marker_name,
            stable_seconds=0,
            max_files=config.worker.max_files,
            max_total_bytes=config.worker.max_total_bytes,
        )
        _, output = resolve_output_directory(
            job["output"], config.worker.output_roots, str(job["job_id"]), str(job["run_id"]),
        )
        print(json.dumps({"input_dir": str(validated.directory), "file_count": len(validated.files),
                          "total_bytes": validated.total_bytes, "output_dir": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
