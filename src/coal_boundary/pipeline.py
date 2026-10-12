from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import shutil
import time

import numpy as np

from .config import AppConfig
from .cache_lock import cache_file_lock
from .crs import validate_crs_config_and_bounds
from .display_cache import (
    DISPLAY_GEOMETRY_METHOD,
    DISPLAY_GEOMETRY_PARAMETERS,
    DisplayGeometryLoadResult,
    display_geometry_cache_key,
    load_display_geometry_cache,
    load_display_geometry_from_result,
    save_display_geometry_cache,
)
from .evaluate import evaluate_results
from .export import atomic_write_json, make_result_documents, make_service_fence_document
from .features import FEATURE_NAMES, build_cell_features
from .geometry import build_pile_results
from .ground import estimate_ground_surface, fill_nearest
from .integration import NoOpReporter, ProgressReporter
from .integration.stages import stage_fraction_progress
from .las import discover_las_files, read_las_header, dataset_fingerprint, summarize_headers
from .model import CellClassifier, LegacyKerasSegmenter, infer_probability_grid
from .pointcloud_boundary import refine_polygons_from_point_cloud
from .raster import SurfaceGrid, build_surface_grid
from .reference import load_reference_workbook
from .roi import load_projected_roi, rasterize_geometry
from .segmentation import (
    candidate_mask,
    merge_nearby_toe_fragments,
    recover_concave_slopes,
    split_by_height_valleys,
)


def _config_hash(config: AppConfig) -> str:
    values = asdict(config)
    # Backend connectivity does not affect algorithm output and therefore must
    # not change the algorithm configuration hash.
    values.pop("integration", None)
    values.pop("worker", None)
    values.pop("config_path", None)
    # These identify where/how a run is launched, not its algorithm.  Worker
    # outputs are isolated by job_id/run_id, so including output_dir would make
    # the same algorithm hash different for every backend task.  LAS content is
    # independently protected by dataset_fingerprint.
    project = values.get("project") or {}
    for field_name in ("name", "las_dir", "output_dir"):
        project.pop(field_name, None)
    raw = json.dumps(values, default=str, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _surface_cache_is_valid(
    metadata: dict,
    config: AppConfig,
    dataset_fingerprint_value: str,
    config_hash: str,
    *,
    require_config_hash: bool,
) -> bool:
    if metadata.get("dataset_fingerprint") != dataset_fingerprint_value:
        return False
    if metadata.get("cell_size_m") != config.raster.cell_size_m:
        return False
    if require_config_hash and metadata.get("config_hash") != config_hash:
        return False
    for key, expected in (
        ("minimum_z_m", config.raster.minimum_z_m),
        ("maximum_z_m", config.raster.maximum_z_m),
    ):
        value = metadata.get(key)
        if value is not None and value != expected:
            return False
    return True


def _materialize_cache_artifact(source: Path, destination: Path) -> None:
    """Copy a shared cache into the immutable run directory atomically."""
    if source.resolve() == destination.resolve():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp-copy")
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _shared_cache_directory(
    cache_root: str | Path,
    dataset_fingerprint_value: str,
    config_hash: str,
) -> Path:
    for label, value in (
        ("dataset_fingerprint", dataset_fingerprint_value),
        ("config_hash", config_hash),
    ):
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            raise ValueError(f"{label} 不是有效 SHA-256")
    namespace = hashlib.sha256(
        f"{dataset_fingerprint_value.lower()}:{config_hash.lower()}".encode("ascii")
    ).hexdigest()
    # Keep paths below legacy Windows MAX_PATH even when cache_root is deep.
    # Full fingerprints remain in cache metadata and are always revalidated.
    return Path(cache_root).resolve() / namespace[:32]


def _package_version() -> str:
    try:
        return version("coal-boundary")
    except PackageNotFoundError:
        return "0.1.0"


def _report_stage(reporter: ProgressReporter, code: str, name: str) -> None:
    print(f"[stage] {name}", flush=True)
    reporter.stage_started(code, name)


def inspect_project(config: AppConfig, max_files: int | None = None) -> dict:
    files = discover_las_files(config.project.las_dir)
    if max_files:
        files = files[:max_files]
    headers = [read_las_header(path) for path in files]
    summary = summarize_headers(headers)
    summary["crs_validation"] = validate_crs_config_and_bounds(
        config.crs, summary["bounds"]["min"], summary["bounds"]["max"],
    )
    summary["dataset_fingerprint"] = dataset_fingerprint(headers)
    summary["model_weights"] = str(config.model.weights_path) if config.model.weights_path else None
    summary["reference_workbook"] = str(config.project.reference_xlsx) if config.project.reference_xlsx else None
    return summary


def run_pipeline(
    config: AppConfig,
    max_files: int | None = None,
    use_cache: bool = True,
    reporter: ProgressReporter | None = None,
    cache_root: str | Path | None = None,
) -> dict:
    active_reporter = reporter or NoOpReporter()
    try:
        return _run_pipeline_impl(config, max_files, use_cache, active_reporter, cache_root)
    except BaseException as exc:
        active_reporter.fail(exc)
        raise
    finally:
        active_reporter.close()


def _run_pipeline_impl(
    config: AppConfig,
    max_files: int | None,
    use_cache: bool,
    reporter: ProgressReporter,
    cache_root: str | Path | None,
) -> dict:
    started = time.time()
    print("[stage] 读取并校验数据", flush=True)
    output_dir = config.project.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "full" if max_files is None else f"first_{max_files}"
    business_path = output_dir / f"coal_piles_{suffix}.json"
    surface_artifact_path = output_dir / f"surface_{suffix}.npz"
    files = discover_las_files(config.project.las_dir)
    if max_files:
        files = files[:max_files]
    headers = [read_las_header(path) for path in files]
    las_summary = summarize_headers(headers)
    crs_validation = validate_crs_config_and_bounds(
        config.crs, las_summary["bounds"]["min"], las_summary["bounds"]["max"],
    )
    fingerprint = dataset_fingerprint(headers)
    config_hash = _config_hash(config)
    shared_cache = cache_root is not None
    if shared_cache:
        cache_directory = _shared_cache_directory(cache_root, fingerprint, config_hash)
        cache_directory.mkdir(parents=True, exist_ok=True)
    else:
        cache_directory = output_dir
    surface_cache_path = cache_directory / f"surface_{suffix}.npz"
    reporter.start_run({
        "run_type": "EXTRACTION",
        "project_name": config.project.name,
        "dataset": {
            "fingerprint": fingerprint,
            "file_count": int(las_summary["file_count"]),
            "point_count": int(las_summary["point_count"]),
            "source_epsg": config.crs.source_epsg,
        },
        "algorithm": {
            "name": "coal-boundary-system",
            "version": _package_version(),
            "git_commit": None,
            "config_hash": config_hash,
            "model_version": config.model.cell_classifier_path.name if config.model.cell_classifier_path else None,
        },
    })
    reporter.stage_started("READ_VALIDATE", "读取并校验数据")
    _report_stage(reporter, "SURFACE_GRID", "建立或加载表面栅格")
    grid = None
    surface_cache_hit = False
    surface_lock = cache_file_lock(surface_cache_path) if shared_cache else nullcontext()
    with surface_lock:
        if use_cache and surface_cache_path.exists():
            try:
                candidate, cache_meta = SurfaceGrid.load(surface_cache_path)
                if _surface_cache_is_valid(
                    cache_meta, config, fingerprint, config_hash,
                    require_config_hash=shared_cache,
                ):
                    grid = candidate
                    surface_cache_hit = True
            except (OSError, ValueError, KeyError, EOFError) as exc:
                reporter.log(
                    "WARNING", "表面栅格缓存无效，将重新生成",
                    {"error_type": type(exc).__name__},
                )
        if grid is None:
            def progress(i, total, path):
                print(f"[surface] {i}/{total} {path.name}", flush=True)
                reporter.progress(
                    "SURFACE_GRID", stage_fraction_progress("SURFACE_GRID", i, total),
                    current=i, total=total, message="正在处理 LAS 文件",
                    context={"file_name": path.name},
                )
            grid = build_surface_grid(
                headers, config.raster.cell_size_m, config.raster.chunk_points,
                config.raster.minimum_z_m, config.raster.maximum_z_m, progress,
            )
            grid.save(surface_cache_path, {
                "dataset_fingerprint": fingerprint,
                "cell_size_m": config.raster.cell_size_m,
                "minimum_z_m": config.raster.minimum_z_m,
                "maximum_z_m": config.raster.maximum_z_m,
                "config_hash": config_hash,
                "ground_opening_radius_m": config.ground.opening_radius_m,
                "ground_smoothing_sigma_m": config.ground.smoothing_sigma_m,
            })

    if shared_cache:
        _materialize_cache_artifact(surface_cache_path, surface_artifact_path)
    if surface_cache_hit:
        print("[cache] 已加载表面栅格缓存", flush=True)
        reporter.progress("SURFACE_GRID", 31.0, message="已加载有效表面缓存")

    _report_stage(reporter, "GROUND_ESTIMATION", "估计坡地地面曲面")
    valid = grid.valid_mask
    mean_surface = fill_nearest(grid.mean_z, valid)
    ground = estimate_ground_surface(
        grid.min_z, valid, grid.cell_size,
        config.ground.opening_radius_m, config.ground.smoothing_sigma_m,
    )
    relative_height = np.maximum(mean_surface - ground, 0.0).astype(np.float32)
    roi_mask = valid.copy()
    roi_status = {"configured": config.project.roi_geojson is not None, "used": False}
    if config.project.roi_geojson is not None:
        roi_geometry = load_projected_roi(config.project.roi_geojson, config.crs.source_epsg)
        roi_mask = rasterize_geometry(
            roi_geometry, grid.rows, grid.cols, grid.x_min, grid.y_min, grid.cell_size,
        )
        roi_status.update({"used": True, "path": str(config.project.roi_geojson)})

    legacy_probability = None
    probability = None
    model_status = {"enabled": config.model.enabled, "legacy_used": False, "cell_classifier_used": False}
    _report_stage(reporter, "MODEL_INFERENCE", "执行模型推理")
    if config.model.enabled:
        try:
            segmenter = LegacyKerasSegmenter(config.model.weights_path, config.model.window_size)
            legacy_probability = infer_probability_grid(
                segmenter, mean_surface, grid.mean_rgb, valid, grid.cell_size,
                config.model.voxel_size_m, config.model.batch_size,
            )
            model_status["legacy_used"] = True
        except Exception as exc:
            model_status["legacy_error"] = f"{type(exc).__name__}: {exc}"
            print(f"[model] legacy model unavailable: {model_status['legacy_error']}", flush=True)
            reporter.log("WARNING", "旧模型不可用", {"error_type": type(exc).__name__})
        if config.model.cell_classifier_path is not None:
            try:
                features = build_cell_features(
                    relative_height, grid.mean_z, grid.min_z, grid.max_z,
                    grid.mean_rgb, grid.point_count, legacy_probability,
                )
                classifier = CellClassifier(config.model.cell_classifier_path)
                probability = classifier.predict_grid(features, valid & roi_mask)
                model_status["cell_classifier_used"] = True
                model_status["cell_feature_names"] = list(FEATURE_NAMES)
            except Exception as exc:
                model_status["cell_classifier_error"] = f"{type(exc).__name__}: {exc}"
                print(f"[model] cell classifier unavailable: {model_status['cell_classifier_error']}", flush=True)
                reporter.log("WARNING", "栅格分类模型不可用", {"error_type": type(exc).__name__})
        if probability is None:
            probability = legacy_probability
    model_status["used"] = bool(model_status["legacy_used"] or model_status["cell_classifier_used"])
    _report_stage(reporter, "CANDIDATE_EXTRACTION", "提取煤堆候选区")
    mask = candidate_mask(
        relative_height, valid & roi_mask, grid.cell_size,
        config.segmentation.minimum_height_m, config.segmentation.core_height_m,
        config.segmentation.minimum_area_m2, config.segmentation.maximum_hole_area_m2,
        config.segmentation.closing_radius_m, probability,
        config.model.probability_low, config.model.probability_high,
        config.segmentation.closing_min_gradient,
        config.segmentation.closing_min_local_relief_m,
        config.segmentation.closing_min_vertical_relief_m,
    )
    _report_stage(reporter, "VALLEY_SPLIT", "按低洼沟槽分堆")
    labels = split_by_height_valleys(
        relative_height, mask, grid.cell_size,
        config.segmentation.valley_depth_m, config.segmentation.minimum_peak_distance_m,
        config.segmentation.minimum_area_m2, config.segmentation.core_height_m,
        config.segmentation.peak_smoothing_m,
        config.segmentation.minimum_valley_depth_ratio,
    )
    _report_stage(reporter, "SLOPE_RECOVERY", "补全连续坡面并剔除结构误入")
    labels_before_recovery = labels.copy()
    labels = recover_concave_slopes(
        labels, relative_height, valid & roi_mask, grid.cell_size,
        config.segmentation.minimum_height_m,
        config.segmentation.concavity_recovery_buffer_m,
        probability,
        config.segmentation.concavity_probability_floor,
        config.segmentation.concavity_min_gradient,
        config.segmentation.concavity_min_local_relief_m,
        config.segmentation.concavity_min_component_area_m2,
        config.segmentation.concavity_min_peak_height_m,
        config.segmentation.concavity_min_vertical_relief_m,
    )
    labels_before_fragment_merge = labels.copy()
    labels = merge_nearby_toe_fragments(
        labels, relative_height, valid & roi_mask, grid.cell_size,
        config.segmentation.fragment_merge_gap_m,
        config.segmentation.fragment_max_area_m2,
        config.segmentation.fragment_max_area_ratio,
        config.segmentation.fragment_max_peak_height_m,
        probability,
        config.segmentation.fragment_min_probability,
        config.segmentation.peak_smoothing_m,
    )
    mask_before_recovery = mask
    mask = labels > 0
    _report_stage(reporter, "GEOMETRY_VOLUME", "生成边界与体积")
    results = build_pile_results(
        labels, relative_height, probability, grid.x_min, grid.y_min, grid.cell_size,
        config.segmentation.boundary_smoothing_m, config.segmentation.simplify_tolerance_m,
    )
    display_parameters = {
        **DISPLAY_GEOMETRY_PARAMETERS,
        "fine_cell_size_m": config.display_geometry.fine_cell_size_m,
        "search_margin_m": config.display_geometry.search_margin_m,
        "height_threshold_m": config.display_geometry.height_threshold_m,
        "field_smoothing_m": config.display_geometry.field_smoothing_m,
        "max_gap_fill_m": config.display_geometry.max_gap_fill_m,
        "toe_height_floor_m": config.display_geometry.toe_height_floor_m,
        "area_tolerance_percent": config.display_geometry.area_tolerance_percent,
        "selection_strategy": config.display_geometry.selection_strategy,
        "corridor_buffer_m": config.display_geometry.corridor_buffer_m,
        "ground_offset_inner_m": config.display_geometry.ground_offset_inner_m,
        "ground_offset_outer_m": config.display_geometry.ground_offset_outer_m,
    }
    if shared_cache:
        display_cache_path = cache_directory / (
            f"display_geometry_{suffix}_{display_geometry_cache_key(results, display_parameters)[:32]}.json"
        )
    else:
        display_cache_path = cache_directory / f"display_geometry_{suffix}.json"
    display_geometry_status = {
        "method": DISPLAY_GEOMETRY_METHOD,
        "used": False,
        "calculation_geometry_unchanged": True,
        **display_parameters,
        "cache_enabled": use_cache,
        "cache_scope": "shared" if shared_cache else "output-directory",
    }
    display_lock = cache_file_lock(display_cache_path) if shared_cache else nullcontext()
    with display_lock:
        display_cache_result = DisplayGeometryLoadResult(False, "已禁用缓存")
        if use_cache:
            display_cache_result = load_display_geometry_cache(
                display_cache_path, results, fingerprint, config_hash, display_parameters,
            )
            if not display_cache_result.hit:
                display_cache_result = load_display_geometry_from_result(
                    business_path, results, fingerprint, config_hash, config.crs, display_parameters,
                )
                if display_cache_result.hit:
                    try:
                        save_display_geometry_cache(
                            display_cache_path, results, fingerprint, config_hash, display_parameters,
                        )
                    except Exception as exc:
                        reporter.log(
                            "WARNING", "展示曲线缓存迁移后写入失败",
                            {"error_type": type(exc).__name__},
                        )
        if display_cache_result.hit:
            print(
                f"[cache] 已加载连续坡脚展示曲线（{display_cache_result.pile_count} 座煤堆）",
                flush=True,
            )
            reporter.progress(
                "GEOMETRY_VOLUME", 87.0,
                current=len(headers), total=len(headers), message="已加载连续坡脚展示曲线缓存",
            )
            cache_source = display_cache_result.source
            if shared_cache and cache_source == "display-cache":
                cache_source = "shared-display-cache"
            display_geometry_status.update({
                "used": True,
                "pile_count": display_cache_result.pile_count,
                "cache_hit": True,
                "cache_source": cache_source,
            })
        else:
            display_geometry_status.update({
                "cache_hit": False,
                "cache_miss_reason": display_cache_result.reason,
            })
            print(
                f"[display-geometry] 首次生成连续坡脚展示曲线（需扫描 {len(headers)} 个 LAS 文件）",
                flush=True,
            )
            last_progress_bucket = -1

            def display_progress(i, total, path):
                nonlocal last_progress_bucket
                percent = int(i * 100 / max(total, 1))
                bucket = min(100, (percent // 10) * 10)
                if i == 1 or i == total or bucket > last_progress_bucket:
                    print(f"[display-geometry] {percent}% ({i}/{total})", flush=True)
                    last_progress_bucket = bucket
                reporter.progress(
                    "GEOMETRY_VOLUME", stage_fraction_progress("GEOMETRY_VOLUME", i, total),
                    current=i, total=total, message="正在按原始点云拟合连续坡脚曲线",
                    context={"file_name": path.name},
                )
            try:
                refined = refine_polygons_from_point_cloud(
                    {result.pile_id: result.polygon for result in results},
                    headers, ground, labels, grid.x_min, grid.y_min, grid.cell_size,
                    chunk_points=config.raster.chunk_points,
                    minimum_z_m=config.raster.minimum_z_m,
                    maximum_z_m=config.raster.maximum_z_m,
                    progress=display_progress,
                    **display_parameters,
                )
                for result in results:
                    display_polygon, _, display_metrics = refined[result.pile_id]
                    result.display_polygon = display_polygon
                    result.display_geometry_metrics = asdict(display_metrics)
                display_geometry_status["used"] = True
                display_geometry_status["pile_count"] = len(refined)
                try:
                    save_display_geometry_cache(
                        display_cache_path, results, fingerprint, config_hash, display_parameters,
                    )
                    display_geometry_status["cache_saved"] = True
                except Exception as exc:
                    display_geometry_status["cache_save_error"] = f"{type(exc).__name__}: {exc}"
                    reporter.log(
                        "WARNING", "连续坡脚展示曲线缓存写入失败",
                        {"error_type": type(exc).__name__},
                    )
            except Exception as exc:
                display_geometry_status["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[display-geometry] unavailable: {display_geometry_status['error']}", flush=True)
                reporter.log(
                    "WARNING", "连续坡脚展示曲线生成失败，保留计算边界显示",
                    {"error_type": type(exc).__name__},
                )
    references = []
    evaluation = None
    if config.project.reference_xlsx and config.project.reference_xlsx.exists():
        _report_stage(reporter, "EVALUATION", "评估人工参考边界")
        references = load_reference_workbook(
            config.project.reference_xlsx, config.crs.central_meridian_deg,
            config.crs.false_easting_m, config.crs.false_northing_m,
        )
        evaluation = evaluate_results(
            results, references, config.evaluation.true_total_volume_m3,
            config.evaluation.target_total_volume_error_percent,
        )
    metadata = {
        "project_name": config.project.name, "dataset_fingerprint": fingerprint,
        "config_hash": config_hash, "las_summary": las_summary,
        "crs_validation": crs_validation,
        "model": model_status, "roi": roi_status,
        "display_geometry": display_geometry_status,
        "cache": {
            "enabled": use_cache,
            "scope": "shared" if shared_cache else "output-directory",
            "surface_hit": surface_cache_hit,
            "display_geometry_hit": bool(display_geometry_status.get("cache_hit")),
        },
        "runtime_seconds": time.time() - started,
        "partial_run_max_files": max_files,
    }
    _report_stage(reporter, "EXPORT", "输出 JSON/GeoJSON")
    result_doc, geojson = make_result_documents(results, metadata, config.crs, evaluation)
    service_fences = make_service_fence_document(results, config.crs)
    geojson_path = output_dir / f"coal_piles_{suffix}.geojson"
    service_path = output_dir / f"coal_piles_service_{suffix}.json"
    evaluation_path = output_dir / f"evaluation_{suffix}.json"
    diagnostics_path = output_dir / f"diagnostics_{suffix}.npz"
    atomic_write_json(business_path, result_doc)
    atomic_write_json(geojson_path, geojson)
    atomic_write_json(service_path, service_fences)
    atomic_write_json(evaluation_path, evaluation or {})
    np.savez_compressed(
        diagnostics_path,
        ground=ground, relative_height=relative_height, probability=probability,
        legacy_probability=legacy_probability, roi_mask=roi_mask,
        mask=mask, mask_before_recovery=mask_before_recovery,
        labels_before_recovery=labels_before_recovery, labels=labels,
        labels_before_fragment_merge=labels_before_fragment_merge,
        x_min=grid.x_min, y_min=grid.y_min, cell_size=grid.cell_size,
        dataset_fingerprint=np.asarray(fingerprint), config_hash=np.asarray(config_hash),
        ground_opening_radius_m=np.float64(config.ground.opening_radius_m),
        ground_smoothing_sigma_m=np.float64(config.ground.smoothing_sigma_m),
    )
    result_sha256 = reporter.submit_result("BOUNDARY_RESULT", result_doc)
    artifact_paths = {
        "BUSINESS_JSON": business_path,
        "GEOJSON": geojson_path,
        "SERVICE_JSON": service_path,
        "EVALUATION_JSON": evaluation_path,
    }
    for artifact_type in config.integration.upload_artifacts:
        artifact_path = artifact_paths.get(artifact_type)
        if artifact_path is not None:
            reporter.upload_artifact(artifact_type, artifact_path)
    if config.integration.upload_large_diagnostics:
        reporter.upload_artifact("DIAGNOSTICS_NPZ", diagnostics_path)
        reporter.upload_artifact("SURFACE_NPZ", surface_artifact_path)
    reporter.complete(result_doc.get("summary", {}), metadata["runtime_seconds"], result_sha256)
    return result_doc
