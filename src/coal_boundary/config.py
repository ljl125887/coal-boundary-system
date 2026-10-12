from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


ALLOWED_ARTIFACT_TYPES = {
    "BUSINESS_JSON", "GEOJSON", "SERVICE_JSON", "EVALUATION_JSON",
    "TRAINING_REPORT_JSON", "MODEL_KERAS", "LOG_JSONL",
    "DIAGNOSTICS_NPZ", "SURFACE_NPZ", "PILE_VOLUME_JSON",
    "EFFECTIVE_CONFIG", "RUN_MANIFEST",
}

ALLOWED_ARTIFACT_MODES = {"DISABLED", "LOCAL_REGISTER", "OBJECT_STORAGE_UPLOAD"}


@dataclass(frozen=True)
class ProjectConfig:
    name: str
    las_dir: Path
    output_dir: Path
    reference_xlsx: Path | None = None
    roi_geojson: Path | None = None


@dataclass(frozen=True)
class CrsConfig:
    source_epsg: int = 4549
    output_epsg: int = 4326
    central_meridian_deg: float = 120.0
    false_easting_m: float = 500000.0
    false_northing_m: float = 0.0


@dataclass(frozen=True)
class RasterConfig:
    cell_size_m: float = 1.0
    chunk_points: int = 1_500_000
    minimum_z_m: float = -20.0
    maximum_z_m: float = 60.0


@dataclass(frozen=True)
class GroundConfig:
    opening_radius_m: float = 35.0
    smoothing_sigma_m: float = 4.0


@dataclass(frozen=True)
class DisplayGeometryConfig:
    """展示用连续坡脚轮廓的参数（只影响三维/二维展示，不影响面积/体积核算）。"""

    # toe_first（默认，贴近地面优先）| area_first（面积优先，旧行为）
    selection_strategy: str = "toe_first"
    # 等高线允许的最低“离地高度”（m）。越小越贴地，0.05 = 离地 5 cm。
    toe_height_floor_m: float = 0.05
    # 贴地优先时允许的轮廓面积偏差（%）：在此容差内取最低（最贴地）的等高线，
    # 超出容差则退回“面积最接近审计边界”的选择。
    area_tolerance_percent: float = 15.0
    fine_cell_size_m: float = 0.20
    search_margin_m: float = 3.0
    height_threshold_m: float = 0.35
    field_smoothing_m: float = 0.50
    max_gap_fill_m: float = 0.60
    # 坡脚等高线允许出现的范围（source 外扩多少米）。必须小于 search_margin_m，
    # 否则等高线会沿着证据带边缘闭合，形成面积巨大的假坡脚。
    corridor_buffer_m: float = 2.8
    # “地面归零”环带：用 source 外这些米范围内的地面相对高度中位数作为零点。
    # 地面估计算法会把煤堆外围地面压低约 0.3 m，不归零时 0 m 等高线不存在，
    # 边界只能靠抬高 level 圈定，于是看起来“不贴地”。
    ground_offset_inner_m: float = 1.0
    ground_offset_outer_m: float = 3.0


@dataclass(frozen=True)
class SegmentationConfig:
    minimum_height_m: float = 0.4
    core_height_m: float = 0.8
    valley_depth_m: float = 1.2
    minimum_valley_depth_ratio: float = 0.6
    minimum_peak_distance_m: float = 12.0
    peak_smoothing_m: float = 4.0
    minimum_area_m2: float = 35.0
    maximum_hole_area_m2: float = 80.0
    closing_radius_m: float = 1.0
    closing_min_gradient: float = 0.45
    closing_min_local_relief_m: float = 0.8
    closing_min_vertical_relief_m: float = 0.5
    concavity_recovery_buffer_m: float = 3.0
    concavity_probability_floor: float = 0.4
    concavity_min_gradient: float = 0.45
    concavity_min_local_relief_m: float = 0.8
    concavity_min_component_area_m2: float = 25.0
    concavity_min_peak_height_m: float = 3.0
    concavity_min_vertical_relief_m: float = 2.0
    fragment_merge_gap_m: float = 4.0
    fragment_max_area_m2: float = 600.0
    fragment_max_area_ratio: float = 0.3
    fragment_max_peak_height_m: float = 4.0
    fragment_min_probability: float = 0.75
    boundary_smoothing_m: float = 0.75
    simplify_tolerance_m: float = 0.35


@dataclass(frozen=True)
class ModelConfig:
    enabled: bool = True
    weights_path: Path | None = None
    cell_classifier_path: Path | None = None
    voxel_size_m: float = 2.0
    probability_low: float = 0.35
    probability_high: float = 0.8
    window_size: int = 1024
    batch_size: int = 64


@dataclass(frozen=True)
class EvaluationConfig:
    true_total_volume_m3: float = 1.0  # Placeholder; provide a reference before evaluation.
    target_total_volume_error_percent: float = 2.0


@dataclass(frozen=True)
class IntegrationConfig:
    enabled: bool = False
    base_url: str = "http://localhost:8080"
    token_env: str = "COAL_BOUNDARY_API_TOKEN"
    login_path: str = "/login"
    username_env: str = "COAL_BOUNDARY_API_USERNAME"
    password_env: str = "COAL_BOUNDARY_API_PASSWORD"
    connect_timeout_seconds: float = 3.0
    read_timeout_seconds: float = 15.0
    verify_tls: bool = True
    event_batch_size: int = 50
    flush_interval_seconds: float = 1.0
    heartbeat_interval_seconds: float = 15.0
    max_retries: int = 5
    spool_dir: Path = Path("integration_spool")
    upload_artifacts: tuple[str, ...] = (
        "BUSINESS_JSON", "GEOJSON", "SERVICE_JSON", "EVALUATION_JSON",
    )
    upload_large_diagnostics: bool = False
    artifact_mode: str = "OBJECT_STORAGE_UPLOAD"
    # 手工运行（GUI / CLI run）时的本地成果登记根：同服务器部署应与后端 coal-results
    # 根目录一致，且要能覆盖 project.output_dir，否则该次运行的附件不会登记。
    artifact_root_id: str | None = None
    artifact_root: Path | None = None


@dataclass(frozen=True)
class WorkerConfig:
    enabled: bool = False
    worker_id: str = "coal-worker-01"
    poll_interval_seconds: float = 5.0
    lease_seconds: int = 90
    lease_renew_interval_seconds: float = 20.0
    max_concurrent_jobs: int = 1
    accepted_job_types: tuple[str, ...] = ("EXTRACTION", "PILE_VOLUME_RECALC")
    accepted_profiles: tuple[str, ...] = ("default", "production")
    input_roots: dict[str, Path] = field(default_factory=dict)
    output_roots: dict[str, Path] = field(default_factory=dict)
    # Shared algorithm cache.  When omitted, Worker derives
    # ``<selected output_root>/_algorithm_cache`` for each claimed job.
    cache_root: Path | None = None
    allowed_extensions: tuple[str, ...] = (".las",)
    require_ready_marker: bool = False
    ready_marker_name: str = "_READY"
    stable_file_seconds: float = 60.0
    max_files: int = 1000
    max_total_bytes: int = 2_000_000_000_000
    local_log_dir: Path = Path("worker_logs")


@dataclass(frozen=True)
class AppConfig:
    project: ProjectConfig
    crs: CrsConfig = field(default_factory=CrsConfig)
    raster: RasterConfig = field(default_factory=RasterConfig)
    ground: GroundConfig = field(default_factory=GroundConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    display_geometry: DisplayGeometryConfig = field(default_factory=DisplayGeometryConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    integration: IntegrationConfig = field(default_factory=IntegrationConfig)
    worker: WorkerConfig = field(default_factory=WorkerConfig)
    config_path: Path | None = None


def _resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return (base / p).resolve() if not p.is_absolute() else p.resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base = config_path.parent
    project_raw = raw.get("project", {})
    model_raw = raw.get("model", {})
    project = ProjectConfig(
        name=str(project_raw.get("name", "coal-boundary")),
        las_dir=_resolve(base, project_raw.get("las_dir")),
        output_dir=_resolve(base, project_raw.get("output_dir")),
        reference_xlsx=_resolve(base, project_raw.get("reference_xlsx")),
        roi_geojson=_resolve(base, project_raw.get("roi_geojson")),
    )
    if project.las_dir is None or project.output_dir is None:
        raise ValueError("project.las_dir 和 project.output_dir 必须配置")
    if not project.las_dir.exists():
        raise FileNotFoundError(f"LAS 目录不存在: {project.las_dir}")
    raster = RasterConfig(**raw.get("raster", {}))
    if raster.cell_size_m <= 0 or raster.chunk_points <= 0:
        raise ValueError("栅格尺寸和分块点数必须大于零")
    model = ModelConfig(
        **{k: v for k, v in model_raw.items() if k not in {"weights_path", "cell_classifier_path"}},
        weights_path=_resolve(base, model_raw.get("weights_path")),
        cell_classifier_path=_resolve(base, model_raw.get("cell_classifier_path")),
    )
    if model.enabled and (model.weights_path is None or not model.weights_path.exists()):
        raise FileNotFoundError(f"模型权重不存在: {model.weights_path}")
    if model.enabled and model.cell_classifier_path is not None and not model.cell_classifier_path.exists():
        raise FileNotFoundError(f"栅格分类模型不存在: {model.cell_classifier_path}")
    if project.roi_geojson is not None and not project.roi_geojson.exists():
        raise FileNotFoundError(f"作业区 GeoJSON 不存在: {project.roi_geojson}")
    segmentation = SegmentationConfig(**raw.get("segmentation", {}))
    if not 0.0 <= segmentation.minimum_valley_depth_ratio <= 1.0:
        raise ValueError("segmentation.minimum_valley_depth_ratio 必须在 0 到 1 之间")
    if not 0.0 < segmentation.fragment_max_area_ratio <= 1.0:
        raise ValueError("segmentation.fragment_max_area_ratio 必须在 0 到 1 之间")
    if not 0.0 <= segmentation.fragment_min_probability <= 1.0:
        raise ValueError("segmentation.fragment_min_probability 必须在 0 到 1 之间")
    if not 0.0 <= segmentation.concavity_probability_floor <= 1.0:
        raise ValueError("segmentation.concavity_probability_floor 必须在 0 到 1 之间")
    if min(
        segmentation.minimum_height_m,
        segmentation.valley_depth_m,
        segmentation.minimum_area_m2,
        segmentation.closing_radius_m,
        segmentation.concavity_recovery_buffer_m,
        segmentation.fragment_merge_gap_m,
    ) < 0:
        raise ValueError("分割距离、面积和高度参数不能为负数")
    integration_raw = dict(raw.get("integration", {}) or {})
    upload_artifacts = tuple(integration_raw.pop("upload_artifacts", IntegrationConfig.upload_artifacts))
    spool_dir = _resolve(base, integration_raw.pop("spool_dir", "integration_spool"))
    artifact_root = _resolve(base, integration_raw.pop("artifact_root", None))
    integration = IntegrationConfig(
        **integration_raw,
        spool_dir=spool_dir or (base / "integration_spool").resolve(),
        upload_artifacts=upload_artifacts,
        artifact_root=artifact_root,
    )
    if integration.enabled and not integration.base_url.strip():
        raise ValueError("integration.enabled=true 时 base_url 不能为空")
    if integration.enabled and not integration.login_path.strip():
        raise ValueError("integration.enabled=true 时 login_path 不能为空")
    if (
        integration.event_batch_size <= 0
        or integration.flush_interval_seconds <= 0
        or integration.heartbeat_interval_seconds <= 0
    ):
        raise ValueError("integration 事件批量、刷新间隔和心跳间隔必须大于零")
    if integration.connect_timeout_seconds <= 0 or integration.read_timeout_seconds <= 0:
        raise ValueError("integration 超时时间必须大于零")
    if integration.max_retries < 0:
        raise ValueError("integration.max_retries 不能为负数")
    unknown_artifacts = set(integration.upload_artifacts) - ALLOWED_ARTIFACT_TYPES
    if unknown_artifacts:
        raise ValueError(f"integration.upload_artifacts 包含未知类型: {sorted(unknown_artifacts)}")
    integration_mode = integration.artifact_mode.upper()
    if integration_mode not in ALLOWED_ARTIFACT_MODES:
        raise ValueError(f"integration.artifact_mode 必须是 {sorted(ALLOWED_ARTIFACT_MODES)} 之一")
    if integration_mode != integration.artifact_mode:
        integration = IntegrationConfig(**{**integration.__dict__, "artifact_mode": integration_mode})

    worker_raw = dict(raw.get("worker", {}) or {})
    input_roots_raw = dict(worker_raw.pop("input_roots", {}) or {})
    output_roots_raw = dict(worker_raw.pop("output_roots", {}) or {})
    accepted_job_types = tuple(worker_raw.pop("accepted_job_types", WorkerConfig.accepted_job_types))
    accepted_profiles = tuple(worker_raw.pop("accepted_profiles", WorkerConfig.accepted_profiles))
    allowed_extensions = tuple(
        str(item).lower() if str(item).startswith(".") else f".{str(item).lower()}"
        for item in worker_raw.pop("allowed_extensions", WorkerConfig.allowed_extensions)
    )
    local_log_dir = _resolve(base, worker_raw.pop("local_log_dir", "worker_logs"))
    cache_root = _resolve(base, worker_raw.pop("cache_root", None))
    worker = WorkerConfig(
        **worker_raw,
        accepted_job_types=accepted_job_types,
        accepted_profiles=accepted_profiles,
        allowed_extensions=allowed_extensions,
        input_roots={str(k): _resolve(base, str(v)) for k, v in input_roots_raw.items()},
        output_roots={str(k): _resolve(base, str(v)) for k, v in output_roots_raw.items()},
        cache_root=cache_root,
        local_log_dir=local_log_dir or (base / "worker_logs").resolve(),
    )
    valid_job_types = {"EXTRACTION", "PILE_VOLUME_RECALC"}
    if not worker.accepted_job_types or set(worker.accepted_job_types) - valid_job_types:
        raise ValueError(f"worker.accepted_job_types 只能包含 {sorted(valid_job_types)}")
    if not worker.accepted_profiles or any(not str(item).strip() for item in worker.accepted_profiles):
        raise ValueError("worker.accepted_profiles 不能为空")
    if worker.poll_interval_seconds <= 0 or worker.lease_renew_interval_seconds <= 0:
        raise ValueError("worker 轮询和续租间隔必须大于零")
    if not 30 <= worker.lease_seconds <= 600:
        raise ValueError("worker.lease_seconds 必须在 30 到 600 之间")
    if worker.lease_renew_interval_seconds >= worker.lease_seconds:
        raise ValueError("worker.lease_renew_interval_seconds 必须小于 lease_seconds")
    if worker.max_concurrent_jobs != 1:
        raise ValueError("当前版本 worker.max_concurrent_jobs 必须为 1")
    if worker.max_files <= 0 or worker.max_total_bytes <= 0 or worker.stable_file_seconds < 0:
        raise ValueError("worker 文件数量、总字节限制必须大于零，稳定等待时间不能为负数")
    if worker.enabled:
        if not integration.enabled:
            raise ValueError("worker.enabled=true 时 integration.enabled 也必须为 true")
        if not worker.input_roots or not worker.output_roots:
            raise ValueError("worker.enabled=true 时必须配置 input_roots 和 output_roots")
        for root_id, root_path in {**worker.input_roots, **worker.output_roots}.items():
            if not root_id or root_path is None or not root_path.is_absolute():
                raise ValueError(f"Worker 根目录配置无效: {root_id}")
        if worker.cache_root is not None and not worker.cache_root.is_absolute():
            raise ValueError("worker.cache_root 必须是绝对路径")
    return AppConfig(
        project=project,
        crs=CrsConfig(**raw.get("crs", {})),
        raster=raster,
        ground=GroundConfig(**raw.get("ground", {})),
        segmentation=segmentation,
        display_geometry=DisplayGeometryConfig(**raw.get("display_geometry", {})),
        model=model,
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        integration=integration,
        worker=worker,
        config_path=config_path,
    )
