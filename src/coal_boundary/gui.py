from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import mimetypes
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import urllib.request
import uuid
import webbrowser

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml

from .config import load_config
from .integration.client import BackendApiClient
from .integration import build_reporter


APP_TITLE = "煤堆点云智能边界提取"
DEFAULT_VIEWER_URL = "http://localhost:5173/"
DEFAULT_UPLOAD_URL = "http://localhost:3000/upload"
DEFAULT_WINDOWS_PYTHON = Path(sys.executable)


def utf8_subprocess_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Build an environment that makes redirected Python output UTF-8 on Windows."""
    environment = dict(os.environ if base is None else base)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUTF8"] = "1"
    # 只使用当前环境(conda env)中的包，避免用户级 site-packages 遮蔽导致环境不可迁移
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def resolve_cli_python() -> Path:
    """Select the coal-boundary environment even if the GUI was launched elsewhere."""
    override = os.environ.get("COAL_BOUNDARY_PYTHON")
    candidates: list[Path] = []
    if override:
        candidates.append(Path(override))
    if os.name == "nt":
        # Also support a Conda installation on another drive beside the
        # environment that happened to launch this GUI.
        current_python = Path(sys.executable)
        if current_python.parent.parent.name.lower() == "envs":
            candidates.append(current_python.parent.parent / "pointcloud_env" / "python.exe")
        candidates.append(DEFAULT_WINDOWS_PYTHON)
    candidates.append(Path(sys.executable))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return Path(sys.executable).resolve()


@dataclass(frozen=True)
class FieldSpec:
    group: str
    label: str
    section: str
    key: str
    value_type: type = str
    browse: str | None = None
    optional: bool = False
    hint: str = ""


@dataclass
class PileView:
    pile_id: str
    coordinates: list[tuple[float, float]]
    area_m2: float | None = None
    volume_m3: float | None = None
    max_height_m: float | None = None
    mean_height_m: float | None = None
    quality_score: float | None = None
    quality_grade: str = ""
    boundary_confidence: float | None = None


FIELD_SPECS = [
    FieldSpec("工程与数据", "项目名称", "project", "name", str),
    FieldSpec("工程与数据", "LAS 文件夹", "project", "las_dir", str, "directory"),
    FieldSpec("工程与数据", "输出文件夹", "project", "output_dir", str, "directory"),
    FieldSpec("工程与数据", "参考台账 XLSX", "project", "reference_xlsx", str, "xlsx", True),
    FieldSpec("工程与数据", "作业区 ROI GeoJSON", "project", "roi_geojson", str, "geojson", True),
    FieldSpec("空间与栅格", "源坐标系 EPSG", "crs", "source_epsg", int),
    FieldSpec("空间与栅格", "输出坐标系 EPSG", "crs", "output_epsg", int),
    FieldSpec("空间与栅格", "中央经线（度）", "crs", "central_meridian_deg", float),
    FieldSpec("空间与栅格", "假东（m）", "crs", "false_easting_m", float),
    FieldSpec("空间与栅格", "假北（m）", "crs", "false_northing_m", float),
    FieldSpec("空间与栅格", "栅格分辨率（m）", "raster", "cell_size_m", float),
    FieldSpec("空间与栅格", "LAS 分块点数", "raster", "chunk_points", int),
    FieldSpec("空间与栅格", "最低有效 Z（m）", "raster", "minimum_z_m", float),
    FieldSpec("空间与栅格", "最高有效 Z（m）", "raster", "maximum_z_m", float),
    FieldSpec("地面与边界", "地面开运算半径（m）", "ground", "opening_radius_m", float),
    FieldSpec("地面与边界", "地面平滑尺度（m）", "ground", "smoothing_sigma_m", float),
    FieldSpec("地面与边界", "最低相对高度（m）", "segmentation", "minimum_height_m", float),
    FieldSpec("地面与边界", "核心相对高度（m）", "segmentation", "core_height_m", float),
    FieldSpec("地面与边界", "沟槽谷深（m）", "segmentation", "valley_depth_m", float),
    FieldSpec("地面与边界", "最小相对谷深（0-1）", "segmentation", "minimum_valley_depth_ratio", float),
    FieldSpec("地面与边界", "最小峰间距（m）", "segmentation", "minimum_peak_distance_m", float),
    FieldSpec("地面与边界", "峰值平滑尺度（m）", "segmentation", "peak_smoothing_m", float),
    FieldSpec("地面与边界", "最小煤堆面积（m²）", "segmentation", "minimum_area_m2", float),
    FieldSpec("地面与边界", "最大孔洞面积（m²）", "segmentation", "maximum_hole_area_m2", float),
    FieldSpec("地面与边界", "闭运算半径（m）", "segmentation", "closing_radius_m", float),
    FieldSpec("地面与边界", "闭合最小坡度", "segmentation", "closing_min_gradient", float),
    FieldSpec("地面与边界", "闭合最小局部起伏（m）", "segmentation", "closing_min_local_relief_m", float),
    FieldSpec("地面与边界", "闭合最小垂直起伏（m）", "segmentation", "closing_min_vertical_relief_m", float),
    FieldSpec("地面与边界", "凹坡恢复缓冲（m）", "segmentation", "concavity_recovery_buffer_m", float),
    FieldSpec("地面与边界", "凹坡最低概率", "segmentation", "concavity_probability_floor", float),
    FieldSpec("地面与边界", "凹坡最小坡度", "segmentation", "concavity_min_gradient", float),
    FieldSpec("地面与边界", "凹坡最小局部起伏（m）", "segmentation", "concavity_min_local_relief_m", float),
    FieldSpec("地面与边界", "凹坡最小斑块（m²）", "segmentation", "concavity_min_component_area_m2", float),
    FieldSpec("地面与边界", "凹坡最小峰高（m）", "segmentation", "concavity_min_peak_height_m", float),
    FieldSpec("地面与边界", "凹坡最小垂直起伏（m）", "segmentation", "concavity_min_vertical_relief_m", float),
    FieldSpec("地面与边界", "坡脚碎片回接距离（m）", "segmentation", "fragment_merge_gap_m", float),
    FieldSpec("地面与边界", "坡脚碎片最大面积（m²）", "segmentation", "fragment_max_area_m2", float),
    FieldSpec("地面与边界", "坡脚碎片最大面积比", "segmentation", "fragment_max_area_ratio", float),
    FieldSpec("地面与边界", "坡脚碎片最大峰高（m）", "segmentation", "fragment_max_peak_height_m", float),
    FieldSpec("地面与边界", "坡脚碎片最低概率", "segmentation", "fragment_min_probability", float),
    FieldSpec("地面与边界", "边界平滑（m）", "segmentation", "boundary_smoothing_m", float),
    FieldSpec("地面与边界", "边界简化容差（m）", "segmentation", "simplify_tolerance_m", float),
    FieldSpec("模型与评估", "旧模型权重", "model", "weights_path", str, "model", True),
    FieldSpec("模型与评估", "栅格分类模型", "model", "cell_classifier_path", str, "model", True),
    FieldSpec("模型与评估", "模型体素尺寸（m）", "model", "voxel_size_m", float),
    FieldSpec("模型与评估", "概率低阈值", "model", "probability_low", float),
    FieldSpec("模型与评估", "概率高阈值", "model", "probability_high", float),
    FieldSpec("模型与评估", "模型窗口点数", "model", "window_size", int),
    FieldSpec("模型与评估", "推理批量大小", "model", "batch_size", int),
    FieldSpec("模型与评估", "真实总体积（m³）", "evaluation", "true_total_volume_m3", float),
    FieldSpec("模型与评估", "目标总体积误差（%）", "evaluation", "target_total_volume_error_percent", float),
    FieldSpec("后端同步", "后端 API 地址", "integration", "base_url", str),
]


STAGE_PROGRESS = {
    "读取并校验数据": 5,
    "建立或加载表面栅格": 15,
    "估计坡地地面曲面": 32,
    "执行模型推理": 45,
    "提取煤堆候选区": 58,
    "按低洼沟槽分堆": 70,
    "补全连续坡面并剔除结构误入": 76,
    "生成边界与体积": 80,
    "评估人工参考边界": 88,
    "输出 JSON/GeoJSON": 96,
}


def read_raw_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("配置文件根节点必须是对象")
    return raw


def field_values_from_config(raw: dict) -> dict[tuple[str, str], object]:
    values: dict[tuple[str, str], object] = {}
    for spec in FIELD_SPECS:
        values[(spec.section, spec.key)] = raw.get(spec.section, {}).get(spec.key, "")
    values[("model", "enabled")] = bool(raw.get("model", {}).get("enabled", True))
    values[("integration", "enabled")] = bool(raw.get("integration", {}).get("enabled", False))
    return values


def _convert_field_value(spec: FieldSpec, value: object):
    text = str(value).strip()
    if not text:
        if spec.optional:
            return None
        raise ValueError(f"{spec.label} 不能为空")
    if spec.value_type is int:
        return int(float(text))
    if spec.value_type is float:
        number = float(text)
        if not math.isfinite(number):
            raise ValueError(f"{spec.label} 必须是有限数字")
        return number
    return text


def update_raw_config(raw: dict, values: dict[tuple[str, str], object]) -> dict:
    updated = dict(raw)
    for spec in FIELD_SPECS:
        section = dict(updated.get(spec.section, {}) or {})
        section[spec.key] = _convert_field_value(spec, values.get((spec.section, spec.key), ""))
        updated[spec.section] = section
    model = dict(updated.get("model", {}) or {})
    model["enabled"] = bool(values.get(("model", "enabled"), True))
    updated["model"] = model
    integration = dict(updated.get("integration", {}) or {})
    integration["enabled"] = bool(values.get(("integration", "enabled"), False))
    updated["integration"] = integration
    return updated


def atomic_save_config(path: str | Path, raw: dict, *, create_backup: bool = True) -> Path:
    config_path = Path(path).resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = yaml.safe_dump(raw, allow_unicode=True, sort_keys=False)
    if config_path.exists() and config_path.read_text(encoding="utf-8") == serialized:
        return config_path
    if create_backup and config_path.exists():
        shutil.copy2(config_path, config_path.with_suffix(config_path.suffix + ".gui-backup"))
    temporary = config_path.with_suffix(config_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(config_path)
    return config_path


def resolve_config_path(config_path: str | Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    base = Path(config_path).resolve().parent
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _ring_area(ring: list) -> float:
    if len(ring) < 3:
        return 0.0
    total = 0.0
    for index, point in enumerate(ring):
        nxt = ring[(index + 1) % len(ring)]
        total += float(point[0]) * float(nxt[1]) - float(nxt[0]) * float(point[1])
    return abs(total) / 2.0


def exterior_ring(geometry: dict | None) -> list:
    if not isinstance(geometry, dict):
        raise ValueError("缺少 geometry")
    coordinates = geometry.get("coordinates")
    if geometry.get("type") == "Polygon" and isinstance(coordinates, list) and coordinates:
        return coordinates[0]
    if geometry.get("type") == "MultiPolygon" and isinstance(coordinates, list) and coordinates:
        polygons = [polygon for polygon in coordinates if polygon and polygon[0]]
        if not polygons:
            raise ValueError("MultiPolygon 不包含外环")
        return max(polygons, key=lambda polygon: _ring_area(polygon[0]))[0]
    raise ValueError(f"不支持的几何类型: {geometry.get('type')}")


def _optional_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_coordinates(raw_coordinates: list) -> list[tuple[float, float]]:
    coordinates: list[tuple[float, float]] = []
    for point in raw_coordinates:
        if isinstance(point, dict):
            lon, lat = point.get("lon"), point.get("lat")
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            lon, lat = point[0], point[1]
        else:
            continue
        try:
            lon_value, lat_value = float(lon), float(lat)
        except (TypeError, ValueError):
            continue
        if math.isfinite(lon_value) and math.isfinite(lat_value):
            coordinates.append((lon_value, lat_value))
    if len(coordinates) > 1 and coordinates[0] == coordinates[-1]:
        coordinates.pop()
    if len(coordinates) < 3:
        raise ValueError("有效边界点少于 3 个")
    return coordinates


def boundary_view_transform(
    base_bounds: tuple[float, float, float, float],
    center: tuple[float, float],
    zoom: float,
    canvas_width: int,
    canvas_height: int,
    padding: int = 28,
) -> tuple[float, float, float, float, float, int, int]:
    """Return the actual world extent visible in a letterboxed canvas.

    The dataset and canvas usually have very different aspect ratios.  Culling
    against the unadjusted dataset extent hides valid piles in the letterboxed
    portion of the screen, so visibility must use this aspect-corrected extent.
    """
    base_min_x, base_min_y, base_max_x, base_max_y = base_bounds
    target_width = max(base_max_x - base_min_x, 1e-12) / max(zoom, 1e-12)
    target_height = max(base_max_y - base_min_y, 1e-12) / max(zoom, 1e-12)
    canvas_width = max(canvas_width, 100)
    canvas_height = max(canvas_height, 100)
    available_width = max(canvas_width - 2 * padding, 1)
    available_height = max(canvas_height - 2 * padding, 1)
    scale = min(available_width / target_width, available_height / target_height)
    visible_width = available_width / max(scale, 1e-12)
    visible_height = available_height / max(scale, 1e-12)
    center_x, center_y = center
    return (
        center_x - visible_width / 2,
        center_y - visible_height / 2,
        center_x + visible_width / 2,
        center_y + visible_height / 2,
        scale,
        canvas_height,
        padding,
    )


def load_result_document(path: str | Path) -> tuple[list[PileView], dict]:
    result_path = Path(path)
    document = json.loads(result_path.read_text(encoding="utf-8"))
    piles: list[PileView] = []
    summary: dict = {}

    if isinstance(document, dict) and isinstance(document.get("piles"), list):
        source_items = [
            (pile, pile, pile.get("display_geometry") or pile.get("geometry"))
            for pile in document["piles"]
        ]
        summary = dict(document.get("summary") or {})
        summary["evaluation"] = document.get("evaluation")
        summary["display_geometry_count"] = sum(
            isinstance(pile.get("display_geometry"), dict) for pile in document["piles"]
        )
    elif isinstance(document, dict) and document.get("type") == "FeatureCollection":
        source_items = [
            (
                feature,
                feature.get("properties") or {},
                feature.get("display_geometry")
                or (feature.get("properties") or {}).get("display_geometry")
                or feature.get("geometry"),
            )
            for feature in document.get("features", [])
        ]
        summary["display_geometry_count"] = sum(
            isinstance(feature.get("display_geometry"), dict)
            or isinstance((feature.get("properties") or {}).get("display_geometry"), dict)
            for feature in document.get("features", [])
        )
    elif isinstance(document, list):
        source_items = [(item, item, None) for item in document]
        summary["display_geometry_count"] = sum(
            isinstance(item.get("display_points"), list) for item in document
        )
    else:
        raise ValueError("不支持的结果文件结构")

    for index, (item, properties, geometry) in enumerate(source_items, 1):
        try:
            if geometry is not None:
                coordinates = _normalize_coordinates(exterior_ring(geometry))
            else:
                coordinates = _normalize_coordinates(
                    item.get("display_points") or item.get("points") or []
                )
            pile_id = properties.get("pile_id") or item.get("id") or properties.get("num") or index
            piles.append(PileView(
                pile_id=str(pile_id),
                coordinates=coordinates,
                area_m2=_optional_float(properties.get("area_m2")),
                volume_m3=_optional_float(properties.get("volume_m3") or properties.get("volume")),
                max_height_m=_optional_float(properties.get("max_relative_height_m")),
                mean_height_m=_optional_float(properties.get("mean_relative_height_m")),
                quality_score=_optional_float(properties.get("quality_score")),
                quality_grade=str(properties.get("quality_grade") or ""),
                boundary_confidence=_optional_float(properties.get("boundary_confidence")),
            ))
        except ValueError:
            continue

    if not piles:
        raise ValueError("结果文件中没有可显示的有效煤堆边界")
    if "pile_count" not in summary:
        summary["pile_count"] = len(piles)
    if "total_area_m2" not in summary:
        summary["total_area_m2"] = sum(pile.area_m2 or 0.0 for pile in piles)
    if "total_volume_m3" not in summary:
        summary["total_volume_m3"] = sum(pile.volume_m3 or 0.0 for pile in piles)
    return piles, summary


def find_result_file(config_path: str | Path, raw_config: dict, max_files: int | None = None) -> Path | None:
    output_dir = resolve_config_path(config_path, raw_config.get("project", {}).get("output_dir"))
    if output_dir is None or not output_dir.exists():
        return None
    suffix = "full" if not max_files else f"first_{max_files}"
    candidates = [
        output_dir / f"coal_piles_{suffix}.json",
        output_dir / f"coal_piles_{suffix}.geojson",
        output_dir / f"coal_piles_service_{suffix}.json",
    ]
    return next((path for path in candidates if path.exists()), None)


def build_multipart_payload(file_path: str | Path, field_name: str = "file") -> tuple[bytes, str]:
    path = Path(file_path)
    boundary = "----CoalBoundary" + uuid.uuid4().hex
    mime_type = mimetypes.guess_type(path.name)[0] or "application/json"
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{path.name}"\r\n'
        f"Content-Type: {mime_type}\r\n\r\n"
    ).encode("utf-8")
    footer = f"\r\n--{boundary}--\r\n".encode("ascii")
    return header + path.read_bytes() + footer, boundary


def upload_result_file(file_path: str | Path, upload_url: str, timeout: int = 45) -> dict:
    payload, boundary = build_multipart_payload(file_path)
    request = urllib.request.Request(
        upload_url,
        data=payload,
        method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content = response.read().decode("utf-8", errors="replace")
        if response.status < 200 or response.status >= 300:
            raise RuntimeError(f"上传失败，HTTP {response.status}: {content}")
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {"success": True, "message": content or "上传成功"}


class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, background="#f5f7fa")
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas, style="Page.TFrame")
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel, add="+")

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def _on_mousewheel(self, event):
        if self.winfo_containing(event.x_root, event.y_root) in (self.canvas, self.inner) or self._is_descendant(
            self.winfo_containing(event.x_root, event.y_root)
        ):
            self.canvas.yview_scroll(int(-event.delta / 120), "units")

    def _is_descendant(self, widget):
        while widget is not None:
            if widget == self.inner:
                return True
            widget = widget.master
        return False


class BoundaryPreview(ttk.Frame):
    COLORS = {"A": "#1f9d72", "B": "#d69e2e", "C": "#d6455d"}

    def __init__(self, master):
        super().__init__(master)
        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(controls, text="二维边界总览", style="PanelTitle.TLabel").pack(side="left")
        ttk.Button(controls, text="适应窗口", command=self.fit).pack(side="right")
        self.canvas = tk.Canvas(self, background="#111827", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.canvas.bind("<Configure>", lambda _event: self._schedule_redraw(40))
        self.canvas.bind("<MouseWheel>", self._zoom)
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan_move)
        self.canvas.bind("<ButtonRelease-1>", self._pan_end)
        self.piles: list[PileView] = []
        self.selected_id: str | None = None
        self.base_bounds = (0.0, 0.0, 1.0, 1.0)
        self.center = (0.5, 0.5)
        self.zoom = 1.0
        self.pan_anchor: tuple[int, int] | None = None
        self._redraw_after_id: str | None = None
        self._pile_bounds: dict[str, tuple[float, float, float, float]] = {}
        self._pile_centres: dict[str, tuple[float, float]] = {}
        self._lod_coordinates: dict[str, dict[int, list[tuple[float, float]]]] = {}

    def set_piles(self, piles: list[PileView]):
        self.piles = piles
        self._pile_bounds.clear()
        self._pile_centres.clear()
        self._lod_coordinates.clear()
        for pile in piles:
            xs = [point[0] for point in pile.coordinates]
            ys = [point[1] for point in pile.coordinates]
            self._pile_bounds[pile.pile_id] = (min(xs), min(ys), max(xs), max(ys))
            self._pile_centres[pile.pile_id] = (
                sum(xs) / len(xs), sum(ys) / len(ys),
            )
            levels = {}
            for target in (240, 480, 960, 1600):
                step = max(1, math.ceil(len(pile.coordinates) / target))
                levels[target] = pile.coordinates[::step]
            self._lod_coordinates[pile.pile_id] = levels
        xs = [point[0] for pile in piles for point in pile.coordinates]
        ys = [point[1] for pile in piles for point in pile.coordinates]
        self.base_bounds = (min(xs), min(ys), max(xs), max(ys))
        self.fit()

    def fit(self):
        min_x, min_y, max_x, max_y = self.base_bounds
        self.center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
        self.zoom = 1.0
        self.redraw()

    def select(self, pile_id: str | None):
        self.selected_id = pile_id
        self.redraw()

    def _view_bounds(self):
        return self._view_transform()[:4]

    def _screen(self, lon: float, lat: float):
        min_x, min_y, _, _, scale, canvas_height, padding = self._view_transform()
        x = padding + (lon - min_x) * scale
        y = canvas_height - padding - (lat - min_y) * scale
        return x, y

    def _view_transform(self):
        return boundary_view_transform(
            self.base_bounds,
            self.center,
            self.zoom,
            self.canvas.winfo_width(),
            self.canvas.winfo_height(),
        )

    def _schedule_redraw(self, delay_ms: int = 16):
        if self._redraw_after_id is None:
            self._redraw_after_id = self.after(delay_ms, self._run_scheduled_redraw)

    def _run_scheduled_redraw(self):
        self._redraw_after_id = None
        self.redraw()

    def _cancel_scheduled_redraw(self):
        if self._redraw_after_id is not None:
            self.after_cancel(self._redraw_after_id)
            self._redraw_after_id = None

    def _lod_target(self) -> int:
        if self.zoom < 1.75:
            return 240
        if self.zoom < 3.5:
            return 480
        if self.zoom < 7.0:
            return 960
        return 1600

    @staticmethod
    def _bounds_intersect(first, second) -> bool:
        return not (
            first[2] < second[0] or first[0] > second[2]
            or first[3] < second[1] or first[1] > second[3]
        )

    def redraw(self):
        self._cancel_scheduled_redraw()
        self.canvas.delete("all")
        if not self.piles:
            self.canvas.create_text(
                max(self.canvas.winfo_width(), 300) / 2,
                max(self.canvas.winfo_height(), 200) / 2,
                text="运行提取或打开结果文件后显示边界",
                fill="#94a3b8",
                font=("Microsoft YaHei UI", 13),
            )
            return
        min_x, min_y, max_x, max_y, scale, canvas_height, padding = self._view_transform()
        view_width = max_x - min_x
        view_height = max_y - min_y
        # Keep a full-screen margin so adjacent piles are already present while
        # the user drags. Far-away piles are skipped only beyond that margin.
        visible_bounds = (
            min_x - view_width,
            min_y - view_height,
            max_x + view_width,
            max_y + view_height,
        )
        lod_target = self._lod_target()
        for pile in self.piles:
            if not self._bounds_intersect(self._pile_bounds[pile.pile_id], visible_bounds):
                continue
            display_points = self._lod_coordinates[pile.pile_id][lod_target]
            flattened = []
            for lon, lat in display_points:
                flattened.append(padding + (lon - min_x) * scale)
                flattened.append(canvas_height - padding - (lat - min_y) * scale)
            selected = pile.pile_id == self.selected_id
            color = self.COLORS.get(pile.quality_grade, "#38bdf8")
            self.canvas.create_polygon(
                *flattened,
                fill="#274060" if not selected else "#7c3aed",
                outline="#22d3ee" if not selected else "#f8fafc",
                width=1.4 if not selected else 3.0,
                tags=("world", f"pile-{pile.pile_id}"),
            )
            if selected:
                cx, cy = self._pile_centres[pile.pile_id]
                x = padding + (cx - min_x) * scale
                y = canvas_height - padding - (cy - min_y) * scale
                self.canvas.create_oval(
                    x - 4, y - 4, x + 4, y + 4,
                    fill=color, outline="#ffffff", tags=("world",),
                )
                self.canvas.create_text(
                    x + 8,
                    y - 8,
                    text=f"#{pile.pile_id}",
                    fill="#ffffff",
                    anchor="sw",
                    font=("Microsoft YaHei UI", 10, "bold"),
                    tags=("world",),
                )
        self._draw_legend()

    def _draw_legend(self):
        x, y = 16, 18
        for grade, color in (("A", "#1f9d72"), ("B", "#d69e2e"), ("C", "#d6455d"), ("无等级", "#38bdf8")):
            self.canvas.create_rectangle(x, y, x + 11, y + 11, fill=color, outline="")
            self.canvas.create_text(x + 16, y + 6, text=grade, fill="#e5e7eb", anchor="w", font=("Microsoft YaHei UI", 9))
            x += 62 if grade != "无等级" else 88

    def _zoom(self, event):
        if not self.piles:
            return
        factor = 1.25 if event.delta > 0 else 0.8
        self.zoom = min(20.0, max(1.0, self.zoom * factor))
        self._schedule_redraw(16)

    def _pan_start(self, event):
        if not self.piles or self.zoom <= 1.0:
            return
        self.pan_anchor = (event.x, event.y)
        self.canvas.configure(cursor="fleur")

    def _pan_move(self, event):
        if self.pan_anchor is None or not self.piles or self.zoom <= 1.0:
            return
        previous_x, previous_y = self.pan_anchor
        dx, dy = event.x - previous_x, event.y - previous_y
        self.pan_anchor = (event.x, event.y)
        _, _, _, _, scale, _, _ = self._view_transform()
        cx, cy = self.center
        self.center = (
            cx - dx / max(scale, 1e-12),
            cy + dy / max(scale, 1e-12),
        )
        # Moving existing canvas items is O(number of items), whereas deleting
        # and rebuilding all boundary vertices on every mouse event is O(points).
        self.canvas.move("world", dx, dy)

    def _pan_end(self, _event):
        if self.pan_anchor is None:
            return
        self.pan_anchor = None
        self.canvas.configure(cursor="crosshair")
        self.redraw()


class CoalBoundaryApp(tk.Tk):
    def __init__(self, config_path: str | Path):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1320x850")
        self.minsize(1100, 700)
        self.configure(background="#e9eef5")
        self.config_path_var = tk.StringVar(value=str(Path(config_path).resolve()))
        self.field_vars: dict[tuple[str, str], tk.Variable] = {}
        self.model_enabled_var = tk.BooleanVar(value=True)
        self.integration_enabled_var = tk.BooleanVar(value=False)
        self.max_files_var = tk.StringVar(value="0")
        self.no_cache_var = tk.BooleanVar(value=False)
        self.upload_url_var = tk.StringVar(value=DEFAULT_UPLOAD_URL)
        self.viewer_url_var = tk.StringVar(value=DEFAULT_VIEWER_URL)
        self.status_var = tk.StringVar(value="就绪")
        self.sync_status_var = tk.StringVar(value="未启用")
        self.backend_run_id_var = tk.StringVar(value="—")
        self.backend_username_var = tk.StringVar(value="")
        self.backend_password_var = tk.StringVar(value="")
        self.backend_session_token: str | None = None
        self.backend_session_url = ""
        self.summary_vars = {
            "pile_count": tk.StringVar(value="—"),
            "total_volume": tk.StringVar(value="—"),
            "volume_error": tk.StringVar(value="—"),
            "quality": tk.StringVar(value="—"),
        }
        self.event_queue: queue.Queue = queue.Queue()
        self.process: subprocess.Popen | None = None
        self.current_result_path: Path | None = None
        self.current_piles: list[PileView] = []
        self.raw_config: dict = {}
        self._configure_styles()
        self._build_ui()
        self._load_config(show_errors=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._poll_events)

    def _configure_styles(self):
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Page.TFrame", background="#f5f7fa")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Header.TFrame", background="#17324d")
        style.configure("HeaderTitle.TLabel", background="#17324d", foreground="#ffffff", font=("Microsoft YaHei UI", 19, "bold"))
        style.configure("HeaderSub.TLabel", background="#17324d", foreground="#cbd5e1", font=("Microsoft YaHei UI", 10))
        style.configure("PanelTitle.TLabel", font=("Microsoft YaHei UI", 11, "bold"), foreground="#17324d")
        style.configure("CardValue.TLabel", background="#ffffff", font=("Microsoft YaHei UI", 16, "bold"), foreground="#17324d")
        style.configure("CardLabel.TLabel", background="#ffffff", font=("Microsoft YaHei UI", 9), foreground="#64748b")
        style.configure("Accent.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(12, 7))
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=(8, 5))
        style.configure("TLabel", font=("Microsoft YaHei UI", 9))
        style.configure("TEntry", padding=5)
        style.configure("Treeview", rowheight=26, font=("Microsoft YaHei UI", 9))
        style.configure("Treeview.Heading", font=("Microsoft YaHei UI", 9, "bold"))

    def _build_ui(self):
        header = ttk.Frame(self, style="Header.TFrame", padding=(20, 14))
        header.pack(fill="x")
        title_area = ttk.Frame(header, style="Header.TFrame")
        title_area.pack(side="left", fill="x", expand=True)
        ttk.Label(title_area, text=APP_TITLE, style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(title_area, text="坡地基准面｜低洼沟槽分堆｜JSON / GeoJSON 输出与预览", style="HeaderSub.TLabel").pack(anchor="w", pady=(3, 0))
        ttk.Button(header, text="打开点云页面", command=self._open_viewer).pack(side="right", padx=(8, 0))
        ttk.Button(header, text="打开输出目录", command=self._open_output_dir).pack(side="right")

        config_bar = ttk.Frame(self, padding=(14, 8))
        config_bar.pack(fill="x")
        ttk.Label(config_bar, text="配置文件：").pack(side="left")
        ttk.Entry(config_bar, textvariable=self.config_path_var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(config_bar, text="浏览", command=self._browse_config).pack(side="left", padx=3)
        ttk.Button(config_bar, text="重新加载", command=lambda: self._load_config(show_errors=True)).pack(side="left", padx=3)
        ttk.Button(config_bar, text="保存配置", command=self._save_config).pack(side="left", padx=3)

        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.run_tab = ttk.Frame(self.notebook, style="Page.TFrame")
        self.result_tab = ttk.Frame(self.notebook, style="Page.TFrame")
        self.log_tab = ttk.Frame(self.notebook, style="Page.TFrame")
        self.notebook.add(self.run_tab, text=" 运行与配置 ")
        self.notebook.add(self.result_tab, text=" 结果预览 ")
        self.notebook.add(self.log_tab, text=" 运行日志 ")
        self._build_run_tab()
        self._build_result_tab()
        self._build_log_tab()

        status = ttk.Frame(self, padding=(14, 5))
        status.pack(fill="x")
        ttk.Label(status, textvariable=self.status_var).pack(side="left")
        self.progress = ttk.Progressbar(status, mode="determinate", maximum=100, value=0)
        self.progress.pack(side="right", fill="x", expand=True, padx=(16, 0))

    def _build_run_tab(self):
        action_bar = ttk.Frame(self.run_tab, padding=(10, 10))
        action_bar.pack(fill="x")
        self.inspect_button = ttk.Button(action_bar, text="① 检查数据", style="Accent.TButton", command=lambda: self._start_command("inspect"))
        self.inspect_button.pack(side="left", padx=(0, 6))
        self.run_button = ttk.Button(action_bar, text="② 开始提取边界", style="Accent.TButton", command=lambda: self._start_command("run"))
        self.run_button.pack(side="left", padx=6)
        self.cancel_button = ttk.Button(action_bar, text="取消任务", command=self._cancel_process, state="disabled")
        self.cancel_button.pack(side="left", padx=6)
        ttk.Separator(action_bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Label(action_bar, text="试运行 LAS 数（0=全部）：").pack(side="left")
        ttk.Spinbox(action_bar, from_=0, to=9999, width=7, textvariable=self.max_files_var).pack(side="left", padx=(2, 10))
        ttk.Checkbutton(action_bar, text="忽略全部缓存并重新计算", variable=self.no_cache_var).pack(side="left")

        scroller = ScrollableFrame(self.run_tab)
        scroller.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        groups: dict[str, ttk.LabelFrame] = {}
        for group_name in dict.fromkeys(spec.group for spec in FIELD_SPECS):
            frame = ttk.LabelFrame(scroller.inner, text=group_name, padding=10)
            frame.pack(fill="x", padx=8, pady=7)
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=1)
            groups[group_name] = frame

        group_counts: dict[str, int] = {name: 0 for name in groups}
        for spec in FIELD_SPECS:
            index = group_counts[spec.group]
            group_counts[spec.group] += 1
            row, column = divmod(index, 2)
            card = ttk.Frame(groups[spec.group], padding=(4, 3))
            card.grid(row=row, column=column, sticky="ew", padx=6, pady=3)
            card.columnconfigure(0, weight=1)
            ttk.Label(card, text=spec.label).grid(row=0, column=0, sticky="w", columnspan=2)
            variable = tk.StringVar()
            self.field_vars[(spec.section, spec.key)] = variable
            ttk.Entry(card, textvariable=variable).grid(row=1, column=0, sticky="ew", pady=(3, 0))
            if spec.browse:
                ttk.Button(card, text="…", width=3, command=lambda current=spec: self._browse_field(current)).grid(row=1, column=1, padx=(4, 0), pady=(3, 0))

        model_frame = groups["模型与评估"]
        model_row = math.ceil(group_counts["模型与评估"] / 2)
        ttk.Checkbutton(model_frame, text="启用模型推理", variable=self.model_enabled_var).grid(row=model_row, column=0, sticky="w", padx=10, pady=6)

        integration_frame = groups["后端同步"]
        integration_row = math.ceil(group_counts["后端同步"] / 2)
        ttk.Checkbutton(integration_frame, text="启用任务、进度和结果同步", variable=self.integration_enabled_var).grid(
            row=integration_row, column=0, sticky="w", padx=10, pady=6,
        )
        integration_actions = ttk.Frame(integration_frame)
        integration_actions.grid(row=integration_row, column=1, sticky="ew", padx=10, pady=6)
        ttk.Button(integration_actions, text="测试连接", command=self._test_backend_connection).pack(side="left")
        ttk.Button(integration_actions, text="补传待同步", command=self._start_pending_sync).pack(side="left", padx=(6, 0))
        ttk.Label(integration_actions, text="状态：").pack(side="left", padx=(10, 2))
        ttk.Label(integration_actions, textvariable=self.sync_status_var).pack(side="left")
        ttk.Label(integration_actions, text="任务 ID：").pack(side="left", padx=(16, 2))
        ttk.Label(integration_actions, textvariable=self.backend_run_id_var).pack(side="left")

        credentials = ttk.Frame(integration_frame)
        credentials.grid(row=integration_row + 1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 7))
        credentials.columnconfigure(1, weight=1)
        credentials.columnconfigure(3, weight=1)
        ttk.Label(credentials, text="登录账号").grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(credentials, textvariable=self.backend_username_var).grid(row=0, column=1, sticky="ew", padx=(0, 12))
        ttk.Label(credentials, text="登录密码").grid(row=0, column=2, sticky="w", padx=(0, 6))
        ttk.Entry(credentials, textvariable=self.backend_password_var, show="●").grid(row=0, column=3, sticky="ew")
        ttk.Button(
            credentials,
            text="清空登录",
            command=lambda: (self.backend_username_var.set(""), self.backend_password_var.set("")),
        ).grid(row=0, column=4, padx=(8, 0))
        ttk.Label(
            credentials,
            text="账号和密码仅保存在本次 GUI 内存中，不写入 config.yaml、日志或补传文件。",
            foreground="#64748b",
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(5, 0))

        service_frame = ttk.LabelFrame(scroller.inner, text="点云服务", padding=10)
        service_frame.pack(fill="x", padx=8, pady=7)
        service_frame.columnconfigure(1, weight=1)
        ttk.Label(service_frame, text="上传接口").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(service_frame, textvariable=self.upload_url_var).grid(row=0, column=1, sticky="ew")
        ttk.Button(service_frame, text="上传当前结果", command=self._upload_current_result).grid(row=0, column=2, padx=(8, 0))
        ttk.Label(service_frame, text="点云页面").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(6, 0))
        ttk.Entry(service_frame, textvariable=self.viewer_url_var).grid(row=1, column=1, sticky="ew", pady=(6, 0))

    def _build_result_tab(self):
        cards = ttk.Frame(self.result_tab, style="Page.TFrame", padding=(10, 10))
        cards.pack(fill="x")
        card_defs = [
            ("pile_count", "煤堆数量"),
            ("total_volume", "总体积（m³）"),
            ("volume_error", "总体积误差"),
            ("quality", "质量等级"),
        ]
        for key, label in card_defs:
            card = ttk.Frame(cards, style="Card.TFrame", padding=(16, 10))
            card.pack(side="left", fill="x", expand=True, padx=5)
            ttk.Label(card, text=label, style="CardLabel.TLabel").pack(anchor="w")
            ttk.Label(card, textvariable=self.summary_vars[key], style="CardValue.TLabel").pack(anchor="w", pady=(4, 0))

        toolbar = ttk.Frame(self.result_tab, padding=(12, 4))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="打开结果文件", command=self._browse_result).pack(side="left")
        ttk.Button(toolbar, text="上传当前结果", command=self._upload_current_result).pack(side="left", padx=6)
        self.result_path_label = ttk.Label(toolbar, text="尚未加载结果", foreground="#64748b")
        self.result_path_label.pack(side="left", fill="x", expand=True, padx=8)

        pane = ttk.Panedwindow(self.result_tab, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        table_frame = ttk.Frame(pane)
        preview_frame = ttk.Frame(pane)
        pane.add(table_frame, weight=2)
        pane.add(preview_frame, weight=3)

        columns = ("id", "area", "volume", "max_h", "mean_h", "grade", "confidence")
        self.result_tree = ttk.Treeview(table_frame, columns=columns, show="headings", selectmode="browse")
        headings = {
            "id": "编号",
            "area": "面积(m²)",
            "volume": "体积(m³)",
            "max_h": "最高(m)",
            "mean_h": "平均(m)",
            "grade": "等级",
            "confidence": "置信度",
        }
        widths = {"id": 58, "area": 92, "volume": 105, "max_h": 78, "mean_h": 78, "grade": 58, "confidence": 75}
        for column in columns:
            self.result_tree.heading(column, text=headings[column])
            self.result_tree.column(column, width=widths[column], anchor="center", stretch=column in ("area", "volume"))
        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)
        self.result_tree.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.result_tree.bind("<<TreeviewSelect>>", self._on_result_select)

        self.preview = BoundaryPreview(preview_frame)
        self.preview.pack(fill="both", expand=True)

    def _build_log_tab(self):
        toolbar = ttk.Frame(self.log_tab, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Label(toolbar, text="实时运行日志", style="PanelTitle.TLabel").pack(side="left")
        ttk.Button(toolbar, text="清空", command=lambda: self.log_text.delete("1.0", "end")).pack(side="right")
        text_frame = ttk.Frame(self.log_tab)
        text_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.log_text = tk.Text(
            text_frame,
            wrap="word",
            background="#0f172a",
            foreground="#dbeafe",
            insertbackground="#ffffff",
            font=("Consolas", 10),
            relief="flat",
            padx=10,
            pady=8,
        )
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        self.log_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _browse_config(self):
        selected = filedialog.askopenfilename(title="选择配置文件", filetypes=[("YAML", "*.yaml *.yml"), ("全部文件", "*.*")])
        if selected:
            self.config_path_var.set(selected)
            self._load_config(show_errors=True)

    def _browse_field(self, spec: FieldSpec):
        if spec.browse == "directory":
            selected = filedialog.askdirectory(title=f"选择{spec.label}")
        else:
            filetypes = {
                "xlsx": [("Excel", "*.xlsx *.xls"), ("全部文件", "*.*")],
                "geojson": [("GeoJSON", "*.geojson *.json"), ("全部文件", "*.*")],
                "model": [("模型文件", "*.hdf5 *.h5 *.keras"), ("全部文件", "*.*")],
            }.get(spec.browse, [("全部文件", "*.*")])
            selected = filedialog.askopenfilename(title=f"选择{spec.label}", filetypes=filetypes)
        if selected:
            self.field_vars[(spec.section, spec.key)].set(selected)

    def _load_config(self, *, show_errors=False):
        try:
            self.raw_config = read_raw_config(self.config_path_var.get())
            values = field_values_from_config(self.raw_config)
            for key, variable in self.field_vars.items():
                value = values.get(key, "")
                variable.set("" if value is None else str(value))
            self.model_enabled_var.set(bool(values.get(("model", "enabled"), True)))
            self.integration_enabled_var.set(bool(values.get(("integration", "enabled"), False)))
            integration_raw = self.raw_config.get("integration") or {}
            username_env = str(integration_raw.get("username_env", "COAL_BOUNDARY_API_USERNAME"))
            password_env = str(integration_raw.get("password_env", "COAL_BOUNDARY_API_PASSWORD"))
            if not self.backend_username_var.get() and username_env:
                self.backend_username_var.set(os.environ.get(username_env, ""))
            if not self.backend_password_var.get() and password_env:
                self.backend_password_var.set(os.environ.get(password_env, ""))
            self.sync_status_var.set("已启用，尚未连接" if self.integration_enabled_var.get() else "未启用")
            self.status_var.set("配置已加载")
            self._append_log(f"[GUI] 已加载配置：{self.config_path_var.get()}\n")
            result_path = find_result_file(self.config_path_var.get(), self.raw_config)
            if result_path:
                self._load_result(result_path, show_errors=False)
            if self.integration_enabled_var.get():
                token, username, password = self._integration_auth_values(self.raw_config)
                if token or (username and password):
                    self._start_pending_sync(silent=True)
                else:
                    self.sync_status_var.set("等待登录")
                    self._append_log("[GUI] 后端同步已启用；请输入登录账号和密码后测试连接或补传。\n")
        except Exception as exc:
            self.status_var.set("配置加载失败")
            if show_errors:
                messagebox.showerror("配置错误", str(exc))

    def _collect_values(self):
        values = {key: variable.get() for key, variable in self.field_vars.items()}
        values[("model", "enabled")] = self.model_enabled_var.get()
        values[("integration", "enabled")] = self.integration_enabled_var.get()
        return values

    def _save_config(self, *, show_message=True) -> bool:
        try:
            raw = read_raw_config(self.config_path_var.get())
            updated = update_raw_config(raw, self._collect_values())
            atomic_save_config(self.config_path_var.get(), updated, create_backup=True)
            self.raw_config = updated
            self.status_var.set("配置已保存")
            self._append_log(f"[GUI] 配置已保存；备份：{self.config_path_var.get()}.gui-backup\n")
            if show_message:
                messagebox.showinfo("保存成功", "配置已保存，并保留 .gui-backup 备份。")
            return True
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return False

    def _max_files(self) -> int | None:
        value = int(float(self.max_files_var.get() or "0"))
        return value if value > 0 else None

    def _start_command(self, command: str):
        if self.process is not None:
            messagebox.showwarning("任务正在运行", "请等待当前任务结束，或先点击取消任务。")
            return
        if not self._save_config(show_message=False):
            return
        try:
            max_files = self._max_files()
        except ValueError:
            messagebox.showerror("参数错误", "试运行 LAS 数必须是整数。")
            return
        config_path = Path(self.config_path_var.get()).resolve()
        cli_python = resolve_cli_python()
        process_environment = self._backend_subprocess_environment()
        command_line = [
            str(cli_python),
            "-m",
            "coal_boundary.cli",
            "--config",
            str(config_path),
            command,
        ]
        if max_files:
            command_line.extend(["--max-files", str(max_files)])
        if command == "run" and self.no_cache_var.get():
            command_line.append("--no-cache")
        self._set_running(True)
        self.progress.configure(mode="determinate")
        self.progress["value"] = 2
        self.status_var.set("正在检查数据…" if command == "inspect" else "正在提取边界…")
        self.notebook.select(self.log_tab)
        self._append_log("\n" + "=" * 76 + "\n")
        self._append_log("[GUI] 启动命令：" + subprocess.list2cmdline(command_line) + "\n")
        project_src = config_path.parent / "src"
        cwd = project_src if project_src.exists() else config_path.parent
        thread = threading.Thread(
            target=self._run_subprocess,
            args=(command_line, cwd, command, max_files, process_environment),
            daemon=True,
        )
        thread.start()

    def _run_subprocess(self, command_line, cwd: Path, command: str, max_files: int | None, process_environment: dict[str, str]):
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                command_line,
                cwd=str(cwd),
                env=process_environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
            )
            self.process = process
            assert process.stdout is not None
            for line in process.stdout:
                self.event_queue.put(("log", line))
            return_code = process.wait()
            self.event_queue.put(("done", return_code, command, max_files))
        except Exception as exc:
            self.event_queue.put(("process_error", str(exc)))

    def _cancel_process(self):
        process = self.process
        if process is None:
            return
        if not messagebox.askyesno("取消任务", "确定终止当前任务吗？已经生成的表面缓存会保留。"):
            return
        try:
            process.terminate()
            self._append_log("[GUI] 已发送终止请求。\n")
            self.status_var.set("正在取消任务…")
        except Exception as exc:
            messagebox.showerror("取消失败", str(exc))

    def _set_running(self, running: bool):
        self.inspect_button.configure(state="disabled" if running else "normal")
        self.run_button.configure(state="disabled" if running else "normal")
        self.cancel_button.configure(state="normal" if running else "disabled")

    def _poll_events(self):
        try:
            while True:
                event = self.event_queue.get_nowait()
                kind = event[0]
                if kind == "log":
                    self._handle_log_line(event[1])
                elif kind == "done":
                    self._handle_process_done(event[1], event[2], event[3])
                elif kind == "process_error":
                    self.process = None
                    self._set_running(False)
                    self.status_var.set("任务启动失败")
                    messagebox.showerror("运行失败", event[1])
                elif kind == "upload_done":
                    self.status_var.set("边界上传成功")
                    messagebox.showinfo("上传成功", event[1])
                elif kind == "upload_error":
                    self.status_var.set("边界上传失败")
                    messagebox.showerror("上传失败", event[1])
                elif kind == "backend_test_done":
                    self.backend_session_url = event[1]
                    self.backend_session_token = event[2]
                    self.sync_status_var.set("连接成功")
                    self._append_log("[GUI] 后端 API 自动登录并连接成功。\n")
                elif kind == "backend_test_error":
                    self.sync_status_var.set("连接失败")
                    self._append_log(f"[GUI] 后端 API 连接失败：{event[1]}\n")
                elif kind == "backend_sync_done":
                    remaining, error = event[1], event[2]
                    if len(event) > 3 and event[3]:
                        self.backend_session_token = event[3]
                        self.backend_session_url = self.field_vars[("integration", "base_url")].get().strip()
                    if error:
                        self.sync_status_var.set(f"待补传 {remaining}")
                        self._append_log(f"[GUI] 后端补传未完成：{error}\n")
                    else:
                        self.sync_status_var.set("同步完成" if remaining == 0 else f"待补传 {remaining}")
                        self._append_log(f"[GUI] 后端补传结束，剩余任务 {remaining} 个。\n")
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _handle_log_line(self, line: str):
        self._append_log(line)
        run_match = re.search(r"\[integration\]\s*后端任务 ID：\s*(\S+)", line)
        if run_match:
            self.backend_run_id_var.set(run_match.group(1))
            self.sync_status_var.set("同步中")
        if "后端同步暂不可用" in line:
            self.sync_status_var.set("待补传")
        stage_match = re.search(r"\[stage\]\s*(.+)", line)
        if stage_match:
            stage = stage_match.group(1).strip()
            if stage in STAGE_PROGRESS:
                self.progress["value"] = STAGE_PROGRESS[stage]
                self.status_var.set(stage + "…")
        surface_match = re.search(r"\[surface\]\s*(\d+)/(\d+)", line)
        if surface_match:
            current, total = map(int, surface_match.groups())
            self.progress["value"] = 15 + 17 * current / max(total, 1)
            self.status_var.set(f"建立表面栅格 {current}/{total}…")

    def _handle_process_done(self, return_code: int, command: str, max_files: int | None):
        self.process = None
        self._set_running(False)
        if return_code == 0:
            self.progress["value"] = 100
            self.status_var.set("数据检查完成" if command == "inspect" else "边界提取完成")
            self._append_log(f"[GUI] 任务完成，退出码 {return_code}。\n")
            if command == "run":
                try:
                    self.raw_config = read_raw_config(self.config_path_var.get())
                    result_path = find_result_file(self.config_path_var.get(), self.raw_config, max_files)
                    if result_path is None:
                        raise FileNotFoundError("任务成功，但未找到结果 JSON")
                    self._load_result(result_path, show_errors=True)
                    self.notebook.select(self.result_tab)
                except Exception as exc:
                    messagebox.showwarning("结果加载失败", str(exc))
            else:
                messagebox.showinfo("检查完成", "数据、配置和 CRS 检查已完成，请查看运行日志。")
        else:
            self.status_var.set(f"任务失败（退出码 {return_code}）")
            self._append_log(f"[GUI] 任务失败，退出码 {return_code}。\n")
            messagebox.showerror("任务失败", "请查看“运行日志”中的错误信息。")

    def _append_log(self, text: str):
        self.log_text.insert("end", text)
        self.log_text.see("end")

    def _browse_result(self):
        selected = filedialog.askopenfilename(
            title="选择边界结果",
            filetypes=[("边界结果", "*.json *.geojson"), ("全部文件", "*.*")],
        )
        if selected:
            self._load_result(Path(selected), show_errors=True)

    def _load_result(self, path: Path, *, show_errors=True):
        try:
            piles, summary = load_result_document(path)
            self.current_result_path = path.resolve()
            self.current_piles = piles
            display_count = int(summary.get("display_geometry_count") or 0)
            display_note = (
                f"  ｜  原始点云平滑边界 {display_count}/{len(piles)}"
                if display_count else "  ｜  计算边界"
            )
            self.result_path_label.configure(text=f"{self.current_result_path}{display_note}")
            for item in self.result_tree.get_children():
                self.result_tree.delete(item)
            for pile in piles:
                values = (
                    pile.pile_id,
                    self._format_number(pile.area_m2, 2),
                    self._format_number(pile.volume_m3, 2),
                    self._format_number(pile.max_height_m, 2),
                    self._format_number(pile.mean_height_m, 2),
                    pile.quality_grade or "—",
                    self._format_number(pile.boundary_confidence, 3),
                )
                self.result_tree.insert("", "end", iid=pile.pile_id, values=values)
            self.preview.set_piles(piles)
            self._update_summary(summary, piles)
            self.status_var.set(
                f"已加载 {len(piles)} 座煤堆；"
                + ("正在显示原始点云平滑边界" if display_count else "正在显示计算边界")
            )
        except Exception as exc:
            if show_errors:
                messagebox.showerror("结果文件错误", str(exc))

    @staticmethod
    def _format_number(value: float | None, digits: int) -> str:
        return "—" if value is None else f"{value:,.{digits}f}"

    def _update_summary(self, summary: dict, piles: list[PileView]):
        self.summary_vars["pile_count"].set(str(summary.get("pile_count", len(piles))))
        total_volume = _optional_float(summary.get("total_volume_m3"))
        self.summary_vars["total_volume"].set(self._format_number(total_volume, 2))
        evaluation = summary.get("evaluation") or {}
        error = _optional_float(evaluation.get("total_volume_error_percent")) if isinstance(evaluation, dict) else None
        self.summary_vars["volume_error"].set("—" if error is None else f"{error:.4f}%")
        grades = {grade: sum(pile.quality_grade == grade for pile in piles) for grade in ("A", "B", "C")}
        self.summary_vars["quality"].set(f"A {grades['A']}  B {grades['B']}  C {grades['C']}")

    def _on_result_select(self, _event):
        selection = self.result_tree.selection()
        self.preview.select(selection[0] if selection else None)

    def _open_output_dir(self):
        try:
            raw = self.raw_config or read_raw_config(self.config_path_var.get())
            output_dir = resolve_config_path(self.config_path_var.get(), raw.get("project", {}).get("output_dir"))
            if output_dir is None:
                raise ValueError("未配置输出目录")
            output_dir.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(output_dir)
            else:
                subprocess.Popen(["xdg-open", str(output_dir)])
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def _open_viewer(self):
        webbrowser.open(self.viewer_url_var.get().strip() or DEFAULT_VIEWER_URL)

    def _integration_auth_values(self, raw: dict | None = None) -> tuple[str | None, str | None, str | None]:
        raw = raw or self.raw_config or read_raw_config(self.config_path_var.get())
        integration_raw = raw.get("integration") or {}
        token_env = str(integration_raw.get("token_env", "COAL_BOUNDARY_API_TOKEN"))
        username_env = str(integration_raw.get("username_env", "COAL_BOUNDARY_API_USERNAME"))
        password_env = str(integration_raw.get("password_env", "COAL_BOUNDARY_API_PASSWORD"))
        base_url = self.field_vars[("integration", "base_url")].get().strip()
        session_token = self.backend_session_token if self.backend_session_url == base_url else None
        token = session_token or (os.environ.get(token_env) if token_env else None)
        username = self.backend_username_var.get().strip() or (os.environ.get(username_env) if username_env else None)
        password = self.backend_password_var.get() or (os.environ.get(password_env) if password_env else None)
        return token or None, username or None, password or None

    def _backend_subprocess_environment(self) -> dict[str, str]:
        raw = self.raw_config or read_raw_config(self.config_path_var.get())
        integration_raw = raw.get("integration") or {}
        token, username, password = self._integration_auth_values(raw)
        environment = dict(os.environ)
        if token:
            environment[str(integration_raw.get("token_env", "COAL_BOUNDARY_API_TOKEN"))] = token
        if username:
            environment[str(integration_raw.get("username_env", "COAL_BOUNDARY_API_USERNAME"))] = username
        if password:
            environment[str(integration_raw.get("password_env", "COAL_BOUNDARY_API_PASSWORD"))] = password
        return utf8_subprocess_environment(environment)

    def _upload_current_result(self):
        result_path = self.current_result_path
        if result_path is None or not result_path.exists():
            messagebox.showwarning("没有结果", "请先运行提取或打开一个 JSON/GeoJSON 结果文件。")
            return
        upload_url = self.upload_url_var.get().strip() or DEFAULT_UPLOAD_URL
        self.status_var.set("正在上传边界…")
        self._append_log(f"[GUI] 上传 {result_path} -> {upload_url}\n")
        threading.Thread(target=self._upload_worker, args=(result_path, upload_url), daemon=True).start()

    def _test_backend_connection(self):
        try:
            raw = read_raw_config(self.config_path_var.get())
            base_url = str(self.field_vars[("integration", "base_url")].get()).strip()
            if not base_url:
                raise ValueError("后端 API 地址不能为空")
            integration_raw = raw.get("integration") or {}
            token, username, password = self._integration_auth_values(raw)
            if not token and not (username and password):
                raise ValueError("请输入后端登录账号和密码；或者由后端配置长期服务令牌。")
            if bool(username) != bool(password):
                raise ValueError("登录账号和密码必须同时填写。")
            verify_tls = bool(integration_raw.get("verify_tls", True))
            login_path = str(integration_raw.get("login_path", "/login"))
            self.sync_status_var.set("连接中…")
            threading.Thread(
                target=self._backend_test_worker,
                args=(base_url, token, username, password, verify_tls, login_path), daemon=True,
            ).start()
        except Exception as exc:
            messagebox.showerror("连接参数错误", str(exc))

    def _backend_test_worker(
        self,
        base_url: str,
        token: str | None,
        username: str | None,
        password: str | None,
        verify_tls: bool,
        login_path: str,
    ):
        try:
            client = BackendApiClient(
                base_url,
                token,
                timeout=5.0,
                verify_tls=verify_tls,
                login_path=login_path,
                username=username,
                password=password,
            )
            client.request_json("GET", "/api/v1/coal-boundary/runs?page=1&size=1")
            self.event_queue.put(("backend_test_done", base_url, client.token))
        except Exception as exc:
            self.event_queue.put(("backend_test_error", str(exc)))

    def _start_pending_sync(self, *, silent=False):
        if not self.integration_enabled_var.get():
            if not silent:
                messagebox.showinfo("后端同步", "请先勾选“启用任务、进度和结果同步”并保存配置。")
            return
        raw = self.raw_config or read_raw_config(self.config_path_var.get())
        token, username, password = self._integration_auth_values(raw)
        if not token and not (username and password):
            self.sync_status_var.set("等待登录")
            if not silent:
                messagebox.showinfo("后端登录", "请输入登录账号和密码后再补传，或配置长期服务令牌。")
            return
        if bool(username) != bool(password):
            if not silent:
                messagebox.showerror("后端登录", "登录账号和密码必须同时填写。")
            return
        self.sync_status_var.set("正在补传…")
        threading.Thread(
            target=self._pending_sync_worker,
            args=(token, username, password),
            daemon=True,
        ).start()

    def _pending_sync_worker(self, token: str | None, username: str | None, password: str | None):
        try:
            config = load_config(self.config_path_var.get())
            reporter = build_reporter(
                config.integration,
                token=token,
                username=username,
                password=password,
            )
            reporter.close()
            spool_dir = Path(config.integration.spool_dir)
            remaining = sum(1 for path in spool_dir.iterdir() if path.is_dir()) if spool_dir.exists() else 0
            error = getattr(getattr(reporter, "status", None), "last_error", None)
            session_token = getattr(getattr(reporter, "client", None), "token", None)
            self.event_queue.put(("backend_sync_done", remaining, error, session_token))
        except Exception as exc:
            self.event_queue.put(("backend_sync_done", -1, str(exc)))

    def _upload_worker(self, result_path: Path, upload_url: str):
        try:
            response = upload_result_file(result_path, upload_url)
            message = response.get("message") or "上传成功"
            self.event_queue.put(("upload_done", message))
        except Exception as exc:
            self.event_queue.put(("upload_error", str(exc)))

    def _on_close(self):
        if self.process is not None:
            if not messagebox.askyesno("退出", "任务仍在运行，确定终止任务并退出吗？"):
                return
            try:
                self.process.terminate()
            except Exception:
                pass
        self.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=APP_TITLE + " GUI")
    default_config = Path(__file__).resolve().parents[2] / "config.yaml"
    parser.add_argument("--config", default=str(default_config), help="YAML 配置文件")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    app = CoalBoundaryApp(args.config)
    app.mainloop()


if __name__ == "__main__":
    main()
