from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from shapely.geometry import Polygon

from coal_boundary.config import AppConfig, ModelConfig, ProjectConfig
from coal_boundary.geometry import PileResult
from coal_boundary.pointcloud_boundary import RefinedBoundaryDiagnostics
from coal_boundary.raster import SurfaceGrid
import coal_boundary.pipeline as pipeline


def test_two_isolated_runs_reuse_shared_surface_and_display_cache(tmp_path, monkeypatch):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    fake_las = las_dir / "one.las"
    fake_las.write_bytes(b"LASF")
    cache_root = tmp_path / "shared-cache"
    counters = {"surface": 0, "display": 0}

    header = SimpleNamespace(path=fake_las)
    monkeypatch.setattr(pipeline, "discover_las_files", lambda _path: [fake_las])
    monkeypatch.setattr(pipeline, "read_las_header", lambda _path: header)
    monkeypatch.setattr(pipeline, "dataset_fingerprint", lambda _headers: "a" * 64)
    monkeypatch.setattr(pipeline, "summarize_headers", lambda _headers: {
        "file_count": 1,
        "point_count": 16,
        "bounds": {"min": [500_000.0, 4_000_000.0, 0.0],
                   "max": [500_004.0, 4_000_004.0, 5.0]},
    })
    monkeypatch.setattr(pipeline, "validate_crs_config_and_bounds", lambda *_args: {
        "validated": True,
    })

    def build_surface(*_args, **_kwargs):
        counters["surface"] += 1
        values = np.ones((4, 4), dtype=np.float32)
        return SurfaceGrid(
            x_min=500_000.0,
            y_min=4_000_000.0,
            cell_size=1.0,
            rows=4,
            cols=4,
            point_count=np.ones((4, 4), dtype=np.uint32),
            mean_z=values,
            min_z=np.zeros((4, 4), dtype=np.float32),
            max_z=values,
            mean_rgb=np.zeros((4, 4, 3), dtype=np.float32),
        )

    monkeypatch.setattr(pipeline, "build_surface_grid", build_surface)
    monkeypatch.setattr(pipeline, "estimate_ground_surface", lambda *_args: np.zeros((4, 4), dtype=np.float32))
    monkeypatch.setattr(pipeline, "candidate_mask", lambda *_args: np.ones((4, 4), dtype=bool))
    monkeypatch.setattr(pipeline, "split_by_height_valleys", lambda *_args: np.ones((4, 4), dtype=np.int32))
    monkeypatch.setattr(pipeline, "recover_concave_slopes", lambda labels, *_args: labels)
    monkeypatch.setattr(pipeline, "merge_nearby_toe_fragments", lambda labels, *_args: labels)

    source = Polygon([
        (500_000.0, 4_000_000.0),
        (500_004.0, 4_000_000.0),
        (500_004.0, 4_000_004.0),
        (500_000.0, 4_000_004.0),
    ])

    def build_results(*_args, **_kwargs):
        return [PileResult(
            pile_id=1,
            polygon=source,
            volume_m3=16.0,
            area_m2=16.0,
            max_relative_height_m=1.0,
            mean_relative_height_m=1.0,
            quality_score=0.9,
            quality_grade="A",
            boundary_confidence=0.8,
            pixel_count=16,
        )]

    monkeypatch.setattr(pipeline, "build_pile_results", build_results)

    def refine(sources, *_args, **_kwargs):
        counters["display"] += 1
        display = source.buffer(0.1, join_style="round")
        metrics = RefinedBoundaryDiagnostics(
            selected_las_files=1,
            selected_points=16,
            fine_cell_size_m=0.2,
            height_threshold_m=0.35,
            source_area_m2=source.area,
            refined_area_m2=display.area,
            area_delta_percent=(display.area - source.area) / source.area * 100.0,
            source_vertices=len(source.exterior.coords),
            evidence_vertices=len(display.exterior.coords),
            display_vertices=len(display.exterior.coords),
            evidence_to_display_hausdorff_m=0.0,
            source_to_display_hausdorff_m=0.1,
        )
        return {1: (display, None, metrics)}

    monkeypatch.setattr(pipeline, "refine_polygons_from_point_cloud", refine)

    first_config = AppConfig(
        project=ProjectConfig("same-project", las_dir, tmp_path / "job-a" / "run-a"),
        model=ModelConfig(enabled=False),
    )
    first = pipeline.run_pipeline(first_config, cache_root=cache_root)
    second_config = replace(
        first_config,
        project=replace(first_config.project, output_dir=tmp_path / "job-b" / "run-b"),
    )
    second = pipeline.run_pipeline(second_config, cache_root=cache_root)

    assert counters == {"surface": 1, "display": 1}
    assert first["metadata"]["cache"] == {
        "enabled": True,
        "scope": "shared",
        "surface_hit": False,
        "display_geometry_hit": False,
    }
    assert second["metadata"]["cache"] == {
        "enabled": True,
        "scope": "shared",
        "surface_hit": True,
        "display_geometry_hit": True,
    }
    assert second["metadata"]["display_geometry"]["cache_source"] == "shared-display-cache"
    assert (second_config.project.output_dir / "surface_full.npz").is_file()

    third_config = replace(
        first_config,
        project=replace(first_config.project, output_dir=tmp_path / "job-c" / "run-c"),
    )
    third = pipeline.run_pipeline(third_config, use_cache=False, cache_root=cache_root)
    assert counters == {"surface": 2, "display": 2}
    assert third["metadata"]["cache"]["surface_hit"] is False
    assert third["metadata"]["cache"]["display_geometry_hit"] is False
