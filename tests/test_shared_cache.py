from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from coal_boundary.cache_lock import CacheLockTimeout, cache_file_lock
from coal_boundary.config import AppConfig, ProjectConfig, SegmentationConfig, load_config
from coal_boundary.pipeline import _config_hash, _shared_cache_directory


def _app_config(output_dir: Path) -> AppConfig:
    return AppConfig(
        project=ProjectConfig(
            name="test-project",
            las_dir=output_dir.parent / "las",
            output_dir=output_dir,
        ),
    )


def test_algorithm_hash_ignores_per_run_identity_and_output_path(tmp_path):
    first = _app_config(tmp_path / "job-a" / "run-a")
    second = replace(
        first,
        project=replace(
            first.project,
            name="renamed-project",
            las_dir=tmp_path / "same-content-mounted-elsewhere",
            output_dir=tmp_path / "job-b" / "run-b",
        ),
        config_path=tmp_path / "different-config-file.yaml",
    )
    assert _config_hash(first) == _config_hash(second)

    changed = replace(
        second,
        segmentation=replace(
            SegmentationConfig(), minimum_height_m=0.9,
        ),
    )
    assert _config_hash(first) != _config_hash(changed)


def test_shared_cache_directory_is_namespaced_by_dataset_and_algorithm(tmp_path):
    dataset = "a" * 64
    algorithm = "b" * 64
    directory = _shared_cache_directory(tmp_path, dataset, algorithm)
    assert directory.parent == tmp_path.resolve()
    assert len(directory.name) == 32
    assert directory != _shared_cache_directory(tmp_path, dataset, "c" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        _shared_cache_directory(tmp_path, "../escape", algorithm)


def test_cache_lock_blocks_a_second_generator_and_releases(tmp_path):
    target = tmp_path / "surface_full.npz"
    with cache_file_lock(target, timeout_seconds=1.0, poll_seconds=0.01):
        with pytest.raises(CacheLockTimeout):
            with cache_file_lock(target, timeout_seconds=0.03, poll_seconds=0.01):
                pass
    with cache_file_lock(target, timeout_seconds=0.1, poll_seconds=0.01):
        assert target.with_name(target.name + ".lock").exists()
    assert not target.with_name(target.name + ".lock").exists()


def test_worker_cache_root_is_resolved_from_config(tmp_path):
    las_dir = tmp_path / "las"
    las_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  name: test
  las_dir: las
  output_dir: output
model:
  enabled: false
integration:
  enabled: false
worker:
  enabled: false
  cache_root: persistent-cache
""".strip(),
        encoding="utf-8",
    )
    config = load_config(config_path)
    assert config.worker.cache_root == (tmp_path / "persistent-cache").resolve()
