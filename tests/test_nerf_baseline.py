"""Reject stale or corrupted mass checkpoints before reusing inference."""

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def checkpoint(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/run_nerf_cached_baseline.py"
    spec = importlib.util.spec_from_file_location("nerf_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    source = tmp_path / "scene"
    for name in (
        "info.json",
        "transforms.json",
        "ns/point_cloud.ply",
        "ns/dataparser_transforms.json",
        "features/patch_features_ps56.pt",
        "features/is_visible_ps56.pt",
        "features/voxel_size_ps56.json",
    ):
        file = source / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(b"synthetic checkpoint fixture")
    for folder in ("images", "ns/renders/depth"):
        (source / folder).mkdir(parents=True, exist_ok=True)
        for index in range(30):
            (source / folder / f"{index:02d}").write_bytes(b"fixture")
    output = tmp_path / "checkpoint"
    output.mkdir()
    (output / "input_hashes.json").write_text(json.dumps(runner.input_hashes(source)))
    row = {"scene_id": "scene", "mass_interval_kg": [1.0, 3.0], "mass_kg": 2.0}
    (output / "prediction.json").write_text(json.dumps(row))
    return runner, source, output, row


def test_valid_checkpoint_is_reused(checkpoint):
    runner, source, output, row = checkpoint
    assert runner.load_checkpoint(output, source, "scene") == row


def test_changed_cache_is_rejected(checkpoint):
    runner, source, output, _ = checkpoint
    (source / "info.json").write_text("changed material proposal")
    with pytest.raises(ValueError, match="输入变化"):
        runner.load_checkpoint(output, source, "scene")


def test_added_view_is_rejected(checkpoint):
    runner, source, output, _ = checkpoint
    (source / "images/extra").write_bytes(b"extra view")
    with pytest.raises(ValueError, match="30视角"):
        runner.load_checkpoint(output, source, "scene")


@pytest.mark.parametrize("mass", [0, 1.5, float("nan")])
def test_corrupt_point_mass_is_rejected(checkpoint, mass):
    runner, source, output, row = checkpoint
    row["mass_kg"] = mass
    (output / "prediction.json").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="预测无效"):
        runner.load_checkpoint(output, source, "scene")
