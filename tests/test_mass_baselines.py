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


@pytest.fixture
def pugs():
    path = Path(__file__).resolve().parents[1] / "scripts/run_pugs_baseline.py"
    spec = importlib.util.spec_from_file_location("pugs_runner", path)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    answer = {
        "description": "fixture",
        "pure_volume": 0.02,
        "materials": [
            {"name": name, "mass density (kg/m^3)": value}
            for name, value in [
                ("fabric", "100-300"),
                ("foam", "30-100"),
                ("wood", "300-600"),
            ]
        ],
    }
    return runner, answer


def test_pugs_accepts_variable_candidate_count_like_upstream_parser(pugs):
    runner, answer = pugs
    info = runner.proposal_info(answer, 8)
    assert info["pure_volume"] == 0.02
    assert info["idx_to_caption"] == "8"
    assert info["candidate_materials_density"].count("kg/m^3") == 3


@pytest.mark.parametrize("volume", [0, -1, float("nan"), float("inf")])
def test_pugs_rejects_invalid_solid_volume(pugs, volume):
    runner, answer = pugs
    answer["pure_volume"] = volume
    with pytest.raises(ValueError, match="volume"):
        runner.proposal_info(answer, 8)


def test_pugs_rejects_reversed_density(pugs):
    runner, answer = pugs
    answer["materials"][0]["mass density (kg/m^3)"] = "300-100"
    with pytest.raises(ValueError, match="density"):
        runner.proposal_info(answer, 8)


def test_pugs_catalog_export_uses_unique_fallback_and_converts_units(pugs):
    runner, _ = pugs
    catalog = {
        "objects": {
            "A_market": {
                "preferred_domain": "missing",
                "candidates": [
                    {
                        "domain_name": "available",
                        "dimensions_m": {"length": 1.0, "width": 0.5, "height": 0.25},
                    }
                ],
            }
        }
    }
    assert runner.catalog_metadata(catalog) == {
        "A": {"item_dimensions": [100.0, 50.0, 25.0]}
    }
    catalog["objects"]["A_market"]["candidates"] *= 2
    with pytest.raises(ValueError, match="Ambiguous"):
        runner.catalog_metadata(catalog)


def test_pugs_volume_normalization_and_mass_conservation(pugs):
    import numpy as np

    runner, _ = pugs
    probabilities = [[1.0, 0.0], [0.0, 1.0]]
    density = [[100.0, 200.0], [500.0, 800.0]]
    assigned, mass = runner.mass_decomposition([1.0, 3.0], probabilities, density, 0.1)
    assert assigned.sum() == pytest.approx(0.1)
    np.testing.assert_allclose(mass.sum(axis=0), [40.0, 65.0])
    _, scaled = runner.mass_decomposition([10.0, 30.0], probabilities, density, 0.1)
    np.testing.assert_allclose(scaled, mass)
    with pytest.raises(ValueError, match="decomposition"):
        runner.mass_decomposition([1.0, 3.0], [[0.0, 0.0], [0.0, 1.0]], density, 0.1)
