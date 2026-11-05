from __future__ import annotations

import json
from pathlib import Path

import pytest

from tactile3d.abo500.dataset import ABOScene
from tactile3d.abo500.demo import make_demo_dataset
from tactile3d.abo500.pipeline import ABO500MeshPipeline
from tactile3d.experiments.five_object_pilot import read_cohort, run_pilot
from tactile3d.physics.part_io import file_sha256, write_json


def test_preparation_requires_no_dataset_or_model(tmp_path: Path) -> None:
    output = tmp_path / "plan"
    result = run_pilot(
        tmp_path / "missing", tmp_path / "missing", output, prepare_only=True
    )
    assert result["status"] == "prepared_not_executed"
    assert len(result["object_ids"]) == 5
    assert not (output / "predictions.json").exists()
    with pytest.raises(ValueError):
        read_cohort(Path("configs/five_object_pilot.json"), 6)


def test_fixed_five_retains_missing_object_and_seals_label_free_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, meshes = make_demo_dataset(tmp_path / "dataset")
    ids = ["scene0000", "missing_object", "scene0001", "scene0002", "scene0003"]
    cohort = tmp_path / "cohort.json"
    write_json(cohort, {"object_ids": ids})
    original = ABO500MeshPipeline.infer_scene

    def checked_inference(
        self: ABO500MeshPipeline, scene: ABOScene, output_dir: Path
    ) -> tuple[dict, dict]:
        assert scene.mass_kg is None
        return original(self, scene, output_dir)

    monkeypatch.setattr(ABO500MeshPipeline, "infer_scene", checked_inference)
    output = tmp_path / "experiment"
    report = run_pilot(root, meshes, output, cohort=cohort, unit_to_meter=1.0)
    assert report["status"] == "incomplete"
    assert report["num_requested"] == 5
    assert report["num_succeeded"] == 4
    assert report["failures"][0]["scene_id"] == "missing_object"
    selection = json.loads((output / "selection_before_evaluation.json").read_text())
    assert selection["object_ids"] == ids
    predictions = json.loads((output / "predictions.json").read_text())
    assert all("target_mass_kg" not in row for row in predictions)
    seal = json.loads((output / "prediction_seal.json").read_text())
    assert file_sha256(output / "predictions.json") == seal["sha256"]
    assert (output / "metrics_valid_objects.json").exists()
