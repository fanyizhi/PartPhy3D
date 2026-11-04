import json
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from tactile3d.abo500.dataset import ABO500Dataset
from tactile3d.abo500.pipeline import ABO500MeshPipeline
from tactile3d.config import load_config, resolve_config_path
from tactile3d.eval.abo500 import evaluate_abo500_predictions
from tactile3d.models.calibration import fit_mass_calibration
from tactile3d.physics.materials import load_material_priors
from tactile3d.utils.io import write_json, write_jsonl


def _make_mini_abo500(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "abo_500"
    scene_dir = root / "scenes" / "scene0000"
    images = scene_dir / "images"
    mesh_root = tmp_path / "meshes"
    images.mkdir(parents=True)
    mesh_root.mkdir()

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = [170, 175, 180]
    Image.fromarray(image).save(images / "image0000.jpg")
    transforms = {
        "w": 64,
        "h": 64,
        "fl_x": 55.0,
        "fl_y": 55.0,
        "cx": 32.0,
        "cy": 32.0,
        "frames": [
            {
                "file_path": "images/image0000.jpg",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    write_json(scene_dir / "transforms.json", transforms)
    write_json(root / "filtered_product_weights.json", {"scene0000": {"weight": 1, "unit": "kg"}})
    write_json(root / "splits.json", {"train": ["scene0000"], "val": [], "test": ["scene0000"]})
    trimesh.creation.box(extents=[0.1, 0.08, 0.06]).export(mesh_root / "scene0000.ply")
    return root, mesh_root


def test_abo500_mesh_pipeline(tmp_path):
    root, mesh_root = _make_mini_abo500(tmp_path)
    dataset = ABO500Dataset(root, mesh_root, mesh_unit_to_meter=1.0)
    report = dataset.validate("test")
    assert report["num_valid"] == 1

    cfg = load_config(Path("configs/abo500_mesh_smoke.yaml"))
    prediction, features = ABO500MeshPipeline(cfg).infer_scene(
        dataset.get_scene("scene0000", "test"), tmp_path / "output"
    )
    assert prediction["mesh"]["watertight"]
    assert prediction["mesh"]["volume_m3"] > 0
    assert prediction["object"]["mass_kg"]["mean"] > 0
    assert len(prediction["parts"]) > 0
    assert "interior" in prediction["parts"][0]
    assert prediction["parts"][0]["occupied_volume_m3"]["mean"] > 0
    assert abs(sum(part["volume_m3"] for part in features["parts"]) - 0.00048) < 1e-6
    assert (tmp_path / "output" / "mesh_parts.ply").exists()
    assert (tmp_path / "output" / "interior_proposals.json").exists()


def test_calibration_and_metrics(tmp_path):
    cfg = load_config(Path("configs/abo500_mesh_smoke.yaml"))
    priors = load_material_priors(
        resolve_config_path(cfg["materials"]["priors"], cfg["_config_dir"])
    )
    material_names = [prior.name for prior in priors]
    row = {
        "scene_id": "scene0000",
        "target_mass_kg": 1.0,
        "material_names": material_names,
        "parts": [{"volume_m3": 0.001, "mean_logits": [0.0] * len(priors)}],
    }
    features = tmp_path / "features.jsonl"
    write_jsonl(features, [row])
    checkpoint = fit_mass_calibration(features, priors, tmp_path / "calibration.json", max_iters=20)
    assert checkpoint["density_scale"] > 0

    predictions = [
        {
            "scene_id": "scene0000",
            "target_mass_kg": 1.0,
            "object": {"mass_kg": {"mean": 0.8, "q05": 0.6, "q95": 1.1}},
        }
    ]
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(json.dumps(predictions), encoding="utf-8")
    result = evaluate_abo500_predictions(prediction_path)
    assert result["metrics"]["ade_kg"] == pytest.approx(0.2)
    assert result["metrics"]["coverage_90"] == 1.0
