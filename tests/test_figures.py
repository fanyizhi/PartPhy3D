import json

import numpy as np
import trimesh

from tactile3d.eval.import_baseline import import_mass_baseline
from tactile3d.figures.comparison import generate_comparison_figures
from tactile3d.figures.object_maps import generate_object_figures


def _predictions(path, scale):
    rows = []
    for index, target in enumerate((0.5, 1.0, 2.0, 4.0)):
        mean = target * scale[index]
        rows.append(
            {
                "scene_id": f"scene{index}",
                "target_mass_kg": target,
                "object": {
                    "mass_kg": {
                        "mean": mean,
                        "q05": mean * 0.8,
                        "q50": mean,
                        "q95": mean * 1.2,
                    }
                },
            }
        )
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_comparison_figures_export_metrics_and_source_data(tmp_path):
    baseline = tmp_path / "baseline.json"
    ours = tmp_path / "ours.json"
    _predictions(baseline, (1.4, 0.6, 1.3, 0.7))
    _predictions(ours, (1.05, 0.95, 1.1, 0.9))
    report = generate_comparison_figures(
        {"Baseline": baseline, "Ours": ours},
        tmp_path / "figures",
        bootstrap_samples=100,
        formats=("png",),
    )
    assert report["num_common_scenes"] == 4
    assert report["methods"]["Ours"]["ape"]["mean"] < report["methods"]["Baseline"]["ape"]["mean"]
    assert report["pairwise_differences"]["Baseline - Ours"]["ape"]["mean"] > 0
    assert (tmp_path / "figures" / "mass_metrics_comparison.png").is_file()
    assert (tmp_path / "figures" / "uncertainty_calibration.png").is_file()
    assert (tmp_path / "figures" / "comparison_source_data.csv").is_file()
    assert report["uncertainty"]["Ours"]["coverage_90"]["mean"] == 1.0


def test_import_pugs_interval_predictions(tmp_path):
    predictions = tmp_path / "pugs.json"
    targets = tmp_path / "targets.json"
    output = tmp_path / "normalized.json"
    predictions.write_text(json.dumps({"chair_view0": [1.0, 1.4]}), encoding="utf-8")
    targets.write_text(json.dumps({"chair": 1.25}), encoding="utf-8")
    summary = import_mass_baseline(
        predictions,
        targets,
        output,
        method="PUGS",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary["num_imported"] == 1
    assert payload["predictions"][0]["scene_id"] == "chair"
    assert payload["predictions"][0]["object"]["mass_kg"] == {
        "mean": 1.2,
        "range_low": 1.0,
        "range_high": 1.4,
    }
    assert payload["predictions"][0]["interval_semantics"] == "range_bounds_not_quantiles"


def test_object_figures_render_mesh_physics_maps(tmp_path):
    mesh = trimesh.creation.box()
    mesh_path = tmp_path / "mesh.ply"
    mesh.export(mesh_path)
    labels = np.arange(len(mesh.faces), dtype=np.int64) % 2
    labels_path = tmp_path / "labels.npy"
    np.save(labels_path, labels)
    parts = []
    for part_id in range(2):
        parts.append(
            {
                "part_id": part_id,
                "density_kg_m3": {"mean": 500.0 + 500.0 * part_id, "q05": 400.0, "q95": 1200.0},
                "youngs_modulus_pa": {"mean": 1e7 * (part_id + 1), "q05": 5e6, "q95": 3e7},
                "mass_kg": {"mean": 0.5 + part_id, "q05": 0.4 + part_id, "q95": 0.7 + part_id},
                "interior": {"posterior_structure_probabilities": {"solid": 0.7, "hollow": 0.3}},
            }
        )
    prediction = {
        "scene_id": "box",
        "object": {"mass_kg": {"mean": 2.0, "q05": 1.6, "q50": 2.0, "q95": 2.4}},
        "parts": parts,
    }
    prediction_path = tmp_path / "prediction.json"
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    result = generate_object_figures(
        prediction_path,
        mesh_path,
        labels_path,
        tmp_path / "object_figures",
        formats=("png",),
    )
    assert result["scene_id"] == "box"
    assert (tmp_path / "object_figures" / "object_physics_maps.png").is_file()
    assert (tmp_path / "object_figures" / "part_posterior_summary.png").is_file()
