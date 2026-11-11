"""Matched-output ablation of geometric, VLM and fused functional grounding."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from run_reference_mesh_study import load_prepared, savefig

from tactile3d.physics.part_io import write_json
from tactile3d.vision.role_constraints import furniture_role_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--answers", type=Path)
    args = parser.parse_args()
    out = args.source / "views_contacts"
    payload = json.loads(
        (args.answers or (out / "view_count_results.json")).read_text()
    )
    selected = payload["selected_views"]
    calls = [
        r for r in payload["records"] if r["view_count"] == selected and r["valid"]
    ]
    mesh, _labels, pieces, _texture = load_prepared(args.source)
    predictions = {"mesh_geometry": [], "vlm": [], "fusion": []}
    for i, part in enumerate(pieces):
        rows = [
            next(c for c in r["answer"]["components"] if c["component_id"] == i)
            for r in calls
        ]
        material = Counter(c["surface_material"] for c in rows).most_common(1)[0][0]
        role = Counter(c["functional_role"] for c in rows).most_common(1)[0][0]
        geometry = furniture_role_candidates(
            part.bounds, mesh.bounds, "geometry_only", part.is_watertight
        )
        fused = furniture_role_candidates(
            part.bounds, mesh.bounds, material, part.is_watertight
        )
        predictions["mesh_geometry"].append(
            {
                "component_id": i,
                "role": geometry["allowed_roles"][0],
                "surface_material": None,
                "evidence": geometry["evidence"],
            }
        )
        predictions["vlm"].append(
            {"component_id": i, "role": role, "surface_material": material}
        )
        predictions["fusion"].append(
            {
                "component_id": i,
                "role": fused["allowed_roles"][0],
                "surface_material": material,
                "evidence": fused["evidence"],
            }
        )
    # Evaluation labels are loaded only after all three predictions are fixed.
    truth = json.loads((out / "role_review.json").read_text())["roles"]
    metrics = []
    for method, rows in predictions.items():
        correct = sum(r["role"] == truth[str(r["component_id"])] for r in rows)
        metrics.append(
            {
                "method": method,
                "role_accuracy": correct / len(truth),
                "component_count": len(truth),
                "correct_components": correct,
            }
        )
    report = {
        "scope": "one previously exposed reference object; development ablation",
        "original_views": selected,
        "vlm_calls": len(calls),
        "vlm_and_fusion_reuse_identical_raw_answers": True,
        "mesh_geometry_receives_rgb_or_vlm_materials": False,
        "geometry_baseline": "same geometry rules with appearance gating removed",
        "metrics": metrics,
        "predictions": predictions,
    }
    write_json(out / "grounding_ablation.json", report)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "grounding_ablation.json", report)
    fig, ax = plt.subplots(figsize=(3.6, 2.8), layout="constrained")
    bars = ax.bar(
        ["Mesh geometry", "VLM", "Fusion"],
        [100 * m["role_accuracy"] for m in metrics],
        color=["#8e969e", "#588cb2", "#bd805c"],
        width=0.6,
    )
    for bar, row in zip(bars, metrics):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{row['correct_components']}/{row['component_count']}",
            ha="center",
            fontsize=8,
        )
    ax.set(
        ylabel="Functional-role accuracy (%)",
        ylim=(0, 112),
        title="Reference-component grounding",
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    savefig(fig, args.output / "grounding_ablation")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
