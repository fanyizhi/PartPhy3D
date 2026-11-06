"""同一五物体的已运行对照；不把不同输入预算称作公平排名。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.figures.bound_material import raster_face_ids, shade_face_colors
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.part_io import file_sha256, write_json

COLORS = ("#5A76A8", "#C3916C", "#8573B5", "#4FA38F")


def compare(args: argparse.Namespace) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = json.loads(
        Path("configs/five_object_pilot.json").read_text(encoding="utf-8")
    )["object_ids"]
    labels = json.loads(
        Path("data/abo500_five/filtered_product_weights.json").read_text(
            encoding="utf-8"
        )
    )
    models = {}
    nerf_sources = {}
    nerf = []
    for folder in (args.nerf_gate, args.nerf_remaining):
        seal = json.loads(
            (folder / "prediction_seal.json").read_text(encoding="utf-8")
        )["sha256"]
        if file_sha256(folder / "predictions.json") != seal:
            raise ValueError("基线封存校验失败")
        current = json.loads((folder / "predictions.json").read_text(encoding="utf-8"))
        nerf.extend(current)
        nerf_sources.update({r["scene_id"]: folder / r["scene_id"] for r in current})
    models["NeRF2Physics cached (30 views)"] = {
        r["scene_id"]: r["mass_kg"] for r in nerf
    }
    qwen = json.loads(
        (args.qwen / "sealed_predictions.json").read_text(encoding="utf-8")
    )
    models["Qwen BF16 (1 view)"] = {
        r["scene_id"]: r["mass_kg"]["estimate"] for r in qwen
    }
    for phase, name in (
        ("before", "Qwen NF4 (1 view)"),
        ("after", "QLoRA NF4 (1 view)"),
    ):
        path = args.sft / phase
        seal = json.loads((path / "prediction_seal.json").read_text(encoding="utf-8"))[
            "sha256"
        ]
        if file_sha256(path / "predictions.json") != seal:
            raise ValueError("SFT封存校验失败")
        models[name] = {
            r["scene_id"]: r["predicted_mass_kg"]
            for r in json.loads((path / "predictions.json").read_text(encoding="utf-8"))
        }
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    metrics = {}
    truth = np.array([labels[i] for i in ids])
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    for j, ((name, values), color) in enumerate(
        zip(models.items(), COLORS, strict=True)
    ):
        prediction = np.array([values[i] for i in ids])
        error = np.abs(prediction - truth) / truth * 100
        metrics[name] = mass_metrics(truth.tolist(), prediction.tolist())
        axes[0].scatter(
            truth,
            prediction,
            color=color,
            label=name,
            s=27,
            marker=("o", "s", "^", "D")[j],
        )
        axes[1].plot(
            np.arange(5) + 1,
            error,
            marker=("o", "s", "^", "D")[j],
            color=color,
            label=name,
            lw=1,
        )
        rows.extend(
            {
                "method": name,
                "scene_id": i,
                "target_mass_kg": labels[i],
                "predicted_mass_kg": values[i],
            }
            for i in ids
        )
    low = min(truth.min(), min(r["predicted_mass_kg"] for r in rows)) * 0.7
    high = max(truth.max(), max(r["predicted_mass_kg"] for r in rows)) * 1.5
    axes[0].plot([low, high], [low, high], "--", color=".5", lw=1)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="ABO mass label (kg)",
        ylabel="Predicted mass (kg)",
    )
    axes[1].set(
        xlabel="Frozen object index",
        ylabel="Absolute relative mass error (%)",
        xticks=range(1, 6),
    )
    axes[0].legend(fontsize=7, frameon=False)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Shared development objects; input budgets differ | "
        "complete physical method and PUGS pending",
        fontsize=10,
    )
    for suffix in ("svg", "png"):
        figure.savefig(
            args.output / f"available_mass_baselines.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    figure, axes = plt.subplots(5, 2, figsize=(6.8, 13.5), layout="constrained")
    norm = matplotlib.colors.LogNorm(20, 8000)
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#BBBBBB")
    for row, scene in enumerate(ids):
        token = json.loads(
            (args.fields / scene / "part_tokens.json").read_text(encoding="utf-8")
        )
        mesh = trimesh.load_mesh(token["mesh"], process=False)
        surface = SimpleNamespace(vertices_m=mesh.vertices, faces=mesh.faces)
        face_ids = raster_face_ids(surface, -55, 300)
        with np.load(nerf_sources[scene] / "density_on_generated_mesh.npz") as f:
            baseline = f["density_interval_kg_m3"].mean(1)
        with np.load(args.fields / scene / "surface_fields.npz") as f:
            part_labels = f["labels"]
        ours = np.full(len(mesh.faces), np.nan)
        for p in token["parts"]:
            prior = token["material_priors"].get(p["surface_material"])
            if prior:
                ours[part_labels == p["part_index"]] = prior["density_mean_kg_m3"]
        if len(baseline) != len(mesh.faces):
            raise ValueError("公共Mesh的查询面数不匹配")
        for column, values in enumerate((baseline, ours)):
            rgba = np.rint(cmap(norm(np.ma.masked_invalid(values))) * 255).astype(
                np.uint8
            )
            axes[row, column].imshow(shade_face_colors(surface, rgba, face_ids))
            axes[row, column].set(xticks=[], yticks=[])
            if row == 0:
                axes[row, column].set_title(
                    ("NeRF2Physics cached field", "Part-conditioned surface prior")[
                        column
                    ],
                    fontsize=9,
                )
            if column == 0:
                axes[row, column].set_ylabel(scene.split("_")[0], fontsize=8)
    figure.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes.ravel().tolist(),
        label="Density hypothesis / prior (kg/m³)",
        fraction=0.025,
    )
    figure.suptitle(
        "Same generated Mesh query positions | gray = unknown\n"
        "Different evidence budgets; no local-density ground truth or accuracy claim",
        fontsize=10,
    )
    for suffix in ("svg", "png"):
        figure.savefig(
            args.output / f"density_field_comparison.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)
    with (args.output / "mass_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "metrics": metrics,
        "object_ids": ids,
        "input_budgets_equal": False,
        "PUGS": {"status": "not_reproduced", "score": None},
        "full_PartPhy3D": {"status": "not_complete", "score": None},
        "density_accuracy_established": False,
    }
    write_json(args.output / "report.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("nerf-gate", "nerf-remaining", "qwen", "sft", "fields", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(compare(a), indent=2))


if __name__ == "__main__":
    main()
