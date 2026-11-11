"""Collect auditable source data and export final matched-budget study figures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from run_reference_mesh_study import savefig

from tactile3d.physics.part_io import write_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out, args_source = args.output, args.source
    sources = {
        "vlm_raw_records.json": args_source / "vlm" / "records.json",
        "material_priors_raw.json": args_source / "property_queries" / "records.json",
        "grounding_visual_evidence_answers.json": args_source
        / "views_contacts"
        / "color_control"
        / "answers.json",
        "view_selection.json": args_source / "views_contacts" / "view_selection.json",
        "material_library_snapshot.yaml": Path("configs/material_priors.yaml"),
    }
    for name, source in sources.items():
        shutil.copyfile(source, out / name)
    plt.rcParams.update(
        {
            "font.size": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    pairs = read(args_source / "matched_qem.json")
    grounding = read(out / "grounding_ablation.json")
    contacts = read(args_source / "views_contacts" / "contact_geometry.json")
    state = read(args_source / "part_state_geometry_constrained.json")
    data = read(out / "source_data.json")
    data.update(
        matched_qem=pairs,
        grounding_ablation=grounding,
        contact_geometry=contacts,
        unified_part_state=state,
        scope="single exposed development object; reference geometry assisted",
    )
    data["edit"][
        "scope"
    ] = "four closed wooden legs; open rails and upholstery excluded"
    write_json(out / "source_data.json", data)
    write_json(out / "part_state_geometry_constrained.json", state)
    write_json(out / "contact_geometry.json", contacts)
    rows = []
    for i, pair in enumerate(pairs["pairs"]):
        for row in (pair["matched_global"], pair["component_method"]):
            error = row["wood_errors"]
            rows.append(
                {
                    "comparison": i,
                    "method": row["method"],
                    "actual_faces": row["actual_faces"],
                    "surface_p95_mm": row["surface_p95_mm"],
                    "leg_mass_error_percent": (
                        100 * error["mass_relative"] if error else None
                    ),
                    "leg_com_error_mm": error["com_error_mm"] if error else None,
                    "leg_inertia_error_percent": (
                        100 * error["inertia_relative"] if error else None
                    ),
                    "median_qem_and_mapping_ms": 1000 * row["median_seconds"],
                }
            )
    with (out / "matched_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), layout="constrained")
    series = {
        "QEM": [
            p["matched_global"]
            for p in pairs["pairs"]
            if p["component_method"]["method"] == "sensitivity_qem"
        ],
        "Component QEM": [
            p["component_method"]
            for p in pairs["pairs"]
            if p["component_method"]["method"] == "component_qem"
        ],
        "Physics budget": [
            p["component_method"]
            for p in pairs["pairs"]
            if p["component_method"]["method"] == "sensitivity_qem"
        ],
    }
    for (name, values), color in zip(series.items(), ["#8e969e", "#588cb2", "#bd805c"]):
        values = sorted(values, key=lambda r: r["actual_faces"])
        x = [r["actual_faces"] for r in values]
        axes[0].plot(
            x,
            [r["surface_p95_mm"] for r in values],
            "o-",
            color=color,
            label=name,
            markersize=4,
        )
        axes[1].plot(
            x,
            [
                (
                    100 * r["wood_errors"]["inertia_relative"]
                    if r["wood_errors"]
                    else np.nan
                )
                for r in values
            ],
            "o-",
            color=color,
            markersize=4,
        )
        y = np.array([r["median_seconds"] for r in values]) * 1000
        quantiles = np.array([r["timing_iqr_seconds"] for r in values]) * 1000
        axes[2].errorbar(
            x,
            y,
            yerr=np.stack((y - quantiles[:, 0], quantiles[:, 1] - y)),
            fmt="o-",
            color=color,
            markersize=4,
            capsize=2,
        )
    for ax, title, ylabel in zip(
        axes,
        ["Surface fidelity", "Four-leg inertia fidelity", "Measured QEM + mapping"],
        ["P95 surface distance (mm)", "Tensor error (%)", "Median / IQR (ms)"],
    ):
        ax.set(title=title, xlabel="Actual triangles", ylabel=ylabel)
    axes[0].legend(fontsize=6)
    savefig(fig, out / "matched_accuracy_cost")
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    boxes = [
        (
            0.2,
            2.3,
            2.2,
            1.0,
            "VLM multi-round inference\nmaterial / construction",
            "#dce8f0",
        ),
        (0.2, 0.6, 2.2, 1.0, "Mesh geometry\nroles / interfaces", "#e8e0d7"),
        (
            3.5,
            1.4,
            2.3,
            1.1,
            "Unified part state\nIDs · evidence · parameters",
            "#e4e6ec",
        ),
        (7.0, 2.3, 2.7, 1.0, "Task-aware QEM\nverified collapse lineage", "#eee0d4"),
        (7.0, 0.6, 2.7, 1.0, "Closed-domain moments\nlocal physical edit", "#dce8df"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.12",
                facecolor=color,
                edgecolor="#697580",
                linewidth=0.7,
            )
        )
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8)
    for start, end in [
        ((2.55, 2.75), (3.35, 2.2)),
        ((2.55, 1.1), (3.35, 1.7)),
        ((5.95, 2.2), (6.85, 2.75)),
        ((5.95, 1.7), (6.85, 1.1)),
    ]:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=10,
                color="#697580",
                linewidth=1,
            )
        )
    savefig(fig, out / "method_overview")
    for row in contacts["legs"]:
        pair = row["exact_surface_pair"]
        assert np.isclose(
            np.linalg.norm(np.array(pair["point_first_m"]) - pair["point_second_m"]),
            pair["distance_m"],
            atol=1e-9,
        )
        assert abs(row["ground_contact_center_m"][2]) < 1e-7
    for key in ("before", "after"):
        inertia = np.array(data["edit"][key]["inertia_kg_m2"])
        assert np.allclose(inertia, inertia.T) and np.linalg.eigvalsh(inertia).min() > 0
    assert max(p["absolute_face_count_difference"] for p in pairs["pairs"]) <= 2
    files = {}
    for path in sorted(out.iterdir()):
        if path.name == "artifact_validation.json":
            continue
        if path.suffix == ".svg":
            ET.parse(path)
        elif path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF")
        elif path.suffix in (".png", ".tiff"):
            with Image.open(path) as im:
                im.verify()
        files[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    write_json(
        out / "artifact_validation.json",
        {
            "numeric_checks": "passed",
            "vector_and_raster_file_checks": "passed",
            "files": files,
        },
    )
    caption_path = out / "CAPTIONS.md"
    caption_path.write_text(
        caption_path.read_text().split("\nContact panels ghost upholstery")[0]
    )
    with caption_path.open("a") as stream:
        stream.write(
            "\nContact panels ghost upholstery to show supports. Red circles mark exact surface pairs; "
            "colored samples are projected interface neighborhoods. Distances refer to the digital reference.\n"
            "Grounding ablation uses identical VLM outputs for VLM and fusion; geometry-only removes appearance gating. "
            "The selected-view control also supplies per-region sampled texture RGB as visual evidence.\n"
            "Matched timing uses one warmup and 15 alternating repeats per pair; bars show IQR. "
            "Physical metrics concern four closed legs. Missing closed-domain values remain unavailable.\n"
        )
    path = out / "CAPTIONS.md"
    files[path.name] = {
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    write_json(
        out / "artifact_validation.json",
        {
            "numeric_checks": "passed",
            "vector_and_raster_file_checks": "passed",
            "files": files,
        },
    )
    print("Final source data, matched table, editable figures and numeric checks saved")


if __name__ == "__main__":
    main()
