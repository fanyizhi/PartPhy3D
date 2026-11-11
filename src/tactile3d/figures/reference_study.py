"""Publication panels exported from the same saved study arrays."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image
from scipy.spatial import ConvexHull

from tactile3d.experiments.reference_study import (
    MATERIAL_COLORS,
    active_states,
    closed_properties,
    digest,
    load_prepared,
    read,
)
from tactile3d.figures.mesh_fields import project_points as project
from tactile3d.figures.mesh_fields import render_mesh
from tactile3d.mesh.assembly import crop_mesh
from tactile3d.physics.part_io import write_json


def configure_style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def savefig(fig, path):
    for suffix in ("png", "svg", "pdf", "tiff"):
        fig.savefig(
            path.with_suffix("." + suffix),
            dpi=600 if suffix == "tiff" else 250,
            facecolor="white",
            bbox_inches="tight",
            **({"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}),
        )
    plt.close(fig)


def draw_properties(args):
    from matplotlib.colors import to_rgb

    out = args.output
    dest = args.figures
    dest.mkdir(parents=True, exist_ok=True)
    mesh, labels, _pieces, _texture = load_prepared(out)
    states = active_states(out)
    state = states["progressive"]
    lookup = {c["component_id"]: c["surface_material"] for c in state["components"]}
    material = np.array([to_rgb(MATERIAL_COLORS[lookup[int(i)]]) for i in labels])
    unknown = np.array(to_rgb(MATERIAL_COLORS["unknown"]))
    rendered = render_mesh(mesh.vertices, mesh.faces, material, resolution=600)
    fields = {}
    for key in ("density_kg_m3", "youngs_modulus_pa", "friction_coefficient"):
        fields[key] = np.array(
            [
                state["materials"].get(lookup[int(i)], {}).get(key) or np.nan
                for i in labels
            ]
        )
    # Hardness is comparable only on an explicitly shared scale.
    hardness_records = [
        (name, r)
        for name, values in state["materials"].items()
        for r in values.get("hardness_records", [])
        if type(r.get("value")) in (int, float)
        and np.isfinite(r["value"])
        and r["value"] >= 0
        and isinstance(r.get("scale"), str)
        and r["scale"].strip()
    ]
    scales = Counter(r["scale"].strip().lower() for _, r in hardness_records)
    hardness_scale = scales.most_common(1)[0][0] if scales else None
    hardness = {}
    for name in state["materials"]:
        values = [
            r["value"]
            for mat, r in hardness_records
            if mat == name and r["scale"].strip().lower() == hardness_scale
        ]
        hardness[name] = float(np.median(values)) if values else np.nan
    fields["hardness"] = np.array(
        [hardness.get(lookup[int(i)], np.nan) for i in labels]
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), layout="constrained")
    axes[0, 0].imshow(Image.open(out / "observation_00.png"))
    axes[0, 0].set_title("a  Original observation")
    axes[0, 1].imshow(rendered)
    axes[0, 1].set_title("b  Material grounding")
    axes[0, 1].legend(
        handles=[
            Patch(color=color, label=name)
            for name, color in MATERIAL_COLORS.items()
            if name in lookup.values()
        ],
        loc="lower center",
        ncol=3,
        fontsize=7,
    )
    for ax, key, title, unit, factor in zip(
        (axes[0, 2], *axes[1]),
        fields,
        (
            "c  Material density prior",
            "d  Material modulus prior",
            "e  Friction prior",
            "f  Hardness prior",
        ),
        ("kg m$^{-3}$", "GPa", "dimensionless", hardness_scale or "unresolved scale"),
        (1, 1e-9, 1, 1),
    ):
        values = fields[key] * factor
        finite = np.isfinite(values)
        rgb = np.tile(unknown, (len(mesh.faces), 1))
        if finite.any():
            lo, hi = values[finite].min(), values[finite].max()
            if hi <= lo:
                lo, hi = max(lo * 0.8, 1e-9), hi * 1.2
            norm = LogNorm(lo, hi) if key == "youngs_modulus_pa" else Normalize(lo, hi)
            rgb[finite] = plt.get_cmap("viridis")(norm(values[finite]))[:, :3]
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                ax=ax,
                shrink=0.65,
                label=unit,
                pad=0.01,
            )
        ax.imshow(render_mesh(mesh.vertices, mesh.faces, rgb, resolution=600))
        ax.set_title(title)
        if not finite.any():
            ax.text(
                0.5,
                0.02,
                (
                    "Unresolved contact pair"
                    if key == "friction_coefficient"
                    else "No supported hardness scale"
                ),
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
            )
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(
        "Reference geometry · VLM material labels + explicit property priors",
        fontsize=9,
    )
    savefig(fig, dest / "dense_property_maps")
    rows = read(out / "lod" / "metrics.json")
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.2), layout="constrained")
    for ax, row in zip(axes.flat, rows):
        data = np.load(out / "lod" / f"{row['method']}_{row['target_faces']}.npz")
        cols = np.array(
            [to_rgb(MATERIAL_COLORS[lookup[int(i)]]) for i in data["component_ids"]]
        )
        ax.imshow(render_mesh(data["vertices"], data["faces"], cols, resolution=500))
        ax.set_title(
            f"{row['method']} · {row['actual_faces']:,} faces\n"
            f"surface P95 {row['surface_p95_mm']:.2f} mm",
            fontsize=8,
        )
        ax.axis("off")
    savefig(fig, dest / "qem_mesh_comparison")
    edit = read(out / "edit.json")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5), layout="constrained")
    axes = axes.ravel()
    edited = material.copy()
    edited[labels == edit["component_id"]] = to_rgb("#c0c6ce")
    axes[0].imshow(rendered)
    axes[0].set_title("Before · wooden supports")
    axes[0].axis("off")
    axes[1].imshow(render_mesh(mesh.vertices, mesh.faces, edited, resolution=600))
    axes[1].set_title(f"Edit component {edit['component_id']} to aluminum")
    axes[1].axis("off")
    axes[2].bar(
        ["Before", "After"],
        [edit[k]["mass_kg"] for k in ("before", "after")],
        color=["#bd8656", "#a0a8b2"],
    )
    axes[2].set(title="Four-leg domain mass", ylabel="Mass (kg)")
    moments = [
        np.linalg.eigvalsh(edit[k]["inertia_kg_m2"]) for k in ("before", "after")
    ]
    for i, (name, vals) in enumerate(zip(("Before", "After"), moments)):
        axes[3].bar(np.arange(3) + (i - 0.5) * 0.35, vals, 0.35, label=name)
    axes[3].set(
        title="Leg-domain principal moments",
        ylabel="kg m$^2$",
        xticks=range(3),
        xticklabels=["I1", "I2", "I3"],
    )
    axes[3].legend(fontsize=7)
    centers = (
        np.array([edit[k]["center_of_mass_m"] for k in ("before", "after")]) * 1000
    )
    for i, name in enumerate(("Before", "After")):
        axes[4].bar(np.arange(3) + (i - 0.5) * 0.35, centers[i], 0.35, label=name)
    axes[4].set(
        title="Leg-domain center of mass",
        ylabel="Coordinate (mm)",
        xticks=range(3),
        xticklabels=["x", "y", "z"],
    )
    displacement = centers[1] - centers[0]
    axes[5].bar(range(3), displacement, color="#bd805c")
    axes[5].set(
        title="Center-of-mass displacement",
        ylabel="After − before (mm)",
        xticks=range(3),
        xticklabels=["x", "y", "z"],
    )
    savefig(fig, dest / "local_material_edit")
    # Conditional sensitivity on the actual closed frame, with one shared wood
    # density across parts. These curves are not calibrated confidence intervals.
    rho0 = edit["before_density_kg_m3"]
    rho_after = edit["edited_density_kg_m3"]
    frame = edit["before"]
    selected = closed_properties(_pieces, [edit["component_id"]], 1.0)
    frame_volume = frame["mass_kg"] / rho0
    frame_center = np.array(frame["center_of_mass_m"])
    part_center = np.array(selected["center_of_mass_m"])

    def origin_tensor(properties):
        c = np.array(properties["center_of_mass_m"])
        return np.array(properties["inertia_kg_m2"]) + properties["mass_kg"] * (
            np.dot(c, c) * np.eye(3) - np.outer(c, c)
        )

    density_values = np.linspace(300.0, 950.0, 80)
    sweep = []
    for rho in density_values:
        mass_before = rho * frame_volume
        delta_mass = (rho_after - rho) * selected["mass_kg"]
        mass_after = mass_before + delta_mass
        center_after = (
            mass_before * frame_center + delta_mass * part_center
        ) / mass_after
        origin_after = origin_tensor(frame) * rho / rho0 + origin_tensor(selected) * (
            rho_after - rho
        )
        inertia_after = origin_after - mass_after * (
            np.dot(center_after, center_after) * np.eye(3)
            - np.outer(center_after, center_after)
        )
        sweep.append(
            {
                "wood_density_kg_m3": float(rho),
                "before_mass_kg": float(mass_before),
                "after_mass_kg": float(mass_after),
                "com_displacement_mm": float(
                    np.linalg.norm(center_after - frame_center) * 1000
                ),
                "after_principal_inertia_kg_m2": np.linalg.eigvalsh(
                    inertia_after
                ).tolist(),
            }
        )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), layout="constrained")
    axes[0].plot(
        density_values,
        [r["before_mass_kg"] for r in sweep],
        label="Before",
        color="#bd8656",
    )
    axes[0].plot(
        density_values,
        [r["after_mass_kg"] for r in sweep],
        label="One aluminum leg",
        color="#6588a2",
    )
    axes[0].set(title="Four-leg domain mass", ylabel="kg")
    axes[0].legend(fontsize=6)
    axes[1].plot(
        density_values, [r["com_displacement_mm"] for r in sweep], color="#6588a2"
    )
    axes[1].set(title="Edit-induced COM shift", ylabel="mm")
    for index in range(3):
        axes[2].plot(
            density_values,
            [r["after_principal_inertia_kg_m2"][index] for r in sweep],
            label=f"I{index+1}",
        )
    axes[2].set(title="Edited leg-domain inertia", ylabel="kg m$^2$")
    axes[2].legend(fontsize=6)
    for ax in axes:
        ax.set_xlabel("Shared wood density (kg m$^{-3}$)")
        ax.axvline(rho0, color="0.6", linestyle=":", linewidth=0.7)
    savefig(fig, dest / "physical_parameter_sensitivity")
    write_json(dest / "density_sensitivity.json", sweep)
    np.savez_compressed(dest / "surface_fields.npz", component_ids=labels, **fields)
    write_json(
        dest / "source_data.json",
        {
            "input": read(out / "input_manifest.json"),
            "consensus": states,
            "qem": rows,
            "edit": edit,
            "vlm_seal": read(out / "vlm" / "seal.json"),
            "geometry_arrays_sha256": digest(out / "reference_arrays.npz"),
        },
    )
    (dest / "CAPTIONS.md").write_text(
        "# Chaise reference-geometry study\n\n"
        "Artist reference geometry is an additional input, not reconstructed geometry. "
        "Component assignments and surface property fields are VLM predictions/priors. "
        "Density maps describe surface material and do not assert homogeneous upholstery. "
        "Missing VLM density/modulus values bind to the existing configured material library; "
        "per-parameter provenance is recorded in source_data.json. "
        "Unknown properties remain gray. Hardness is not pooled across incompatible scales.\n\n"
        "QEM comparison uses common requested budgets and reports actual face counts. "
        "Surface distances are evaluated in both directions using 12000 samples per surface. "
        "Physical errors concern only VLM-assigned closed wooden regions with fixed nominal density. "
        "Invalid domains have no physical-error value. Times include reduction and component mapping, "
        "exclude VLM, and summarize three warmed runs.\n\n"
        "Local edit changes one closed wooden component to specified aluminum density. "
        "Mass and inertia refer to four closed wooden legs, excluding open rails, upholstery and hidden supports.\n"
        "RGB and reference mesh: Amazon Berkeley Objects, Collins et al., CVPR 2022, CC BY 4.0. "
        "These renders, overlays and simplifications are derived analyses.\n",
        encoding="utf-8",
    )


def draw_contacts(args):
    out = args.output / "views_contacts"
    mesh, _labels, pieces, texture = load_prepared(args.output)
    report = json.loads((out / "contact_geometry.json").read_text())
    arrays = np.load(out / "contact_points.npz")
    rails = trimesh.util.concatenate([pieces[i] for i in [10, 11, 12, 13]])
    fig = plt.figure(figsize=(7.2, 4.6), layout="constrained")
    grid = fig.add_gridspec(2, 4)
    ax = fig.add_subplot(grid[:, :2])
    full = render_mesh(mesh.vertices, mesh.faces, texture, azimuth=55, resolution=600)
    supports = np.isin(_labels, [6, 7, 8, 9, 10, 11, 12, 13])
    assembly = render_mesh(
        mesh.vertices,
        mesh.faces[supports],
        texture[supports],
        azimuth=55,
        resolution=600,
    )
    ghosted = np.rint(0.20 * full + 0.80 * 255).astype(np.uint8)
    visible = np.any(assembly < 250, axis=2)
    ghosted[visible] = assembly[visible]
    ax.imshow(ghosted)
    points = np.array([r["ground_contact_center_m"] for r in report["legs"]])
    uv = project(mesh.vertices, points, 55)
    order = ConvexHull(points[:, :2]).vertices
    polygon = uv[np.r_[order, order[0]]]
    ax.plot(polygon[:, 0], polygon[:, 1], "--", color="#d28b45", lw=1)
    ax.scatter(uv[:, 0], uv[:, 1], s=24, c="#d05a42", edgecolors="white", zorder=4)
    for point, row in zip(uv, report["legs"]):
        ax.annotate(
            str(row["leg_component_id"]),
            point,
            xytext=(0, -10),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax.axis("off")
    ax.set_title("a  Support assembly and ground contacts", fontsize=8)
    for k, row in enumerate(report["legs"]):
        ax = fig.add_subplot(grid[k // 2, 2 + k % 2])
        index = row["leg_component_id"]
        leg = pieces[index]
        center = (leg.bounds[0] + leg.bounds[1]) / 2
        center[2] = leg.bounds[1, 2] - 0.02
        clipped_leg = crop_mesh(leg, center, np.array([0.075, 0.075, 0.055]))
        clipped_rails = crop_mesh(rails, center, np.array([0.075, 0.075, 0.055]))
        clipped = trimesh.util.concatenate([clipped_leg, clipped_rails])
        colors = np.vstack(
            [
                np.tile([0.74, 0.53, 0.34], (len(clipped_leg.faces), 1)),
                np.tile([0.56, 0.65, 0.72], (len(clipped_rails.faces), 1)),
            ]
        )
        az = 45 if center[0] < 0 else 135
        ax.imshow(
            render_mesh(
                clipped.vertices, clipped.faces, colors, azimuth=az, resolution=500
            )
        )
        samples = arrays[f"leg_{index}_samples"]
        distance = arrays[f"leg_{index}_distances_m"]
        chosen = np.flatnonzero(distance <= 0.003)
        chosen = chosen[:: max(len(chosen) // 160, 1)]
        visible = project(clipped.vertices, samples[chosen], az, resolution=500)
        ax.scatter(
            visible[:, 0],
            visible[:, 1],
            s=1.3,
            c=distance[chosen] * 1000,
            cmap="viridis",
            vmin=0,
            vmax=3,
        )
        exact = row["exact_surface_pair"]
        uv = project(
            clipped.vertices, np.array([exact["point_first_m"]]), az, resolution=500
        )[0]
        ax.scatter(*uv, s=22, facecolors="none", edgecolors="#cf4339", linewidths=0.8)
        gap = (
            "<0.001"
            if exact["distance_m"] < 1e-6
            else f"{exact['distance_m']*1000:.3f}"
        )
        ax.set_title(
            f"{chr(98+k)}  Leg {index}–frame\nSurface gap {gap} mm", fontsize=7
        )
        ax.axis("off")
    fig.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 3), cmap="viridis"),
        ax=fig.axes[1:],
        orientation="horizontal",
        fraction=0.025,
        pad=0.03,
        label="Sample-to-frame distance (mm)",
    )
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    args.figures.mkdir(parents=True, exist_ok=True)
    savefig(fig, args.figures / "contact_geometry")


def finalize_assets(args):
    out, args_source = args.figures, args.output
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
            "VLM inference\nmulti-round hypotheses\nmaterial / structure",
            "#dce8f0",
        ),
        (0.2, 0.6, 2.2, 1.0, "Mesh geometry\nroles / interfaces", "#e8e0d7"),
        (
            3.5,
            1.4,
            2.3,
            1.1,
            "Unified part state\nIDs · evidence\nmaterial parameters",
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
