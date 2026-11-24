"""Draw same-object density comparisons and a component-level property plate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize, to_rgb
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.figures.mesh_fields import render_mesh
from tactile3d.figures.bound_material import raster_face_ids


INK = "#182535"
FABRIC, WOOD = "#438794", "#a5633d"
NO_MESH, UNASSIGNED = "#dce2e9", "#a3adb8"


def save(fig, stem):
    for fmt in ("png", "svg", "pdf"):
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=360, facecolor="white")
    plt.close(fig)


def named_density_grid(paths, output):
    names = {
        "B07TWQTVXL_ATVPDKIKX0DER": "Metal shoe rack",
        "B07DBDX15Y_ATVPDKIKX0DER": "Patterned accent chair",
        "B07QFRSC43_ATVPDKIKX0DER": "Blue accent chair",
    }
    methods = (
        ("RGB input", "rgb"),
        ("NeRF2Physics", "nerf"),
        ("PUGS–Qwen", "pugs"),
        ("PartPhy3D", "propagated"),
    )
    norm, cmap = LogNorm(10, 10000), mpl.colormaps["plasma"]
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.65))
    fig.subplots_adjust(
        left=0.10, right=0.965, top=0.84, bottom=0.205, wspace=0.14, hspace=0.34
    )
    fig.text(
        0.072,
        0.968,
        "Material-conditioned surface density",
        weight="bold",
        size=13,
        va="top",
    )
    fig.text(0.072, 0.907, "ABO-500  ·  Shared mesh and camera per object", size=7.5)
    results = []
    for index, (path, row) in enumerate(zip(paths, axes, strict=True)):
        with np.load(path) as data:
            fg = data["foreground"]
            supports = (data["face_ids"] >= 0) & fg
            y, x = np.nonzero(fg)
            counts = {}
            for ax, (label, key) in zip(row, methods, strict=True):
                if key == "rgb":
                    layer = data[key]
                else:
                    values = data[key]
                    available = supports & np.isfinite(values) & (values > 0)
                    counts[label] = float(100 * available.sum() / fg.sum())
                    layer = np.ones(fg.shape + (3,))
                    layer[fg] = to_rgb(NO_MESH)
                    layer[supports] = to_rgb(UNASSIGNED)
                    layer[available] = cmap(norm(values[available]))[:, :3]
                    ax.text(
                        0.5,
                        -0.034,
                        f"{counts[label]:.1f}%",
                        transform=ax.transAxes,
                        ha="center",
                        va="top",
                        size=7,
                        color="#526171",
                    )
                ax.imshow(layer, interpolation="nearest")
                ax.set(xlim=(x.min() - 6, x.max() + 6), ylim=(y.max() + 6, y.min() - 6))
                if index == 0:
                    ax.set_title(label, fontsize=8.3, weight="bold", pad=7)
                ax.axis("off")
            row[0].text(
                0.5,
                -0.034,
                names[path.stem],
                transform=row[0].transAxes,
                ha="center",
                va="top",
                size=7,
                weight="bold",
            )
            results.append(
                {
                    "scene_id": path.stem,
                    "object": names[path.stem],
                    "shared_mesh_coverage_pct": float(100 * supports.sum() / fg.sum()),
                    "density_coverage_pct": counts,
                }
            )
    fig.legend(
        handles=[
            Patch(color=NO_MESH, label="Missing mesh surface"),
            Patch(color=UNASSIGNED, label="Unassigned surface"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.065, 0.071),
        frameon=False,
        fontsize=7,
    )
    cax = fig.add_axes([0.48, 0.117, 0.445, 0.022])
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
        ticks=[10, 100, 1000, 10000],
    )
    bar.set_label("Estimated density (kg m$^{-3}$)", size=7.4, labelpad=2)
    bar.ax.tick_params(labelsize=7, length=2, pad=2)
    fig.text(
        0.072,
        0.033,
        "Percentages are foreground pixels with a density estimate.",
        size=7,
        color="#526171",
    )
    save(fig, output / "shared_density")
    return results


def component_fields(reference, fields, state, edit, output):
    parts = json.loads(state.read_text(encoding="utf-8"))["parts"]
    trials = edit["trials"]
    mass = {
        p["component_id"]: p["mass_kg"]
        for p in trials[edit["figure_trial"]]["before"]["parts"]
    }
    with np.load(reference) as arrays:
        vertices, faces, ids = (
            arrays[k] for k in ("vertices", "faces", "component_ids")
        )
    with np.load(fields) as records:
        if not np.array_equal(records["component_ids"], ids):
            raise ValueError(
                "Surface fields and displayed mesh use different face ordering"
            )
        density, modulus, friction = (
            records[k]
            for k in ("density_kg_m3", "youngs_modulus_pa", "friction_coefficient")
        )
    roles = {row["component_id"]: row for row in parts}
    palette = {"fabric": FABRIC, "wood": WOOD}
    material = np.array([to_rgb(palette[roles[i]["surface_material"]]) for i in ids])
    camera = edit["shared_camera_azimuth_deg"]
    projected = raster_face_ids(
        SimpleNamespace(vertices_m=vertices, faces=faces), camera, 650
    )
    visible = projected >= 0
    y, x = np.nonzero(visible)
    estimators = [
        ("Surface material", material, None, None),
        ("Density", density, LogNorm(100, 10000), mpl.colormaps["plasma"]),
        ("Young’s modulus", modulus, LogNorm(1e7, 1e11), mpl.colormaps["cividis"]),
        ("Slider friction", friction, Normalize(0, 1), mpl.colormaps["viridis"]),
    ]
    fig = plt.figure(figsize=(7.2, 4.8))
    gs = fig.add_gridspec(
        3,
        4,
        left=0.045,
        right=0.954,
        top=0.825,
        bottom=0.12,
        height_ratios=(1.14, 0.24, 1.04),
        wspace=0.25,
        hspace=0.38,
    )
    fig.text(
        0.045,
        0.967,
        "Upholstered chaise lounge",
        va="top",
        size=13,
        weight="bold",
    )
    fig.text(
        0.045,
        0.900,
        "Same reference mesh across maps  ·  Material-conditioned surface priors",
        size=7.6,
    )
    for column, (title, values, norm, cmap) in enumerate(estimators):
        ax = fig.add_subplot(gs[0, column])
        legend = fig.add_subplot(gs[1, column])
        if norm is None:
            rgb = values
            legend.axis("off")
            legend.legend(
                handles=[
                    Patch(color=FABRIC, label="Fabric"),
                    Patch(color=WOOD, label="Wood"),
                ],
                loc="upper center",
                ncols=2,
                frameon=False,
                fontsize=6.7,
                handlelength=1,
            )
        else:
            valid = np.isfinite(values) & (
                (values > 0) if isinstance(norm, LogNorm) else True
            )
            rgb = np.tile(to_rgb(UNASSIGNED), (len(faces), 1))
            rgb[valid] = cmap(norm(values[valid]))[:, :3]
            legend.axis("off")
            cax = legend.inset_axes([0.09, 0.57, 0.82, 0.29])
            bar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=cax,
                orientation="horizontal",
            )
            unit, ticks = [
                ("kg m⁻³", [100, 1000, 10000]),
                ("Pa", [1e7, 1e9, 1e11]),
                ("dimensionless", [0, 0.5, 1]),
            ][column - 1]
            bar.set_ticks(ticks)
            bar.ax.tick_params(labelsize=6, length=2, pad=1)
            bar.set_label(unit, fontsize=6, labelpad=1)
        # Keep map colors numerically comparable with their printed color bars.
        raster = np.ones((*projected.shape, 3))
        raster[visible] = rgb[projected[visible]]
        ax.imshow(raster)
        ax.set(xlim=(x.min() - 8, x.max() + 8), ylim=(y.max() + 8, y.min() - 8))
        ax.set_title(title, fontsize=8.1, pad=5, weight="bold")
        ax.axis("off")

    samples = [(1, "Seat"), (3, "Armrest"), (6, "Leg"), (12, "Frame rail")]
    for column, (i, label) in enumerate(samples):
        ax = fig.add_subplot(gs[2, column])
        keep = faces[ids == i]
        unique, local = np.unique(keep, return_inverse=True)
        # Each inset enlarges one component. Mass uses its original metric scale.
        render = render_mesh(
            vertices[unique], local.reshape(-1, 3), material[ids == i], camera, 600
        )
        ax.imshow(render)
        ax.set_title(f"{label}  ·  {i:02d}", fontsize=8, pad=2, weight="bold")
        ax.text(
            0.5,
            -0.02,
            f"{mass[i]:.2f} kg assigned",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )
        ax.axis("off")
    fig.text(
        0.045,
        0.035,
        "Selected components shown at individual scales  ·  Mass from construction hypothesis 1",
        fontsize=7,
        color="#526171",
    )
    save(fig, output / "part_property_fields")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("reference", "fields", "state", "edit-data", "density-pair", "output"):
        parser.add_argument(
            f"--{arg}",
            type=Path,
            required=True,
            **({"nargs": 3} if arg == "density-pair" else {}),
        )
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.labelcolor": INK,
            "figure.facecolor": "white",
        }
    )
    args.output.mkdir(parents=True, exist_ok=True)
    stats = named_density_grid(args.density_pair, args.output)
    (args.output / "shared_density_data.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    component_fields(
        args.reference,
        args.fields,
        args.state,
        json.loads(args.edit_data.read_text(encoding="utf-8")),
        args.output,
    )


if __name__ == "__main__":
    main()
