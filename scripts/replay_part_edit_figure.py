"""Audit a sealed ABO chaise material edit and draw its paper evidence plate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib as mpl
import numpy as np
from PIL import Image

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tactile3d.experiments.vlm_mass import properties
from tactile3d.figures.bound_material import raster_face_ids
from tactile3d.figures.mesh_fields import render_mesh


WOOD, ALUMINUM = "#a5633d", "#187c88"
NEUTRAL, INK = "#c9ced6", "#182535"
TARGET_ID, EDITED_DENSITY = 6, 2700.0
AZIMUTH, SIZE = -135, 740


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def translated_inertia(mass, center):
    return mass * ((center @ center) * np.eye(3) - np.outer(center, center))


def replay(construction, state):
    protocol = read(construction / "protocol.json")
    seal = read(construction / "prediction_seal.json")
    if any(digest(construction / name) != sha for name, sha in seal.items()):
        raise ValueError("Construction predictions failed the seal check")
    if digest(state) != protocol["part_state_sha256"]:
        raise ValueError("Part state differs from the sealed construction input")
    parts = read(state)["parts"]
    target = next(part for part in parts if part["component_id"] == TARGET_ID)
    if target["surface_material"] != "wood" or target["functional_role"] != "leg":
        raise ValueError("The edit requires a wood leg at the expected part ID")
    densities = protocol["densities_kg_m3"]
    rows = []
    for index in range(protocol["rounds"]):
        frozen = read(construction / f"hypothesis_{index}.json")
        partition, answer = frozen["partition"], frozen["answer"]
        before = properties(partition, parts, answer, densities)
        for key in ("mass_kg", "center_of_mass_m", "inertia_kg_m2"):
            np.testing.assert_allclose(before[key], frozen[key], rtol=1e-8, atol=1e-10)
        after = properties(
            partition, parts, answer, densities, TARGET_ID, EDITED_DENSITY
        )
        domain = next(
            p["assigned"] for p in partition["parts"] if p["component_id"] == TARGET_ID
        )
        delta_density = EDITED_DENSITY - densities["wood"]
        m0, c0 = before["mass_kg"], np.asarray(before["center_of_mass_m"])
        dm = delta_density * domain["volume_m3"]
        m1 = m0 + dm
        c1 = (m0 * c0 + delta_density * np.asarray(domain["first_m4"])) / m1
        inertia1 = (
            np.asarray(before["inertia_kg_m2"])
            + translated_inertia(m0, c0)
            + delta_density * np.asarray(domain["inertia_origin_m5"])
            - translated_inertia(m1, c1)
        )
        res = {
            "mass_kg": abs(m1 - after["mass_kg"]),
            "com_m": float(np.linalg.norm(c1 - after["center_of_mass_m"])),
            "inertia_kg_m2": float(np.linalg.norm(inertia1 - after["inertia_kg_m2"])),
        }
        if max(res.values()) > 1e-9:
            raise ValueError("Local moment update diverges from full recomputation")
        original_mass = {p["component_id"]: p["mass_kg"] for p in before["parts"]}
        changed_mass = {p["component_id"]: p["mass_kg"] for p in after["parts"]}
        if any(
            changed_mass[i] != value
            for i, value in original_mass.items()
            if i != TARGET_ID
        ):
            raise ValueError("A protected component's mass changed")
        rows.append(
            {
                "trial": index,
                "before": before,
                "after": after,
                "delta_mass_kg": dm,
                "com_shift_mm": float(1000 * np.linalg.norm(c1 - c0)),
                "local_vs_full_residual": res,
                "target_domain_volume_m3": domain["volume_m3"],
            }
        )
    return {
        "scene": protocol["scene"],
        "target": target["part_id"],
        "component_id": TARGET_ID,
        "before_density_kg_m3": densities["wood"],
        "after_density_kg_m3": EDITED_DENSITY,
        "geometry_sha256": read(state)["geometry_sha256"],
        "construction_seal_sha256": digest(construction / "prediction_seal.json"),
        "figure_trial": 0,
        "surface_geometry_reused": True,
        "mass_labels_read": False,
        "property_scope": "Fixed domain density edit; all masses are model estimates.",
        "trials": rows,
    }


def mesh_views(vertices, faces, component_ids):
    geometry = SimpleNamespace(vertices_m=vertices, faces=faces)
    # Pick the shared camera for which the edited component is best exposed.
    cameras = (AZIMUTH, -45, 45, 135)

    def visible_count(azimuth):
        ids = raster_face_ids(geometry, azimuth, SIZE)
        return np.count_nonzero((ids >= 0) & (component_ids[ids] == TARGET_ID))

    camera = max(cameras, key=visible_count)
    colors = np.tile(to_rgb(NEUTRAL), (len(faces), 1))
    selected = component_ids == TARGET_ID
    before_colors, after_colors = colors.copy(), colors.copy()
    before_colors[selected], after_colors[selected] = to_rgb(WOOD), to_rgb(ALUMINUM)
    before = render_mesh(vertices, faces, before_colors, camera, SIZE)
    after = render_mesh(vertices, faces, after_colors, camera, SIZE)
    pixels = raster_face_ids(geometry, camera, SIZE)
    changed = np.any(before != after, axis=-1)
    if np.any(changed & ((pixels < 0) | ~selected[pixels])):
        raise ValueError("Pre/post render changed outside the projected target")
    y, x = np.nonzero((pixels >= 0) & selected[pixels])
    return (
        before,
        after,
        camera,
        (float(np.median(x)), float(np.median(y))),
        int(changed.sum()),
    )


def make_figure(geometry, rgb_path, result, output):
    vertices, faces, ids = (
        geometry[key] for key in ("vertices", "faces", "component_ids")
    )
    before, after, camera, target_xy, changed_pixels = mesh_views(vertices, faces, ids)
    result.update(
        {
            "shared_camera_azimuth_deg": camera,
            "render_resolution_px": SIZE,
            "changed_target_pixels": changed_pixels,
            "changed_protected_pixels": 0,
        }
    )
    row = result["trials"][result["figure_trial"]]
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "text.color": INK,
            "axes.labelcolor": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(7.2, 4.65), facecolor="white")
    grid = fig.add_gridspec(
        2,
        3,
        height_ratios=(1.0, 0.85),
        left=0.075,
        right=0.955,
        top=0.875,
        bottom=0.158,
        wspace=0.24,
        hspace=0.40,
    )
    panels = [fig.add_subplot(grid[0, i]) for i in range(3)]
    rgb = Image.open(rgb_path).convert("RGBA")
    if rgb.getchannel("A").getextrema()[0] < 255:
        rgb = rgb.crop(rgb.getchannel("A").getbbox())
    else:
        mask = np.any(np.asarray(rgb)[:, :, :3] < 235, axis=-1)
        y, x = np.nonzero(mask)
        rgb = rgb.crop((x.min(), y.min(), x.max() + 1, y.max() + 1))
    panels[0].imshow(rgb)
    for ax, raster in zip(panels[1:], (before, after)):
        ax.imshow(raster)
        y, x = np.nonzero(np.any(before != 255, axis=-1))
        ax.set(xlim=(x.min() - 12, x.max() + 12), ylim=(y.max() + 17, y.min() - 17))
    titles = ("ABO input", "Wood leg selected", "Aluminum assigned")
    for label, ax, title in zip("abc", panels, titles):
        ax.set_title(
            f"{label}   {title}", loc="left", fontsize=9, fontweight="bold", pad=8
        )
        ax.axis("off")
    for ax, color in zip(panels[1:], (WOOD, ALUMINUM)):
        ax.scatter(*target_xy, s=22, facecolor=color, edgecolor="white", linewidth=0.85)
    panels[1].text(
        0.5,
        -0.03,
        f"{result['before_density_kg_m3']:,.0f} kg/m³  ·  {result['target']}",
        transform=panels[1].transAxes,
        ha="center",
        va="top",
        color=WOOD,
        fontsize=7.4,
    )
    panels[2].text(
        0.5,
        -0.03,
        f"{result['after_density_kg_m3']:,.0f} kg/m³  ·  {result['target']}",
        transform=panels[2].transAxes,
        ha="center",
        va="top",
        color=ALUMINUM,
        fontsize=7.4,
    )

    ax = fig.add_subplot(grid[1, :2])
    c0, c1 = (
        np.asarray(row[key]["center_of_mass_m"])[:2] * 1000
        for key in ("before", "after")
    )
    target = vertices[faces[ids == TARGET_ID].ravel(), :2].mean(axis=0) * 1000
    triangles = vertices[faces]
    order = np.argsort(triangles[:, :, 2].mean(axis=1))
    colors = np.where(np.isin(ids, [0, 2, 3, 4, 5]), "#dce3e9", "#eef1f4")
    ax.add_collection(
        PolyCollection(
            triangles[order, :, 1::-1] * 1000,
            facecolors=colors[order],
            edgecolors="face",
            linewidths=0.07,
        )
    )
    ax.scatter(target[1], target[0], s=17, marker="s", color=ALUMINUM, zorder=3)
    ax.scatter(c0[1], c0[0], s=10, color=INK, zorder=3)
    ax.set_aspect("equal")
    ax.set(
        xlim=(-1000, 1850), xlabel="Longitudinal position (mm)", ylabel="Lateral (mm)"
    )
    ax.set_xticks([-800, 0, 800])
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title(
        "d   Whole-object center of mass", loc="left", fontsize=9, fontweight="bold"
    )
    ax.annotate(
        "Edited leg",
        xy=(target[1], target[0]),
        xytext=(155, -335),
        fontsize=7,
        arrowprops={"arrowstyle": "-", "color": ALUMINUM, "lw": 0.7},
        color=ALUMINUM,
    )
    inset = ax.inset_axes([0.66, 0.23, 0.32, 0.68])
    inset.set_facecolor("#f3f6f7")
    inset.annotate(
        "",
        xy=c1[::-1],
        xytext=c0[::-1],
        arrowprops={"arrowstyle": "->", "lw": 0.9, "color": "#667480"},
    )
    inset.scatter(
        [c0[1], c1[1]], [c0[0], c1[0]], color=[WOOD, ALUMINUM], s=25, zorder=3
    )
    inset.set(
        xlim=(min(c0[1], c1[1]) - 12, max(c0[1], c1[1]) + 12),
        ylim=(min(c0[0], c1[0]) - 16, max(c0[0], c1[0]) + 16),
        xticks=[],
        yticks=[],
    )
    ax.annotate(
        "",
        xy=c0[::-1],
        xytext=(880, 120),
        arrowprops={"arrowstyle": "-", "ls": "--", "color": "#8a9aaa", "lw": 0.7},
    )
    inset.spines[:].set_color("#b3bdc6")
    inset.set_title(f"{row['com_shift_mm']:.1f} mm shift", fontsize=7.4, pad=3)

    ax = fig.add_subplot(grid[1, 2])
    ax.set_title("e   Whole-object mass", loc="left", fontsize=9, fontweight="bold")
    mass = [row[key]["mass_kg"] for key in ("before", "after")]
    ax.bar([0, 1], mass, width=0.53, color=[WOOD, ALUMINUM])
    for i, value in enumerate(mass):
        ax.text(i, value + 0.85, f"{value:.2f}", ha="center", fontsize=8)
    ax.set(
        xticks=(0, 1),
        xticklabels=("Before", "After"),
        ylabel="Mass (kg)",
        ylim=(0, max(mass) * 1.21),
    )
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", color="#e8edf0", linewidth=0.6)
    ax.set_axisbelow(True)
    fig.text(
        0.075,
        0.968,
        "Upholstered chaise lounge · local material edit",
        fontsize=13,
        weight="bold",
        va="top",
    )
    fig.text(
        0.075,
        0.035,
        "Fixed mesh and occupancy  ·  Whole-object estimates from construction hypothesis 1",
        size=7.1,
        color="#526171",
    )
    for extension in ("png", "pdf", "svg", "tiff"):
        options = (
            {"pil_kwargs": {"compression": "tiff_lzw"}} if extension == "tiff" else {}
        )
        fig.savefig(
            output / f"material_edit.{extension}",
            dpi=600 if extension == "tiff" else 360,
            **options,
        )
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "construction", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.construction / "protocol.json")
    state = args.reference / protocol.get(
        "part_state_name", "part_state_geometry_constrained.json"
    )
    result = replay(args.construction, state)
    geometry = args.reference / "reference_arrays.npz"
    if digest(geometry) != result["geometry_sha256"]:
        raise ValueError("Reference mesh changed since part assignment")
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(geometry) as arrays:
        make_figure(arrays, args.reference / "observation_00.png", result, args.output)
    (args.output / "material_edit_data.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
