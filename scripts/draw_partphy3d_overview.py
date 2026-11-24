"""Compose the method and region-evidence figures from recorded experiment arrays."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
from PIL import Image

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.figures.mesh_fields import render_mesh


INK, FABRIC, FOAM, WOOD, METAL = "#162535", "#377eaa", "#79ad8c", "#b88355", "#7667a7"
GRAY, VOID = "#bdc6cf", "#eceff2"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def show(ax, rgb, padding=5):
    rgb = np.asarray(rgb, dtype=float)
    rgb = rgb / 255 if rgb.max() > 1 else rgb
    if rgb.shape[2] == 4:
        rgb = rgb[..., :3] * rgb[..., 3:] + 1 - rgb[..., 3:]
    y, x = np.nonzero(np.any(rgb < 0.97, axis=-1))
    if len(y):
        ax.imshow(rgb, interpolation="nearest")
        ax.set(
            xlim=(x.min() - padding, x.max() + padding),
            ylim=(y.max() + padding, y.min() - padding),
        )
    ax.axis("off")


def save(fig, out):
    for fmt in ("png", "pdf", "svg"):
        p = out.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=360, facecolor="white")
        if fmt == "svg":
            p.write_text(
                "\n".join(
                    line.rstrip() for line in p.read_text(encoding="utf-8").splitlines()
                )
                + "\n",
                encoding="utf-8",
            )
    plt.close(fig)


def mesh_picture(vertices, faces, ids, colors, camera, only=None):
    if only is not None:
        keep = ids == only
        faces, colors = faces[keep], colors[keep]
        retained, new_ids = np.unique(faces, return_inverse=True)
        vertices, faces = vertices[retained], new_ids.reshape(-1, 3)
    return render_mesh(vertices, faces, colors, camera, 640)


def pipeline(args):
    edit = read(args.edit_data)
    trial = edit["trials"][0]
    construction = read(args.construction)
    seat = next(
        part for part in construction["answer"]["upholstery"] if part["role"] == "seat"
    )
    phases = [seat["foam_fraction"], seat["hidden_wood_fraction"]]
    phases.append(1 - sum(phases))
    regions = {
        0: "#4b6f9a",
        1: FABRIC,
        2: "#647aac",
        3: "#478f91",
        4: "#478f91",
        5: "#647aac",
    }
    with np.load(args.reference) as data:
        vertices, faces, ids = (data[k] for k in ("vertices", "faces", "component_ids"))
    camera = edit["shared_camera_azimuth_deg"]
    region_colors = np.array([to_rgb(regions.get(i, WOOD)) for i in ids])
    fig = plt.figure(figsize=(7.2, 5.0))
    fig.text(
        0.056,
        0.968,
        "PartPhy3D links surface evidence to editable physics",
        va="top",
        fontsize=12.8,
        weight="bold",
    )
    fig.text(
        0.056,
        0.875,
        "Upholstered chaise lounge",
        fontsize=8.5,
        weight="bold",
        color="#476276",
    )
    top = fig.add_gridspec(
        1, 4, top=0.80, bottom=0.515, left=0.056, right=0.963, wspace=0.34
    )
    stages = [fig.add_subplot(top[i]) for i in range(4)]
    for i, (ax, label) in enumerate(
        zip(
            stages,
            (
                "Multi-view RGB",
                "Bound mesh parts",
                "Seat material domain",
                "Local material edit",
            ),
        )
    ):
        ax.set_title(
            f"{'abcd'[i]}   {label}", loc="left", fontsize=8, weight="bold", pad=5
        )
        ax.axis("off")
        if i:
            fig.text(
                ax.get_position().x0 - 0.022,
                0.686,
                "→",
                fontsize=11,
                ha="center",
                color="#6a7c8b",
            )
    for file, rect in zip(
        args.views,
        ((0.09, 0.31, 0.84, 0.66), (0.06, 0.01, 0.43, 0.32), (0.52, 0.01, 0.43, 0.32)),
        strict=True,
    ):
        with Image.open(file) as img:
            show(stages[0].inset_axes(rect), np.asarray(img.convert("RGBA")))
    image = mesh_picture(vertices, faces, ids, region_colors, camera)
    show(stages[1].inset_axes((0, 0.23, 1, 0.74)), image)
    stages[1].text(
        0.50,
        0.15,
        f"{len(np.unique(ids))} mesh components",
        ha="center",
        fontsize=7,
        transform=stages[1].transAxes,
    )

    show(
        stages[2].inset_axes((0.08, 0.44, 0.84, 0.50)),
        mesh_picture(vertices, faces, ids, region_colors, camera, 1),
    )
    ax = stages[2].inset_axes((0.10, 0.19, 0.82, 0.18))
    start = 0
    for fraction, color in zip(phases, (FOAM, WOOD, VOID), strict=True):
        ax.barh([0], [fraction], left=start, color=color, height=0.52)
        start += fraction
    ax.set(xlim=(0, 1), ylim=(-0.44, 0.44))
    ax.axis("off")
    stages[2].text(
        0.5,
        0.39,
        "Fabric shell · 3.5 mm",
        ha="center",
        fontsize=6.8,
        transform=stages[2].transAxes,
    )
    stages[2].text(
        0.5,
        0.10,
        "Core volume fractions",
        ha="center",
        fontsize=6.7,
        transform=stages[2].transAxes,
    )
    stages[2].text(
        0.5,
        -0.005,
        " / ".join(f"{v:.0%}" for v in phases),
        ha="center",
        color="#476276",
        fontsize=6.7,
        transform=stages[2].transAxes,
    )
    for i, color in enumerate((WOOD, "#197f87")):
        leg_colors = np.tile(to_rgb(color), (len(faces), 1))
        show(
            stages[3].inset_axes((0.08 + 0.48 * i, 0.40, 0.34, 0.53)),
            mesh_picture(
                vertices, faces, ids, leg_colors, camera, edit["component_id"]
            ),
        )
    stages[3].text(
        0.50, 0.65, "→", ha="center", fontsize=11, transform=stages[3].transAxes
    )
    stages[3].text(
        0.5,
        0.28,
        "Wood → aluminum",
        ha="center",
        fontsize=7,
        transform=stages[3].transAxes,
    )
    stages[3].text(
        0.5,
        0.09,
        "%+.2f kg  ·  %.1f mm CoM" % (trial["delta_mass_kg"], trial["com_shift_mm"]),
        ha="center",
        fontsize=7,
        transform=stages[3].transAxes,
    )

    fig.text(
        0.056, 0.453, "Striped cushion", fontsize=8.5, weight="bold", color="#476276"
    )
    fig.legend(
        handles=[
            Patch(color=FOAM, label="Foam"),
            Patch(color=WOOD, label="Wood"),
            Patch(color=VOID, label="Void"),
        ],
        loc="center",
        bbox_to_anchor=(0.64, 0.469),
        ncols=3,
        frameon=False,
        fontsize=6,
        handlelength=0.75,
        columnspacing=1,
    )
    lower = fig.add_gridspec(
        1,
        3,
        top=0.38,
        bottom=0.13,
        left=0.074,
        right=0.95,
        wspace=0.37,
        width_ratios=(1, 1, 1.18),
    )
    construction_ax, section, response = [fig.add_subplot(lower[i]) for i in range(3)]
    with np.load(args.particles) as data:
        pos, material = data["position_m"], data["material_names"][data["material_id"]]
        volume = data["volume_m3"]
        times = data["time_ms"]
        heights = np.ptp(data["trajectory_m"][:, :, 2], axis=1) * 1000
    points = (pos - pos.min(axis=0)) * 1000
    table = {m: float(volume[material == m].sum()) for m in ("fabric", "foam")}
    projects = np.column_stack(
        (points[:, 0] - 0.31 * points[:, 1], points[:, 2] + 0.26 * points[:, 1])
    )
    order = points[:, 1].argsort()[::-1]
    color = np.where(material == "fabric", FABRIC, FOAM)
    construction_ax.scatter(
        *projects[order].T, s=0.60, c=color[order], lw=0, alpha=0.75, rasterized=True
    )
    take = np.abs(points[:, 1] - np.median(points[:, 1])) < 17
    section.scatter(
        points[take, 0], points[take, 2], s=1.35, c=color[take], lw=0, rasterized=True
    )
    for ax, title in (
        (construction_ax, "e   Adaptive particles"),
        (section, "f   Material-point section"),
    ):
        fig.text(ax.get_position().x0, 0.393, title, weight="bold", fontsize=8)
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.text(0.17, 0.158, f"{len(points):,} material points", ha="center", fontsize=6.8)
    fig.legend(
        handles=[Patch(color=FABRIC, label="Fabric"), Patch(color=FOAM, label="Foam")],
        loc="lower center",
        bbox_to_anchor=(0.485, 0.146),
        ncols=2,
        fontsize=6,
        frameon=False,
        handlelength=0.8,
    )
    response.plot(times, heights - heights[0], lw=1.2, color="#197f87")
    response.axhline(0, color="#c6d0d7", lw=0.65)
    low = heights.argmin()
    response.scatter(
        times[low], heights[low] - heights[0], s=16, color="#197f87", zorder=5
    )
    response.set(xlabel="Time (ms)", ylabel="Height change (mm)")
    fig.text(
        response.get_position().x0,
        0.393,
        "g   MLS-MPM impact",
        weight="bold",
        fontsize=8,
    )
    response.spines[["top", "right"]].set_visible(False)
    response.tick_params(labelsize=6.6)
    fig.text(
        0.056,
        0.035,
        "Recorded reference-object replay above  ·  Recorded cushion solver output below",
        size=6.8,
        color="#526171",
    )
    save(fig, args.output / "partphy3d_pipeline")
    return {
        "seat_core_fractions": dict(zip(("foam", "wood", "void"), phases)),
        "cushion_particle_count": len(pos),
        "represented_volume_m3": table,
        "time_ms": times.tolist(),
        "height_change_mm": (heights - heights[0]).tolist(),
    }


def material_grid(args):
    palette = {
        "unknown": GRAY,
        "fabric": FABRIC,
        "wood": WOOD,
        "steel": METAL,
        "metal": METAL,
        "foam": FOAM,
    }
    materials = (
        ("Fabric", FABRIC),
        ("Wood", WOOD),
        ("Metal", METAL),
        ("Foam", FOAM),
        ("Unassigned", GRAY),
    )
    names = ("Metal shoe rack", "Patterned accent chair", "Blue accent chair")
    fig, axes = plt.subplots(
        5, 3, figsize=(7.2, 6.6), gridspec_kw={"height_ratios": (1, 1, 1, 1, 1)}
    )
    fig.subplots_adjust(
        left=0.207, right=0.965, top=0.86, bottom=0.12, wspace=0.13, hspace=0.13
    )
    fig.text(
        0.06,
        0.968,
        "Region evidence and material segmentation",
        weight="bold",
        size=13,
        va="top",
    )
    for i, (title, name) in enumerate(
        zip(
            ("RGB input", "Face evidence", "PUGS–Qwen", "PartPhy3D", "PhysSeg target"),
            "abcde",
        )
    ):
        bounds = axes[i, 0].get_position()
        fig.text(
            0.068,
            (bounds.y0 + bounds.y1) / 2,
            f"{name}\n{title}",
            ha="center",
            va="center",
            size=7.4,
            weight="bold",
        )
    for column, (record, title) in enumerate(
        zip(read(args.region_data / "manifest.json"), names, strict=True)
    ):
        stem = f"{record['scene_id']}_{record['view']:02d}.npz"
        with np.load(args.region_data / stem) as data:
            rgba = data["rgba"]
            fg = rgba[:, :, 3] >= 128
            layers = [rgba, data["evidence"]]
            for key in ("pugs", "ours"):
                colors = np.array(
                    [to_rgb(palette[n]) for n in ("unknown", *data[f"{key}_names"])]
                )
                layer = np.ones(fg.shape + (3,))
                layer[fg] = colors[data[key][fg]]
                layers.append(layer)
        annotation = args.annotations / stem
        if annotation.is_file():
            with np.load(annotation) as reference:
                if np.mean(fg == (reference["rgba"][:, :, 3] >= 128)) < 0.995:
                    raise ValueError(
                        "Material annotation is not registered to the displayed view"
                    )
                target = reference["target"]
            colors = np.array([to_rgb(c) for c in ("white", FABRIC, WOOD)])
            layers.append(colors[target])
        y, x = np.nonzero(fg)
        for row, ax in enumerate(axes[:, column]):
            if row < len(layers):
                ax.imshow(layers[row], interpolation="nearest")
                ax.set(xlim=(x.min() - 7, x.max() + 7), ylim=(y.max() + 7, y.min() - 7))
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No annotation",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    size=7,
                    color="#607080",
                )
            ax.axis("off")
        axes[0, column].set_title(title, size=8, weight="bold", pad=7)
    fig.legend(
        handles=[Patch(color=c, label=n) for n, c in materials],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.075),
        ncols=5,
        frameon=False,
        fontsize=7,
        handlelength=1,
    )
    fig.text(
        0.06,
        0.032,
        "Face evidence shows projected multi-view votes in PCA colors. Material colors are shared across methods.",
        size=6.8,
        color="#526171",
    )
    save(fig, args.output / "region_material_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "reference",
        "edit-data",
        "construction",
        "particles",
        "region-data",
        "annotations",
        "output",
    ):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--views", nargs=3, type=Path, required=True)
    args = parser.parse_args()
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "text.color": INK,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    data = pipeline(args)
    material_grid(args)
    (args.output / "partphy3d_pipeline_data.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
