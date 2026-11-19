"""Run and visualize spatial material allocation and an MPM integration pilot."""

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from tactile3d.experiments.bound_material_demo import BOXES, PITCH_M
from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.mesh_particles import Construction, sample_mesh_particles
from tactile3d.physics.mpm_validation import coarsen_material_points, run_mpm
from tactile3d.physics.part_io import file_sha256

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"fabric": "#526D96", "foam": "#65AD9F", "wood": "#B96B30"}
POISSON = {"fabric": 0.33, "foam": 0.25, "wood": 0.30}


def sofa_hypothesis(mesh, proposal):
    parts = proposal["parts"]
    shell = [
        p for p in parts if p["material"] == "fabric" and p["construction"] == "shell"
    ]
    foam = [
        p for p in parts if p["material"] == "foam" and p["construction"] == "solid"
    ]
    frame = [
        p for p in parts if p["name"] == "internal frame" and p["material"] == "wood"
    ]
    legs = [
        p
        for p in parts
        if "leg" in p["name"]
        and p["material"] == "wood"
        and p["construction"] == "solid"
        and p["evidence"].lower().startswith("visible")
    ]
    if len(shell) != 1 or not foam or len(frame) != 1 or not legs:
        raise ValueError(
            "Expected separate fabric, foam and inferred timber frame evidence"
        )
    lo, hi = mesh.bounds
    support_top = None
    for z in np.arange(lo[2] + 0.01, lo[2] + 0.35 * mesh.extents[2], 0.0025):
        section = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        if section is not None and any(
            np.all(np.ptp(loop[:, :2], axis=0) > [0.8, 0.5] * mesh.extents[:2])
            for loop in section.discrete
        ):
            support_top = float(z)
            break
    if support_top is None:
        raise ValueError(
            "Upholstered body could not be separated from visible wooden legs"
        )
    width = frame[0]["thickness_mm"] / 1000
    x0, x1 = lo[0] + 0.12 * mesh.extents[0], hi[0] - 0.12 * mesh.extents[0]
    y0, y1 = lo[1] + 0.23 * mesh.extents[1], hi[1] - 0.17 * mesh.extents[1]
    z0, z1 = lo[2] + 0.24 * mesh.extents[2], lo[2] + 0.33 * mesh.extents[2]
    rails = (
        ((x0, y0, z0), (x1, y0 + width, z1)),
        ((x0, y1 - width, z0), (x1, y1, z1)),
        ((x0, y0, z0), (x0 + width, y1, z1)),
        ((x1 - width, y0, z0), (x1, y1, z1)),
    )
    return Construction.from_evidence(
        surface="fabric",
        solid=False,
        core="foam",
        thickness_m=shell[0]["thickness_mm"] / 1000,
        frame="wood",
        frame_boxes=rails,
        support_top_m=support_top,
    )


def stool_fixture():
    meshes = []
    for _, lower, count in BOXES:
        size = np.array(count) * PITCH_M
        meshes.append(
            trimesh.creation.box(
                size,
                transform=trimesh.transformations.translation_matrix(lower + size / 2),
            )
        )
    return trimesh.boolean.union(meshes, engine="manifold")


def serial_metrics(result):
    return {key: val for key, val in result.items() if np.isscalar(val)} | {
        "min_J": float(result["jacobian"].min()),
        "max_J": float(result["jacobian"].max()),
    }


def run_case(name, mesh, layout, priors, output, finest_m):
    fine = sample_mesh_particles(
        mesh, layout, priors, finest_m=finest_m, bulk_m=0.035, poisson=POISSON
    )
    coarse = sample_mesh_particles(
        mesh, layout, priors, finest_m=finest_m * 2, bulk_m=0.035, poisson=POISSON
    )
    np.savez_compressed(output / f"{name}_quadrature.npz", **fine)
    pitch = max(mesh.extents.max() / 26, 0.018)
    grid = pitch * 2
    export = output / f"{name}_mpm_input.npz"
    points = coarsen_material_points(fine, pitch)
    # Exact mesh volume is available for a uniform solid, including thin box legs.
    normalization = (
        float(mesh.volume / fine["volume_m3"].sum()) if layout.thickness_m == 0 else 1.0
    )
    points["volume_m3"] *= normalization
    points["mass_kg"] *= normalization
    np.savez_compressed(export, **points)
    with np.load(export, allow_pickle=False) as loaded:
        points = dict(loaded)
    if not np.allclose(
        np.bincount(points["material_id"], weights=points["mass_kg"]),
        np.bincount(fine["material_id"], weights=fine["mass_kg"]) * normalization,
        rtol=1e-12,
    ):
        raise ValueError("MPM export changed a material's mass")
    fall = run_mpm(points, dx=grid, steps=12)
    load = run_mpm(points, dx=grid, steps=120, strain_rate=10)
    ablation = None
    if layout.frame:
        # Suppress stiffness contrast at fixed particle masses, locations and timestep.
        uniform = points | {
            "youngs_pa": np.full_like(points["youngs_pa"], min(points["youngs_pa"])),
            "poisson": np.full_like(points["poisson"], POISSON["foam"]),
        }
        ablation = serial_metrics(
            run_mpm(
                uniform, dx=grid, steps=120, strain_rate=10, timestep_s=load["dt_s"]
            )
        )
    if (
        fall["max_deviation_from_ballistic_m"] > 1e-10
        or load["max_p2g_mass_residual_kg"] > 1e-10 * points["mass_kg"].sum()
        or load["elastic_energy_j"] <= 0
    ):
        raise ValueError("MPM acceptance conditions failed")
    np.savez_compressed(
        output / f"{name}_mpm_response.npz",
        position_m=load["position_m"],
        jacobian=load["jacobian"],
    )
    return (
        fine,
        points,
        load,
        {
            "quadrature_n": len(fine["position_m"]),
            "mpm_n": len(points["position_m"]),
            "mpm_mass_kg": float(points["mass_kg"].sum()),
            "mpm_volume_normalization": normalization,
            "mesh_volume_m3": float(mesh.volume),
            "mpm_grid_m": grid,
            "quadrature_volume_m3": float(fine["volume_m3"].sum()),
            "mesh_volume_relative_error": float(
                fine["volume_m3"].sum() / mesh.volume - 1
            ),
            "coarse_volume_m3": float(coarse["volume_m3"].sum()),
            "material_volume_m3": dict(
                zip(
                    fine["material_names"],
                    np.bincount(fine["material_id"], weights=fine["volume_m3"]),
                )
            ),
            "coarse_material_volume_m3": dict(
                zip(
                    coarse["material_names"],
                    np.bincount(coarse["material_id"], weights=coarse["volume_m3"]),
                )
            ),
            "material_mass_kg": dict(
                zip(
                    fine["material_names"],
                    np.bincount(fine["material_id"], weights=fine["mass_kg"]),
                )
            ),
            "free_fall": serial_metrics(fall),
            "compression": serial_metrics(load),
            "uniform_foam_stiffness_ablation": ablation,
        },
    )


def draw_section(ax, mesh, fine, axis, value):
    other = [i for i in range(3) if i != axis]
    pos = fine["position_m"]
    sizes = np.cbrt(fine["volume_m3"])
    section = (pos[:, axis] - sizes / 2 <= value) & (value < pos[:, axis] + sizes / 2)
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    patches = pos[section][:, None, other] + sizes[section, None, None] * corners
    colors = [
        COLORS[label] for label in fine["material_names"][fine["material_id"][section]]
    ]
    ax.add_collection(
        PolyCollection(
            patches,
            facecolors=colors,
            edgecolors="face",
            linewidths=0.15,
            rasterized=True,
        )
    )
    normal = np.eye(3)[axis]
    outline = mesh.section(plane_origin=normal * value, plane_normal=normal)
    if outline is not None:
        for segment in outline.discrete:
            ax.plot(*segment[:, other].T, color="#283746", lw=0.45)
    ax.autoscale_view()
    ax.set_aspect("equal")


def draw_points(ax, points, load=None):
    pos = points["position_m"]
    if load is not None:
        pos = pos + 35 * load["displacement_m"]
    for i, label in enumerate(points["material_names"]):
        take = points["material_id"] == i
        ax.scatter(
            *pos[take].T,
            s=1.3,
            color=COLORS[label],
            depthshade=False,
            rasterized=True,
            linewidths=0,
        )
    ax.view_init(elev=19, azim=-61)
    ax.set_box_aspect(np.ptp(points["position_m"], axis=0))
    original = points["position_m"]
    padding = 0.05 * np.ptp(original, axis=0)
    for setter, low, high in zip(
        (ax.set_xlim, ax.set_ylim, ax.set_zlim),
        original.min(0) - padding,
        original.max(0) + padding,
    ):
        setter(low, high)


def figure(output, rgb, meshes, results):
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(7.2, 4.4))
    grid = fig.add_gridspec(
        2, 4, left=0.09, right=0.99, bottom=0.12, top=0.9, wspace=0.13, hspace=0.29
    )
    titles = ("Input / reference", "Material cutaway", "MPM input", "After compression")
    for index, title in enumerate(titles):
        fig.text(0.195 + index * 0.233, 0.942, title, ha="center", weight="bold")
    for row, (name, mesh, (fine, points, load, _)) in enumerate(
        zip(("ABO sofa", "Solid wood\nstool fixture"), meshes, results)
    ):
        axes = [
            fig.add_subplot(
                grid[row, col],
                projection="3d" if col >= 2 or (row == 1 and col == 0) else None,
            )
            for col in range(4)
        ]
        if row == 0:
            fg = np.any(rgb[..., :3] < 235, axis=2)
            yy, xx = np.nonzero(fg)
            axes[0].imshow(rgb[yy.min() : yy.max() + 1, xx.min() : xx.max() + 1])
            draw_section(axes[1], mesh, fine, axis=0, value=mesh.bounds[:, 0].mean())
        else:
            axes[0].plot_trisurf(
                *mesh.vertices.T,
                triangles=mesh.faces,
                color=COLORS["wood"],
                shade=True,
                linewidth=0,
            )
            axes[0].view_init(elev=19, azim=-61)
            axes[0].set_box_aspect(np.array(mesh.extents, copy=True))
            draw_section(axes[1], mesh, fine, axis=1, value=-0.12)
        fig.text(
            0.017, 0.72 - row * 0.40, name, rotation=90, va="center", weight="bold"
        )
        draw_points(axes[2], points)
        draw_points(axes[3], points, load)
        for ax in axes:
            ax.set_axis_off()
        axes[3].text2D(
            0.5,
            -0.015,
            f"{1e3*load['duration_s']:.2f} ms · motion ×35",
            transform=axes[3].transAxes,
            fontsize=6.5,
            ha="center",
        )
    fig.legend(
        [
            Line2D([], [], color=c, marker="s", linestyle="None")
            for c in COLORS.values()
        ],
        ["Upholstery", "Foam", "Wood"],
        ncol=3,
        loc="lower center",
        frameon=False,
    )
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"material_points_mpm.{suffix}", dpi=350, facecolor="white"
        )
    plt.close(fig)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mesh", type=Path, required=True)
    cli.add_argument("--proposal", type=Path, required=True)
    cli.add_argument("--rgb-npz", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    priors = load_material_priors(ROOT / "configs/material_priors.yaml")
    proposal = json.loads(args.proposal.read_text(encoding="utf-8"))
    sofa = trimesh.load_mesh(args.mesh)
    sofa.apply_scale(proposal["max_length_m"] / sofa.extents.max())
    stool = stool_fixture()
    layouts = (
        sofa_hypothesis(sofa, proposal),
        Construction.from_evidence(surface="wood", solid=True),
    )
    names = ("sofa", "stool")
    meshes = (sofa, stool)
    results = []
    for name, mesh, layout, resolution in zip(names, meshes, layouts, (0.004, 0.001)):
        results.append(run_case(name, mesh, layout, priors, args.output, resolution))
        print(name, results[-1][-1], flush=True)
    report = {name: result[-1] for name, result in zip(names, results)}
    report["inputs"] = {
        key: file_sha256(getattr(args, key)) for key in ("mesh", "proposal", "rgb_npz")
    }
    report["sofa"]["construction"] = vars(layouts[0])
    report["stool"]["construction"] = vars(layouts[1])
    (args.output / "mpm_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with np.load(args.rgb_npz, allow_pickle=False) as rgb_data:
        figure(args.output, rgb_data["rgb"], meshes, results)


if __name__ == "__main__":
    main()
