"""Render saved geometric contact coordinates with readable local mesh panels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from run_reference_mesh_study import load_prepared, savefig
from run_view_and_contact_study import project
from scipy.spatial import ConvexHull

from tactile3d.figures.mesh_fields import render_mesh
from tactile3d.mesh.assembly import crop_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.source / "views_contacts"
    mesh, _labels, pieces, texture = load_prepared(args.source)
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
    args.output.mkdir(parents=True, exist_ok=True)
    savefig(fig, args.output / "contact_geometry")


if __name__ == "__main__":
    main()
