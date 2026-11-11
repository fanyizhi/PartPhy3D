"""Three/four actual-view comparison and measurable geometric contact panels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image
from run_reference_mesh_study import SCENE, load_prepared, savefig
from scipy.spatial import ConvexHull

from tactile3d.figures.bound_material import ELEVATION_DEGREES, MARGIN_RATIO
from tactile3d.figures.mesh_fields import render_mesh
from tactile3d.mesh.contacts import closest_surface_pair
from tactile3d.physics.part_io import write_json
from tactile3d.vision.view_selection import select_azimuth_views


def project(vertices, points, azimuth, resolution=600):
    az, el = np.deg2rad([azimuth, ELEVATION_DEGREES])
    camera = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.0])
    up = np.cross(camera, right)
    center = (vertices.min(0) + vertices.max(0)) / 2
    plane = (vertices - center) @ np.column_stack((right, up))
    uv = (points - center) @ np.column_stack((right, up))
    uv *= resolution * (1 - 2 * MARGIN_RATIO) / np.ptp(plane, axis=0).max()
    uv[:, 1] *= -1
    return uv + resolution / 2


def prepare(args):
    out = args.source / "views_contacts"
    out.mkdir(exist_ok=False)
    mesh, _labels, pieces, texture = load_prepared(args.source)
    centers = np.array(
        [np.average(p.triangles_center, weights=p.area_faces, axis=0) for p in pieces]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    for ax, az in zip(axes, [55, -125]):
        ax.imshow(
            render_mesh(mesh.vertices, mesh.faces, texture, azimuth=az, resolution=600)
        )
        uv = project(mesh.vertices, centers, az)
        for side in (0, 1):
            ids = np.flatnonzero((uv[:, 0] >= 300) == bool(side))
            ids = ids[np.argsort(uv[ids, 1])]
            for i, y in zip(ids, np.linspace(70, 530, len(ids))):
                ax.annotate(
                    str(i),
                    xy=uv[i],
                    xytext=(20 if side == 0 else 580, y),
                    fontsize=12,
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "circle,pad=0.2", "fc": "white", "ec": "#426980"},
                    arrowprops={"arrowstyle": "-", "color": "#426980", "lw": 0.8},
                )
        ax.axis("off")
        ax.set_title(f"Reference component positions · azimuth {az}°")
    fig.savefig(out / "global_component_positions.png", dpi=110, facecolor="white")
    plt.close(fig)
    camera = json.loads((args.root / "scenes" / SCENE / "transforms.json").read_text())
    poses = np.array([f["transform_matrix"] for f in camera["frames"]])
    selections = {}
    for count in (3, 4):
        selection = select_azimuth_views(poses, count)
        selections[str(count)] = selection
        fig, axes = plt.subplots(
            1, count, figsize=(3 * count, 3.2), layout="constrained"
        )
        for row, ax in zip(selection, axes):
            frame = camera["frames"][row["frame_index"]]
            image = Image.open(
                args.root / "scenes" / SCENE / frame["file_path"]
            ).convert("RGBA")
            rgb = Image.alpha_composite(
                Image.new("RGBA", image.size, "white"), image
            ).convert("RGB")
            rgb.save(out / f"view_{row['frame_index']:02d}.png")
            ax.imshow(rgb)
            ax.set_title(
                f"View {row['frame_index']:02d} · az {row['actual_azimuth_degrees']:.0f}°"
            )
            ax.axis("off")
        fig.savefig(out / f"selected_{count}_views.png", dpi=140)
        plt.close(fig)
    write_json(out / "view_selection.json", selections)
    write_json(
        out / "role_review.json",
        {
            "scope": "single-reviewer reference-component functional roles",
            "roles": {
                str(i): role
                for i, role in enumerate(
                    [
                        "cushion",
                        "seat",
                        "backrest",
                        "armrest",
                        "armrest",
                        "backrest",
                        "leg",
                        "leg",
                        "leg",
                        "leg",
                        "frame_rail",
                        "frame_rail",
                        "frame_rail",
                        "frame_rail",
                        "underside",
                    ]
                )
            },
            "material_review": "configs/evidence/chaise_surface_material_review.json",
            "interior_ground_truth": False,
        },
    )


def run(args):
    from tactile3d.vision.local_vlm import LocalVLM
    from tactile3d.vision.vlm_interior import _parse_response_content

    out = args.source / "views_contacts"
    selections = json.loads((out / "view_selection.json").read_text())
    roles = json.loads((out / "role_review.json").read_text())["roles"]
    material_review = json.loads(
        Path("configs/evidence/chaise_surface_material_review.json").read_text()
    )["component_materials"]
    model = LocalVLM(args.model, max_tokens=1400, max_pixels=768**2, quantized=True)
    write_json(out / "view_model.json", model.provenance)
    prompt = (
        "Identify numbered components of this chaise. The first images are actual calibrated "
        "observations. The final two images are an isolated numbered-component sheet and a "
        "whole-reference-mesh position diagram; numbers in these diagrams denote the SAME "
        "components. Use spatial position and shape to distinguish the loose back cushion "
        "from the long horizontal seat. Do not name all fabric panels headrests. "
        "Classify all 15 IDs exactly once. Functional role vocabulary is "
        "cushion, seat, backrest, armrest, leg, frame_rail, underside, unknown. "
        "Material vocabulary is wood, fabric, metal, plastic, unknown. "
        "Return strict JSON {components:[{component_id:integer,functional_role:string,"
        "surface_material:string,evidence:string}]}. Describe visible evidence briefly."
    )
    records = []
    for trial in range(3):
        for count in (3, 4):
            paths = [
                out / f"view_{row['frame_index']:02d}.png"
                for row in selections[str(count)]
            ]
            paths += [
                args.source / "component_sheet.png",
                out / "global_component_positions.png",
            ]
            response = model.generate(
                prompt,
                [Image.open(p).convert("RGB") for p in paths],
                "Assign component roles from observed geometry and context; output JSON.",
                seed=1201 + trial,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            record = {
                "trial": trial,
                "view_count": count,
                "prompt": prompt,
                "images": [p.name for p in paths],
                "response": asdict(response),
                "valid": False,
            }
            try:
                answer = _parse_response_content(response.text)
                rows = answer["components"]
                if sorted(c["component_id"] for c in rows) != list(range(15)):
                    raise ValueError("Missing or duplicated component IDs")
                prediction = {str(c["component_id"]): c for c in rows}
                record["answer"] = answer
                record["valid"] = True
                record["role_accuracy"] = sum(
                    prediction[i]["functional_role"] == role
                    for i, role in roles.items()
                ) / len(roles)
                known = {i: m for i, m in material_review.items() if m != "unknown"}
                record["material_accuracy"] = sum(
                    prediction[i]["surface_material"] == mat for i, mat in known.items()
                ) / len(known)
                record["wrong_role_ids"] = [
                    i
                    for i, role in roles.items()
                    if prediction[i]["functional_role"] != role
                ]
            except (KeyError, ValueError, TypeError) as error:
                record["error"] = str(error)
            write_json(out / f"view_count_{count}_trial_{trial}.json", record)
            records.append(record)
            print(
                f"{count} views trial {trial}: valid={record['valid']}, role={record.get('role_accuracy')}, {response.seconds:.1f}s",
                flush=True,
            )
    summary = []
    for count in (3, 4):
        selected = [r for r in records if r["view_count"] == count]
        summary.append(
            {
                "views": count,
                "trials": 3,
                "valid": sum(r["valid"] for r in selected),
                "role_accuracy": float(
                    np.mean([r.get("role_accuracy", 0) for r in selected])
                ),
                "material_accuracy": float(
                    np.mean([r.get("material_accuracy", 0) for r in selected])
                ),
                "mean_seconds": float(
                    np.mean([r["response"]["seconds"] for r in selected])
                ),
            }
        )
    winner = min(
        summary,
        key=lambda r: (
            -r["role_accuracy"],
            -r["material_accuracy"],
            -r["valid"],
            r["mean_seconds"],
        ),
    )
    write_json(
        out / "view_count_results.json",
        {
            "scope": "development role/material comparison with shared context diagrams",
            "summary": summary,
            "selected_views": winner["views"],
            "selection_rule": "role accuracy, material accuracy, validity, then runtime",
            "records": records,
        },
    )


def contacts(args):
    out = args.source / "views_contacts"
    mesh, _labels, pieces, texture = load_prepared(args.source)
    leg_ids = [6, 7, 8, 9]
    rail_ids = [10, 11, 12, 13]
    rails = trimesh.util.concatenate([pieces[i] for i in rail_ids])
    points, summaries, arrays = [], [], {}
    for index in leg_ids:
        leg = pieces[index]
        ground = leg.vertices[np.abs(leg.vertices[:, 2] - leg.bounds[0, 2]) < 1e-7]
        hull = ConvexHull(ground[:, :2])
        footprint = ground[hull.vertices]
        center = footprint.mean(0)
        points.append(center)
        band = leg.slice_plane([0, 0, leg.bounds[1, 2] - 0.05], [0, 0, 1], cap=False)
        samples, _ = trimesh.sample.sample_surface(band, 4096, seed=1400 + index)
        paired, distance, _ = trimesh.proximity.closest_point(rails, samples)
        nearest = int(np.argmin(distance))
        exact = closest_surface_pair(leg, rails)
        summaries.append(
            {
                "leg_component_id": index,
                "ground_contact_center_m": center.tolist(),
                "ground_footprint_area_m2": float(hull.volume),
                "ground_footprint_vertices_m": footprint.tolist(),
                "interface_sample_count": len(samples),
                "exact_surface_pair": exact,
                "interface_min_sample_distance_mm": float(distance.min() * 1000),
                "interface_p05_distance_mm": float(np.quantile(distance, 0.05) * 1000),
                "near_surface_fraction_1mm": float(np.mean(distance <= 0.001)),
                "near_surface_fraction_3mm": float(np.mean(distance <= 0.003)),
                "closest_sample_on_leg_m": samples[nearest].tolist(),
                "paired_point_on_frame_m": paired[nearest].tolist(),
            }
        )
        arrays[f"leg_{index}_samples"] = samples
        arrays[f"leg_{index}_paired_points"] = paired
        arrays[f"leg_{index}_distances_m"] = distance
    points = np.array(points)
    fig = plt.figure(figsize=(7.2, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(grid[:, 0])
    az = 55
    ax.imshow(
        render_mesh(mesh.vertices, mesh.faces, texture, azimuth=az, resolution=600)
    )
    projected = project(mesh.vertices, points, az)
    order = ConvexHull(points[:, :2]).vertices
    polygon = projected[np.r_[order, order[0]]]
    ax.plot(polygon[:, 0], polygon[:, 1], "--", color="#d28b45", lw=1.5)
    ax.scatter(
        projected[:, 0],
        projected[:, 1],
        s=35,
        c="#d05a42",
        edgecolors="white",
        zorder=4,
    )
    for uv, index in zip(projected, leg_ids):
        ax.annotate(
            f"Leg {index}",
            uv,
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    ax.axis("off")
    ax.set_title("a  Ground footprints\nand support polygon", fontsize=8)
    for k, index in enumerate(leg_ids):
        ax = fig.add_subplot(grid[k // 2, 1 + k % 2], projection="3d")
        leg = pieces[index]
        samples = arrays[f"leg_{index}_samples"]
        paired = arrays[f"leg_{index}_paired_points"]
        distance = arrays[f"leg_{index}_distances_m"]
        # Crop triangle centroids to the upper interface neighborhood.
        center = (leg.bounds[0] + leg.bounds[1]) / 2
        center[2] = leg.bounds[1, 2] - 0.02
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        for part, color in [(leg, "#bd8656"), (rails, "#8d9ba7")]:
            near = np.all(
                np.abs(part.triangles_center - center) < [0.11, 0.11, 0.08], axis=1
            )
            ax.add_collection3d(
                Poly3DCollection(
                    part.triangles[near], facecolor=color, edgecolor="none", alpha=0.23
                )
            )
        visible = distance <= 0.003
        chosen = np.flatnonzero(visible)[:: max(int(visible.sum() / 180), 1)]
        ax.scatter(
            *samples[chosen].T,
            s=3,
            c=distance[chosen] * 1000,
            cmap="viridis_r",
            vmin=0,
            vmax=3,
            depthshade=False,
        )
        exact = summaries[k]["exact_surface_pair"]
        line = np.array([exact["point_first_m"], exact["point_second_m"]])
        ax.plot(*line.T, color="#c4473d", lw=2)
        ax.scatter(*line.T, s=12, c="#c4473d", depthshade=False)
        ax.set(
            xlim=(center[0] - 0.07, center[0] + 0.07),
            ylim=(center[1] - 0.07, center[1] + 0.07),
            zlim=(center[2] - 0.05, center[2] + 0.05),
            xlabel="x (m)",
            ylabel="y (m)",
            zlabel="z (m)",
        )
        ax.set_title(
            f"{chr(98+k)}  Leg {index} — frame\nminimum surface gap {exact['distance_m']*1000:.4f} mm",
            fontsize=7,
        )
        ax.tick_params(labelsize=5, pad=0)
        ax.view_init(elev=24, azim=45 if center[0] < 0 else 135)
        ax.set_box_aspect([1, 1, 0.8])
    savefig(fig, out / "contact_geometry")
    np.savez_compressed(out / "contact_points.npz", **arrays)
    write_json(
        out / "contact_geometry.json",
        {
            "scope": "reference-mesh geometric proximity and horizontal-ground footprints",
            "reference_unit": "glTF meters, not laboratory calibration",
            "sampled_min_is_not_exact_surface_min": True,
            "frame_interfaces_force_or_bond_strength_evaluated": False,
            "ground_plane_z_m": 0.0,
            "legs": summaries,
        },
    )
    print(
        "Saved ground footprints and four measured geometric interface panels",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "run", "contacts"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/abo500_five"))
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"),
    )
    args = parser.parse_args()
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7})
    {"prepare": prepare, "run": run, "contacts": contacts}[args.stage](args)


if __name__ == "__main__":
    main()
