"""Recover missing material support from nearby appearance-matched source pixels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from prepare_abo50_benchmark import digest, read_json, write_json
from replay_abo50_material_coverage import assign
from run_abo50_density_maps import CASES, SIZE
from tactile3d.vision.part_lifting import add_view_votes

VIEWS = (0, 10, 20)
COLOR_RADIUS = 0.18
SPATIAL_RADIUS = 0.14
COMPETITOR_RATIO = 1.25


def fill_foreground(rgb, foreground, seed, material_count):
    ys, xs = np.nonzero(foreground)
    if not len(xs):
        raise ValueError("Empty calibrated foreground")
    span = max(np.ptp(xs), np.ptp(ys), 1)
    step = max(span * SPATIAL_RADIUS, 1)
    features = np.column_stack(
        (ys / step, xs / step, rgb[ys, xs] / (255 * COLOR_RADIUS))
    )
    known = seed[ys, xs]
    missing = known == 0
    if not missing.any():
        return seed.copy()
    distance = np.full((material_count, int(missing.sum())), np.inf)
    for index in range(1, material_count + 1):
        sample = known == index
        if sample.any():
            distance[index - 1] = cKDTree(features[sample]).query(
                features[missing], distance_upper_bound=1.25
            )[0]
    primary = distance.min(axis=0)
    secondary = (
        np.partition(distance, 1, axis=0)[1]
        if material_count > 1
        else np.full_like(primary, np.inf)
    )
    selected = (primary <= 1) & (secondary > COMPETITOR_RATIO * primary)
    output = seed.copy()
    output[ys[missing][selected], xs[missing][selected]] = (
        distance[:, selected].argmin(axis=0) + 1
    )
    return output


def complete_by_mesh_appearance(mesh, labels):
    """Extend accepted face labels within a conservative 3D appearance neighborhood."""
    if mesh.visual.kind != "vertex":
        raise ValueError(
            "3D appearance completion requires reconstructed vertex colors"
        )
    unknown = np.flatnonzero(labels == 0)
    known_materials = np.unique(labels[labels > 0])
    if not len(unknown) or not len(known_materials):
        return labels.copy()
    colors = mesh.visual.vertex_colors[mesh.faces, :3].mean(axis=1) / 255.0
    features = np.column_stack(
        (mesh.triangles_center / (0.20 * mesh.extents.max()), colors / COLOR_RADIUS)
    )
    costs = np.array(
        [
            cKDTree(features[labels == category]).query(
                features[unknown], distance_upper_bound=1.25
            )[0]
            for category in known_materials
        ]
    )
    first = costs.min(axis=0)
    second = (
        np.partition(costs, 1, axis=0)[1]
        if len(known_materials) > 1
        else np.full_like(first, np.inf)
    )
    replace = (first <= 1) & (second > COMPETITOR_RATIO * first)
    output = labels.copy()
    output[unknown[replace]] = known_materials[costs[:, replace].argmin(axis=0)]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--complete-mesh", action="store_true")
    args = parser.parse_args()
    args.output.mkdir()
    meshes = read_json(args.source / "meshes.json")
    write_json(args.output / "meshes.json", meshes)
    manifest = read_json(args.observations / "cohort.json")
    reports = []
    density_reports = {
        row["scene_id"]: row
        for row in read_json(args.source / "density_manifest.json")["scenes"]
    }
    for scene in CASES:
        assets = args.source / "surface" / "fields" / scene
        tokens = read_json(assets / "part_tokens.json")
        mesh_path = Path(meshes[scene]["path"])
        if digest(mesh_path) != tokens["mesh_sha256"]:
            raise ValueError("Original mesh does not match the source masks")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        materials = sorted({row["surface_material"] for row in tokens["parts"]})
        indices = {name: i + 1 for i, name in enumerate(materials)}
        part_to_material = np.zeros(len(tokens["parts"]) + 1, dtype=int)
        for row in tokens["parts"]:
            part_to_material[row["part_index"]] = indices[row["surface_material"]]
        grouped_tokens = {
            **tokens,
            "parts": [
                {"part_index": i + 1, "surface_material": name}
                for i, name in enumerate(materials)
            ],
        }
        source = args.observations / "scenes" / scene
        camera_file = source / "transforms.json"
        if (
            digest(camera_file)
            != manifest["files_sha256"][f"scenes/{scene}/transforms.json"]
        ):
            raise ValueError("Camera changed since the benchmark was frozen")
        camera = read_json(camera_file)
        votes = np.zeros((len(mesh.faces), len(materials) + 1))
        source_coverage = []
        for view in VIEWS:
            with np.load(assets / f"view_{view:02d}.npz") as original:
                seed = part_to_material[original["labels"]]
                face_ids = original["face_ids"]
            frame = camera["frames"][view]
            path = source / frame["file_path"]
            if (
                digest(path)
                != manifest["files_sha256"][f"scenes/{scene}/{frame['file_path']}"]
            ):
                raise ValueError(f"Image {view} changed since the benchmark was frozen")
            with Image.open(path) as image:
                rgba = image.convert("RGBA").resize((SIZE, SIZE))
                foreground = np.asarray(rgba.getchannel("A")) >= 128
                rgb = np.asarray(rgba.convert("RGB"))
            completed = fill_foreground(rgb, foreground, seed, len(materials))
            source_coverage.append(
                {
                    "view": view,
                    "original": float(
                        np.count_nonzero((seed > 0) & foreground) / foreground.sum()
                    ),
                    "filled": float(
                        np.count_nonzero((completed > 0) & foreground)
                        / foreground.sum()
                    ),
                }
            )
            add_view_votes(
                votes,
                mesh,
                np.asarray(frame["transform_matrix"]),
                camera,
                completed,
                cached_face_ids=face_ids,
            )
        _, _, labels = assign(mesh, grouped_tokens, votes, abstain_unknown=True)
        if args.complete_mesh:
            labels = complete_by_mesh_appearance(mesh, labels)
        with np.load(args.source / f"{scene}.npz") as target_grid:
            maps = dict(target_grid)
        fids = maps["face_ids"]
        foreground = maps["foreground"]
        valid = (fids >= 0) & foreground
        density = np.array(
            [
                np.nan,
                *[
                    tokens["material_priors"][name]["density_mean_kg_m3"]
                    for name in materials
                ],
            ]
        )
        updated = np.full(foreground.shape, np.nan)
        updated[valid] = density[labels[fids[valid]]]
        maps["propagated"] = updated
        np.savez_compressed(args.output / f"{scene}.npz", **maps)
        final_coverage = float(np.isfinite(updated).sum() / foreground.sum())
        density_reports[scene]["propagated_foreground_coverage"] = final_coverage
        reports.append(
            {
                "scene_id": scene,
                "view_08_density_coverage": final_coverage,
                "material_source_view_coverage": source_coverage,
            }
        )
        field = args.output / "surface" / "fields" / scene
        field.mkdir(parents=True)
        write_json(field / "part_tokens.json", grouped_tokens)
        np.savez_compressed(field / "surface_fields.npz", labels=labels)
    write_json(
        args.output / "report.json",
        {
            "scenes": reports,
            "input_views": VIEWS,
            "color_radius": COLOR_RADIUS,
            "spatial_radius": SPATIAL_RADIUS,
            "competitor_ratio": COMPETITOR_RATIO,
            "mesh_appearance_completion": args.complete_mesh,
        },
    )
    write_json(
        args.output / "density_manifest.json",
        {
            "scenes": [density_reports[scene] for scene in CASES],
            "units": "kg/m^3",
            "density_variant_label": (
                "Ours · appearance completion"
                if args.complete_mesh
                else "Ours · input mask completion"
            ),
            "density_variant_filename": (
                "appearance_completion" if args.complete_mesh else "mask_completion"
            ),
        },
    )


if __name__ == "__main__":
    main()
