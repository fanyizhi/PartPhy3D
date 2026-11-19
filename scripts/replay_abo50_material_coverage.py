"""Audit fixed material-group voting and local propagation against the original maps."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo50_density_maps import CASES
from tactile3d.vision.part_lifting import NORMAL_COSINE, finalize_votes


def assign(mesh, tokens, votes):
    parts = tokens["parts"]
    names = sorted({p["surface_material"] for p in parts})
    material_indices = {name: i + 1 for i, name in enumerate(names)}
    grouped = np.zeros((len(votes), len(names) + 1), dtype=float)
    for part in parts:
        grouped[:, material_indices[part["surface_material"]]] += votes[
            :, part["part_index"]
        ]
    labels, _, _ = finalize_votes(mesh, grouped)
    # One view-equivalent of positive evidence is required for a seed.
    labels[grouped[:, 1:].sum(axis=1) < 1] = 0
    edges = mesh.face_adjacency
    normal_dot = np.einsum(
        "ij,ij->i", mesh.face_normals[edges[:, 0]], mesh.face_normals[edges[:, 1]]
    )
    edges = edges[
        (normal_dot >= NORMAL_COSINE) & (votes.sum(axis=1)[edges] > 0).all(axis=1)
    ]
    rows, cols = edges.T
    distance = np.linalg.norm(
        mesh.triangles_center[rows] - mesh.triangles_center[cols], axis=1
    )
    limit = 0.03 * mesh.extents.max()
    graph = csr_matrix((distance, (rows, cols)), shape=(len(votes), len(votes)))
    costs = np.full((len(names), len(votes)), np.inf)
    for index in material_indices.values():
        seeds = np.flatnonzero(labels == index)
        if len(seeds):
            costs[index - 1] = dijkstra(
                graph, directed=False, indices=seeds, min_only=True, limit=limit
            )
    closest = costs.min(axis=0)
    competitor = (
        np.partition(costs, 1, axis=0)[1]
        if len(names) > 1
        else np.full(len(votes), np.inf)
    )
    fill = (labels == 0) & np.isfinite(closest) & (competitor > 1.5 * closest)
    propagated = labels.copy()
    propagated[fill] = costs[:, fill].argmin(axis=0) + 1
    return names, labels, propagated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    baseline = read_json(args.source / "density_manifest.json")
    meshes = read_json(args.source / "meshes.json")
    args.output.mkdir()
    write_json(args.output / "meshes.json", meshes)
    reports = []
    for scene, current in zip(CASES, baseline["scenes"], strict=True):
        if scene != current["scene_id"]:
            raise ValueError("Density manifest differs from frozen examples")
        mesh_path = Path(meshes[scene]["path"])
        if digest(mesh_path) != meshes[scene]["sha256"]:
            raise ValueError("Shared mesh changed")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        field = args.source / "surface" / "fields" / scene
        tokens = read_json(field / "part_tokens.json")
        if tokens["mesh_sha256"] != meshes[scene]["sha256"]:
            raise ValueError("Votes are bound to a different mesh")
        with np.load(field / "surface_fields.npz") as stored:
            names, grouped, propagated = assign(mesh, tokens, stored["votes"])
        density = np.array(
            [
                np.nan,
                *[tokens["material_priors"][n]["density_mean_kg_m3"] for n in names],
            ]
        )
        with np.load(args.source / f"{scene}.npz") as original:
            arrays = dict(original)
        foreground = arrays["foreground"]
        visible = foreground & (arrays["face_ids"] >= 0)
        for key, labels in (("grouped", grouped), ("propagated", propagated)):
            image = np.full(foreground.shape, np.nan)
            image[visible] = density[labels[arrays["face_ids"][visible]]]
            arrays[key] = image
        reports.append(
            {
                **current,
                "grouped_foreground_coverage": float(
                    np.isfinite(arrays["grouped"]).sum() / foreground.sum()
                ),
                "propagated_foreground_coverage": float(
                    np.isfinite(arrays["propagated"]).sum() / foreground.sum()
                ),
            }
        )
        np.savez_compressed(args.output / f"{scene}.npz", **arrays)
        field = args.output / "surface" / "fields" / scene
        field.mkdir(parents=True)
        write_json(
            field / "part_tokens.json",
            {
                **tokens,
                "parts": [
                    {"part_index": i + 1, "surface_material": name}
                    for i, name in enumerate(names)
                ],
            },
        )
        np.savez_compressed(field / "surface_fields.npz", labels=propagated)
    write_json(args.output / "density_manifest.json", {**baseline, "scenes": reports})


if __name__ == "__main__":
    main()
