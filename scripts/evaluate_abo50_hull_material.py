"""Measure material assignment on the geometry recovered by visual-hull carving."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json
from replay_abo50_material_coverage import assign
from run_abo50_density_maps import SIZE
from tactile3d.vision.part_lifting import add_view_votes, rasterize_faces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "hull-test", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()
    scene = args.scene
    source = args.benchmark / "observations" / "scenes" / scene
    camera = read_json(source / "transforms.json")
    geometry = read_json(args.hull_test / scene / "report.json")
    hull_path = args.hull_test / scene / "visual_hull.ply"
    if digest(hull_path) != geometry["visual_hull_sha256"]:
        raise ValueError("Hull changed since evaluation")
    hull = trimesh.load_mesh(hull_path, process=False)
    field = args.benchmark / "density_maps_v1" / "surface" / "fields" / scene
    tokens = read_json(field / "part_tokens.json")
    original_path = Path(tokens["mesh"])
    if digest(original_path) != tokens["mesh_sha256"]:
        raise ValueError("Original mesh changed since material assignment")
    original = trimesh.load_mesh(original_path, process=False)
    original_field = (
        args.benchmark / "density_material_replay" / "surface" / "fields" / scene
    )
    with np.load(original_field / "surface_fields.npz") as data:
        original_labels = data["labels"]
    used = (0, 10, 20)
    votes = np.zeros((len(hull.faces), len(tokens["parts"]) + 1))
    for view in used:
        pose = np.asarray(camera["frames"][view]["transform_matrix"])
        with np.load(field / f"view_{view:02d}.npz") as data:
            add_view_votes(votes, hull, pose, camera, data["labels"])
    _, _, hull_labels = assign(hull, tokens, votes)
    materials, _, abstention_labels = assign(hull, tokens, votes, abstain_unknown=True)
    view_support = read_json(field / "report.json")["views"]
    high_coverage_views = [
        row["view"]
        for row in view_support
        if row["labeled_foreground_fraction"] >= 0.90
    ]
    single_material = len(materials) == 1 and len(high_coverage_views) >= 2
    complete_labels = (
        np.ones_like(hull_labels) if single_material else abstention_labels
    )
    eligible = sorted(
        set(range(len(camera["frames"]))) - set(geometry["input_views"]) - set(used)
    )
    views = [eligible[i] for i in np.linspace(0, len(eligible) - 1, 5).astype(int)]
    rows = []
    for view in views:
        frame = camera["frames"][view]
        pose = np.asarray(frame["transform_matrix"])
        with Image.open(source / frame["file_path"]) as image:
            foreground = np.asarray(image.getchannel("A").resize((SIZE, SIZE))) >= 128
        result = {"view": view}
        original_ids, _ = rasterize_faces(original, pose, camera, SIZE)
        hull_ids, _ = rasterize_faces(hull, pose, camera, SIZE)
        for method, ids, labels in (
            ("instantmesh", original_ids, original_labels),
            ("hull", hull_ids, hull_labels),
            ("hull_abstention", hull_ids, abstention_labels),
            ("hull_material_consensus", hull_ids, complete_labels),
        ):
            matched = foreground & (ids >= 0)
            valid = np.count_nonzero(labels[ids[matched]])
            result[method] = {
                "geometry_coverage": float(matched.sum() / foreground.sum()),
                "density_coverage": float(valid / foreground.sum()),
            }
        print(result, flush=True)
        rows.append(result)
    write_json(
        args.output,
        {
            "scene": scene,
            "rows": rows,
            "material_input_views": used,
            "geometry_input_views": geometry["input_views"],
            "high_coverage_material_input_views": high_coverage_views,
            "single_material_prior_applied": single_material,
            "metrics": {
                method: {
                    key: float(np.mean([row[method][key] for row in rows]))
                    for key in rows[0][method]
                }
                for method in (
                    "instantmesh",
                    "hull",
                    "hull_abstention",
                    "hull_material_consensus",
                )
            },
        },
    )


if __name__ == "__main__":
    main()
