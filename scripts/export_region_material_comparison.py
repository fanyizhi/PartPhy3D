"""Register PartPhy3D face evidence and PUGS material probabilities in identical views."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.vision.part_lifting import rasterize_faces


OBJECTS = (
    ("B07TWQTVXL_ATVPDKIKX0DER", 8),
    ("B07DBDX15Y_ATVPDKIKX0DER", 1),
    ("B07QFRSC43_ATVPDKIKX0DER", 3),
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_evidence(votes):
    """Per-face PCA of positive view votes, with deterministic channel signs."""
    support = votes.sum(axis=1)
    known = support > 0
    if not known.any():
        raise ValueError("No positive part votes to display")
    features = votes[known] / support[known, None]
    features -= features.mean(axis=0)
    _, _, axes = np.linalg.svd(features, full_matrices=False)
    axes = axes[:3]
    signs = np.sign(axes[np.arange(len(axes)), np.abs(axes).argmax(axis=1)])
    coordinates = features @ (axes * signs[:, None]).T
    rgb = np.full((len(votes), 3), 0.68)
    for i, channel in enumerate(coordinates.T):
        scale = np.max(np.abs(channel))
        if scale > 1e-9:
            rgb[known, i] = 0.53 + 0.34 * channel / scale
    # Lighten single-view evidence without interpreting PCA colors as materials.
    weight = np.minimum(1.0, support[known] / 2.0)[:, None]
    rgb[known] = 0.87 * (1 - weight) + rgb[known] * weight
    return rgb, known


def export(scene, view, benchmark, pugs, registered, output):
    completion = benchmark / "appearance_fill_final"
    source = benchmark / "observations" / "scenes" / scene
    manifest = read(completion / "meshes.json")[scene]
    mesh_path = Path(manifest["path"])
    original = benchmark / "density_maps_v1" / "surface" / "fields" / scene
    field = completion / "surface" / "fields" / scene
    if (
        digest(mesh_path) != manifest["sha256"]
        or read(original / "part_tokens.json")["mesh_sha256"] != manifest["sha256"]
    ):
        raise ValueError("Votes, labels and rendering must use the same geometry")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    camera = read(source / "transforms.json")
    frame = camera["frames"][view]
    path = source / frame["file_path"]
    dataset_hashes = read(benchmark / "observations" / "cohort.json")["files_sha256"]
    if digest(path) != dataset_hashes[f"scenes/{scene}/{frame['file_path']}"]:
        raise ValueError("RGB differs from frozen camera cohort")
    ids, _ = rasterize_faces(mesh, np.array(frame["transform_matrix"]), camera, 384)
    with Image.open(path) as im:
        rgba = np.array(im.convert("RGBA").resize((384, 384)))
    visible = (ids >= 0) & (rgba[..., 3] >= 128)
    with np.load(original / "surface_fields.npz") as arrays:
        votes = arrays["votes"][:, 1:]
    rgb, known = project_evidence(votes)
    feature_map = np.ones((384, 384, 3))
    feature_map[visible] = rgb[ids[visible]]
    feature_map[visible & ~known[ids]] = 0.81
    ours = read(field / "part_tokens.json")["parts"]
    ours = [
        part["surface_material"] for part in sorted(ours, key=lambda p: p["part_index"])
    ]
    with np.load(field / "surface_fields.npz") as arrays:
        labels = arrays["labels"]
    ours_map = np.zeros_like(ids)
    ours_map[visible] = labels[ids[visible]]
    run = pugs / "scenes" / scene / "pred_result_density"
    provenance = read(
        registered / f"pugs_registered_{scene}" / "density_manifest.json"
    )["scenes"][0]
    points_path = run / "dense_points.npy"
    if digest(points_path) != provenance["pugs_source_points_sha256"]:
        raise ValueError("PUGS coordinates differ from published registration")
    points = np.load(points_path)
    scores_path = run / "dense_pred_probs.npy"
    scores = np.load(scores_path)
    materials = [
        m["name"]
        for m in read(pugs / "reports" / scene / "mass_decomposition.json")["materials"]
    ]
    if scores.shape != (len(points), len(materials)):
        raise ValueError("PUGS score rows or columns do not match the run")
    use = np.isfinite(points).all(axis=1) & np.isfinite(scores).all(axis=1)
    faces, inverse = np.unique(ids[visible], return_inverse=True)
    distances, index = cKDTree(points[use]).query(mesh.triangles_center[faces])
    pred = 1 + scores[use][index].argmax(axis=1)
    pred[distances > provenance["pugs_max_transfer_distance_camera_units"]] = 0
    pugs_map = np.zeros_like(ids)
    pugs_map[visible] = pred[inverse]
    np.savez_compressed(
        output / f"{scene}_{view:02}.npz",
        rgba=rgba,
        evidence=feature_map,
        ours=ours_map,
        ours_names=ours,
        pugs=pugs_map,
        pugs_names=materials,
    )
    return {
        "scene_id": scene,
        "view": view,
        "pugs_scores_sha256": digest(scores_path),
        "pugs_classes": materials,
        "partphy3d_classes": ours,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "pugs", "registered", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    metadata = [
        export(scene, view, args.benchmark, args.pugs, args.registered, args.output)
        for scene, view in OBJECTS
    ]
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
