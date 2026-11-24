"""Export aligned ABO material views with official NeRF2Physics and optional PUGS–Qwen."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from export_region_material_comparison import digest, project_evidence, read


CASES = (
    ("B076YJFGCN_A1F83G8C2ARO7P", "USB hub"),
    ("B075YQ46MP_ATVPDKIKX0DER", "Wood shelving unit"),
    ("B07TWQTVXL_ATVPDKIKX0DER", "Metal shoe rack"),
    ("B07DBDX15Y_ATVPDKIKX0DER", "Patterned accent chair"),
    ("B07QFRSC43_ATVPDKIKX0DER", "Blue accent chair"),
    ("B07Y3Q7MDR", "Purple desk chair"),
    ("B07K6S2G5Z_A1F83G8C2ARO7P", "Writing desk"),
)
PAPER_CASES = (
    ("B07DBDX15Y_ATVPDKIKX0DER", "Patterned accent chair"),
    ("B07QFP4PV5_ATVPDKIKX0DER", "Gray upholstered sofa"),
    ("B07TWQTVXL_ATVPDKIKX0DER", "Metal shoe rack"),
)

# Select from the original RGB turntables before inspecting any method output.
FRONT_CASES = (
    ("B07DBDX15Y_ATVPDKIKX0DER", "Accent chair", 26),
    ("B07QFRSC43_ATVPDKIKX0DER", "Counter stool", 9),
    ("B07QFP4PV5_ATVPDKIKX0DER", "Upholstered sofa", 10),
    ("B07TWQTVXL_ATVPDKIKX0DER", "Shoe rack", 10),
)


def camera_raster(mesh, scene_dir, view, size=512):
    from PIL import Image
    from tactile3d.vision.part_lifting import rasterize_faces

    camera = read(scene_dir / "transforms.json")
    frame = camera["frames"][view]
    ids, _ = rasterize_faces(mesh, np.asarray(frame["transform_matrix"]), camera, size)
    with Image.open(scene_dir / frame["file_path"]) as source:
        rgba = source.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    fg = np.asarray(rgba.getchannel("A")) >= 128
    rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    return np.asarray(rgb.convert("RGB")), fg, ids


def pugs_projection(root, scene, mesh, face_ids, inverse, visible):
    report = root / "reports" / scene
    prediction = report / "official_predictions.json"
    if digest(prediction) != read(report / "prediction_seal.json")[prediction.name]:
        raise ValueError(f"PUGS mass prediction seal differs for {scene}")
    protocol = read(report / "integral_protocol.json")
    if protocol["method"] != "PUGS-Qwen3-VL-8B-NF4" or protocol["official_GPT4o"]:
        raise ValueError(f"Expected the recorded Qwen proposal protocol for {scene}")
    names = [p["name"] for p in read(report / "mass_decomposition.json")["materials"]]
    result = root / "scenes" / scene / "pred_result_density"
    points_path = result / "dense_points.npy"
    scores_path = result / "dense_pred_probs.npy"
    points, scores = np.load(points_path), np.load(scores_path)
    if points.shape != (len(scores), 3) or scores.shape[1:] != (len(names),):
        raise ValueError(f"PUGS points and scores do not correspond for {scene}")
    finite = np.isfinite(points).all(1) & np.isfinite(scores).all(1)
    if not finite.any():
        raise ValueError(f"PUGS returned no finite predictions for {scene}")
    distance, index = cKDTree(points[finite]).query(mesh.triangles_center[face_ids])
    transfer_limit = 0.02 * mesh.extents.max()
    predictions = 1 + scores[finite][index].argmax(axis=1)
    predictions[distance > transfer_limit] = 0
    labels = np.zeros(visible.shape, dtype=np.int16)
    labels[visible] = predictions[inverse]
    return (
        labels,
        names,
        {
            "pugs_scores_sha256": digest(scores_path),
            "pugs_points_sha256": digest(points_path),
            "pugs_proposal": protocol["method"],
            "pugs_upstream_commit": protocol["upstream_commit"],
            "pugs_transfer_limit_camera_units": float(transfer_limit),
        },
    )


def export(case, args, query, model, tokenizer, torch, view=8):
    scene, name = case
    if (args.benchmark / "appearance_fill_final" / f"{scene}.npz").exists():
        base, completion, cache = (
            "density_maps_v1",
            "appearance_fill_final",
            "heatmap_cache",
        )
    elif (args.benchmark / "density_sofa_completion" / f"{scene}.npz").exists():
        base, completion, cache = (
            "density_sofa",
            "density_sofa_completion",
            "sofa_density_cache",
        )
    else:
        base, completion, cache = (
            "density_paperlike",
            "density_paperlike_completion",
            "extra_density_cache",
        )
    base, completion, cache = (args.benchmark / p for p in (base, completion, cache))
    field = completion / "surface" / "fields" / scene
    source_field = base / "surface" / "fields" / scene
    data = read(completion / "meshes.json")[scene]
    mesh_path = Path(data["path"])
    if (
        digest(mesh_path) != data["sha256"]
        or read(source_field / "part_tokens.json")["mesh_sha256"] != data["sha256"]
    ):
        raise ValueError("Mesh and surface field have different provenance")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    scene_path = args.benchmark / "observations" / "scenes" / scene
    if args.front:
        rgb, fg, ids = camera_raster(mesh, scene_path, view)
    else:
        with np.load(completion / f"{scene}.npz") as maps:
            rgb, fg, ids = (maps[key] for key in ("rgb", "foreground", "face_ids"))
    visible = fg & (ids >= 0)
    original = cache / "scenes" / scene
    records = list(args.nerf_runs.glob(f"batch_*/{scene}/input_hashes.json"))
    if len(records) != 1:
        raise ValueError(f"Missing NeRF2Physics provenance for {scene}")
    for key, checksum in read(records[0]).items():
        if (
            not key.startswith("ns/renders/depth/")
            and digest(original / key) != checksum
        ):
            raise ValueError(f"Cached input changed: {scene} / {key}")
    ids_used, inverse = np.unique(ids[visible], return_inverse=True)
    settings = SimpleNamespace(
        mats_load_name="info",
        property_name="density",
        feature_load_name="ps56",
        device="cuda",
        temperature=0.1,
        sample_voxel_size=0.005,
        show_mat_seg=False,
    )
    query_points = torch.as_tensor(
        mesh.triangles_center[ids_used], dtype=torch.float32, device="cuda"
    )
    with torch.inference_mode():
        prediction = query(
            settings, query_points, str(original), model, tokenizer, return_all=True
        )
    nerf = np.zeros_like(ids)
    nerf[visible] = 1 + prediction["query_pred_probs"].argmax(axis=1)[inverse]
    with np.load(source_field / "surface_fields.npz") as arrays:
        features, _ = project_evidence(arrays["votes"][:, 1:])
    evidence = np.ones_like(rgb, dtype=float)
    evidence[fg] = 0.86
    evidence[visible] = features[ids[visible]]
    with np.load(field / "surface_fields.npz") as arrays:
        labels = arrays["labels"]
    ours = np.zeros_like(ids)
    ours[visible] = labels[ids[visible]]
    parts = read(field / "part_tokens.json")["parts"]
    ours_names = [
        r["surface_material"] for r in sorted(parts, key=lambda p: p["part_index"])
    ]
    names = prediction["mat_names"]
    arrays = dict(
        rgb=rgb,
        foreground=fg,
        evidence=evidence,
        nerf=nerf,
        nerf_names=names,
        ours=ours,
        ours_names=ours_names,
    )
    metadata = {
        "scene_id": scene,
        "object": name,
        "view": view,
        "mesh_sha256": data["sha256"],
        "nerf_classes": list(names),
        "partphy3d_classes": ours_names,
    }
    if args.pugs is not None:
        scene_path = args.benchmark / "observations" / "scenes" / scene
        pugs_scene = args.pugs / "scenes" / scene
        camera_file = scene_path / "transforms.json"
        frame = read(camera_file)["frames"][metadata["view"]]["file_path"]
        if any(
            digest(scene_path / path) != digest(pugs_scene / path)
            for path in ("transforms.json", frame)
        ):
            raise ValueError(f"PUGS camera or RGB frame differs for {scene}")
        pugs, pugs_names, provenance = pugs_projection(
            args.pugs, scene, mesh, ids_used, inverse, visible
        )
        arrays.update(pugs=pugs, pugs_names=pugs_names)
        metadata.update(
            pugs_classes=pugs_names,
            camera_sha256=digest(camera_file),
            rgb_sha256=digest(scene_path / frame),
            **provenance,
        )
    raster_path = args.output / f"{scene}.npz"
    np.savez_compressed(raster_path, **arrays)
    if args.pugs is not None:
        metadata["raster_sha256"] = digest(raster_path)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("benchmark", "nerf", "nerf-runs", "clip", "lock", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument(
        "--pugs", type=Path, help="Export scenes with PUGS–Qwen outputs"
    )
    parser.add_argument(
        "--front",
        action="store_true",
        help="Project the frozen fields into front-facing ABO cameras",
    )
    args = parser.parse_args()
    import open_clip
    import torch

    lock = read(args.lock)
    if any(
        digest(args.nerf / key) != expected for key, expected in lock["files"].items()
    ):
        raise ValueError("NeRF2Physics source differs from the project lock")
    if digest(args.clip) != lock["clip_sha256"]:
        raise ValueError("Unexpected CLIP weight")
    sys.path.insert(0, str(args.nerf))
    from predict_property import predict_physical_property_query

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.clip)
    )
    model = model.cuda().eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    args.output.mkdir(exist_ok=False)
    cases = PAPER_CASES if args.pugs is not None else CASES
    views = FRONT_CASES if args.front else [(scene, name, 8) for scene, name in cases]
    manifest = [
        export(
            (scene, name),
            args,
            predict_physical_property_query,
            model,
            tokenizer,
            torch,
            view,
        )
        for scene, name, view in views
    ]
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
