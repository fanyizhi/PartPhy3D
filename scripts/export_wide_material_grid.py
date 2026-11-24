"""Replay the official NeRF2Physics query for seven ABO material examples."""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh

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


def export(case, args, query, model, tokenizer, torch):
    scene, name = case
    main = (args.benchmark / "appearance_fill_final" / f"{scene}.npz").exists()
    base, completion, cache = (
        ("density_maps_v1", "appearance_fill_final", "heatmap_cache")
        if main
        else (
            "density_paperlike",
            "density_paperlike_completion",
            "extra_density_cache",
        )
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
    np.savez_compressed(
        args.output / f"{scene}.npz",
        rgb=rgb,
        foreground=fg,
        evidence=evidence,
        nerf=nerf,
        nerf_names=names,
        ours=ours,
        ours_names=ours_names,
    )
    return {
        "scene_id": scene,
        "object": name,
        "view": 8,
        "mesh_sha256": data["sha256"],
        "nerf_classes": list(names),
        "partphy3d_classes": ours_names,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("benchmark", "nerf", "nerf-runs", "clip", "lock", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
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
    manifest = [
        export(case, args, predict_physical_property_query, model, tokenizer, torch)
        for case in CASES
    ]
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
