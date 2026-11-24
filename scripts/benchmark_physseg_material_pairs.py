"""Compare the complete surface field and two runnable baselines on shared PhysSeg views."""

import argparse
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

from export_region_material_comparison import digest, read, rasterize_faces


OBJECTS = {
    "B07DBDX15Y_ATVPDKIKX0DER": "Patterned accent chair",
    "B07QFRSC43_ATVPDKIKX0DER": "Blue accent chair",
}
CLASSES = ("fabric", "wood")
METHODS = ("NeRF2Physics", "PUGS–Qwen", "Ours")
SIZE = 384


def score(target, prediction, vocabulary):
    # Score only pixels with exactly one released material annotation.
    domain = target > 0
    counts = {}
    for i, name in enumerate(CLASSES, start=1):
        expected = target == i
        if name in vocabulary:
            predicted = domain & (prediction == vocabulary.index(name) + 1)
        else:
            predicted = np.zeros_like(domain)
        counts[name] = [
            int((expected & predicted).sum()),
            int((expected | predicted).sum()),
        ]
    return counts


def aggregate(rows):
    totals = np.array([list(row.values()) for row in rows]).sum(axis=0)
    iou = totals[:, 0] / totals[:, 1]
    return {
        "class_iou": dict(zip(CLASSES, iou.tolist(), strict=True)),
        "miou": float(iou.mean()),
    }


def target_map(archive, paths, camera):
    masks = []
    for name in CLASSES:
        with Image.open(io.BytesIO(archive.read(paths[name]))) as im:
            if im.size != (camera["w"], camera["h"]):
                raise ValueError(
                    "Annotation dimensions differ from the calibrated camera"
                )
            masks.append(
                np.array(
                    im.getchannel("A").resize((SIZE, SIZE), Image.Resampling.NEAREST)
                )
                >= 128
            )
    fabric, wood = masks
    return np.where(fabric ^ wood, np.where(fabric, 1, 2), 0).astype(np.uint8)


def scene_predictions(scene, archive, args, baseline, torch):
    benchmark = args.benchmark
    shared = benchmark / "appearance_fill_final"
    source = benchmark / "observations/scenes" / scene
    camera = read(source / "transforms.json")
    manifest = read(shared / "meshes.json")[scene]
    field = shared / "surface/fields" / scene
    tokens = read(field / "part_tokens.json")
    mesh_path = Path(manifest["path"])
    if (
        digest(mesh_path) != manifest["sha256"]
        or tokens["mesh_sha256"] != manifest["sha256"]
    ):
        raise ValueError("Part labels use a different mesh")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    hull = None
    if args.hull is not None:
        directory = args.hull / scene
        hull_path = directory / "visual_hull.ply"
        hull_info = read(directory / "report.json")
        if digest(hull_path) != hull_info["visual_hull_sha256"]:
            raise ValueError("Subpixel hull differs from its recorded run")
        hull = trimesh.load_mesh(hull_path, process=False)
    with np.load(field / "surface_fields.npz") as stored:
        ours_labels = stored["labels"]
    ours_names = [
        row["surface_material"]
        for row in sorted(tokens["parts"], key=lambda r: r["part_index"])
    ]

    paths = {}
    for path in archive.namelist():
        fields = path.split("/")
        if (
            len(fields) == 5
            and fields[2] == scene
            and fields[4] in {f"{name}.png" for name in CLASSES}
        ):
            paths.setdefault(int(fields[3]), {})[Path(fields[4]).stem] = path
    if len(paths) != 5:
        raise ValueError(f"Expected five labelled views for {scene}")
    cohort_hashes = read(benchmark / "observations/cohort.json")["files_sha256"]
    if (
        digest(source / "transforms.json")
        != cohort_hashes[f"scenes/{scene}/transforms.json"]
    ):
        raise ValueError("Camera provenance differs")
    views = []
    for view, files in sorted(paths.items()):
        frame = camera["frames"][view]
        ids, _ = rasterize_faces(
            mesh, np.asarray(frame["transform_matrix"]), camera, SIZE
        )
        rgb_path = source / frame["file_path"]
        if digest(rgb_path) != cohort_hashes[f"scenes/{scene}/{frame['file_path']}"]:
            raise ValueError("Observation provenance differs")
        with Image.open(rgb_path) as im:
            rgba = np.array(im.convert("RGBA").resize((SIZE, SIZE)))
        target = target_map(archive, files, camera)
        views.append((view, ids, rgba, target))

    # Query the union of visible faces once so both baselines see the same 3D support.
    selected = np.unique(np.concatenate([ids[ids >= 0] for _, ids, _, _ in views]))
    points = mesh.triangles_center[selected]
    original = benchmark / "heatmap_cache/scenes" / scene
    (provenance,) = args.nerf_runs.glob(f"batch_*/{scene}/input_hashes.json")
    for key, checksum in read(provenance).items():
        if (
            not key.startswith("ns/renders/depth/")
            and digest(original / key) != checksum
        ):
            raise ValueError(f"NeRF2Physics cached input changed: {key}")
    query, settings, model, tokenizer = baseline
    with torch.inference_mode():
        nerf = query(
            settings,
            torch.as_tensor(points, dtype=torch.float32, device="cuda"),
            str(original),
            model,
            tokenizer,
            return_all=True,
        )
    nerf_face = np.zeros(len(mesh.faces), dtype=np.int16)
    nerf_face[selected] = 1 + nerf["query_pred_probs"].argmax(axis=1)

    run = args.pugs / "scenes" / scene / "pred_result_density"
    link = read(args.registration / f"pugs_registered_{scene}/density_manifest.json")[
        "scenes"
    ][0]
    report = args.pugs / "reports" / scene
    prior_names = [
        row["name"] for row in read(report / "mass_decomposition.json")["materials"]
    ]
    score_path = run / "dense_pred_probs.npy"
    points_path = run / "dense_points.npy"
    if (
        link["mesh_sha256"] != manifest["sha256"]
        or digest(points_path) != link["pugs_source_points_sha256"]
    ):
        raise ValueError("PUGS registration or source points differ")
    pugs_material_digest = digest(score_path)
    expected = next(row for row in read(args.pugs_manifest) if row["scene_id"] == scene)
    if (
        pugs_material_digest != expected["pugs_scores_sha256"]
        or prior_names != expected["pugs_classes"]
    ):
        raise ValueError(
            "PUGS probabilities or column labels differ from the recorded run"
        )
    probabilities = np.load(score_path)
    dense = np.load(points_path)
    if probabilities.shape != (len(dense), len(prior_names)):
        raise ValueError("Unexpected PUGS probability shape")
    valid = np.isfinite(dense).all(axis=1) & np.isfinite(probabilities).all(axis=1)
    distances, indices = cKDTree(dense[valid]).query(points)
    pugs_face = np.zeros(len(mesh.faces), dtype=np.int16)
    accept = distances <= link["pugs_max_transfer_distance_camera_units"]
    pugs_face[selected[accept]] = 1 + probabilities[valid][indices[accept]].argmax(
        axis=1
    )

    vocabularies = (nerf["mat_names"], prior_names, ours_names)
    all_rows = []
    for view, ids, rgba, target in views:
        domain = target > 0
        raster_domain = domain | (rgba[..., 3] >= 128)
        visible = raster_domain & (ids >= 0)
        rasters = []
        for field in (nerf_face, pugs_face, ours_labels):
            raster = np.zeros(ids.shape, dtype=np.int16)
            raster[visible] = field[ids[visible]]
            rasters.append(raster)
        pairs = zip(METHODS, rasters, vocabularies, strict=True)
        wood = target == 2
        ours = rasters[-1]
        on_mesh = ids >= 0
        metal_ids = [
            i + 1
            for i, label in enumerate(ours_names)
            if label in {"metal", "steel", "aluminum"}
        ]
        wood_ids = [i + 1 for i, label in enumerate(ours_names) if label == "wood"]
        as_metal, as_wood = np.isin(ours, metal_ids), np.isin(ours, wood_ids)
        foreground = rgba[..., 3] >= 128
        geometry = {
            "original": {
                "wood_rays_hit": int((wood & on_mesh).sum()),
                "rgb_foreground_true_positive": int((foreground & on_mesh).sum()),
                "rgb_foreground_predicted": int(on_mesh.sum()),
            }
        }
        if hull is not None and view not in hull_info["input_views"]:
            pose = camera["frames"][view]["transform_matrix"]
            hull_ids, _ = rasterize_faces(hull, np.asarray(pose), camera, SIZE)
            hit = hull_ids >= 0
            geometry["subpixel_hull"] = {
                "wood_rays_hit": int((wood & hit).sum()),
                "rgb_foreground_true_positive": int((foreground & hit).sum()),
                "rgb_foreground_predicted": int(hit.sum()),
            }
        all_rows.append(
            dict(
                scene_id=scene,
                object=OBJECTS[scene],
                view=view,
                counts={
                    name: score(target, raster, names) for name, raster, names in pairs
                },
                wood_pixels=int(wood.sum()),
                wood_missing_geometry=int((wood & ~on_mesh).sum()),
                wood_assigned_metal=int((wood & as_metal).sum()),
                wood_unassigned_on_mesh=int((wood & on_mesh & (ours == 0)).sum()),
                wood_other_material=int(
                    (wood & (ours > 0) & ~as_wood & ~as_metal).sum()
                ),
                wood_correct=int((wood & as_wood).sum()),
                rgb_foreground_pixels=int(foreground.sum()),
                geometry=geometry,
            )
        )
        if view == views[0][0]:
            np.savez_compressed(
                args.output / f"{scene}_{view:02}.npz",
                rgba=rgba,
                target=target,
                nerf=rasters[0],
                pugs=rasters[1],
                ours=rasters[2],
                nerf_names=vocabularies[0],
                pugs_names=vocabularies[1],
                ours_names=vocabularies[2],
            )
    return all_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "benchmark",
        "pugs",
        "registration",
        "pugs-manifest",
        "nerf",
        "nerf-runs",
        "clip",
        "lock",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--hull", type=Path)
    args = parser.parse_args()
    import open_clip
    import torch

    lock = read(args.lock)
    if digest(args.clip) != lock["clip_sha256"] or any(
        digest(args.nerf / name) != sha for name, sha in lock["files"].items()
    ):
        raise ValueError("NeRF2Physics source or CLIP checkpoint differs from the lock")
    sys.path.insert(0, str(args.nerf))
    from predict_property import predict_physical_property_query

    settings = SimpleNamespace(
        mats_load_name="info",
        property_name="density",
        feature_load_name="ps56",
        device="cuda",
        temperature=0.1,
        sample_voxel_size=0.005,
        show_mat_seg=False,
    )
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.clip)
    )
    baseline = (
        predict_physical_property_query,
        settings,
        model.cuda().eval(),
        open_clip.get_tokenizer("ViT-B-16"),
    )
    args.output.mkdir(parents=True, exist_ok=False)
    source = args.benchmark / "PhysSeg-15.zip"
    with ZipFile(source) as archive:
        rows = [
            row
            for scene in OBJECTS
            for row in scene_predictions(scene, archive, args, baseline, torch)
        ]
    result = {
        "annotation_sha256": digest(source),
        "objects": len(OBJECTS),
        "views": len(rows),
        "classes": CLASSES,
        "metrics": {m: aggregate([r["counts"][m] for r in rows]) for m in METHODS},
        "rows": rows,
    }
    (args.output / "paired_material.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
