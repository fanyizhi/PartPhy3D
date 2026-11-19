"""Run a shared-mesh surface density comparison on the frozen ABO-50 examples."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import trimesh
from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json
from run_nerf_cached_baseline import CLIP_SHA256

from tactile3d.physics.materials import load_material_priors
from tactile3d.vision.part_lifting import rasterize_faces
from tactile3d.vision.vlm_interior import _parse_response_content

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "B07TWQTVXL_ATVPDKIKX0DER",
    "B07DBDX15Y_ATVPDKIKX0DER",
    "B07QFRSC43_ATVPDKIKX0DER",
)
VIEW = 8
SIZE = 384


def prepare(args: argparse.Namespace) -> None:
    frozen = args.benchmark / "observations" / "cohort.json"
    if not set(CASES) <= set(read_json(frozen)["object_ids"]):
        raise ValueError("Density examples must belong to the frozen cohort")
    predictions = args.benchmark / "mass_hypotheses"
    if read_json(predictions / "prediction_seal.json")["sha256"] != digest(
        predictions / "predictions.json"
    ):
        raise ValueError("Mass hypothesis seal mismatch")
    by_scene = {
        row["scene_id"]: row for row in read_json(predictions / "predictions.json")
    }
    priors = {
        item.name: {
            "density_mean_kg_m3": item.density_kg_m3.mean,
            "youngs_mean_pa": item.youngs_modulus_pa.mean,
        }
        for item in load_material_priors(ROOT / "configs/material_priors.yaml")
    }
    meshes = {}
    inputs = {}
    for scene in CASES:
        mesh = args.benchmark / "silhouettes" / scene / "corrected_camera_space.ply"
        meshes[scene] = {"path": str(mesh.resolve()), "sha256": digest(mesh)}
        record = by_scene[scene]
        raw_sha = None
        if "parts" not in record:
            raw = predictions / scene / "construction.txt"
            record = _parse_response_content(raw.read_text(encoding="utf-8"))
            raw_sha = digest(raw)
        parts = [
            {
                "part_id": f"p{index}",
                "name": part["name"],
                "surface_material": part["material"],
            }
            for index, part in enumerate(record["parts"])
            if not part["evidence"]
            .lower()
            .startswith(("inferred", "internal", "hidden"))
            and part["material"] in priors
        ]
        if not parts:
            raise ValueError(f"No visible candidates in sealed prediction for {scene}")
        inputs[scene] = {
            "parts": parts,
            "material_priors": priors,
            "raw_hypothesis_sha256": raw_sha,
        }
    args.output.mkdir()
    for scene, prediction in inputs.items():
        directory = args.output / "objects" / scene
        directory.mkdir(parents=True)
        write_json(directory / "prediction.json", prediction)
    write_json(args.output / "cohort.json", {"object_ids": CASES})
    write_json(args.output / "meshes.json", meshes)


def export(args: argparse.Namespace) -> None:
    import open_clip
    import torch

    lock = read_json(ROOT / "configs/nerf2physics_lock.json")
    if digest(args.clip) != CLIP_SHA256:
        raise ValueError("CLIP checkpoint differs from the pinned baseline")
    for name, checksum in lock["files"].items():
        if digest(args.nerf / name) != checksum:
            raise ValueError(f"NeRF2Physics file differs: {name}")
    sys.path.insert(0, str(args.nerf.resolve()))
    from predict_property import predict_physical_property_query

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.clip)
    )
    model = model.to("cuda").eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    settings = SimpleNamespace(
        mats_load_name="info",
        property_name="density",
        feature_load_name="ps56",
        device="cuda",
        temperature=0.1,
        sample_voxel_size=0.005,
        show_mat_seg=False,
    )
    meshes = read_json(args.output / "meshes.json")
    observations = args.benchmark / "observations"
    reports = []
    for scene in CASES:
        source = args.cache / "scenes" / scene
        reference_hashes = list(
            args.nerf_runs.glob(f"batch_*/{scene}/input_hashes.json")
        )
        if len(reference_hashes) != 1:
            raise ValueError(
                f"Expected one official cache provenance record for {scene}"
            )
        hashes = read_json(reference_hashes[0])
        for key, checksum in hashes.items():
            if (
                not key.startswith("ns/renders/depth/")
                and digest(source / key) != checksum
            ):
                raise ValueError(f"Official cache differs for {scene}: {key}")
        mesh_path = Path(meshes[scene]["path"])
        mesh_hash = digest(mesh_path)
        if mesh_hash != meshes[scene]["sha256"]:
            raise ValueError("Shared mesh changed")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        directory = args.output / "surface" / "fields" / scene
        tokens = read_json(directory / "part_tokens.json")
        if tokens["mesh_sha256"] != mesh_hash:
            raise ValueError("Ours used a different mesh")
        with np.load(directory / "surface_fields.npz") as fields:
            labels = fields["labels"]
        face_density = np.full(len(mesh.faces), np.nan)
        for part in tokens["parts"]:
            prior = tokens["material_priors"].get(part["surface_material"])
            if prior is not None:
                face_density[labels == part["part_index"]] = prior["density_mean_kg_m3"]
        camera = read_json(observations / "scenes" / scene / "transforms.json")
        frame = camera["frames"][VIEW]
        face_ids, _ = rasterize_faces(
            mesh, np.asarray(frame["transform_matrix"]), camera, SIZE
        )
        with Image.open(observations / "scenes" / scene / frame["file_path"]) as image:
            rgba = image.convert("RGBA").resize((SIZE, SIZE))
            foreground = np.asarray(rgba.getchannel("A")) >= 128
            rgb = np.asarray(
                Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba
                ).convert("RGB")
            )
        valid = foreground & (face_ids >= 0)
        unique, inverse = np.unique(face_ids[valid], return_inverse=True)
        if not len(unique):
            raise ValueError("Shared projected mesh does not overlap object foreground")
        points = torch.as_tensor(
            mesh.triangles_center[unique], dtype=torch.float32, device="cuda"
        )
        density = predict_physical_property_query(
            settings, points, str(source), model, tokenizer
        )
        base = np.full((SIZE, SIZE), np.nan)
        ours = base.copy()
        base[valid] = density.mean(axis=1)[inverse]
        ours[valid] = face_density[face_ids[valid]]
        np.savez_compressed(
            args.output / f"{scene}.npz",
            rgb=rgb,
            ours=ours,
            nerf=base,
            foreground=foreground,
            face_ids=face_ids,
        )
        reports.append(
            {
                "scene_id": scene,
                "view": VIEW,
                "mesh_sha256": mesh_hash,
                "shared_mesh_foreground_coverage": float(
                    valid.sum() / foreground.sum()
                ),
                "ours_foreground_coverage": float(
                    np.isfinite(ours).sum() / foreground.sum()
                ),
            }
        )
    write_json(
        args.output / "density_manifest.json",
        {
            "scenes": reports,
            "units": "kg/m^3",
            "label": "reported density estimates; no measured density target",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "export"))
    for name in ("benchmark", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("clip", "nerf", "cache", "nerf-runs"):
        parser.add_argument(f"--{name}", type=Path)
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        if any(
            getattr(args, attr) is None
            for attr in ("clip", "nerf", "cache", "nerf_runs")
        ):
            parser.error("export requires --clip, --nerf, --cache and --nerf-runs")
        export(args)


if __name__ == "__main__":
    main()
