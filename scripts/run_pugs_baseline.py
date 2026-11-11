"""Run locked PUGS stages with a separately labelled local-Qwen proposal."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.physics.part_io import file_sha256, write_json


def source_prompt(path: Path) -> str:
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "PRED_CAND_MATS_DENSITY_SYS_MSG_4V_ENHANCED"
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Missing official PUGS density/volume prompt")


def proposal_info(answer: dict, index: int) -> dict:
    volume = float(answer["pure_volume"])
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Pure material volume must be positive and finite")
    materials = answer["materials"]
    if not isinstance(materials, list) or not materials:
        raise ValueError("Material candidates must be nonempty")
    pairs = []
    for material in materials:
        name = str(material["name"]).strip().lower()
        values = str(material["mass density (kg/m^3)"]).split("-")
        bounds = [float(v) for v in values]
        if len(bounds) == 1:
            bounds *= 2
        if (
            len(bounds) != 2
            or not np.isfinite(bounds).all()
            or not 0 < bounds[0] <= bounds[1]
            or not name
            or any(char in name for char in ":;()")
        ):
            raise ValueError("Invalid material density range")
        pairs.append(f"({name}: {bounds[0]}-{bounds[1]} kg/m^3)")
    return {
        "idx_to_caption": str(index),
        "caption": str(answer["description"]),
        "candidate_materials_density": ";".join(pairs),
        "pure_volume": volume,
    }


def catalog_metadata(catalog: dict) -> dict:
    """Export only dimension records, converting the catalog's meters to cm."""
    result = {}
    for scene, row in catalog["objects"].items():
        candidates = [c for c in row.get("candidates", []) if c.get("dimensions_m")]
        preferred = [
            c for c in candidates if c["domain_name"] == row.get("preferred_domain")
        ]
        choices = preferred or candidates
        if len(choices) != 1:
            raise ValueError(f"Ambiguous or missing catalog dimensions: {scene}")
        dimensions = [
            100 * choices[0]["dimensions_m"][key]
            for key in ("length", "width", "height")
        ]
        if not np.isfinite(dimensions).all() or min(dimensions) <= 0:
            raise ValueError(f"Invalid catalog dimensions: {scene}")
        key = scene.split("_")[0]
        if key in result and result[key]["item_dimensions"] != dimensions:
            raise ValueError(f"Conflicting catalog dimensions: {scene}")
        result[key] = {"item_dimensions": dimensions}
    return result


def mass_decomposition(
    volumes, probabilities, density, pure_volume: float
) -> tuple[np.ndarray, np.ndarray]:
    volumes = np.asarray(volumes, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    density = np.asarray(density, dtype=float)
    if (
        volumes.ndim != 1
        or probabilities.ndim != 2
        or probabilities.shape[0] != len(volumes)
        or density.shape != (probabilities.shape[1], 2)
        or not all(np.isfinite(x).all() for x in (volumes, probabilities, density))
        or not np.isfinite(pure_volume)
        or pure_volume <= 0
        or np.any(volumes < 0)
        or volumes.sum() <= 0
        or np.any(probabilities < 0)
        or np.any(density <= 0)
        or np.any(density[:, 0] > density[:, 1])
        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    ):
        raise ValueError("Invalid Gaussian/material volume decomposition")
    assigned_volume = pure_volume * (volumes @ probabilities) / volumes.sum()
    return assigned_volume, assigned_volume[:, None] * density


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("prepare", "reconstruct", "propose", "integrate")
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-lock", type=Path, default=Path("configs/pugs_lock.json")
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--clip-checkpoint", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--view-index", type=int, default=8)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("configs/evidence/abo500_catalog_dimensions.json"),
    )
    args = parser.parse_args()
    args.repository = args.repository.resolve()
    args.scene = args.scene.resolve()
    if args.sam_checkpoint:
        args.sam_checkpoint = args.sam_checkpoint.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    for name, digest in lock["files"].items():
        if file_sha256(args.repository / name) != digest:
            raise ValueError(f"PUGS source changed: {name}")
    if args.stage == "prepare":
        metadata = catalog_metadata(
            json.loads(args.catalog.read_text(encoding="utf-8"))
        )
        destination = args.scene.parents[1] / "filtered_metadata.json"
        if destination.exists() and json.loads(destination.read_text()) != metadata:
            raise ValueError(
                "Existing metadata differs; use an isolated PUGS data root"
            )
        write_json(destination, metadata)
        write_json(
            args.output / "catalog_protocol.json",
            {
                "source_sha256": file_sha256(args.catalog),
                "export_sha256": file_sha256(destination),
                "dimensions_unit": "centimeters",
                "mass_labels_included": False,
                "domain_rule": "preferred if present, otherwise unique available candidate",
            },
        )
    elif args.stage == "reconstruct":
        if (
            not args.sam_checkpoint
            or file_sha256(args.sam_checkpoint) != lock["sam_vit_h_sha256"]
        ):
            parser.error("Provide the verified official SAM ViT-H checkpoint")
        scene, saga = str(args.scene), str(args.scene / "saga")
        jobs = [
            ("train_scene", ["-s", scene, "--model_path", saga]),
            (
                "extract_segment_everything_masks",
                [
                    "--image_root",
                    scene,
                    "--sam_checkpoint_path",
                    str(args.sam_checkpoint),
                ],
            ),
            ("get_scale", ["--image_root", scene, "--model_path", saga]),
            (
                "train_contrastive_feature",
                ["-m", saga, "--iterations", "10000", "--num_sampled_rays", "1000"],
            ),
        ]
        for name, options in jobs:
            command = [sys.executable, "-m", f"reconstruction.{name}", *options]
            started = time.perf_counter()
            with (args.output / f"{name}.log").open("w") as stream:
                subprocess.run(
                    command,
                    cwd=args.repository,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            write_json(
                args.output / f"{name}.json",
                {
                    "command": command,
                    "seconds": time.perf_counter() - started,
                    "upstream_commit": lock["commit"],
                },
            )
    elif args.stage == "propose":
        from tactile3d.vision.local_vlm import LocalVLM
        from tactile3d.vision.vlm_interior import _parse_response_content

        if not args.model:
            parser.error("Provide local Qwen model weights")
        images = sorted((args.scene / "images").glob("*.png"))
        if len(images) != 30 or not 0 <= args.view_index < len(images):
            raise ValueError(
                "Expected 30 original ABO PNG views and a valid view index"
            )
        prompt = source_prompt(args.repository / "utils/gpt_inference_utils.py")
        model = LocalVLM(args.model, max_tokens=300, max_pixels=1024**2, quantized=True)
        user_prompt = "Analyze the image."
        response = model.generate(
            user_prompt, [Image.open(images[args.view_index])], prompt, seed=100
        )
        record = {
            "method": "PUGS-Qwen3-VL-8B-NF4",
            "official_GPT4o": False,
            "model": model.provenance,
            "system_prompt": prompt,
            "user_prompt": user_prompt,
            "view_index": args.view_index,
            "image_sha256": file_sha256(images[args.view_index]),
            "response": asdict(response),
        }
        write_json(args.output / "proposal_raw.json", record)
        info = proposal_info(_parse_response_content(response.text), args.view_index)
        write_json(args.scene / "info_qwen.json", info)
    else:
        import open_clip
        import torch

        if (
            not args.clip_checkpoint
            or file_sha256(args.clip_checkpoint) != lock["clip_sha256"]
        ):
            parser.error("Provide the locked CLIP checkpoint")
        sys.path.insert(0, str(args.repository))
        from clip_feature_fusion import process_scene
        from predict_property import (
            get_gaussian_volume_with_opacity,
            predict_physical_property_integral,
        )
        from utils.enhance_utils import load_saga_point_cloud
        from utils.gpt_inference_utils import parse_material_list

        settings = SimpleNamespace(
            device="cuda",
            mats_load_name="info_qwen",
            property_name="density",
            temperature=0.1,
            sample_voxel_size=0.005,
            volume_method="gaussian",
            feature_voxel_size=0.02,
            occ_thr=0.02,
            different_Ks=False,
            patch_size=56,
            batch_size=16,
        )
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=str(args.clip_checkpoint)
        )
        model = model.to("cuda").eval()
        with torch.no_grad():
            process_scene(settings, str(args.scene), model, preprocess)
            bounds = predict_physical_property_integral(
                settings, str(args.scene), model, open_clip.get_tokenizer("ViT-B-16")
            )
        values = np.asarray(bounds, dtype=float)
        if (
            values.shape != (2,)
            or not np.isfinite(values).all()
            or not 0 < values[0] <= values[1]
        ):
            raise ValueError("Invalid official mass range")
        write_json(args.output / "official_predictions.json", {args.scene.name: bounds})
        write_json(
            args.output / "prediction_seal.json",
            {
                "official_predictions.json": file_sha256(
                    args.output / "official_predictions.json"
                )
            },
        )
        write_json(
            args.output / "integral_protocol.json",
            {
                "method": "PUGS-Qwen3-VL-8B-NF4",
                "official_GPT4o": False,
                "upstream_commit": lock["commit"],
                "settings": vars(settings),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "proposal_sha256": file_sha256(args.scene / "info_qwen.json"),
                "target_mass_passed_to_inference": False,
            },
        )
        _, _, scales, opacity = load_saga_point_cloud(
            str(
                args.scene
                / "saga/point_cloud/iteration_10000/contrastive_feature_point_cloud.ply"
            ),
            ds_size=None,
            return_gaussian_property=True,
        )
        scales = torch.exp(torch.as_tensor(scales))
        opacity = torch.sigmoid(torch.as_tensor(opacity))
        keep = opacity > 0.95
        volumes = get_gaussian_volume_with_opacity(scales[keep], opacity[keep]).numpy()
        info = json.loads((args.scene / "info_qwen.json").read_text())
        names, density = parse_material_list(info["candidate_materials_density"])
        probabilities = np.load(args.scene / "pred_result_density/dense_pred_probs.npy")
        assigned, contributions = mass_decomposition(
            volumes, probabilities, density, info["pure_volume"]
        )
        if not np.allclose(contributions.sum(axis=0), bounds, rtol=1e-5, atol=1e-6):
            raise ValueError("Decomposition differs from the official integral")
        write_json(
            args.output / "mass_decomposition.json",
            {
                "pure_volume_m3": info["pure_volume"],
                "materials": [
                    {
                        "name": name,
                        "assigned_volume_m3": float(volume),
                        "density_range_kg_m3": rho,
                        "mass_range_kg": mass.tolist(),
                    }
                    for name, volume, rho, mass in zip(
                        names, assigned, density, contributions, strict=True
                    )
                ],
                "sum_matches_official_integral": True,
            },
        )


if __name__ == "__main__":
    main()
