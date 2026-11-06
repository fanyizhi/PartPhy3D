"""重跑NeRF2Physics官方属性阶段，使用发布的缓存，不调用在线LLM。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.part_io import file_sha256, write_json

CLIP_SHA256 = "80905176486d952914c1deb3b753b8e653ab54aa4ffb14208bc5007cf7643a16"


def run(args: argparse.Namespace) -> dict[str, Any]:
    """保留默认厚度积分和0.6修正，先封存预测，再读取质量标签。"""
    import open_clip
    import torch
    import trimesh

    if file_sha256(args.clip_checkpoint) != CLIP_SHA256:
        raise ValueError("CLIP检查点与官方固定文件不符")
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.repository.resolve()))
    from predict_property import (
        predict_physical_property_integral,
        predict_physical_property_query,
    )

    ids = json.loads(
        Path("configs/five_object_pilot.json").read_text(encoding="utf-8")
    )["object_ids"]
    ids = ids[args.start : args.start + args.limit]
    settings = SimpleNamespace(
        mats_load_name="info",
        property_name="density",
        feature_load_name="ps56",
        device="cuda",
        temperature=0.1,
        sample_voxel_size=0.005,
        volume_method="thickness",
        correction_factor=0.6,
        show_mat_seg=False,
    )
    protocol = {
        "upstream_commit": "7311210f62b120f6896b3112b231c26003ce70e3",
        "scope": "released_cached_intermediates_official_property_stage",
        "fresh_NeRF_reconstruction": False,
        "fresh_LLM_proposals": False,
        "PUGS_reproduced": False,
        "original_input_views": 30,
        "settings": vars(settings),
        "clip_sha256": CLIP_SHA256,
        "source_predict_sha256": file_sha256(args.repository / "predict_property.py"),
        "cohort": ids,
        "target_mass_passed_to_inference": False,
    }
    write_json(args.output / "protocol.json", protocol)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.clip_checkpoint)
    )
    model = model.to("cuda").eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    meshes = json.loads(args.mesh_manifest.read_text(encoding="utf-8"))
    predictions = []
    for scene in ids:
        directory = args.output / scene
        directory.mkdir()
        source = args.root / "scenes" / scene
        write_json(
            directory / "input_hashes.json",
            {
                str(p.relative_to(source)): file_sha256(p)
                for p in [
                    source / "info.json",
                    source / "ns/point_cloud.ply",
                    source / "ns/dataparser_transforms.json",
                    source / "features/patch_features_ps56.pt",
                    source / "features/is_visible_ps56.pt",
                    source / "features/voxel_size_ps56.json",
                ]
            },
        )
        interval = predict_physical_property_integral(
            settings, str(source), model, tokenizer
        )
        values = np.asarray(interval, dtype=float)
        if (
            values.shape != (2,)
            or not np.isfinite(values).all()
            or not 0 < values[0] <= values[1]
        ):
            raise ValueError("基线质量区间无效")
        predictions.append(
            {
                "scene_id": scene,
                "mass_interval_kg": interval,
                "mass_kg": float(values.mean()),
            }
        )
        mesh_path = Path(meshes[scene]["path"])
        if file_sha256(mesh_path) != meshes[scene]["sha256"]:
            raise ValueError("生成Mesh变化")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        query = torch.as_tensor(
            mesh.triangles_center, dtype=torch.float32, device="cuda"
        )
        field = predict_physical_property_query(
            settings, query, str(source), model, tokenizer
        )
        np.savez_compressed(
            directory / "density_on_generated_mesh.npz", density_interval_kg_m3=field
        )
        print("BASELINE_PREDICTED", scene, interval, flush=True)
    write_json(args.output / "predictions.json", predictions)
    write_json(
        args.output / "prediction_seal.json",
        {"sha256": file_sha256(args.output / "predictions.json")},
    )
    weights = json.loads(args.weights.read_text(encoding="utf-8"))
    target = [
        weights.get(p["scene_id"], weights.get(p["scene_id"].split("_")[0]))
        for p in predictions
    ]
    result = {
        "scope": protocol["scope"],
        "objects": predictions,
        "metrics": mass_metrics(target, [p["mass_kg"] for p in predictions]),
        "comparison_input_budgets_identical": False,
        "density_ground_truth_available": False,
    }
    write_json(args.output / "report.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--clip-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mesh-manifest", type=Path, required=True)
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("data/abo500_five/filtered_product_weights.json"),
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=1)
    a = p.parse_args()
    if a.start < 0 or a.limit < 1 or a.start + a.limit > 5:
        p.error("需要冻结五物体队列的有效范围")
    print(json.dumps(run(a), indent=2))


if __name__ == "__main__":
    main()
