"""重跑NeRF2Physics官方属性阶段，使用发布的缓存，不调用在线LLM。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.part_io import file_sha256, write_json

CLIP_SHA256 = "80905176486d952914c1deb3b753b8e653ab54aa4ffb14208bc5007cf7643a16"


def input_hashes(source: Path) -> dict[str, str]:
    files = [
        source / name
        for name in (
            "info.json",
            "transforms.json",
            "ns/point_cloud.ply",
            "ns/dataparser_transforms.json",
            "features/patch_features_ps56.pt",
            "features/is_visible_ps56.pt",
            "features/voxel_size_ps56.json",
        )
    ]
    for folder in ("images", "ns/renders/depth"):
        views = sorted(p for p in (source / folder).iterdir() if p.is_file())
        if len(views) != 30:
            raise ValueError(f"官方30视角缓存不完整: {source / folder}")
        files.extend(views)
    return {p.relative_to(source).as_posix(): file_sha256(p) for p in files}


def load_checkpoint(directory: Path, source: Path, scene: str) -> dict[str, Any]:
    hashes = json.loads((directory / "input_hashes.json").read_text())
    if hashes != input_hashes(source):
        raise ValueError(f"断点续跑输入变化: {scene}")
    prediction = json.loads((directory / "prediction.json").read_text())
    values = np.asarray(prediction["mass_interval_kg"], dtype=float)
    if (
        values.shape != (2,)
        or not np.isfinite(values).all()
        or not 0 < values[0] <= values[1]
        or prediction["scene_id"] != scene
        or prediction["mass_kg"] != float(values.mean())
    ):
        raise ValueError(f"断点预测无效: {scene}")
    return prediction


def run(args: argparse.Namespace) -> dict[str, Any]:
    """保留默认厚度积分和0.6修正，先封存预测，再读取质量标签。"""
    import open_clip
    import torch
    import trimesh

    if file_sha256(args.clip_checkpoint) != CLIP_SHA256:
        raise ValueError("CLIP检查点与官方固定文件不符")
    lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
    for name, digest in lock["files"].items():
        if file_sha256(args.repository / name) != digest:
            raise ValueError(f"官方源文件校验失败: {name}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    sys.path.insert(0, str(args.repository.resolve()))
    from predict_property import (
        predict_physical_property_integral,
        predict_physical_property_query,
    )

    ids = json.loads(args.cohort.read_text(encoding="utf-8"))["object_ids"]
    if len(set(ids)) != len(ids) or not ids:
        raise ValueError("对象清单必须非空且不重复")
    stop = len(ids) if args.limit is None else args.start + args.limit
    if args.start >= len(ids) or stop > len(ids):
        raise ValueError("对象范围超出冻结清单")
    ids = ids[args.start : stop]
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
        "upstream_commit": lock["commit"],
        "source_lock_sha256": file_sha256(args.source_lock),
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
        "point_prediction": "arithmetic_mean_of_native_range_bounds",
        "interval_semantics": "material_range_bounds_not_probability_quantiles",
    }
    protocol_path = args.output / "protocol.json"
    if (
        args.resume
        and protocol_path.exists()
        and json.loads(protocol_path.read_text(encoding="utf-8")) != protocol
    ):
        raise ValueError("断点续跑协议与原运行不一致")
    write_json(protocol_path, protocol)
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.clip_checkpoint)
    )
    model = model.to("cuda").eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    meshes = (
        json.loads(args.mesh_manifest.read_text(encoding="utf-8"))
        if args.mesh_manifest
        else None
    )
    predictions = []
    for scene in ids:
        directory = args.output / scene
        directory.mkdir(exist_ok=args.resume)
        source = args.root / "scenes" / scene
        if args.resume and (directory / "prediction.json").exists():
            prediction = load_checkpoint(directory, source, scene)
            if meshes is not None:
                raise ValueError("Mesh字段查询请使用独立输出目录")
            predictions.append(prediction)
            print("BASELINE_RESUMED", scene, flush=True)
            continue
        write_json(directory / "input_hashes.json", input_hashes(source))
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        interval = predict_physical_property_integral(
            settings, str(source), model, tokenizer
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
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
                "property_stage_seconds": elapsed,
                "peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_reserved_gpu_mib": torch.cuda.max_memory_reserved() / 2**20,
            }
        )
        write_json(directory / "prediction.json", predictions[-1])
        if meshes is None:
            print("BASELINE_PREDICTED", scene, interval, flush=True)
            continue
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
        args.output / "official_predictions.json",
        {p["scene_id"]: p["mass_interval_kg"] for p in predictions},
    )
    write_json(
        args.output / "prediction_seal.json",
        {
            name: file_sha256(args.output / name)
            for name in ("predictions.json", "official_predictions.json")
        },
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
        "cohort_size": len(ids),
        "property_stage_runtime_excludes": [
            "CLIP_loading",
            "NeRF_training",
            "LLM_calls",
        ],
    }
    from evaluation import ADE, ALDE, APE, MnRE

    native = np.asarray([p["mass_interval_kg"] for p in predictions])
    official = {
        name: float(function(native, np.asarray(target)))
        for name, function in zip(
            ("ade_kg", "alde", "ape", "mnre"), (ADE, ALDE, APE, MnRE), strict=True
        )
    }
    if not all(
        np.isclose(value, result["metrics"][name], rtol=1e-12, atol=1e-12)
        for name, value in official.items()
    ):
        raise ValueError("统一指标与作者评价脚本不一致")
    result["official_metric_parity"] = official
    write_json(args.output / "report.json", result)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--clip-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mesh-manifest", type=Path)
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--cohort", type=Path, default=Path("configs/five_object_pilot.json")
    )
    p.add_argument(
        "--source-lock", type=Path, default=Path("configs/nerf2physics_lock.json")
    )
    p.add_argument(
        "--weights",
        type=Path,
        default=Path("data/abo500_five/filtered_product_weights.json"),
    )
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.start < 0 or (a.limit is not None and a.limit < 1):
        p.error("需要有效的对象范围")
    print(json.dumps(run(a), indent=2))


if __name__ == "__main__":
    main()
