"""固定CLIP图像特征上的尺度基线，300训练/100验证/100测试，不使用质量。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.models.scale_ridge import (
    calibrate_log_scale,
    fit_log_scale,
    scale_metrics,
)
from tactile3d.physics.part_io import file_sha256, write_json

ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
BATCH_SIZE = 32
SEED = 17
INTERVAL_COVERAGE = 0.9


def run(args: argparse.Namespace) -> dict[str, Any]:
    """一次特征提取，验证集选alpha；先封存测试预测，再报告测试误差。"""
    import open_clip
    import torch

    manifest_path = args.dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["images"]
    if manifest.get("cross_split_ASIN_overlap") is not False:
        raise ValueError("数据清单未确认对象跨划分无重叠")
    ids = [r["scene_id"] for r in rows]
    if len(ids) != 500 or len(set(ids)) != 500:
        raise ValueError("尺度基线需要官方完整500对象，不能用五对象子集代替")
    split = np.array([r["split"] for r in rows])
    for name, count in (("train", 300), ("val", 100), ("test", 100)):
        if np.count_nonzero(split == name) != count:
            raise ValueError("对象划分与300/100/100协议不一致")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    if catalog.get("scope") != "catalog_dimension_assisted_input_no_mass_labels":
        raise ValueError("需要排除质量标签的官方商品尺寸输入")
    for scene in ids:
        if catalog["objects"][scene]["status"] != "available":
            raise ValueError(f"{scene}缺少无冲突尺寸；需另定缺失协议，不能悄悄删样本")
    declared = json.loads(
        (ROOT / "configs/pretrained/clip-datacomp_manifest.json").read_text()
    )
    if file_sha256(args.checkpoint) != declared["files"][0]["Sha256"]:
        raise ValueError("CLIP模型校验失败")
    if not torch.cuda.is_available():
        raise RuntimeError("此已执行协议使用CUDA提取CLIP特征")
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "scope": "image_only_scale_baseline_catalog_supervised_training",
        "dataset_sha256": file_sha256(manifest_path),
        "catalog_sha256": file_sha256(args.catalog),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "alphas": list(ALPHAS),
        "validation_selection": "mean_absolute_log_error",
        "seed": SEED,
        "image_views": 1,
        "mass_labels_used": False,
        "test_labels_passed_to_fitting": False,
        "part_latent_training": False,
        "validation_tuning_objects": 50,
        "independent_calibration_objects": 50,
        "nominal_scale_interval_coverage": INTERVAL_COVERAGE,
    }
    write_json(args.output / "protocol.json", protocol)
    torch.manual_seed(SEED)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.checkpoint)
    )
    model = model.to("cuda").eval()
    features, image_hashes = [], {}
    start = time.perf_counter()
    for index in range(0, len(rows), BATCH_SIZE):
        images = []
        for row in rows[index : index + BATCH_SIZE]:
            path = args.dataset / "images" / f"{row['scene_id']}.png"
            digest = file_sha256(path)
            if digest != row["derived_sha256"]:
                raise ValueError(f"图像哈希不匹配：{row['scene_id']}")
            image_hashes[row["scene_id"]] = digest
            with Image.open(path) as image:
                images.append(preprocess(image.convert("RGB")))
        with torch.inference_mode():
            encoded = model.encode_image(torch.stack(images).to("cuda"))
            encoded = encoded / encoded.norm(dim=-1, keepdim=True)
        features.append(encoded.float().cpu().numpy())
        print("SCALE_FEATURES", min(index + BATCH_SIZE, len(rows)), flush=True)
    x = np.concatenate(features).astype(float)
    lengths = np.array([catalog["objects"][scene]["max_dimension_m"] for scene in ids])
    train, val, test = split == "train", split == "val", split == "test"
    validation_indices = np.flatnonzero(val)
    tuning, calibration = validation_indices[:50], validation_indices[50:]
    validations = []
    for alpha in ALPHAS:
        fit = fit_log_scale(x[train], lengths[train], alpha)
        validations.append(
            {"alpha": alpha, **scale_metrics(lengths[tuning], fit.predict(x[tuning]))}
        )
    best = min(validations, key=lambda r: r["mean_absolute_log_error"])["alpha"]
    fit = fit_log_scale(x[train], lengths[train], best)
    radius = calibrate_log_scale(
        lengths[calibration], fit.predict(x[calibration]), INTERVAL_COVERAGE
    )
    prediction = fit.predict(x[test])
    baseline = np.full(len(prediction), np.exp(np.log(lengths[train]).mean()))
    test_ids = np.array(ids)[test].tolist()
    frozen_five = json.loads((ROOT / "configs/five_object_pilot.json").read_text())[
        "object_ids"
    ]
    if test_ids[:5] != frozen_five:
        raise ValueError("历史开发对象顺序与测试清单不一致")
    sealed = [
        {
            "scene_id": scene,
            "max_length_m": float(p),
            "train_mean_baseline_m": float(b),
            "conformal_scale_low_m": float(p * np.exp(-radius)),
            "conformal_scale_high_m": float(p * np.exp(radius)),
        }
        for scene, p, b in zip(test_ids, prediction, baseline, strict=True)
    ]
    write_json(args.output / "test_predictions.json", sealed)
    write_json(
        args.output / "prediction_seal.json",
        {"sha256": file_sha256(args.output / "test_predictions.json")},
    )
    np.savez_compressed(
        args.output / "scale_head.npz",
        feature_mean=fit.feature_mean,
        coefficients=fit.coefficients,
        log_length_mean=fit.log_length_mean,
        alpha=fit.alpha,
    )
    np.savez_compressed(
        args.output / "image_features.npz",
        features=x,
        scene_ids=np.array(ids),
        splits=split,
    )
    write_json(args.output / "image_hashes.json", image_hashes)
    write_json(args.output / "validation.json", validations)
    write_json(
        args.output / "split_roles.json",
        {
            "train": np.array(ids)[train].tolist(),
            "tuning": np.array(ids)[tuning].tolist(),
            "calibration": np.array(ids)[calibration].tolist(),
            "test": test_ids,
        },
    )
    truth = lengths[test]
    residual_test = np.array([scene not in test_ids[:10] for scene in test_ids])
    result = {
        "scope": protocol["scope"],
        "selected_alpha": best,
        "train_objects": 300,
        "validation_objects": 100,
        "test_objects": 100,
        "test_metrics": {
            "train_geometric_mean": scale_metrics(truth, baseline),
            "CLIP_ridge": scale_metrics(truth, prediction),
        },
        "outside_historical_ten_90": {
            "train_geometric_mean": scale_metrics(
                truth[residual_test], baseline[residual_test]
            ),
            "CLIP_ridge": scale_metrics(
                truth[residual_test], prediction[residual_test]
            ),
        },
        "seconds_including_feature_encoding": time.perf_counter() - start,
        "target_is_catalog_dimension_not_lab_measurement": True,
        "split_conformal": {
            "calibration_objects": 50,
            "log_radius": radius,
            "nominal_coverage": INTERVAL_COVERAGE,
            "test_coverage": float(
                np.mean(np.abs(np.log(prediction / truth)) <= radius)
            ),
            "mean_interval_width_m": float(
                np.mean(prediction * (np.exp(radius) - np.exp(-radius)))
            ),
            "requires_exchangeability": True,
            "is_full_physics_confidence_interval": False,
        },
        "full_mass_accuracy_established": False,
    }
    write_json(args.output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
