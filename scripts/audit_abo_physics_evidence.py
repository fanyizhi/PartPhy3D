"""审计本地完整ABO-500的尺度、材料和质量证据，不读取图像像素或改动数据。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.physics.part_io import file_sha256, write_json

DIMENSION_KEYS = {
    "dimensions",
    "item_dimensions",
    "width",
    "height",
    "length",
    "unit_to_meter",
    "mesh_unit_to_meter",
    "camera_unit_to_meter",
    "units",
    "unit",
    "physical_scale",
    "metric_scale",
}


def dimension_fields(value: Any, prefix: str = "") -> list[str]:
    """只找字段存在性；出现scale字样并不自动认定为米制尺度。"""
    if isinstance(value, dict):
        result = []
        for key, item in value.items():
            name = prefix + "/" + key
            if key.lower() in DIMENSION_KEYS:
                result.append(name)
            if key != "frames":
                result.extend(dimension_fields(item, name))
        return result
    return []


def audit(root: Path, output: Path) -> dict[str, Any]:
    """按官方对象划分核对本地文件，区分商品质量标签与未验证材料提议。"""
    if not root.is_dir():
        raise ValueError("ABO数据根目录不存在")
    splits_path, weights_path = (
        root / "splits.json",
        root / "filtered_product_weights.json",
    )
    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    weights = json.loads(weights_path.read_text(encoding="utf-8"))
    records, metadata_keys = [], Counter()
    for split, scenes in splits.items():
        if not isinstance(scenes, list):
            raise TypeError("划分列表格式不合法")
        for scene in scenes:
            directory = root / "scenes" / scene
            camera_path, info_path = (
                directory / "transforms.json",
                directory / "info.json",
            )
            camera = json.loads(camera_path.read_text(encoding="utf-8"))
            info = json.loads(info_path.read_text(encoding="utf-8"))
            metadata_keys.update(info.keys())
            dataparser = directory / "ns/dataparser_transforms.json"
            ns_keys = []
            if dataparser.is_file():
                ns_keys = list(json.loads(dataparser.read_text(encoding="utf-8")))
            mass = weights.get(scene, weights.get(scene.split("_")[0]))
            images = list((directory / "images").glob("*.png"))
            dimensions = dimension_fields(camera) + dimension_fields(info)
            references = [
                str(p.relative_to(directory))
                for suffix in ("*.glb", "*.obj", "*.ply")
                for p in directory.glob(suffix)
            ]
            records.append(
                {
                    "scene_id": scene,
                    "split": split,
                    "image_count": len(images),
                    "camera_frames": len(camera.get("frames", [])),
                    "catalog_mass_label_present": mass is not None,
                    "potential_dimension_fields": dimensions,
                    "root_mesh_files": references,
                    "has_cached_nerf_point_cloud": (
                        directory / "ns/point_cloud.ply"
                    ).is_file(),
                    "nerfstudio_dataparser_keys": ns_keys,
                    "camera_sha256": file_sha256(camera_path),
                    "material_proposals_sha256": file_sha256(info_path),
                }
            )
    result = {
        "scope": "local_full_ABO500_metadata_evidence_audit",
        "splits_sha256": file_sha256(splits_path),
        "weights_sha256": file_sha256(weights_path),
        "objects": len(records),
        "split_counts": dict(Counter(r["split"] for r in records)),
        "image_count": sum(r["image_count"] for r in records),
        "objects_with_catalog_mass": sum(
            r["catalog_mass_label_present"] for r in records
        ),
        "objects_with_candidate_dimension_fields": sum(
            bool(r["potential_dimension_fields"]) for r in records
        ),
        "info_keys": dict(metadata_keys),
        "physical_scale_verified_by_this_audit": False,
        "internal_material_ground_truth_verified": False,
        "material_proposal_values_are_ground_truth": False,
        "nerfstudio_scale_is_independent_physical_measurement": False,
        "records": records,
    }
    if output.exists():
        raise FileExistsError("拒绝覆盖既有数据审计")
    write_json(output, result)
    return {k: v for k, v in result.items() if k != "records"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.root, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
