"""只使用有人工核验记录的PNG评价部件提议；缺少真值时指标保持空。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.part_metrics import part_mask_metrics

VIEWS = (0, 10, 20)


def evaluate(
    fields: Path,
    annotations: Path,
    data_root: Path,
    output: Path,
    prediction_source: str = "mesh_projection",
) -> dict[str, Any]:
    """校验图像哈希、标签字典、人工确认和分辨率，报告标注覆盖率。"""
    if output.exists():
        raise FileExistsError("拒绝覆盖已有分割评价")
    if prediction_source not in {"image", "mesh_projection"}:
        raise ValueError("预测来源只能是image或mesh_projection")
    source = json.loads((fields / "report.json").read_text(encoding="utf-8"))
    cases, pending = [], []
    for obj in source["objects"]:
        scene = obj["scene_id"]
        token = json.loads(
            (fields / scene / "part_tokens.json").read_text(encoding="utf-8")
        )
        mapping = [(p["part_index"], p["part_id"]) for p in token["parts"]]
        for view in VIEWS:
            case_id = f"{scene}_{view:02d}"
            png, meta = (
                annotations / f"{case_id}_gt.png",
                annotations / f"{case_id}_gt.json",
            )
            if not png.is_file() or not meta.is_file():
                pending.append(
                    {"case_id": case_id, "reason": "missing_human_mask_or_record"}
                )
                continue
            review = json.loads(meta.read_text(encoding="utf-8"))
            image_path = data_root / "scenes" / scene / "images" / f"{case_id}.png"
            if (
                review.get("schema") != "partphy3d_human_mask_v1"
                or review.get("case_id") != case_id
                or review.get("human_reviewed") is not True
                or review.get("prediction_used_as_reference") is not False
                or not review.get("annotator")
                or review.get("image_sha256") != file_sha256(image_path)
                or [(p["index"], p["part_id"]) for p in review["parts"]] != mapping
            ):
                raise ValueError(f"{case_id}人工记录、图像或标签字典核对失败")
            with Image.open(png) as image:
                array = np.asarray(image)
                if array.ndim == 3:
                    if not np.array_equal(
                        array[..., :3], np.repeat(array[..., :1], 3, axis=2)
                    ):
                        raise ValueError("需要类别编号PNG，不接受可视化彩色掩码作真值")
                    array = array[..., 0]
                truth = array.copy()
            with Image.open(image_path) as image:
                if truth.shape != (image.height, image.width):
                    raise ValueError("人工PNG尺寸必须与原图一致")
                if (review.get("width"), review.get("height")) != image.size:
                    raise ValueError("人工记录中的图像尺寸不一致")
                foreground = np.asarray(image.convert("RGBA"))[..., 3] >= 128
            valid_foreground = foreground & (truth > 0) & (truth != 255)
            if not valid_foreground.any():
                pending.append(
                    {"case_id": case_id, "reason": "no_human_foreground_labels"}
                )
                continue
            with np.load(fields / scene / f"view_{view:02d}.npz") as data:
                if prediction_source == "image":
                    small = data["labels"].copy()
                else:
                    ids = data["face_ids"].copy()
                    with np.load(fields / scene / "surface_fields.npz") as surface:
                        labels = surface["labels"]
                    if ids.min() < -1 or ids.max() >= len(labels):
                        raise ValueError("可见面缓存与三维部件标签不匹配")
                    small = np.zeros(ids.shape, np.int32)
                    visible = ids >= 0
                    small[visible] = labels[ids[visible]]
            predicted = np.asarray(
                Image.fromarray(small.astype(np.int32)).resize(
                    (truth.shape[1], truth.shape[0]), Image.Resampling.NEAREST
                )
            )
            metrics = part_mask_metrics(predicted, truth, len(mapping))
            cases.append(
                {
                    "case_id": case_id,
                    "mask_sha256": file_sha256(png),
                    "review_sha256": file_sha256(meta),
                    "annotated_foreground_fraction": float(
                        valid_foreground.sum() / max(foreground.sum(), 1)
                    ),
                    "metrics": metrics,
                }
            )
    expected = len(source["objects"]) * len(VIEWS)
    complete = (
        bool(cases)
        and not pending
        and all(c["annotated_foreground_fraction"] == 1 for c in cases)
    )
    result = {
        "scope": "independent_human_mask_evaluation",
        "prediction_source": prediction_source,
        "source_report_sha256": file_sha256(fields / "report.json"),
        "status": (
            "complete"
            if complete
            else ("partial_annotations" if cases else "pending_annotations")
        ),
        "expected_cases": expected,
        "evaluated_cases": len(cases),
        "cases": cases,
        "pending": pending,
        "macro_case_mean_iou": (
            float(np.mean([c["metrics"]["mean_iou"] for c in cases])) if cases else None
        ),
        "complete_foreground_annotation": complete,
        "automatic_predictions_used_as_ground_truth": False,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "data/abo500_five")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prediction-source",
        choices=("image", "mesh_projection"),
        default="mesh_projection",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(
                args.fields,
                args.annotations,
                args.root,
                args.output,
                args.prediction_source,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
