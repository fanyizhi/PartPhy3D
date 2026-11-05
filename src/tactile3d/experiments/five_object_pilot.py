"""固定五物体先导入口；沿用原开发对象，失败不补位。"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from tactile3d.abo500.dataset import ABO500Dataset, ABOScene
from tactile3d.abo500.pipeline import run_abo500_split
from tactile3d.config import load_config
from tactile3d.eval.abo500 import evaluate_abo500_predictions
from tactile3d.physics.part_io import file_sha256, write_json

DEFAULT_COHORT = Path("configs/five_object_pilot.json")
DEFAULT_CONFIG = Path("configs/abo500_mesh_smoke.yaml")
DEFAULT_COUNT = 5


def read_cohort(path: Path, limit: int = DEFAULT_COUNT) -> dict[str, Any]:
    """只取冻结队列的前缀，不按质量、可下载性或成功情况重新选样。"""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    ids = payload.get("object_ids") if isinstance(payload, dict) else None
    if (
        not isinstance(ids, list)
        or not ids
        or not all(isinstance(item, str) and item.strip() for item in ids)
        or len(ids) != len(set(ids))
    ):
        raise ValueError("先导配置需要非空、唯一的对象ID列表")
    if type(limit) is not int or not 1 <= limit <= min(len(ids), DEFAULT_COUNT):
        raise ValueError("当前先导物体数量必须在1到5之间，且不能超过冻结队列")
    return {
        **payload,
        "object_ids": ids[:limit],
        "requested_count": limit,
        "cohort_sha256": file_sha256(path),
        "held_out_benchmark": False,
    }


class PilotDataset(ABO500Dataset):
    def __init__(
        self, root: Path, mesh_root: Path, unit_to_meter: float, object_ids: list[str]
    ) -> None:
        super().__init__(root, mesh_root, unit_to_meter)
        self.object_ids = tuple(object_ids)

    def scene_ids(self, split: str = "pilot") -> list[str]:
        # 不能依照权重元数据过滤队列，否则缺数据的对象会静默消失。
        return list(self.object_ids)

    def get_scene(self, scene_id: str, split: str | None = None) -> ABOScene:
        if scene_id not in self.object_ids:
            raise ValueError("请求的对象不在冻结先导队列中")
        return replace(super().get_scene(scene_id, split), mass_kg=None)


def run_pilot(
    root: Path,
    mesh_root: Path,
    output: Path,
    cohort: Path = DEFAULT_COHORT,
    config: Path = DEFAULT_CONFIG,
    limit: int = DEFAULT_COUNT,
    unit_to_meter: float | None = None,
    prepare_only: bool = False,
) -> dict[str, Any]:
    """先封存队列及预测，再单独接入称重标签进行评价。

    Args:
        root: ABO图像、相机和元数据目录。
        mesh_root: 参考或生成Mesh目录，实验中必须说明来源。
        output: 新目录，拒绝覆盖既有结果。
        cohort: 固定对象列表。
        config: 当前推理实现的配置，不自动替换为其他模型。
        limit: 初次检查可设1，默认正式先导为5。
        unit_to_meter: 独立核验的几何尺度，不能由质量反推。
        prepare_only: 只冻结选择，不加载数据或执行模型。

    Returns:
        完整请求数量、成功/失败记录和评价范围。
    """
    selection = read_cohort(cohort, limit)
    if not prepare_only and (
        unit_to_meter is None or not math.isfinite(unit_to_meter) or unit_to_meter <= 0
    ):
        raise ValueError("实际推理必须通过--mesh-unit-to-meter指定已核验的正尺度")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "selection_before_evaluation.json", selection)
    if prepare_only:
        report = {"status": "prepared_not_executed", **selection}
        write_json(output / "pilot_status.json", report)
        return report
    try:
        cfg = load_config(config)
        dataset = PilotDataset(root, mesh_root, unit_to_meter, selection["object_ids"])
        report = run_abo500_split(dataset, cfg, "pilot", output)
        prediction_path = output / "predictions.json"
        sealed_hash = file_sha256(prediction_path)
        write_json(
            output / "prediction_seal.json",
            {
                "sha256": sealed_hash,
                "config_sha256": file_sha256(config),
                "target_mass_passed_to_inference": False,
                "dataset_reader_loads_weight_metadata": True,
            },
        )
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
        # 评价使用新文件，原始无标签预测保持封存。
        evaluation = [
            {**item, "target_mass_kg": dataset.weights[item["scene_id"]]}
            for item in predictions
        ]
        if evaluation:
            path = output / "evaluation_predictions.json"
            write_json(path, evaluation)
            evaluate_abo500_predictions(path, output / "metrics_valid_objects.json")
        if file_sha256(prediction_path) != sealed_hash:
            raise ValueError("评价期间原始预测被改写")
        report.update(
            status="complete" if report["num_failed"] == 0 else "incomplete",
            held_out_benchmark=False,
            metrics_scope="successful_objects_only",
        )
        write_json(output / "pilot_status.json", report)
        return report
    except Exception as error:
        write_json(
            output / "pilot_status.json",
            {
                "status": "failed",
                "requested_count": limit,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/abo_500"))
    parser.add_argument("--mesh-root", type=Path, default=Path("data/abo500_ten"))
    parser.add_argument("--output", type=Path, default=Path("outputs/abo500_five"))
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--limit", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--mesh-unit-to-meter", type=float)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    try:
        report = run_pilot(
            args.root,
            args.mesh_root,
            args.output,
            args.cohort,
            args.config,
            args.limit,
            args.mesh_unit_to_meter,
            args.prepare_only,
        )
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"先导实验失败：{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "incomplete":
        parser.exit(1, "部分对象失败，已保留原队列和失败记录。\n")


if __name__ == "__main__":
    main()
