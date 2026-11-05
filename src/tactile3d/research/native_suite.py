"""Mesh原生CPU验收入口：已知构造积分、编辑和可微响应学习。"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from tactile3d.experiments.mesh_native_demo import run_mesh_demo
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.research.learning import DEFAULT_EPOCHS
from tactile3d.research.native_learning import run_native_learning
from tactile3d.research.workflow import DEFAULT_COUNT, DEFAULT_SEEDS


def run_native_suite(
    output: Path,
    count: int = DEFAULT_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """先验证解析构造，再执行独立合成学习，所有产物保留来源和哈希。

    Args:
        output: 不存在的结果目录。
        count: 合成诊断对象数，不改变五物体真实开发队列。
        epochs: 预先固定的训练轮数。
        seeds: 同一数据划分上的初始化种子。

    Returns:
        软件诊断状态；真实图像重建、语义分割和VLM训练仍另行验收。
    """
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running",
        "scope": "offline_CPU_diagnostics",
        "representation": "closed_material_region_mesh",
        "voxelization_used": False,
        "python": platform.python_version(),
    }
    write_json(output / "suite.json", report)
    start = time.perf_counter()
    try:
        edit = run_mesh_demo(output / "editing")
        learning = run_native_learning(output / "learning", count, epochs, seeds)
        report.update(
            status="complete",
            seconds=time.perf_counter() - start,
            example_mass_before_kg=edit["before"]["mass_kg"],
            example_mass_after_kg=edit["after"]["mass_kg"],
            learning_seconds=learning["seconds"],
            data="specified_construction_and_independent_analytic_synthetic_labels",
            full_research_pipeline_complete=False,
            real_model_inference=False,
            server_training_started=False,
            remaining=[
                "Real image reconstruction and fine semantic parts",
                "Evidence-backed hidden construction and independent measurements",
                "Qwen adaptation and multimodal part encoders",
                "Error-controlled sparse construction in actual geometry decoder",
                "Fixed five-object development pilot and published baselines",
            ],
        )
        write_json(output / "suite.json", report)
        write_json(
            output / "ARTIFACTS_SHA256.json",
            {
                path.relative_to(output).as_posix(): file_sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        )
        return report
    except Exception as error:
        report.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        write_json(output / "suite.json", report)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/mesh_reproduction")
    )
    parser.add_argument("--objects", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    try:
        report = run_native_suite(
            args.output, args.objects, args.epochs, tuple(args.seeds)
        )
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Mesh验收失败：{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
