"""统一验收入口：学习、编辑、可辨识性与自适应曲面预算实验。"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

from tactile3d.experiments.edit_commands import (
    DEFAULT_COMMANDS,
    export_result,
    synthetic_reference,
)
from tactile3d.experiments.observability import audit_observability
from tactile3d.experiments.part_edit_demo import make_example
from tactile3d.physics.edit_program import execute_program, program_from_commands
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.research.adaptive_demo import run_adaptive
from tactile3d.research.learning import DEFAULT_EPOCHS
from tactile3d.research.workflow import DEFAULT_COUNT, DEFAULT_SEEDS, run_experiment


def run_suite(
    output: Path, count: int, epochs: int, seeds: tuple[int, ...]
) -> dict[str, Any]:
    """所有任务都在本机CPU执行；数据生成包含于各自计时和清单。"""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "status": "running",
        "scope": "offline_CPU_diagnostics",
        "python": platform.python_version(),
    }
    write_json(output / "suite.json", report)
    start = time.perf_counter()
    try:
        learning = run_experiment(output / "learning", count, epochs, seeds)
        report["learning_seconds"] = learning["seconds"]
        source, donor = make_example()
        donor_key = str(Path("synthetic_aluminum_leg").resolve())
        program = program_from_commands(
            source, DEFAULT_COMMANDS, {}, {"aluminum_leg": donor_key}
        )
        result = execute_program(source, program, {donor_key: donor})
        export_result(result, output / "editing", synthetic_reference())
        write_json(output / "observability.json", audit_observability())
        run_adaptive(output / "adaptive")
        report.update(
            status="complete",
            seconds=time.perf_counter() - start,
            real_model_inference=False,
            server_training_started=False,
        )
        write_json(output / "suite.json", report)
        hashes = {
            path.relative_to(output).as_posix(): file_sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        }
        write_json(output / "ARTIFACTS_SHA256.json", hashes)
    except Exception as error:
        report.update(
            status="failed", error=str(error), error_type=type(error).__name__
        )
        write_json(output / "suite.json", report)
        raise
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/full_reproduction")
    )
    parser.add_argument("--objects", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    try:
        report = run_suite(args.output, args.objects, args.epochs, tuple(args.seeds))
    except (OSError, ValueError, ImportError) as error:
        parser.exit(1, f"验收失败：{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
