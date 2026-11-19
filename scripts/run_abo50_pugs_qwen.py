"""Run PUGS's pinned reconstruction and integration with local Qwen proposals."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prepare_abo50_benchmark import digest, read_json, write_json


def run(args: argparse.Namespace) -> None:
    cohort_path = args.observations / "cohort.json"
    cohort = read_json(cohort_path)
    cases = cohort["object_ids"]
    if len(cases) != 50 or len(set(cases)) != 50:
        raise ValueError("Expected 50 distinct frozen scenes")
    args.output.mkdir(parents=True, exist_ok=False)
    stage_runner = Path(__file__).with_name("run_pugs_baseline.py")
    rows = []
    for scene in cases:
        original = args.observations / "scenes" / scene
        workspace = args.output / "scenes" / scene
        report = args.output / "reports" / scene
        report.mkdir(parents=True)
        workspace.mkdir(parents=True)
        (workspace / "images").symlink_to(
            (original / "images").resolve(), target_is_directory=True
        )
        (workspace / "transforms.json").symlink_to(
            (original / "transforms.json").resolve()
        )
        common = [
            "--repository",
            str(args.repository),
            "--scene",
            str(workspace),
            "--output",
            str(report),
        ]
        completed = []
        stages = (
            (sys.executable, "prepare", []),
            (args.pugs_python, "reconstruct", ["--sam-checkpoint", str(args.sam)]),
            (
                sys.executable,
                "propose",
                ["--model", str(args.model), "--view-index", "8"],
            ),
            (args.pugs_python, "integrate", ["--clip-checkpoint", str(args.clip)]),
        )
        row = {"scene_id": scene, "completed_stages": completed}
        for executable, name, options in stages:
            with (report / f"{name}.log").open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    [str(executable), str(stage_runner), name, *common, *options],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if process.returncode:
                row["failed_stage"] = name
                row["exit_code"] = process.returncode
                break
            completed.append(name)
        if len(completed) == len(stages):
            source = report / "official_predictions.json"
            expected = read_json(report / "prediction_seal.json")[source.name]
            if digest(source) != expected:
                raise ValueError(f"PUGS prediction seal mismatch at {scene}")
            lower, upper = read_json(source)[scene]
            row["mass_kg"] = (lower + upper) / 2
        rows.append(row)
        write_json(
            args.output / "progress.json", {"requested": len(cases), "rows": rows}
        )
        print(scene, row.get("failed_stage", "complete"), flush=True)
    result = args.output / "predictions.json"
    write_json(result, rows)
    write_json(
        args.output / "prediction_seal.json",
        {"sha256": digest(result), "cohort_sha256": digest(cohort_path)},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "observations",
        "repository",
        "pugs-python",
        "sam",
        "model",
        "clip",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
