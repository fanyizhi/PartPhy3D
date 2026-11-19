"""Run the frozen InstantMesh front end on each scene in the ABO-50 manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from prepare_abo50_benchmark import digest, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--sparse-surface-weights", action="store_true")
    args = parser.parse_args()
    manifest = read_json(args.observations / "cohort.json")
    cases = manifest["object_ids"]
    if (
        args.expected_count < 1
        or len(cases) != args.expected_count
        or len(set(cases)) != len(cases)
    ):
        raise ValueError(f"Expected {args.expected_count} distinct objects")
    for relative, expected in manifest["files_sha256"].items():
        if digest(args.observations / relative) != expected:
            raise ValueError(f"Observation hash differs: {relative}")

    args.output.mkdir(parents=True, exist_ok=False)
    runner = Path(__file__).with_name("run_observed_instantmesh.py")
    rows = []
    for scene in cases:
        scene_output = args.output / scene
        log = args.output / f"{scene}.log"
        command = [
            sys.executable,
            str(runner),
            "--scene",
            str(args.observations / "scenes" / scene),
            "--repository",
            str(args.repository),
            "--checkpoint",
            str(args.checkpoint),
            "--encoder-config",
            str(args.encoder_config),
            "--output",
            str(scene_output),
        ]
        if args.sparse_surface_weights:
            command.append("--sparse-surface-weights")
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        rows.append({"scene_id": scene, "exit_code": process.returncode})
        write_json(
            args.output / "progress.json", {"requested": len(cases), "rows": rows}
        )
        print(scene, process.returncode, flush=True)
    if any(row["exit_code"] for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
