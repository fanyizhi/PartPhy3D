"""Score the sealed, completed prefix of the ABO-50 PUGS-Qwen queue."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.models.calibration import mass_metrics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sealed(path, seal, key):
    if hashlib.sha256(path.read_bytes()).hexdigest() != read(seal)[key]:
        raise ValueError(f"Unverified predictions in {path.name}")
    return read(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("cohort", "nerf-cache", "pugs-reports", "ours", "output"):
        parser.add_argument(f"--{arg}", type=Path, required=True)
    args = parser.parse_args()
    ids = read(args.cohort)["object_ids"]
    cache = read(args.nerf_cache)
    if hashlib.sha256(args.cohort.read_bytes()).hexdigest() != cache["cohort_sha256"]:
        raise ValueError("NeRF2Physics cache is bound to a different cohort")
    nerf = {row["scene_id"]: row for row in cache["per_scene"]}
    ours = sealed(
        args.ours / "predictions.json", args.ours / "prediction_seal.json", "sha256"
    )
    test_ids = set(read(args.ours / "splits.json")["test"])
    completed = [
        scene
        for scene in ids
        if (args.pugs_reports / scene / "official_predictions.json").exists()
    ]
    if not completed or completed != ids[: len(completed)]:
        raise ValueError("Completed PUGS rows are not a continuous cohort prefix")
    rows = []
    for scene in completed:
        report = args.pugs_reports / scene
        interval = sealed(
            report / "official_predictions.json",
            report / "prediction_seal.json",
            "official_predictions.json",
        )[scene]
        if (
            len(interval) != 2
            or not np.isfinite(interval).all()
            or not 0 < interval[0] <= interval[1]
        ):
            raise ValueError(f"Invalid PUGS interval for {scene}")
        rows.append(
            {
                "scene": scene,
                "ours_test_split": scene in test_ids,
                "target_kg": nerf[scene]["target_mass_kg"],
                "nerf_cached_kg": nerf[scene]["mass_kg"],
                "pugs_qwen_interval_kg": interval,
                "pugs_qwen_kg": float(np.mean(interval)),
                "partphy3d_kg": ours[scene]["updated"],
            }
        )
    targets = [row["target_kg"] for row in rows]
    metrics = {
        key: mass_metrics(targets, [row[key] for row in rows])
        for key in ("nerf_cached_kg", "pugs_qwen_kg", "partphy3d_kg")
    }
    result = {
        "requested": len(ids),
        "completed": len(completed),
        "evaluation": "diagnostic queue prefix",
        "cohort_sha256": cache["cohort_sha256"],
        "pending_ids": ids[len(completed) :],
        "metrics": metrics,
        "per_scene": rows,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
