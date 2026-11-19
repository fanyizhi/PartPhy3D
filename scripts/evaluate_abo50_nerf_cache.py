"""Score sealed NeRF2Physics predictions against the frozen ABO-50 subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prepare_abo50_benchmark import digest, read_json, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.models.calibration import mass_metrics


def evaluate(cohort_path: Path, cache: Path, weights_path: Path, output: Path) -> None:
    cohort = read_json(cohort_path)
    scene_ids = cohort["object_ids"]
    if len(scene_ids) != 50 or len(set(scene_ids)) != 50:
        raise ValueError("Expected 50 distinct frozen test scenes")
    selected = set(scene_ids)
    predictions = {}
    for batch in sorted(cache.glob("batch_*")):
        seal = read_json(batch / "prediction_seal.json")
        source = batch / "predictions.json"
        if digest(source) != seal["predictions.json"]:
            raise ValueError(f"Prediction seal differs in {batch.name}")
        protocol = read_json(batch / "protocol.json")
        if protocol["scope"] != "released_cached_intermediates_official_property_stage":
            raise ValueError(f"Unexpected baseline stage in {batch.name}")
        for row in read_json(source):
            scene = row["scene_id"]
            if scene not in selected:
                continue
            if scene in predictions:
                raise ValueError(f"Duplicate prediction for {scene}")
            hashes = read_json(batch / scene / "input_hashes.json")
            observations = [
                "transforms.json",
                *(f"images/{scene}_{i:02d}.png" for i in range(30)),
            ]
            if any(
                hashes[p] != cohort["files_sha256"][f"scenes/{scene}/{p}"]
                for p in observations
            ):
                raise ValueError(f"Official cache observations differ for {scene}")
            predictions[scene] = row
    if selected != predictions.keys():
        raise ValueError(
            f"Missing predictions: {sorted(selected - predictions.keys())}"
        )

    weights = read_json(weights_path)
    rows = []
    for scene in scene_ids:
        target = weights.get(scene, weights.get(scene.split("_")[0]))
        if target is None:
            raise ValueError(f"Missing catalogue mass for {scene}")
        rows.append({**predictions[scene], "target_mass_kg": float(target)})
    metrics = mass_metrics(
        [row["target_mass_kg"] for row in rows], [row["mass_kg"] for row in rows]
    )
    write_json(
        output,
        {
            "method": "NeRF2Physics official cached property stage",
            "scope": "sealed prediction replay; official camera and image hashes match",
            "cohort_sha256": digest(cohort_path),
            "n": len(rows),
            "metrics": metrics,
            "per_scene": rows,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.cohort, args.cache, args.weights, args.output)


if __name__ == "__main__":
    main()
