"""Seal full-cohort mass predictions, then score each official split separately."""

import argparse
import math
from pathlib import Path
from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo100_role_replay import predict
from run_component_mass_stability import sealed_rows
from tactile3d.models.calibration import mass_metrics


def finish_batch(batch):
    scenes = read_json(batch / "observations/cohort.json")["object_ids"]
    raw = sealed_rows(batch / "selected")
    parts = sealed_rows(batch / "panels")
    rows = []
    for scene in scenes:
        r = raw[scene]
        e = parts[scene]
        physical = r["construction_mass_kg"]
        matches = (
            physical is not None
            and e.get("parts")
            and math.isclose(
                sum(p["mass_kg"] for p in e["parts"]), physical, abs_tol=1e-6
            )
        )
        if matches:
            for p in e["parts"]:
                p.setdefault("thickness_mm", None)
        prediction = predict(r, e if matches else None)
        rows.append(
            dict(
                scene_id=scene,
                image_prior_kg=r["image_prior_kg"],
                construction_mass_kg=physical,
                **prediction,
            )
        )
    target = batch / "role"
    target.mkdir(exist_ok=True)
    write_json(target / "predictions.json", rows)
    write_json(
        target / "prediction_seal.json", {"sha256": digest(target / "predictions.json")}
    )


def finish_all(root):
    splits = read_json(root / "splits.json")
    test_file = root / "test_predictions.json"
    if digest(test_file) != read_json(root / "test_metrics.json")["prediction_sha256"]:
        raise ValueError("Archived test seal mismatch")
    rows = read_json(test_file)
    if set(rows) != set(splits["test"]):
        raise ValueError("Archived test membership mismatch")
    for batch in sorted(root.glob("*_[0-9][0-9][0-9]")):
        current = sealed_rows(batch / "role")
        if rows.keys() & current.keys():
            raise ValueError("Duplicated object")
        rows.update(current)
    ids = sum((splits[k] for k in ("train", "val", "test")), [])
    if len(rows) != len(set(ids)) or len(rows) != 500 or set(ids) != set(rows):
        raise ValueError("Incomplete full cohort")
    target = root / "predictions.json"
    write_json(target, {i: rows[i] for i in ids})
    write_json(root / "prediction_seal.json", {"sha256": digest(target)})
    weights = read_json(root / "weights.json")
    scores = {}
    for split, scenes in {**splits, "all": ids}.items():
        scores[split] = mass_metrics(
            [weights.get(i, weights.get(i.split("_")[0])) for i in scenes],
            [rows[i]["updated"] for i in scenes],
        )
    write_json(root / "score.json", scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["batch", "all"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    (finish_all if args.action == "all" else finish_batch)(args.directory)
