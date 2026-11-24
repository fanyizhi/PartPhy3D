"""Compare archived baselines on identical test objects and audit part-level sums."""

from collections import Counter
import json

import numpy as np
from evaluate_structure_cues import WORK, errors, read


def main():
    records = read(WORK / "structural_ablation.json")["records"]
    parts = read(WORK / "part_mass_calibration.json")["objects"]
    models = read(WORK / "matched_models.json")["results"]
    nerf = read(WORK / "nerf.json")
    pugs = {r["scene"]: r for r in read(WORK / "pugs_prefix.json")["per_scene"]}
    ids = sorted(k for k, r in records.items() if r["split"] == "test")
    results = {}
    for label, keys in [("test", ids), ("pugs_prefix", [k for k in ids if k in pugs])]:
        truth = np.array([records[k]["gt"] for k in keys])
        if label == "pugs_prefix":
            np.testing.assert_allclose(truth, [pugs[k]["target_kg"] for k in keys])
            np.testing.assert_allclose(
                [np.mean(nerf[k]) for k in keys],
                [pugs[k]["nerf_cached_kg"] for k in keys],
            )
        estimates = {
            "NeRF2Physics": [np.mean(nerf[k]) for k in keys],
            "Three-view image": [models["image"]["predictions"][k] for k in keys],
            "Image + structure": [models["joint"]["predictions"][k] for k in keys],
            "Part-calibrated structure": [parts[k]["full_fit_kg"] for k in keys],
        }
        if label == "pugs_prefix":
            estimates["PUGS-Qwen"] = [pugs[k]["pugs_qwen_kg"] for k in keys]
        results[label] = {
            "ids": keys,
            "metrics": {
                m: errors(truth, np.array(v)).mean(0).tolist()
                for m, v in estimates.items()
            },
            "source_counts": dict(Counter(parts[k]["source"] for k in keys)),
        }
    for r in parts.values():
        np.testing.assert_allclose(
            sum(p["calibrated_kg"] for p in r["fitted_parts"]), r["full_fit_kg"]
        )
    (WORK / "paired_baselines.json").write_text(json.dumps(results, indent=2))
    for label, r in results.items():
        print(label, len(r["ids"]), r["source_counts"], r["metrics"])


if __name__ == "__main__":
    main()
