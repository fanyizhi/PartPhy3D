"""Check if archived part records can substantiate unresolved mass totals."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from evaluate_structure_cues import ROOT, WORK, errors, read, sealed


def main():
    geometry = read(WORK / "geometry_predictions.json")
    unified = read(WORK / "unified_comparison_500.json")
    prior50 = Path("/root/autodl-tmp/PartPhy3D/outputs/abo50_20261119")
    test100 = ROOT.parent / "abo100_20261120"
    pending = {k: [] for k, r in geometry.items() if not r["component_evidence"]}
    folders = {
        f.parent
        for d, pattern in (
            (prior50, "*/prediction_seal.json"),
            (test100, "batch_*/*/prediction_seal.json"),
            (test100, "batch_*/repaired_pipeline/*/prediction_seal.json"),
            (ROOT, "*_???/*/prediction_seal.json"),
            (ROOT, "*_???/repaired_pipeline/*/prediction_seal.json"),
        )
        for f in d.glob(pattern)
        if (f.parent / "predictions.json").is_file()
    }
    bad_seals = []
    comparable = 0
    for directory in sorted(folders):
        try:
            payload = sealed(directory)
        except (AssertionError, KeyError, ValueError) as exc:
            bad_seals.append(dict(directory=str(directory), error=type(exc).__name__))
            continue
        if not isinstance(payload, list):
            continue
        comparable += 1
        for row in payload:
            key = row.get("scene_id")
            if key not in pending or not row.get("parts"):
                continue
            parts = row["parts"]
            if not all(
                isinstance(p, dict) and isinstance(p.get("mass_kg"), (float, int))
                for p in parts
            ):
                continue
            total = sum(p["mass_kg"] for p in parts)
            pending[key].append(
                dict(
                    directory=str(directory),
                    part_sum_kg=total,
                    matches=math.isclose(
                        total, geometry[key]["raw"], rel_tol=1e-6, abs_tol=1e-8
                    ),
                )
            )
    counts = Counter(geometry[k]["split"] for k in pending)
    records = unified["objects"]
    comparison = {}
    for split in ("train", "val", "test"):
        ids = [k for k in records if geometry[k]["split"] == split]
        truth = np.array([records[k]["target_kg"] for k in ids])
        forced = np.array(
            [records[k]["image" if k in pending else "joint"] for k in ids]
        )
        comparison[split] = errors(truth, forced).mean(0).tolist()
    report = dict(
        archives_checked=comparable,
        skipped_invalid_seals=bad_seals,
        unresolved_counts=dict(counts),
        forced_image_fallback_metrics=comparison,
        objects={
            k: dict(
                split=geometry[k]["split"],
                archived_total_kg=geometry[k]["raw"],
                alternatives=v,
            )
            for k, v in pending.items()
        },
    )
    (WORK / "component_source_recovery_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    print(comparable, counts, comparison)
    print({k: len([p for p in v if p["matches"]]) for k, v in pending.items()})


if __name__ == "__main__":
    main()
