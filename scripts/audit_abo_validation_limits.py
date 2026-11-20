"""Diagnose validation errors using the frozen fusion range and geometric similarity."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "paper_assets/abo500_20261120"


def read(path):
    return json.loads(path.read_bytes())


def sealed(folder):
    path = folder / "predictions.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != read(folder / "prediction_seal.json")["sha256"]:
        raise ValueError(f"Invalid prediction seal in {folder}")
    return {row["scene_id"]: row for row in read(path)}


def audit(panels, weights):
    ids = read(ROOT / "splits.json")["val"]
    catalog = read(PROJECT / "configs/evidence/abo500_catalog_dimensions.json")["objects"]
    truth = read(weights)
    rows = []
    for start in range(0, len(ids), 25):
        batch = f"val_{start:03d}"
        physical = sealed(panels / batch / "panels")
        role = sealed(ROOT / "batches" / batch)
        if set(role) != set(ids[start:start + 25]):
            raise ValueError(batch)
        for scene, r in role.items():
            y = truth.get(scene, truth.get(scene.split("_")[0]))
            image, mass = r["image_prior_kg"], r["updated"]
            lo, hi = image / math.sqrt(2), image * math.sqrt(2)
            if not lo - 1e-8 <= mass <= hi + 1e-8:
                raise ValueError("Frozen bounded fusion range violated")
            row = dict(scene=scene, target=y, image=image, frozen=mass, absolute_error=abs(y-mass),
                       fusion_lower=lo, fusion_upper=hi, unavoidable_error=max(lo-y, y-hi, 0))
            p = physical.get(scene, {})
            base = r["construction_mass_kg"]
            length = catalog[scene].get("max_dimension_m")
            if length and p.get("predicted_max_length_m") and base and math.isclose(
                    p.get("construction_mass_kg") or 0, base, abs_tol=1e-6):
                # Dimensional thought experiment: all lengths, including walls, scale together.
                predicted = p["predicted_max_length_m"]
                scaled = base * (length / predicted) ** 3
                row.update(predicted_length_m=predicted, catalog_length_m=length,
                           physical=base, similarity_mass=scaled)
            rows.append(row)
    ranked = sorted(rows, key=lambda r: -r["absolute_error"])
    valid = [r for r in ranked if "similarity_mass" in r]
    summary = dict(n=len(rows), below=sum(r["target"] < r["fusion_lower"] for r in rows),
                   above=sum(r["target"] > r["fusion_upper"] for r in rows),
                   unavoidable_ade=np.mean([r["unavoidable_error"] for r in rows]),
                   top10_error_share=sum(r["absolute_error"] for r in ranked[:10]) /
                                     sum(r["absolute_error"] for r in ranked),
                   scale_diagnostic_n=len(valid),
                   scale_improves=sum(abs(r["target"]-r["similarity_mass"]) <
                                      abs(r["target"]-r["physical"]) for r in valid))
    path = ROOT / "validation_failure_audit.json"
    path.write_text(json.dumps(dict(summary=summary, rows=ranked), indent=2), encoding="utf-8")
    print(summary)
    for r in ranked[:3]:
        print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panels", type=Path)
    parser.add_argument("weights", type=Path)
    args = parser.parse_args()
    audit(args.panels, args.weights)
