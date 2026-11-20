"""Validation replay of an explicit unsupported-dominant-part fallback."""
import argparse
import json
import math
import re
from pathlib import Path

from audit_abo_validation_limits import ROOT, read, sealed
from tactile3d.models.calibration import mass_metrics

INFERENCE = re.compile(r"\b(inferred|assumed|presumed|not visible|not directly visible)\b", re.I)


def evaluate(panels, sealed_role):
    audit = read(ROOT / "validation_failure_audit.json")["rows"]
    ids = {r["scene"] for r in audit}
    matched = set()
    triggered = {}
    for folder in sorted(panels.glob("val_*")):
        for scene, p in sealed(folder / "panels").items():
            current = sealed_role[scene]
            parts = p.get("parts", [])
            total = current.get("construction_mass_kg")
            if not parts or not total or not math.isclose(sum(a["mass_kg"] for a in parts), total, abs_tol=1e-6):
                continue
            matched.add(scene)
            for a in parts:
                evidence = a.get("evidence")
                if a["mass_kg"] > total / 2 and isinstance(evidence, str) and INFERENCE.search(evidence):
                    triggered[scene] = dict(part=a["name"], evidence=a["evidence"],
                        share=a["mass_kg"]/total)
    if ids != set(sealed_role):
        raise ValueError("Validation membership mismatch")
    frozen = [sealed_role[r["scene"]]["updated"] for r in audit]
    candidate = [sealed_role[r["scene"]]["image_prior_kg"] if r["scene"] in triggered else f
                 for r, f in zip(audit, frozen)]
    truth = [r["target"] for r in audit]
    output = dict(matched=len(matched), triggers=triggered, frozen=mass_metrics(truth, frozen),
                  fallback=mass_metrics(truth, candidate))
    (ROOT/"structure_cases/fallback_validation.json").write_text(json.dumps(output, indent=2))
    print(len(triggered), output["fallback"],len(matched))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("panels", type=Path)
    args = parser.parse_args()
    sealed_role = {k:v for folder in (ROOT/"batches").glob("val_*") for k,v in sealed(folder).items()}
    evaluate(args.panels, sealed_role)

