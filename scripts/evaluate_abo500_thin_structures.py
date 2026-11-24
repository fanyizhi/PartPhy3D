"""Recompute material occupancy for porous fabrics and narrow trims."""

import json
import math
import re
from itertools import product

import numpy as np

from evaluate_structure_cues import ROOT, WORK, errors, read, sealed


MESH = re.compile(r"\b(mesh|netting)\b", re.I)
TRIM = re.compile(r"\b(edge trim|edge profile|piping|gasket|sealant)\b", re.I)


def classify(part):
    name = part["name"].replace("_", " ").replace("-", " ")
    if part["material"] == "fabric" and MESH.search(name):
        return "mesh_fabric"
    if part["material"] in {
        "rubber",
        "plastic",
        "silicone",
        "fabric",
        "leather",
    } and TRIM.search(name):
        return "narrow_trim"
    return None


def main():
    verified = read(WORK / "abo500_component_ledger.json")["objects"]
    geometry = read(WORK / "geometry_predictions.json")
    previous = read(WORK / "abo500_cover_areal_density.json")["variants"][
        "extreme_cover_only"
    ]["objects"]
    labels = read(ROOT / "weights.json")
    split_record = read(ROOT / "splits.json")
    groups = {s: split_record[s] for s in ("train", "val", "test")}
    keys = [k for cohort in groups.values() for k in cohort]
    assert len(keys) == len(set(keys)) == 500 and set(keys) == set(verified)
    assert tuple(map(len, groups.values())) == (300, 100, 100)

    candidates = read(WORK / "component_evidence.json")["scenes"]
    for split in ("train", "val"):
        for folder in sorted(ROOT.glob(split + "_???")):
            for archive in sealed(folder / "panels"):
                if archive["scene_id"] in groups[split]:
                    candidates[archive["scene_id"]] = archive

    eligible = {}
    for key in keys:
        record = verified[key]
        if record["source_field"] == "geometry_predictions.parts":
            panel = (
                geometry[key]["recovered_panel"]
                if geometry[key]["recovered"]
                else candidates.get(key, {})
            )
            proposed = panel.get("parts", [])
            names = [p["name"] for p in proposed]
            if not proposed or names != [p["name"] for p in record["components"]]:
                continue
            np.testing.assert_allclose(
                [p["mass_kg"] for p in proposed],
                [p["mass_kg"] for p in record["components"]],
            )
            area = panel.get("area_m2", 0.0)
            for i, part in enumerate(proposed):
                role = classify(part)
                surface = area * part["area_share"]
                if role and surface > 0:
                    eligible.setdefault(key, []).append(
                        (i, role, surface, part["density_kg_m3"])
                    )

    def modify(key, mesh_areal, trim_depth):
        src = previous[key]["parts"]
        masses = [p["mass_kg"] for p in src]
        changes = []
        for index, role, area, density in eligible.get(key, []):
            bound = area * (
                mesh_areal if role == "mesh_fabric" else density * trim_depth
            )
            updated = min(masses[index], bound)
            if updated < masses[index]:
                changes.append(
                    dict(
                        part=src[index]["name"],
                        role=role,
                        area_m2=area,
                        before_kg=masses[index],
                        after_kg=updated,
                    )
                )
                masses[index] = updated
        return dict(
            mass_kg=math.fsum(masses),
            names=[p["name"] for p in src],
            parts_kg=masses,
            changes=changes,
        )

    def metric(ids, estimate):
        truth = np.array([labels.get(k, labels.get(k.split("_")[0])) for k in ids])
        return (
            errors(truth, np.array([estimate[k]["mass_kg"] for k in ids]))
            .mean(0)
            .tolist()
        )

    training = groups["train"]
    reference = np.array(metric(training, previous))
    trials = []
    for areal, depth in product(
        (0.12, 0.25, 0.45, math.inf), (0.001, 0.002, 0.003, math.inf)
    ):
        altered = {k: modify(k, areal, depth) for k in training}
        score = metric(training, altered)
        # Selection only uses the training partition and balances the error scales.
        trials.append(
            (float(np.mean(np.array(score[:3]) / reference[:3])), areal, depth, score)
        )
    _, areal, depth, _ = min(trials, key=lambda t: t[0])
    amended = {k: modify(k, areal, depth) for k in keys}
    report = {}
    for split, ids in {**groups, "all": keys}.items():
        report[split] = dict(
            original=metric(ids, previous),
            amended=metric(ids, amended),
            eligible=sum(k in eligible for k in ids),
            modified=sum(bool(amended[k]["changes"]) for k in ids),
        )
    result = dict(
        mesh_areal_kg_m2=str(areal),
        trim_equivalent_depth_m=str(depth),
        metrics=report,
        trials=[
            dict(score=s, mesh_areal=str(a), trim_depth=str(v), metrics=m)
            for s, a, v, m in trials
        ],
        objects=amended,
    )
    (WORK / "abo500_thin_structure_integrals.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(areal, depth, report)
    for key in keys:
        if amended[key]["changes"]:
            print(
                key,
                geometry[key]["split"],
                labels.get(key, labels.get(key.split("_")[0])),
                amended[key]["changes"],
            )


if __name__ == "__main__":
    main()
