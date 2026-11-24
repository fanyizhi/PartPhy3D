"""Test part-specific thickness and section limits on archived ABO structure evidence."""

import json
import math
import re

import numpy as np

from evaluate_structure_cues import ROOT, WORK, errors, read, sealed


# Limit only specifically named materials and components. The estimates use the
# surface allocation already present in the model, not a measured part mask.
PLATE = re.compile(
    r"\b(headboard|footboard|panel|shelf|drawer front|drawer bottom|tabletop|slat)s?\b"
)
ROD = re.compile(r"\b(rail|rod|bar|wire|tube)s?\b")
SKIN = re.compile(
    r"\b(mousepad surface|edge trim|edge profile|lining|slipcover|brand logo)\b"
)
BOARD_MATERIALS = {"wood", "plywood", "particleboard", "mdf", "plastic", "glass"}
ROD_MATERIALS = {"wood", "steel", "metal", "aluminum"}
SOFT_MATERIALS = {"fabric", "leather", "rubber", "foam", "plastic", "silicone"}


def geometry_role(part):
    name = part["name"].lower().replace("_", " ").replace("-", " ")
    material = part["material"]
    if SKIN.search(name) and material in SOFT_MATERIALS:
        return "skin"
    if PLATE.search(name) and material in BOARD_MATERIALS:
        return "plate"
    if ROD.search(name) and material in ROD_MATERIALS:
        return "rod"
    return None


def main():
    original = read(WORK / "geometry_predictions.json")
    bounds = read(WORK / "bounded_components.json")["predictions"]
    splits = read(ROOT / "splits.json")
    cohorts = {k: splits[k] for k in ("train", "val", "test")}
    ids = [key for group in cohorts.values() for key in group]
    assert tuple(map(len, cohorts.values())) == (300, 100, 100)
    assert len(set(ids)) == 500 and set(ids) == set(original) == set(bounds)
    targets = read(ROOT / "weights.json")
    truth = np.array([targets.get(k, targets.get(k.split("_")[0])) for k in ids])
    panels = read(WORK / "component_evidence.json")["scenes"]
    for key in ("train", "val"):
        for folder in sorted(ROOT.glob(key + "_???")):
            for row in sealed(folder / "panels"):
                if row["scene_id"] in cohorts[key]:
                    panels[row["scene_id"]] = row

    source_parts = []
    for key in ids:
        row = original[key]
        panel = row["recovered_panel"] if row["recovered"] else panels.get(key, {})
        scored = bounds[key]["scored_parts"]
        archived = [p for p in panel.get("parts", []) if p["mass_kg"] > 0]
        if row["component_evidence"]:
            assert [p["name"] for p in archived] == [p["name"] for p in scored]
            np.testing.assert_allclose(
                [p["mass_kg"] for p in archived], [p["raw_kg"] for p in scored]
            )
        else:
            archived = [None] * len(scored)
        surface = panel.get("area_m2", 0.0)
        source_parts.append((scored, archived, surface))

    def calculate(thickness, retain):
        output = []
        for scored, parts, surface in source_parts:
            adjusted = []
            for old, p in zip(scored, parts, strict=True):
                value = old["mass_kg"]
                role = geometry_role(p) if p is not None else None
                updated = value
                if role is not None and math.isfinite(thickness):
                    # A two-sided plate has V/A=t/2. A long solid round rod
                    # has V/A=d/4; this bounds a hollow rod with equal diameter.
                    # Thin skins use the assigned one-sided surface.
                    length = {
                        "plate": thickness / 2,
                        "rod": thickness / 4,
                        "skin": thickness / 8,
                    }[role]
                    area = surface * p["area_share"]
                    if area > 0:
                        geometry_limit = area * length * p["density_kg_m3"]
                        updated = max(retain * value, min(value, geometry_limit))
                adjusted.append(
                    dict(
                        name=old["name"], role=role, previous_kg=value, mass_kg=updated
                    )
                )
            output.append(adjusted)
        mass = np.array([sum(p["mass_kg"] for p in row) for row in output])
        return mass, output

    baseline = np.array([bounds[k]["mass_kg"] for k in ids])
    train = slice(0, 300)
    reference = errors(truth[train], baseline[train]).mean(0)
    trials = []
    for thickness, retain in [(math.inf, 1.0)] + [
        (thickness, retain)
        for thickness in (0.012, 0.024, 0.048, 0.096)
        for retain in (0.25, 0.5, 0.75)
    ]:
        masses, _ = calculate(thickness, retain)
        score = errors(truth[train], masses[train]).mean(0)
        objective = np.mean(score[:3] / reference[:3])
        trials.append((float(objective), thickness, retain, score.tolist()))
    _, thickness, retain, _ = min(trials)
    estimate, allocation = calculate(thickness, retain)
    no_limit, _ = calculate(math.inf, 1.0)
    np.testing.assert_allclose(no_limit, baseline)
    metrics = {}
    for k, ix in {
        "train": train,
        "val": slice(300, 400),
        "test": slice(400, 500),
        "all": slice(None),
    }.items():
        metrics[k] = dict(
            previous=errors(truth[ix], baseline[ix]).mean(0).tolist(),
            revised=errors(truth[ix], estimate[ix]).mean(0).tolist(),
            changed=int(np.count_nonzero(np.abs(estimate[ix] - baseline[ix]) > 1e-8)),
        )
    paired = {}
    rng = np.random.default_rng(25)
    for name, ix in (("val", slice(300, 400)), ("test", slice(400, 500))):
        delta = errors(truth[ix], estimate[ix]) - errors(truth[ix], baseline[ix])
        sampled = rng.integers(0, len(delta), (10000, len(delta)))
        paired[name] = np.quantile(
            delta[sampled].mean(1), [0.025, 0.975], axis=0
        ).tolist()
    result = dict(
        thickness_m=str(thickness),
        retain=retain,
        metrics=metrics,
        paired_ci95=paired,
        candidates=[
            dict(score=t[0], thickness_m=str(t[1]), retain=t[2], metrics=t[3])
            for t in trials
        ],
        objects={
            k: dict(
                split=original[k]["split"],
                target_kg=float(truth[i]),
                original_kg=float(baseline[i]),
                estimate_kg=float(estimate[i]),
                localized=original[k]["component_evidence"],
                parts=allocation[i],
            )
            for i, k in enumerate(ids)
        },
    )
    (WORK / "geometry_role_constraints.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(json.dumps({k: v for k, v in result.items() if k != "objects"}, indent=2))


if __name__ == "__main__":
    main()
