"""Evaluate canopy mass from fabric areal density using archived component groups."""

import json
import math
import re
from itertools import product

import numpy as np

from evaluate_structure_cues import ROOT, WORK, errors, read


COVER = re.compile(r"\b(roof|canopy|cover|flysheet)\b", re.IGNORECASE)
FITTING = re.compile(r"\b(fittings?|feet|connectors?)\b", re.IGNORECASE)


def is_cover(part):
    return (
        part["kind"] == "sheet"
        and part["material"] == "fabric"
        and COVER.search(part["name"])
    )


def main():
    records = read(WORK / "abo500_component_ledger.json")["objects"]
    split_ids = read(ROOT / "splits.json")
    groups = {s: split_ids[s] for s in ("train", "val", "test")}
    ids = [k for cohort in groups.values() for k in cohort]
    assert len(ids) == len(set(ids)) == len(records) == 500
    assert tuple(map(len, groups.values())) == (300, 100, 100)
    eligible = {
        k
        for k in ids
        if records[k]["source_field"] == "open_frame_groups"
        and any(is_cover(p) for p in records[k]["components"])
    }
    baseline = {k: records[k]["raw_kg"] for k in ids}

    def replay(key, gsm, fitting_fraction, extreme_only=False):
        parts = []
        for p in records[key]["components"]:
            value = p["mass_kg"]
            if key in eligible and is_cover(p) and gsm is not None:
                area = p["area_m2"] * p["count"]
                # Use the extreme policy only above 3 x 450 g/m2 (reinforced
                # coated fabric reference). This multiplier is a sensitivity choice.
                if not extreme_only or value / area > 1.350:
                    value = area * gsm / 1000.0
            elif key in eligible and p["kind"] == "box" and FITTING.search(p["name"]):
                value *= fitting_fraction
            parts.append(dict(name=p["name"], before_kg=p["mass_kg"], mass_kg=value))
        return dict(mass_kg=math.fsum(p["mass_kg"] for p in parts), parts=parts)

    labels = read(ROOT / "weights.json")
    truth = {k: labels.get(k, labels.get(k.split("_")[0])) for k in ids}

    def score(keys, predictions):
        if not keys:
            return None
        return (
            errors(
                np.array([truth[k] for k in keys]),
                np.array([predictions[k] for k in keys]),
            )
            .mean(0)
            .tolist()
        )

    train = [k for k in groups["train"] if k in eligible]
    reference = np.array(score(train, baseline))
    trials = []
    for gsm, fraction in product(
        (None, 160, 220, 250, 280, 340, 450), (1.0, 0.75, 0.5, 0.25)
    ):
        stats = score(train, {k: replay(k, gsm, fraction)["mass_kg"] for k in train})
        trials.append(
            dict(
                gsm=gsm,
                fitting_fraction=fraction,
                metrics=stats,
                objective=float(np.mean(np.array(stats[:3]) / reference[:3])),
            )
        )
    variants = {
        "raw": dict(gsm=None, fitting_fraction=1.0),
        "cover": min(
            (t for t in trials if t["fitting_fraction"] == 1.0),
            key=lambda t: t["objective"],
        ),
        "cover_and_fittings": min(trials, key=lambda t: t["objective"]),
        "extreme_cover_only": dict(gsm=250, fitting_fraction=1.0, extreme_only=True),
    }
    outputs = {}
    for method, params in variants.items():
        ledger = {
            k: replay(
                k,
                params["gsm"],
                params["fitting_fraction"],
                params.get("extreme_only", False),
            )
            for k in ids
        }
        total = {k: r["mass_kg"] for k, r in ledger.items()}
        if method == "raw":
            np.testing.assert_allclose(list(total.values()), list(baseline.values()))
        outputs[method] = dict(
            parameters=params,
            metrics={
                s: dict(
                    whole=score(keys, total),
                    target_group=score([k for k in keys if k in eligible], total),
                    eligible=sum(k in eligible for k in keys),
                    changed=sum(
                        not math.isclose(total[k], baseline[k], rel_tol=1e-10)
                        for k in keys
                    ),
                )
                for s, keys in {**groups, "all": ids}.items()
            },
            objects=ledger,
        )
    result = dict(
        trials=trials,
        eligible={
            k: dict(target_kg=truth[k], split=records[k]["split"])
            for k in ids
            if k in eligible
        },
        variants=outputs,
    )
    (WORK / "abo500_cover_areal_density.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    for method, data in outputs.items():
        print(method, data["parameters"], data["metrics"])
    for key, record in result["eligible"].items():
        print(key, record, [v["objects"][key]["mass_kg"] for v in outputs.values()])


if __name__ == "__main__":
    main()
