"""Evaluate visual-only, structure-only and confidence-gated fusion on all ABO objects."""

import argparse
import json
from itertools import product

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from evaluate_matched_physical_features import visual
from evaluate_structure_cues import ROOT, WORK, errors, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--areal-replay", action="store_true")
    args = parser.parse_args()
    splits = read(ROOT / "splits.json")
    cohorts = {s: splits[s] for s in ("train", "val", "test")}
    ids = [k for v in cohorts.values() for k in v]
    assert len(ids) == len(set(ids)) == 500
    assert tuple(map(len, cohorts.values())) == (300, 100, 100)
    label = read(ROOT / "weights.json")
    truth = np.array([label.get(k, label.get(k.split("_")[0])) for k in ids])
    raw = read(WORK / "geometry_predictions.json")
    structure_filename = (
        "areal_structure_control.json"
        if args.areal_replay
        else "nonlinear_structure_control.json"
    )
    structure_details = read(WORK / structure_filename)["structure"]
    archived = structure_details["objects"]
    structure = np.array([archived[k]["mass_kg"] for k in ids])
    localized = np.array([raw[k]["component_evidence"] for k in ids])
    if args.areal_replay:
        ledger = read(WORK / "abo500_component_ledger.json")["objects"]
        localized = np.array([bool(ledger[k]["components"]) for k in ids])
    x = visual(ids)
    train = np.arange(300)
    folds = list(KFold(5, shuffle=True, random_state=25).split(train))

    def image_predict(build, hold, alpha):
        regressor = Ridge(alpha=alpha, solver="cholesky").fit(
            x[build], np.log(truth[build])
        )
        return np.exp(regressor.predict(x[hold]))

    candidates = []
    for alpha in (0.001, 0.01, 0.1, 1.0, 10.0):
        image_oof = np.zeros(300)
        for a, b in folds:
            image_oof[b] = image_predict(train[a], train[b], alpha)
        candidates.append(
            (errors(truth[train], image_oof).mean(0)[1], alpha, image_oof)
        )
    _, alpha, image_oof = min(candidates, key=lambda t: t[0])
    image = image_predict(train, np.arange(len(ids)), alpha)
    image[train] = image_oof

    def blend(weight, ratio_cap, fallback):
        log_ratio = np.log(structure / image)
        if not np.isinf(ratio_cap):
            log_ratio = np.clip(log_ratio, -np.log(ratio_cap), np.log(ratio_cap))
        confidence = np.where(localized, 1.0, fallback)
        return image * np.exp(weight * confidence * log_ratio)

    reference = errors(truth[train], image[train]).mean(0)
    trials = []
    for weight, ratio_cap, fallback in product(
        (0.0, 0.25, 0.5, 0.75, 1.0), (1.5, 3.0, np.inf), (0.0, 1.0)
    ):
        result = blend(weight, ratio_cap, fallback)
        stats = errors(truth[train], result[train]).mean(0)
        # Evaluate all three error scales, normalized to the RGB-only training reference.
        trials.append(
            (float(np.mean(stats[:3] / reference[:3])), weight, ratio_cap, fallback)
        )
    _, weight, cap, fallback = min(trials)
    combined = blend(weight, cap, fallback)
    methods = {"structure": structure, "image": image, "joint": combined}
    sections = {
        "train": slice(0, 300),
        "val": slice(300, 400),
        "test": slice(400, 500),
        "all": slice(None),
    }
    results = {}
    rng = np.random.default_rng(25)
    for split, ix in sections.items():
        diff = errors(truth[ix], combined[ix]) - errors(truth[ix], image[ix])
        samples = rng.integers(0, len(diff), (10000, len(diff)))
        results[split] = dict(
            n=len(diff),
            metrics={
                k: errors(truth[ix], v[ix]).mean(0).tolist() for k, v in methods.items()
            },
            joint_minus_image_ci95=np.quantile(
                diff[samples].mean(1), [0.025, 0.975], axis=0
            ).tolist(),
        )
    output = dict(
        parameters=dict(
            image_alpha=alpha,
            structure_weight=weight,
            ratio_cap=str(cap),
            structure_weight_multiplier_if_missing=fallback,
        ),
        splits=results,
        structure_input=structure_filename,
        structure_model={
            k: structure_details[k]
            for k in ("leaf", "features_fraction", "mass_weight_power")
            if k in structure_details
        },
        trials=[
            dict(score=s, weight=w, ratio_cap=str(c), fallback=f)
            for s, w, c, f in trials
        ],
        objects={
            k: dict(
                target_kg=float(truth[i]),
                has_parts=bool(localized[i]),
                **{name: float(v[i]) for name, v in methods.items()},
            )
            for i, k in enumerate(ids)
        },
    )
    assert all(np.isfinite(v).all() and (v > 0).all() for v in methods.values())
    filename = (
        "areal_unified_500.json" if args.areal_replay else "unified_comparison_500.json"
    )
    (WORK / filename).write_text(json.dumps(output, indent=2, allow_nan=False))
    print(output["parameters"], results)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
