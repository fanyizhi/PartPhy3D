"""Audit percentage error and calibrate only on out-of-fold ABO predictions."""

import json
from itertools import product

import numpy as np

from evaluate_structure_cues import ROOT, WORK, errors, read


def main():
    manifest = read(ROOT / "splits.json")
    groups = {s: manifest[s] for s in ("train", "val", "test")}
    ids = [k for keys in groups.values() for k in keys]
    index = {key: i for i, key in enumerate(ids)}
    assert len(index) == len(ids) == 500
    assert tuple(map(len, groups.values())) == (300, 100, 100)
    weight = read(ROOT / "weights.json")
    truth = np.array([weight.get(k, weight.get(k.split("_")[0])) for k in ids])
    raw = read(WORK / "abo500_component_ledger.json")["objects"]
    replay = read(WORK / "abo500_cover_areal_density.json")["variants"][
        "extreme_cover_only"
    ]["objects"]
    physical = np.array([replay[k]["mass_kg"] for k in ids])
    models = {
        "unweighted": read(WORK / "areal_structure_control.json")["structure"],
        "light_weighted": read(WORK / "areal_weighted_structure_control.json")[
            "structure"
        ],
    }
    predictions = {
        name: np.array([r["objects"][k]["mass_kg"] for k in ids])
        for name, r in models.items()
    }
    train = np.array([index[k] for k in groups["train"]])
    reference = errors(truth[train], predictions["unweighted"][train]).mean(0)

    def adjust(estimate, pivot, max_reduction):
        # Use only the estimated mass at inference. The reduction tapers
        # smoothly to zero as the estimated mass increases.
        return estimate * (1.0 - max_reduction / (1.0 + (estimate / pivot) ** 2))

    def calibrated(estimate, pivot, max_reduction, anchor_range, anchor_multiplier):
        result = adjust(estimate, pivot, max_reduction)
        if anchor_range is None:
            return result
        # For particularly small integrated masses, bound increases made by
        # the regression. The bound is conditional on the physical estimate.
        return np.where(
            physical <= anchor_range,
            np.minimum(result, physical * anchor_multiplier),
            result,
        )

    trials = []
    anchors = [(None, None), *product((0.25, 0.5, 1.0), (2.0, 3.0, 5.0))]
    for name, estimate in predictions.items():
        for pivot, reduction, (anchor_range, anchor_multiplier) in product(
            (1.0, 3.0, 10.0), (0.0, 0.15, 0.30, 0.45, 0.60), anchors
        ):
            pred = calibrated(
                estimate, pivot, reduction, anchor_range, anchor_multiplier
            )
            score = errors(truth[train], pred[train]).mean(0)
            trials.append(
                dict(
                    source=name,
                    pivot_kg=pivot,
                    max_reduction=reduction,
                    anchor_range_kg=anchor_range,
                    anchor_multiplier=anchor_multiplier,
                    metrics=score.tolist(),
                    feasible=bool(
                        score[0] <= 1.05 * reference[0]
                        and score[1] <= 1.10 * reference[1]
                    ),
                )
            )
    winner = min(
        (trial for trial in trials if trial["feasible"]), key=lambda t: t["metrics"][2]
    )
    corrected = calibrated(
        predictions[winner["source"]],
        winner["pivot_kg"],
        winner["max_reduction"],
        winner["anchor_range_kg"],
        winner["anchor_multiplier"],
    )
    predictions["corrected"] = corrected
    report = {}
    rng = np.random.default_rng(25)
    for split, keys in {**groups, "all": ids}.items():
        ix = np.array([index[k] for k in keys])
        light = truth[ix] < 1
        base = errors(truth[ix], predictions["unweighted"][ix])
        diff = errors(truth[ix], corrected[ix]) - base
        sample = rng.integers(0, len(ix), (10000, len(ix)))
        worst = np.argsort(-base[:, 2])[:5]
        report[split] = dict(
            metrics={
                name: errors(truth[ix], estimate[ix]).mean(0).tolist()
                for name, estimate in predictions.items()
            },
            light_n=int(light.sum()),
            light_ape={
                name: float(errors(truth[ix][light], estimate[ix][light]).mean(0)[2])
                for name, estimate in predictions.items()
            },
            light_share_of_original_ape=float(base[light, 2].sum() / base[:, 2].sum()),
            top_5_share_of_original_ape=float(base[worst, 2].sum() / base[:, 2].sum()),
            overestimated_light=int(
                np.sum(predictions["unweighted"][ix][light] > truth[ix][light])
            ),
            confidence_95=np.quantile(
                diff[sample].mean(1), [0.025, 0.975], axis=0
            ).tolist(),
            worst=[
                dict(
                    id=ids[ix[i]],
                    truth_kg=float(truth[ix[i]]),
                    raw_kg=raw[ids[ix[i]]]["raw_kg"],
                    estimate_kg=float(predictions["unweighted"][ix[i]]),
                    updated_kg=float(corrected[ix[i]]),
                    geometry=raw[ids[ix[i]]]["components"],
                )
                for i in worst
            ],
        )
    chosen = models[winner["source"]]["objects"]
    objects = {}
    for i, key in enumerate(ids):
        parts = (
            np.array(chosen[key]["component_kg"])
            * corrected[i]
            / chosen[key]["mass_kg"]
        )
        np.testing.assert_allclose(parts.sum(), corrected[i])
        objects[key] = dict(
            target_kg=float(truth[i]),
            mass_kg=float(corrected[i]),
            component_kg=parts.tolist(),
        )
    result = dict(selection=winner, trials=trials, splits=report, objects=objects)
    (WORK / "abo500_structure_ape_audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    print(winner)
    print({s: {k: v for k, v in r.items() if k != "worst"} for s, r in report.items()})
    for row in report["test"]["worst"]:
        print({k: v for k, v in row.items() if k != "geometry"})


if __name__ == "__main__":
    main()
