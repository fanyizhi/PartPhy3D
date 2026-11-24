"""Evaluate directional fusion and light-mass calibration using training-fold predictions."""

import argparse
import json
from itertools import product

import numpy as np

from evaluate_structure_cues import ROOT, WORK, errors, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog-structure", action="store_true")
    args = parser.parse_args()
    archive = read(WORK / "thin_unified_500.json")["objects"]
    manifest = read(ROOT / "splits.json")
    groups = {s: manifest[s] for s in ("train", "val", "test")}
    ids = [k for members in groups.values() for k in members]
    assert len(ids) == len(set(ids)) == 500 and set(ids) == set(archive)
    assert tuple(map(len, groups.values())) == (300, 100, 100)
    image, structure, old, truth = (
        np.array([archive[k][column] for k in ids])
        for column in ("image", "structure", "joint", "target_kg")
    )
    if args.catalog_structure:
        catalog = read(WORK / "weighted_structure_control.json")["catalog_assisted"][
            "objects"
        ]
        structure = np.array([catalog[k]["mass_kg"] for k in ids])
        old = image * np.exp(
            0.25 * np.clip(np.log(structure / image), -np.log(3.0), np.log(3.0))
        )
    log_ratio = np.log(structure / image)
    train = slice(0, 300)
    before = errors(truth[train], old[train]).mean(0)

    def predict(up, down, cap, pivot, shrink):
        weight = np.where(log_ratio >= 0, up, down)
        mixed = image * np.exp(weight * np.clip(log_ratio, -np.log(cap), np.log(cap)))
        return mixed * (1.0 - shrink / (1.0 + (mixed / pivot) ** 2))

    np.testing.assert_allclose(predict(0.25, 0.25, 3.0, 1.0, 0.0), old)
    trials = []
    calibration = [(1.0, 0.0)] + list(product((1.0, 3.0), (0.10, 0.20, 0.30, 0.45)))
    for up, down, cap, (pivot, shrink) in product(
        (0.0, 0.125, 0.25, 0.50, 0.75),
        (0.0, 0.125, 0.25, 0.50, 0.75),
        (1.5, 2.0, 3.0),
        calibration,
    ):
        estimate = predict(up, down, cap, pivot, shrink)
        score = errors(truth[train], estimate[train]).mean(0)
        # Reject large degradation in any training-fold metric before comparing
        # combined error rates. The already published model remains a candidate.
        acceptable = bool(
            score[0] <= 1.02 * before[0]
            and score[1] <= 1.02 * before[1]
            and score[3] >= before[3] - 0.02
        )
        objective = float(np.mean(score[:3] / before[:3]))
        trials.append(
            dict(
                up=up,
                down=down,
                cap=cap,
                pivot=pivot,
                shrink=shrink,
                metric=score.tolist(),
                acceptable=acceptable,
                objective=objective,
            )
        )
    winner = min((t for t in trials if t["acceptable"]), key=lambda t: t["objective"])
    result = predict(**{k: winner[k] for k in ("up", "down", "cap", "pivot", "shrink")})
    # Check whether a mass-dependent adjustment improves absolute error.
    # This search sees the same archived out-of-fold training predictions.
    heavy_trials = []
    for name, base in (("fixed_fusion", old), ("conditional_fusion", result)):
        for pivot, gain in product(
            (10.0, 20.0, 40.0), (-0.20, -0.10, 0.0, 0.10, 0.20, 0.30)
        ):
            estimate = base * (1.0 + gain / (1.0 + (pivot / base) ** 2))
            m = errors(truth[train], estimate[train]).mean(0)
            if m[1] <= 1.02 * before[1] and m[2] <= before[2]:
                heavy_trials.append((float(m[0]), name, pivot, gain, estimate))
    _, heavy_source, heavy_pivot, heavy_gain, heavy = min(
        heavy_trials, key=lambda t: t[0]
    )
    rng = np.random.default_rng(25)
    partitions = {
        "train": train,
        "val": slice(300, 400),
        "test": slice(400, 500),
        "all": slice(None),
    }
    models = {
        "structure": structure,
        "image": image,
        "fixed_fusion": old,
        "conditional_fusion": result,
        "heavy_calibration": heavy,
    }
    summaries = {}
    audit = {}
    for split, ix in partitions.items():
        exact = truth[ix]
        prior = errors(exact, old[ix])
        next_errors = errors(exact, result[ix])
        delta = next_errors - prior
        draws = rng.integers(0, len(exact), (10000, len(exact)))
        summaries[split] = {
            name: errors(exact, estimate[ix]).mean(0).tolist()
            for name, estimate in models.items()
        }
        light = exact < 1
        worst = np.argsort(-prior[:, 2])[:5]
        abs_worst = np.argsort(-prior[:, 0])[:10]
        audit[split] = dict(
            delta=delta.mean(0).tolist(),
            ci95=np.quantile(delta[draws].mean(1), [0.025, 0.975], axis=0).tolist(),
            light_count=int(light.sum()),
            light_ape_before=float(prior[light, 2].mean()),
            light_ape_after=float(next_errors[light, 2].mean()),
            light_ape_share=float(prior[light, 2].sum() / prior[:, 2].sum()),
            top5_ape_share=float(prior[worst, 2].sum() / prior[:, 2].sum()),
            top10_ade_share=float(prior[abs_worst, 0].sum() / prior[:, 0].sum()),
            top_ape_ids=[ids[ix][i] for i in worst],
            top_ade_ids=[ids[ix][i] for i in abs_worst],
        )
    assert np.isfinite(result).all() and (result > 0).all()
    output = dict(
        catalog_structure=args.catalog_structure,
        selection=winner,
        heavy_selection=dict(
            source=heavy_source, pivot_kg=heavy_pivot, gain=heavy_gain
        ),
        selection_candidates=len(trials),
        trials=trials,
        metrics=summaries,
        audit=audit,
        objects={
            k: dict(
                target_kg=float(truth[i]),
                **{name: float(v[i]) for name, v in models.items()},
            )
            for i, k in enumerate(ids)
        },
    )
    filename = (
        "catalog_conditional_fusion_500.json"
        if args.catalog_structure
        else "conditional_fusion_500.json"
    )
    (WORK / filename).write_text(json.dumps(output, indent=2, allow_nan=False))
    print(winner, output["heavy_selection"])
    print(summaries)
    print(audit["test"], audit["all"])


if __name__ == "__main__":
    main()
