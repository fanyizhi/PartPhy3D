"""Fit bounded component corrections using object weights from the training split."""

import argparse
import json

import numpy as np
from scipy.optimize import least_squares
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from evaluate_structure_cues import ROOT, WORK, errors, read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="Fit absolute and relative errors jointly",
    )
    args = parser.parse_args()
    rows = read(WORK / "geometry_predictions.json")
    labels = read(ROOT / "weights.json")
    ids = list(rows)
    groups = {
        s: np.array([i for i, k in enumerate(ids) if rows[k]["split"] == s])
        for s in ("train", "val", "test")
    }
    truth = np.array([labels.get(k, labels.get(k.split("_")[0])) for k in ids])
    parts = [
        r["parts"]
        or [
            dict(
                name="",
                material="unlocated",
                construction="unlocated",
                mass_kg=r["raw"],
            )
        ]
        for r in rows.values()
    ]

    def features(part):
        return {
            "intercept": 1.0,
            "log_mass": np.log(part["mass_kg"]),
            "material=" + part["material"]: 1.0,
            "role=" + part["construction"]: 1.0,
        }

    def fit(train, evaluate, lower, upper, objective):
        vec = DictVectorizer(sparse=False)
        vec.fit([features(p) for i in train for p in parts[i] if p["mass_kg"] > 0])

        def encode(indices):
            lists = [[p for p in parts[i] if p["mass_kg"] > 0] for i in indices]
            flat = [p for group in lists for p in group]
            return (
                vec.transform([features(p) for p in flat]),
                np.array([p["mass_kg"] for p in flat]),
                np.cumsum([0] + [len(g) for g in lists[:-1]]),
            )

        design, mass, start = encode(train)

        def total(coefficients):
            return np.add.reduceat(
                mass * np.exp(np.clip(design @ coefficients, lower, upper)), start
            )

        med = np.median(truth[train])

        def smooth_absolute(residual):
            return residual * np.sqrt(2.0 / (np.hypot(1.0, residual) + 1.0))

        def residual(coefficients):
            estimate = total(coefficients)
            target = truth[train]
            terms = [smooth_absolute(np.log(estimate / target))]
            if objective in {"absolute", "joint"}:
                terms.append(smooth_absolute((estimate - target) / med))
            if objective in {"relative", "joint"}:
                terms.append(smooth_absolute((estimate - target) / target))
            return np.concatenate([*terms, coefficients])

        result = least_squares(residual, np.zeros(design.shape[1]), max_nfev=400)
        if not result.success:
            raise RuntimeError(result.message)
        design, mass, start = encode(evaluate)
        values = mass * np.exp(np.clip(design @ result.x, lower, upper))
        return np.add.reduceat(values, start), np.split(values, start[1:])

    train = groups["train"]
    folds = list(KFold(5, shuffle=True, random_state=25).split(train))
    trials = []
    objectives = ("log", "absolute", "relative", "joint") if args.balanced else ("log",)
    for objective in objectives:
        for maximum in (1.5, 3.0, 6.0):
            bounds = (-np.log(30.0), np.log(maximum))
            train_oof = np.zeros(len(train))
            oof_parts = {}
            for build, held in folds:
                train_oof[held], held_parts = fit(
                    train[build], train[held], *bounds, objective
                )
                oof_parts.update(zip(train[held], held_parts, strict=True))
            stats = errors(truth[train], train_oof).mean(0).tolist()
            trials.append((maximum, stats, train_oof, oof_parts, objective))
    baseline = np.array(
        min((t for t in trials if t[4] == "log"), key=lambda t: t[1][1])[1][:3]
    )

    def rank(trial):
        if args.balanced:
            return np.mean(np.array(trial[1][:3]) / baseline)
        return trial[1][1]

    maximum, _, selected_oof, selected_parts, objective = min(trials, key=rank)
    predicted, individual = fit(
        train, np.arange(len(ids)), -np.log(30.0), np.log(maximum), objective
    )
    full_fit = predicted.copy()
    predicted[train] = selected_oof
    np.testing.assert_allclose(full_fit, [v.sum() for v in individual])
    for indices in groups.values():
        assert np.isfinite(predicted[indices]).all() and (predicted[indices] > 0).all()
    summary = {
        s: errors(truth[idx], predicted[idx]).mean(0).tolist()
        for s, idx in {**groups, "all": np.arange(len(ids))}.items()
    }
    light = {
        s: errors(truth[idx][truth[idx] < 1], predicted[idx][truth[idx] < 1])
        .mean(0)
        .tolist()
        for s, idx in groups.items()
    }
    result = dict(
        metrics=summary,
        below_one_kg=light,
        maximum_multiplier=maximum,
        objective=objective,
        training_candidates={f"{t[4]}/{t[0]}": t[1] for t in trials},
        predictions={
            k: dict(
                split=rows[k]["split"],
                mass_kg=predicted[i],
                full_fit_kg=full_fit[i],
                localized=bool(rows[k]["parts"]),
                full_fit_parts=[
                    dict(name=p["name"], raw_kg=p["mass_kg"], mass_kg=float(v))
                    for p, v in zip(
                        [p for p in parts[i] if p["mass_kg"] > 0],
                        individual[i],
                        strict=True,
                    )
                ],
                scored_parts=[
                    dict(name=p["name"], raw_kg=p["mass_kg"], mass_kg=float(v))
                    for p, v in zip(
                        [p for p in parts[i] if p["mass_kg"] > 0],
                        selected_parts.get(i, individual[i]),
                        strict=True,
                    )
                ],
            )
            for i, k in enumerate(ids)
        },
    )
    output = "balanced_components.json" if args.balanced else "bounded_components.json"
    (WORK / output).write_text(json.dumps(result, indent=2))
    print(objective, maximum, summary, light)
    print(result["training_candidates"])


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
