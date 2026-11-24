"""Fit bounded component corrections using object weights from the training split."""

import json

import numpy as np
from scipy.optimize import least_squares
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from evaluate_structure_cues import ROOT, WORK, errors, read


def main():
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

    def fit(train, evaluate, lower, upper):
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

        def residual(coefficients):
            # Smooth log-ratio, with smaller influence for extreme outliers.
            z = np.log(total(coefficients) / truth[train])
            robust = z * np.sqrt(2.0 / (np.hypot(1.0, z) + 1.0))
            return np.r_[robust, coefficients]

        result = least_squares(residual, np.zeros(design.shape[1]), max_nfev=400)
        if not result.success:
            raise RuntimeError(result.message)
        design, mass, start = encode(evaluate)
        values = mass * np.exp(np.clip(design @ result.x, lower, upper))
        return np.add.reduceat(values, start), np.split(values, start[1:])

    train = groups["train"]
    folds = list(KFold(5, shuffle=True, random_state=25).split(train))
    trials = []
    for maximum in (1.5, 3.0, 6.0):
        bounds = (-np.log(30.0), np.log(maximum))
        train_oof = np.zeros(len(train))
        for build, held in folds:
            train_oof[held] = fit(train[build], train[held], *bounds)[0]
        stats = errors(truth[train], train_oof).mean(0).tolist()
        trials.append((maximum, stats, train_oof))
    maximum, _, selected_oof = min(trials, key=lambda t: t[1][1])
    predicted, individual = fit(
        train, np.arange(len(ids)), -np.log(30.0), np.log(maximum)
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
        training_candidates={str(t[0]): t[1] for t in trials},
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
            )
            for i, k in enumerate(ids)
        },
    )
    (WORK / "bounded_components.json").write_text(json.dumps(result, indent=2))
    print(maximum, summary, light)


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
