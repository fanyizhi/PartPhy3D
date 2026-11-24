"""Fit positive partwise mass multipliers with object mass supervision only."""

import json
import re

import numpy as np
from scipy.optimize import least_squares
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import KFold
from threadpoolctl import threadpool_limits

from evaluate_structure_cues import ROOT, WORK, errors, read, sealed


def part_features(p):
    tokens = set(re.findall(r"[a-z]{3,}", p.get("name", "").lower()))
    return {
        "bias": 1,
        "log_kg": np.log(max(p["mass_kg"], 1e-6)),
        "material/" + str(p.get("material")): 1,
        "role/" + str(p.get("construction")): 1,
        **{"name/" + t: 1 for t in tokens},
    }


def main():
    records = read(WORK / "structural_ablation.json")["records"]
    ev = read(WORK / "component_evidence.json")["scenes"]
    for pattern in ("train_???", "val_???"):
        for folder in ROOT.glob(pattern):
            for r in sealed(folder / "panels"):
                if (
                    r["scene_id"] in records
                    and records[r["scene_id"]]["split"] != "test"
                ):
                    ev[r["scene_id"]] = r
    ids = list(records)
    y = np.array([records[k]["gt"] for k in ids])
    sections = [
        [j for j, k in enumerate(ids) if records[k]["split"] == s]
        for s in ("train", "val", "test")
    ]
    segments = []
    source = {}
    for key in ids:
        parts = ev.get(key, {}).get("parts", [])
        raw = records[key]["raw"]
        resolved = bool(parts) and np.isclose(sum(p["mass_kg"] for p in parts), raw)
        source[key] = "parts" if resolved else "unresolved_total"
        segments.append(
            [p for p in parts if p["mass_kg"] > 0]
            if resolved
            else [dict(name="", mass_kg=raw)]
        )

    def run(train, hold, penalty):
        vec = DictVectorizer(sparse=False)
        vec.fit([part_features(p) for i in train for p in segments[i]])

        def design(indices):
            groups = [segments[i] for i in indices]
            flat = [p for group in groups for p in group]
            return (
                vec.transform([part_features(p) for p in flat]),
                np.array([p["mass_kg"] for p in flat]),
                np.cumsum([0] + [len(group) for group in groups[:-1]]),
            )

        x, m, start = design(train)

        def residual(b):
            estimates = m * np.exp(np.clip(x @ b, -20, 20))
            return np.r_[
                np.log(np.add.reduceat(estimates, start) / y[train]),
                np.sqrt(penalty) * b,
            ]

        def jac(b):
            z = x @ b
            estimates = m * np.exp(np.clip(z, -20, 20))
            rows = np.add.reduceat(
                (estimates * ((z > -20) & (z < 20)))[:, None] * x, start, axis=0
            )
            return np.vstack(
                [
                    rows / np.add.reduceat(estimates, start)[:, None],
                    np.sqrt(penalty) * np.eye(len(b)),
                ]
            )

        fit = least_squares(residual, np.zeros(x.shape[1]), jac=jac, max_nfev=150)
        assert fit.success
        x, m, start = design(hold)
        individual = m * np.exp(np.clip(x @ fit.x, -20, 20))
        return np.add.reduceat(individual, start), np.split(individual, start[1:])

    train = np.array(sections[0])
    folds = list(KFold(5, shuffle=True, random_state=25).split(train))
    trials = []
    for strength in (1.0, 10.0, 100.0):
        oof = np.zeros(len(train))
        for a, b in folds:
            oof[b] = run(train[a], train[b], strength)[0]
        trials.append((errors(y[train], oof).mean(0)[1], strength, oof))
    _, strength, oof = min(trials, key=lambda t: t[0])
    pred, parts = run(train, np.arange(len(ids)), strength)
    assert all(np.isclose(p.sum(), value) for p, value in zip(parts, pred))
    data = {
        k: dict(
            source=source[k],
            fitted_parts=[
                dict(name=p.get("name"), raw_kg=p["mass_kg"], calibrated_kg=float(m))
                for p, m in zip(segments[j], parts[j])
            ],
            full_fit_kg=float(pred[j]),
        )
        for j, k in enumerate(ids)
    }
    # Sensitivity to over-amplifying a component. Select the upper bound
    # using out-of-fold object APE, then apply it unchanged to both holdouts.
    caps = (1.0, 2.0, 4.0, 8.0, 16.0, float("inf"))

    def capped_total(i, values, cap):
        raw = np.array([p["mass_kg"] for p in segments[i]])
        return np.minimum(values, cap * raw).sum()

    cv = np.zeros((len(caps), len(train)))
    for a, b in folds:
        _, held_parts = run(train[a], train[b], strength)
        for j, cap in enumerate(caps):
            cv[j, b] = [capped_total(i, v, cap) for i, v in zip(train[b], held_parts)]
    ape = [errors(y[train], v).mean(0)[2] for v in cv]
    best = int(np.argmin(ape))
    cap = caps[best]
    bounded = np.array([capped_total(i, v, cap) for i, v in enumerate(parts)])
    bounded[train] = cv[best]
    cap_audit = dict(
        selected_cap=str(cap),
        training_ape_by_cap=dict(zip(map(str, caps), ape)),
        metrics={
            name: errors(y[idx], bounded[idx]).mean(0).tolist()
            for name, idx in zip(("train_oof", "val", "test"), sections)
        },
        predictions=dict(zip(ids, bounded.tolist())),
    )
    (WORK / "part_cap_audit.json").write_text(json.dumps(cap_audit, indent=2))
    pred[train] = oof
    metrics = {
        name: errors(y[idx], pred[idx]).mean(0).tolist()
        for name, idx in zip(("train_oof", "val", "test"), sections)
    }
    results = dict(
        penalty=strength,
        metrics=metrics,
        training_oof=dict(zip([ids[j] for j in train], oof.tolist())),
        objects=data,
    )
    (WORK / "part_mass_calibration.json").write_text(json.dumps(results, indent=2))
    print(
        strength,
        metrics,
        {s: list(source.values()).count(s) for s in set(source.values())},
    )


if __name__ == "__main__":
    with threadpool_limits(1):
        main()
