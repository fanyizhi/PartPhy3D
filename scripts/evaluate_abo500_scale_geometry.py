"""Audit scale uncertainty and calibrate ABO mass using structural features only."""

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_structure_cues import ROOT, WORK, errors, read, sealed


def component_masses(panel, scale):
    """Apply similarity to solids, holding shell wall thickness in metric units."""
    if scale == 1.0:
        return [part["mass_kg"] for part in panel["parts"]]
    estimates = []
    for part in panel["parts"]:
        if part["construction"] == "shell":
            volume = min(
                part["outer_volume_m3"] * scale**3,
                panel["area_m2"]
                * part["area_share"]
                * scale**2
                * part["thickness_mm"]
                / 1000,
            )
        else:
            volume = (part["mass_kg"] / part["density_kg_m3"]) * scale**3
        estimates.append(volume * part["density_kg_m3"])
    return estimates


def main():
    rows = read(WORK / "geometry_predictions.json")
    catalog = read(WORK / "abo500_catalog_dimensions.json")["objects"]
    panels = read(WORK / "component_evidence.json")["scenes"]
    for directory in {p["source"] for p in panels.values()}:
        for candidate in sealed(Path(directory)):
            key = candidate["scene_id"]
            current = panels.get(key)
            if current is None or current["source"] != directory:
                continue
            signature = lambda parts: [(p["name"], p["mass_kg"]) for p in parts]
            if signature(candidate.get("parts", [])) == signature(current["parts"]):
                current.update(
                    predicted_max_length_m=candidate.get("predicted_max_length_m")
                )
    for split in ("train", "val"):
        for batch in sorted(ROOT.glob(split + "_???")):
            for panel in sealed(batch / "panels"):
                key = panel["scene_id"]
                if key in rows and rows[key]["split"] == split:
                    panels[key] = panel
    ids = list(rows)
    labels = read(ROOT / "weights.json")
    truth = np.array([labels.get(k, labels.get(k.split("_")[0])) for k in ids])
    groups = {
        s: np.flatnonzero([rows[k]["split"] == s for k in ids])
        for s in ("train", "val", "test")
    }
    assert tuple(map(len, groups.values())) == (300, 100, 100)
    train = groups["train"]
    folds = list(KFold(5, shuffle=True, random_state=25).split(train))
    archive = {k: {} for k in ids}
    for key, row in rows.items():
        panel = panels.get(key, {})
        if row["recovered"]:
            # Recovery carries its own verified parts and geometry.
            panel = row["recovered_panel"]
        parts = panel.get("parts", [])
        if not parts or not math.isclose(
            sum(p["mass_kg"] for p in parts), row["raw"], abs_tol=1e-6
        ):
            panel = {}
        length = panel.get("predicted_max_length_m")
        catalog_length = catalog.get(key, {}).get("max_dimension_m")
        ratio = catalog_length / length if length and catalog_length else None
        if ratio is None:
            status = "missing"
        elif 0.025 <= catalog_length <= 5.0 and 0.25 <= ratio <= 4.0:
            status = "eligible"
        else:
            status = "disagreement_or_range"
        archive[key] = dict(panel=panel, ratio=ratio, status=status)

    def prepare(weight):
        features, bases, parts_per_object = [], [], []
        for key in ids:
            row = rows[key]
            item = archive[key]
            panel = item["panel"]
            parts = panel.get("parts", [])
            scale = 1.0
            if item["status"] == "eligible":
                # Limit extrapolation: even with catalogue support, length changes
                # by at most a factor of two before applying the fitted weight.
                scale = float(np.clip(item["ratio"], 0.5, 2.0) ** weight)
            masses = component_masses(panel, scale) if parts else [row["raw"]]
            total = sum(masses)
            entry = {"log_base": math.log(total), "part_count": math.log1p(len(parts))}
            if parts:
                if panel.get("predicted_max_length_m"):
                    entry["log_length"] = math.log(
                        scale * panel["predicted_max_length_m"]
                    )
                entry["log_area"] = math.log(scale**2 * panel["area_m2"])
                entry["log_volume"] = math.log(
                    scale**3 * sum(p["outer_volume_m3"] for p in parts)
                )
                for part, mass in zip(parts, masses, strict=True):
                    for field in ("material", "construction"):
                        name = f"{field}/{part[field]}"
                        entry[name] = entry.get(name, 0.0) + mass / total
            features.append(entry)
            bases.append(total)
            parts_per_object.append(masses)
        return features, np.array(bases), parts_per_object

    def fit(features, bases, fit_ids, predict_ids, alpha, bound):
        model = make_pipeline(
            DictVectorizer(sparse=False), StandardScaler(), Ridge(alpha=alpha)
        )
        model.fit(
            [features[i] for i in fit_ids], np.log(truth[fit_ids] / bases[fit_ids])
        )
        correction = model.predict([features[i] for i in predict_ids])
        return np.exp(np.clip(correction, -np.log(bound), np.log(bound)))

    def stats(predictions):
        return {
            s: errors(truth[ix], predictions[ix]).mean(0).tolist()
            for s, ix in {**groups, "all": np.arange(len(ids))}.items()
        }

    results = {}
    for title, weights in (
        ("structure", (0.0,)),
        ("catalog_assisted", (0.25, 0.5, 1.0)),
    ):
        trials = []
        for weight in weights:
            features, bases, parts = prepare(weight)
            for alpha in (10.0, 50.0, 150.0):
                for bound in (3.0, 10.0, 30.0):
                    oof = np.zeros(len(train))
                    for a, b in folds:
                        oof[b] = bases[train[b]] * fit(
                            features, bases, train[a], train[b], alpha, bound
                        )
                    score = errors(truth[train], oof).mean(0)
                    trials.append((float(score[1]), weight, alpha, bound, oof))
        _, weight, alpha, bound, oof = min(trials, key=lambda r: r[0])
        features, bases, parts = prepare(weight)
        factors = fit(features, bases, train, np.arange(len(ids)), alpha, bound)
        factors[train] = oof / bases[train]
        estimates = bases * factors
        allocations = [
            [float(m * f) for m in p] for p, f in zip(parts, factors, strict=True)
        ]
        np.testing.assert_allclose([sum(p) for p in allocations], estimates)
        results[title] = dict(
            parameters=dict(weight=weight, alpha=alpha, bound=bound),
            metrics=stats(estimates),
            uncalibrated_metrics=stats(bases),
            trials=[r[:4] for r in trials],
            objects={
                k: dict(
                    mass_kg=float(estimates[i]),
                    component_kg=allocations[i],
                    component_names=(
                        [p["name"] for p in archive[k]["panel"]["parts"]]
                        if archive[k]["panel"]
                        else [""]
                    ),
                    correction=float(factors[i]),
                )
                for i, k in enumerate(ids)
            },
        )
    previous = read(WORK / "bounded_components.json")["predictions"]
    diagnostic = {}
    rng = np.random.default_rng(25)
    for split in ("val", "test"):
        ix = groups[split]
        base = np.array([previous[ids[i]]["mass_kg"] for i in ix])
        revised = np.array(
            [results["catalog_assisted"]["objects"][ids[i]]["mass_kg"] for i in ix]
        )
        delta = errors(truth[ix], revised) - errors(truth[ix], base)
        sampled = rng.integers(0, len(ix), size=(10000, len(ix)))
        worst = np.argsort(delta[:, 0])[-3:][::-1]
        diagnostic[split] = dict(
            mean_delta=delta.mean(0).tolist(),
            ci95=np.quantile(delta[sampled].mean(1), [0.025, 0.975], axis=0).tolist(),
            worst=[
                dict(
                    id=ids[ix[j]],
                    truth=float(truth[ix[j]]),
                    previous=float(base[j]),
                    catalog_assisted=float(revised[j]),
                    ratio=archive[ids[ix[j]]]["ratio"],
                )
                for j in worst
            ],
        )
    result = dict(
        results=results,
        paired_against_previous=diagnostic,
        scale_status_counts={
            s: dict(Counter(archive[ids[i]]["status"] for i in ix))
            for s, ix in groups.items()
        },
        scale_inputs={
            k: dict(status=v["status"], ratio=v["ratio"]) for k, v in archive.items()
        },
    )
    (WORK / "scale_geometry_study.json").write_text(
        json.dumps(result, indent=2, allow_nan=False)
    )
    for k, v in results.items():
        print(k, v["parameters"], v["metrics"], v["uncalibrated_metrics"])
    print(result["scale_status_counts"])
    print(json.dumps(diagnostic))


if __name__ == "__main__":
    main()
