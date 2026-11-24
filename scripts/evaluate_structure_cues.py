"""Score object-matched structural signals using only training labels for calibration."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/root/autodl-fs/PartPhy3D/abo500_20261120")
WORK = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_bytes())


def sealed(folder):
    path = folder / "predictions.json"
    assert (
        hashlib.sha256(path.read_bytes()).hexdigest()
        == read(folder / "prediction_seal.json")["sha256"]
    )
    return read(path)


def errors(truth, estimates):
    ratio = estimates / truth
    return np.column_stack(
        (
            abs(estimates - truth),
            abs(np.log(ratio)),
            100 * abs(ratio - 1),
            np.minimum(ratio, 1 / ratio),
        )
    )


def features(mass, record):
    parts = record.get("parts", [])
    available = bool(parts) and np.isclose(sum(p["mass_kg"] for p in parts), mass)
    parts = parts if available else []
    # Fixed material groups and role features are computed before loading targets.
    categories = ("fabric", "foam", "wood", "steel", "plastic", "aluminum")
    fractions = [
        sum(p["mass_kg"] / mass for p in parts if p.get("material") == k)
        for k in categories
    ]
    roles = [
        sum(p["mass_kg"] / mass for p in parts if p.get("construction") == k)
        for k in ("shell", "partial")
    ]
    return [np.log(mass), int(available), np.log1p(len(parts)), *fractions, *roles]


def main():
    split = read(ROOT / "splits.json")
    combined = sealed(ROOT)
    selected = {r["scene_id"]: r for r in sealed(WORK)}
    evidence = read(WORK / "component_evidence.json")["scenes"]
    for name in ("train", "val"):
        for folder in sorted(ROOT.glob(name + "_???")):
            for row in sealed(folder / "panels"):
                if row["scene_id"] in split[name]:
                    evidence[row["scene_id"]] = row
    assert set(selected) == set(split["test"])
    labels = read(ROOT / "weights.json")
    records = {}
    missing = []
    for name in ("train", "val", "test"):
        for key in split[name]:
            branch = selected[key] if name == "test" else combined[key]
            mass = branch["construction_mass_kg"]
            if mass is None:
                missing.append(key)
                continue
            records[key] = dict(
                split=name,
                raw=mass,
                image=branch["image_prior_kg"],
                original=combined[key]["updated"],
                x=features(mass, evidence.get(key, {})),
            )
    ids = list(records)
    assert len(ids) == 499 and len(missing) == 1
    gt = np.array([labels.get(k, labels.get(k.split("_")[0])) for k in ids])
    x = np.array([records[k]["x"] for k in ids])
    raw = np.exp(x[:, 0])
    image = np.array([records[k]["image"] for k in ids])
    train = np.array([records[k]["split"] == "train" for k in ids])
    metrics = {}
    corrected = {}
    # Fit a multiplicative correction on the physical kg estimate. Both controls
    # use the same fixed ridge penalty, target and out-of-fold training protocol.
    for title, design in (("scale_only", x[:, :1]), ("calibrated_structure", x)):
        model = make_pipeline(StandardScaler(), Ridge(alpha=10))
        residual = np.log(gt[train] / raw[train])
        model.fit(design[train], residual)
        adjustment = model.predict(design)
        adjustment[train] = cross_val_predict(
            model, design[train], residual, cv=KFold(5, shuffle=True, random_state=25)
        )
        corrected[title] = raw * np.exp(adjustment)
    structure = corrected["calibrated_structure"]

    # Test the physical cue at a fixed 25% log weight without fitting on image predictions.
    def mix(i, s):
        return np.exp(0.75 * np.log(i) + 0.25 * np.log(s))

    mixed = mix(image, structure)
    methods = dict(
        raw=raw,
        image=image,
        original=np.array([records[k]["original"] for k in ids]),
        **corrected,
        mixed=mixed,
    )
    comparisons = {}
    for name in ("train", "val", "test", "all"):
        ix = np.array(
            [
                j
                for j, k in enumerate(ids)
                if name == "all" or records[k]["split"] == name
            ]
        )
        metrics[name] = {
            m: errors(gt[ix], v[ix]).mean(0).tolist() for m, v in methods.items()
        }
        if name in ("val", "test"):
            rng = np.random.default_rng(25)
            diff = errors(gt[ix], mixed[ix]) - errors(gt[ix], image[ix])
            ci = np.quantile(
                diff[rng.integers(0, len(ix), (10000, len(ix)))].mean(1),
                [0.025, 0.975],
                axis=0,
            )
            null = np.array(
                [
                    errors(gt[ix], mix(image[ix], rng.permutation(structure[ix]))).mean(
                        0
                    )
                    for _ in range(2000)
                ]
            )
            comparisons[name] = dict(
                paired_diff=diff.mean(0).tolist(),
                ci95=ci.tolist(),
                shuffled_mean=null.mean(0).tolist(),
            )
    for j, k in enumerate(ids):
        records[k].update({m: float(v[j]) for m, v in methods.items()})
        records[k]["gt"] = float(gt[j])
    output = dict(
        metrics_order=["ADE_kg", "ALDE", "APE_percent", "MnRE"],
        metrics=metrics,
        comparisons=comparisons,
        missing=missing,
        records=records,
    )
    (WORK / "structural_ablation.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({"metrics": metrics, "comparisons": comparisons}))


if __name__ == "__main__":
    main()
