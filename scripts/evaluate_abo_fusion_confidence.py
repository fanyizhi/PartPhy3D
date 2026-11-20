"""Fit a disagreement-dependent physical correction on sealed training batches."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar
from tactile3d.models.calibration import mass_error_arrays

ROOT = Path(__file__).resolve().parents[1] / "paper_assets/abo500_20261120"


def read(path):
    return json.loads(path.read_bytes())


def load_split(split, batches, truth):
    ids = set(read(ROOT / "splits.json")[split])
    rows = {}
    for batch in batches:
        folder = ROOT / "batches" / batch
        path = folder / "predictions.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != read(folder / "prediction_seal.json")["sha256"]:
            raise ValueError(f"Invalid seal for {batch}")
        for row in read(path):
            scene = row["scene_id"]
            if scene not in ids or scene in rows:
                raise ValueError(scene)
            prior = row["image_prior_kg"]
            physical = row["physical_after_projection_kg"]
            # Missing structural evidence gives the image prediction full weight.
            correction = np.log(row["updated"] / prior) if physical else 0.
            disagreement = abs(np.log(physical / prior)) if physical else 0.
            rows[scene] = (truth.get(scene, truth.get(scene.split("_")[0])), prior,
                           row["updated"], correction, disagreement)
    return rows


def run(weights):
    truth = read(weights)
    train = load_split("train", [f"train_{i:03d}" for i in range(0, 175, 25)], truth)
    val = load_split("val", [f"val_{i:03d}" for i in range(0, 100, 25)], truth)
    if len(train) != 175 or len(val) != 100:
        raise ValueError("Unexpected cohort size")
    y, prior, _, delta, gap = np.array(list(train.values())).T
    def loss(k):
        residual = np.log(prior / y) + np.exp(-k * gap) * delta
        return np.mean(np.hypot(1., residual) - 1.)
    fit = minimize_scalar(loss, bounds=(0., 4.), method="bounded")
    k = min([0., float(fit.x), 4.], key=loss)
    y, prior, original, delta, gap = np.array(list(val.values())).T
    candidate = prior * np.exp(np.exp(-k * gap) * delta)
    predictions = {"Frozen fusion": original, "Image prior": prior, "Adaptive fusion": candidate}
    errors = {name: mass_error_arrays(y, pred) for name, pred in predictions.items()}
    metrics = {name: {m: float(v.mean()) for m, v in e.items()} for name, e in errors.items()}
    # Resample paired objects so that difficulty is shared between methods.
    draws = np.random.default_rng(42).integers(0, len(y), size=(10000, len(y)))
    intervals = {}
    for m in errors["Frozen fusion"]:
        diff = errors["Adaptive fusion"][m] - errors["Frozen fusion"][m]
        intervals[m] = np.quantile(diff[draws].mean(axis=1), [.025, .975]).tolist()
    result = dict(train_n=len(train), val_n=len(val), k=k, metrics=metrics,
                  paired_bootstrap_difference_95=intervals,
                  validation_predictions={scene: float(p) for scene, p in zip(val, candidate)})
    out = ROOT / "fusion_validation"
    out.mkdir(exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none", "pdf.fonttype": 42})
    fig, (ax, ecdf) = plt.subplots(1, 2, figsize=(7.3, 3.0), layout="constrained")
    ax.scatter(gap, np.exp(-k * gap), s=14, color="#207369", alpha=.65, linewidths=0)
    ax.set(xlabel="Structure–image disagreement |ln ratio|", ylabel="Retained correction fraction",
           ylim=(-.04, 1.06), title="a   Weight of structural evidence")
    for (name, e), color, style in zip(errors.items(), ["#65558b", "#72777d", "#11756c"], ["--", ":", "-"]):
        ecdf.step(np.sort(e["alde"]), np.arange(1, len(y)+1)/len(y), where="post",
                  color=color, linestyle=style, label=name)
    ecdf.set(xlabel="Absolute log mass error", ylabel="Fraction of validation objects",
             title="b   Error across 100 objects", ylim=(0, 1.04))
    ecdf.legend(frameon=False, fontsize=8, loc="lower right")
    for panel in (ax, ecdf):
        panel.spines[["top", "right"]].set_visible(False)
        panel.title.set_fontsize(10)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"validation.{ext}", dpi=300)
    plt.close(fig)
    print(k, metrics, intervals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    run(parser.parse_args().weights)
