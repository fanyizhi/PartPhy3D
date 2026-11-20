"""Score sealed ABO-500 predictions with product-paired, split-stratified intervals."""
import argparse
import json
from pathlib import Path

import numpy as np

from build_abo500_paper_tables import ROOT, sealed
from prepare_abo50_benchmark import read_json
from tactile3d.models.calibration import mass_error_arrays


def evaluate(weights_path):
    splits = read_json(ROOT / "splits.json")
    splits = {k: splits[k] for k in ("train", "val", "test")}
    ids = sum(splits.values(), [])
    if len(set(s.split("_")[0] for s in ids)) != 500:
        raise ValueError("Expected 500 unique products")
    ours, nerf = sealed(ROOT), sealed(ROOT / "nerf_run")
    if set(ours) != set(ids) or set(nerf) != set(ids):
        raise ValueError("Mismatched cohorts")
    prior = {i: r["mass_kg"] for i, r in sealed(ROOT / "control").items()}
    archive = read_json(ROOT.parent / "abo100_extension_20261120/final100/score.json")
    prior.update({r["scene_id"]: r["image_prior_kg"] for r in archive["per_scene"]})
    # Archived role predictions omit the unfused mass, so join the sealed selection.
    selection_path = ROOT.parent / "abo100_extension_20261120/final100/selected"
    archived = {r["scene_id"]: r for r in sealed(selection_path)}
    branches = {i: (archived[i] if i in splits["test"] else ours[i]) for i in ids}
    same_run_prior = np.array([branches[i]["image_prior_kg"] for i in ids])
    valid = np.array([branches[i]["construction_mass_kg"] is not None for i in ids])
    physical = np.array([branches[i]["construction_mass_kg"] if ok else branches[i]["image_prior_kg"]
                         for i, ok in zip(ids, valid)])
    weights = read_json(weights_path)
    gt = np.array([weights.get(i, weights.get(i.split("_")[0])) for i in ids])
    predictors = {"ours": [ours[i]["updated"] for i in ids],
                  "image": [prior[i] for i in ids], "nerf": [np.mean(nerf[i]) for i in ids]}
    errors = {m: mass_error_arrays(gt, pred) for m, pred in predictors.items()}
    ablations = {"image_same_run": mass_error_arrays(gt, same_run_prior),
                 "physical_with_image_fallback": mass_error_arrays(gt, physical)}
    rng = np.random.default_rng(17)
    strata = {k: np.array([ids.index(i) for i in part]) for k, part in splits.items()}
    report = {}
    for split in [*splits, "all"]:
        groups = list(strata.values()) if split == "all" else [strata[split]]
        ix = np.concatenate(groups)
        draws = np.concatenate([rng.choice(g, (10000, len(g))) for g in groups], axis=1)
        comparisons = {}
        for ref in ("image", "nerf"):
            paired = {}
            for key, values in errors["ours"].items():
                delta = values - errors[ref][key]
                paired[key] = {"mean": float(delta[ix].mean()),
                               "ci95": np.quantile(delta[draws].mean(axis=1), [.025, .975]).tolist()}
            comparisons[ref] = paired
        absolute = errors["ours"]["ade_kg"][ix]
        order = np.argsort(-absolute)
        report[split] = {"n": len(ix), "paired_ours_minus": comparisons,
                         "median_ade_kg": float(np.median(absolute)),
                         "top_decile_ade_share": float(sum(absolute[order[:len(ix)//10]]) / sum(absolute)),
                         "largest_error_scene": ids[ix[order[0]]]}
        report[split]["valid_structure"] = int(valid[ix].sum())
        report[split]["cover_edits_triggered"] = sum(bool(ours[ids[j]].get("edits")) for j in ix)
        report[split]["ablations"] = {method: {m: float(v[ix].mean()) for m, v in fields.items()}
                                      for method, fields in ablations.items()}
        server = read_json(ROOT / "score.json")[split]
        for name, values in errors["ours"].items():
            if not np.isclose(server[name], values[ix].mean(), rtol=1e-10):
                raise ValueError(f"Server score mismatch: {split} {name}")
    result = {"seed": 17, "bootstrap_draws": 10000, "ape_unit": "fraction",
              "protocol": "Split-stratified paired percentile intervals, no multiplicity adjustment.",
              "splits": report}
    (ROOT / "paired_review.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    evaluate(parser.parse_args().weights)
