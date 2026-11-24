"""Replay geometry-based thickness limits on all 500 archived ABO objects."""

import argparse
import importlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

from evaluate_structure_cues import ROOT, WORK, errors, read, sealed


def recover_six_parts(project, archive, key):
    sys.path.insert(0, str(project / "scripts"))
    integration = importlib.import_module("run_abo50_mass_hypotheses")
    repair = importlib.import_module("repair_abo50_construction")
    import trimesh

    source = archive / "batch_00" / "weighted" / key
    payload = integration._parse_response_content(
        (source / "attempt_2.txt").read_text()
    )
    mesh_dir = archive / "reconstructions_sparse" / key
    mesh_path = mesh_dir / "camera_space.ply"
    if integration.digest(mesh_path) != read(mesh_dir / "report.json")["mesh_sha256"]:
        raise ValueError("Archived mesh differs from its report")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    if mesh.is_watertight and not mesh.is_winding_consistent:
        trimesh.repair.fix_winding(mesh)
    materials = integration.load_material_priors(
        project / "configs/material_priors.yaml"
    )
    densities = {p.name: p.density_kg_m3.mean for p in materials}
    proposal = repair.decode_weights(payload, list(densities))
    return integration.integrate(proposal, mesh, densities)


def thin_layer_volume(part, area, thickness_mm):
    # Surface area is assigned by the archived model, so this is a first-order
    # bound. Without a verified cross-section we use no edge-length estimate.
    outer = part["outer_volume_m3"]
    return min(outer, area * part["area_share"] * thickness_mm / 1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, required=True)
    args = parser.parse_args()
    splits = {s: read(ROOT / "splits.json")[s] for s in ("train", "val", "test")}
    source = sealed(ROOT)
    test = {r["scene_id"]: r for r in sealed(WORK)}
    evidence = read(WORK / "component_evidence.json")["scenes"]
    for split in ("train", "val"):
        for folder in sorted(ROOT.glob(split + "_???")):
            evidence.update((p["scene_id"], p) for p in sealed(folder / "panels"))
    members = [key for ids in splits.values() for key in ids]
    if len(members) != 500 or len(set(members)) != 500:
        raise ValueError("Expected the full disjoint ABO cohort")
    rows = {}
    candidates = (1.0, 2.0, 4.0, 8.0, 16.0, math.inf)
    thin_roles = re.compile(
        r"\b(mouse[ _-]?pad surface|edge trim|edge profile|lining|slipcover|brand logo)\b",
        re.I,
    )
    soft_materials = {"fabric", "leather", "rubber", "plastic", "silicone", "foam"}
    for split, ids in splits.items():
        for key in ids:
            branch = test[key] if split == "test" else source[key]
            raw = branch["construction_mass_kg"]
            recovered = raw is None
            panel = evidence.get(key, {})
            if recovered:
                panel = recover_six_parts(args.project, args.test_archive, key)
                raw = panel["construction_mass_kg"]
            parts = panel.get("parts", [])
            usable = bool(parts) and math.isclose(
                sum(p["mass_kg"] for p in parts), raw, abs_tol=1e-6
            )
            eligible = (
                [
                    p
                    for p in parts
                    if p["material"] in soft_materials and thin_roles.search(p["name"])
                ]
                if usable
                else []
            )
            estimates = {}
            for cap in candidates:
                reductions = []
                if math.isfinite(cap):
                    for p in eligible:
                        limit = p["density_kg_m3"] * thin_layer_volume(
                            p, panel["area_m2"], cap
                        )
                        reductions.append((p["name"], max(0.0, p["mass_kg"] - limit)))
                estimates[str(cap)] = dict(
                    mass=raw - sum(v for _, v in reductions), corrections=reductions
                )
            rows[key] = dict(
                split=split,
                raw=raw,
                recovered=recovered,
                component_evidence=usable,
                estimates=estimates,
                parts=(
                    [
                        {
                            field: part[field]
                            for field in ("name", "material", "construction", "mass_kg")
                        }
                        for part in parts
                    ]
                    if usable
                    else []
                ),
                recovered_parts=parts if recovered else None,
            )
    # Seal all geometry predictions before looking up any mass labels.
    (WORK / "geometry_predictions.json").write_text(json.dumps(rows, indent=2))
    labels = read(ROOT / "weights.json")
    metric = (
        lambda keys, candidate: errors(
            np.array([labels.get(k, labels.get(k.split("_")[0])) for k in keys]),
            np.array([rows[k]["estimates"][str(candidate)]["mass"] for k in keys]),
        )
        .mean(0)
        .tolist()
    )
    train_scores = {str(c): metric(splits["train"], c) for c in candidates}
    selected = min(candidates, key=lambda c: train_scores[str(c)][1])
    comparisons = {}
    for split, ids in {**splits, "all": members}.items():
        changed = [
            k
            for k in ids
            if rows[k]["estimates"][str(selected)]["mass"] < rows[k]["raw"] - 1e-10
        ]
        comparisons[split] = dict(
            n=len(ids),
            raw=metric(ids, math.inf),
            constrained=metric(ids, selected),
            corrected_ids=changed,
            localized_count=sum(rows[k]["component_evidence"] for k in ids),
        )
    report = dict(
        thickness_mm=str(selected),
        train_candidates=train_scores,
        comparisons=comparisons,
        recovered_ids=[k for k in members if rows[k]["recovered"]],
    )
    (WORK / "geometry_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
