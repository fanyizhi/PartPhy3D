"""Infer local material volumes in fixed units, then score sealed predictions."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_abo50_mass_hypotheses import paired_comparison
from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo50_mass_hypotheses import DIRECT, ROOT, SYSTEM, VIEWS
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.materials import load_material_priors
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.vlm_interior import _parse_response_content

COMPONENT_PROMPT = """Estimate local physical dimensions of the one complete
unpackaged object across these three views. Infer its longest dimension in cm.
Do not estimate its mass. Decompose into 1–8 nonoverlapping groups of pieces.
Use all load-bearing or mass-bearing pieces, including concealed components
plausibly required for the depicted functional object. Distinguish a bare
enclosure from a working appliance when the views give evidence. Exclude air,
decorative pattern and packaging. A foam cushion's bulk volume includes its cells.
For thin walls use sheet; for hollow round supports use tube; keep thickness
independent of span. For a complicated compact component use a bounding block
and the fraction actually occupied by the specified material. Prefer physically
plausible local dimensions to allocating fractions of the object's outer box.
Material choices: {materials}.
Return one compact JSON object with longest_cm and groups. Each group is exactly
[name, material, shape, integer_count, dims_cm, occupied_fraction, evidence].
Shapes and dimension order: sheet=[length,width,thickness], block=[length,width,height],
ellipsoid=[full_length,full_width,full_height], tube=[length,outer_diameter,wall],
rod=[length,diameter]. All lengths in cm. occupied_fraction in (0,1], 1 for a
fully occupied shape. evidence is exactly visible or inferred. Count each piece
only once. Return JSON only.
"""


def number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field} must be positive and finite")
    return float(value)


def integrate(payload, densities):
    extent = number(payload["longest_cm"], "longest_cm")
    groups = payload["groups"]
    if not isinstance(groups, list) or not 1 <= len(groups) <= 8:
        raise ValueError("Expected 1–8 groups")
    records = []
    names = set()
    for name, material, shape, count, dims, fill, evidence in groups:
        if not isinstance(name, str) or not name.strip() or name.casefold() in names:
            raise ValueError("Each group needs a distinct name")
        names.add(name.casefold())
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or not 1 <= count <= 300
        ):
            raise ValueError("Invalid part count")
        if evidence not in {"visible", "inferred"} or material not in densities:
            raise ValueError("Unknown material or evidence status")
        expected = {"sheet": 3, "block": 3, "ellipsoid": 3, "tube": 3, "rod": 2}
        if not isinstance(dims, list) or len(dims) != expected.get(shape):
            raise ValueError("Invalid shape or dimension count")
        dims = [number(v, "dimension_cm") for v in dims]
        if max(dims) > 1.8 * extent:
            raise ValueError("A part exceeds the object scale")
        fill = number(fill, "occupied_fraction")
        if fill > 1:
            raise ValueError("Occupancy > 1")
        if shape == "tube":
            length, diameter, wall = dims
            if 2 * wall >= diameter:
                raise ValueError("Tube wall leaves no cavity")
            volume_cm3 = math.pi * length * wall * (diameter - wall)
        elif shape == "rod":
            length, diameter = dims
            volume_cm3 = math.pi * length * (diameter / 2) ** 2
        else:
            if shape == "sheet" and dims[2] > min(dims[:2]):
                raise ValueError("Invalid sheet thickness")
            volume_cm3 = math.prod(dims) * (math.pi / 6 if shape == "ellipsoid" else 1)
        volume = volume_cm3 * 1e-6 * count * fill
        records.append(
            dict(
                name=name,
                material=material,
                shape=shape,
                count=count,
                dims_cm=dims,
                occupied_fraction=fill,
                evidence=evidence,
                volume_m3=volume,
                mass_kg=volume * densities[material],
            )
        )
    return records


def sealed_rows(run):
    path = run / "predictions.json"
    if digest(path) != read_json(run / "prediction_seal.json")["sha256"]:
        raise ValueError(f"Invalid prediction seal at {run}")
    rows = {row["scene_id"]: row for row in read_json(path)}
    return rows


def predict(args):
    archive = sealed_rows(args.archived_direct) if args.archived_direct else {}
    paths = [(p, read_json(p / "cohort.json")) for p in args.observations]
    scenes = [(p, scene) for p, cohort in paths for scene in cohort["object_ids"]]
    if len(scenes) != len({scene for _, scene in scenes}):
        raise ValueError("A scene occurs in more than one input cohort")
    for folder, cohort in paths:
        for scene in cohort["object_ids"]:
            for view in VIEWS:
                path = f"scenes/{scene}/images/{scene}_{view:02d}.png"
                if digest(folder / path) != cohort["files_sha256"][path]:
                    raise ValueError(f"Observation hash mismatch: {path}")
    priors = ROOT / "configs/material_priors.yaml"
    densities = {p.name: p.density_kg_m3.mean for p in load_material_priors(priors)}
    prompt = COMPONENT_PROMPT.format(materials=", ".join(densities))
    args.output.mkdir(parents=True, exist_ok=False)
    model = LocalVLM(
        args.model, max_tokens=1200, max_pixels=448**2, quantized=not args.bf16
    )
    write_json(
        args.output / "protocol.json",
        dict(
            cohorts=[
                dict(
                    cohort_sha256=digest(folder / "cohort.json"),
                    object_ids=c["object_ids"],
                )
                for folder, c in paths
            ],
            code_sha256=digest(Path(__file__)),
            model=model.provenance,
            quantized=not args.bf16,
            views=list(VIEWS),
            archived_direct_seal_sha256=(
                digest(args.archived_direct / "predictions.json") if archive else None
            ),
            priors_sha256=digest(priors),
            prompt=prompt,
            direct_prompt=DIRECT,
            system=SYSTEM,
            hypotheses=[
                "visible only",
                "visible plus inferred",
                "equal-weight geometric mean of component and direct",
            ],
            fallback="On component failure after one constrained retry, ensemble uses direct only.",
            mass_labels_used=False,
        ),
    )
    rows = []
    for folder, scene in scenes:
        images = []
        child = args.output / scene
        child.mkdir()
        for view in VIEWS:
            with Image.open(
                folder / f"scenes/{scene}/images/{scene}_{view:02d}.png"
            ) as image:
                images.append(image.copy())
        row = {"scene_id": scene, "errors": []}
        if scene in archive:
            row["direct_mass_kg"] = archive[scene]["direct_mass_kg"]
        else:
            response = model.generate(DIRECT, images, SYSTEM)
            (child / "direct.txt").write_text(response.text, encoding="utf-8")
            row["direct_mass_kg"] = number(
                _parse_response_content(response.text)["mass_kg"], "mass_kg"
            )
        for attempt in range(2):
            request = (
                prompt
                if not attempt
                else prompt
                + f"\nLast answer failed validation: {row['errors'][-1]}. Fix your JSON."
            )
            response = model.generate(request, images, SYSTEM)
            (child / f"components_{attempt}.txt").write_text(
                response.text, encoding="utf-8"
            )
            try:
                payload = _parse_response_content(response.text)
                parts = integrate(payload, densities)
                row.update(
                    parts=parts,
                    longest_cm=payload["longest_cm"],
                    component_mass_kg=sum(p["mass_kg"] for p in parts),
                    visible_mass_kg=sum(
                        p["mass_kg"] for p in parts if p["evidence"] == "visible"
                    ),
                )
                break
            except (AttributeError, KeyError, TypeError, ValueError) as error:
                row["errors"].append(str(error))
        row["ensemble_mass_kg"] = math.sqrt(
            row["direct_mass_kg"] * row.get("component_mass_kg", row["direct_mass_kg"])
        )
        rows.append(row)
        write_json(child / "result.json", row)
        print(
            scene,
            row.get("component_mass_kg"),
            row["direct_mass_kg"],
            row["errors"],
            flush=True,
        )
    result = args.output / "predictions.json"
    write_json(result, rows)
    write_json(args.output / "prediction_seal.json", {"sha256": digest(result)})


def score(args):
    rows = sealed_rows(args.run)
    protocol = read_json(args.run / "protocol.json")
    weights = read_json(args.weights)
    old = sealed_rows(args.archived_direct) if args.archived_direct else {}
    nerf = {r["scene_id"]: r["mass_kg"] for r in read_json(args.nerf)["per_scene"]}
    report = {"prediction_sha256": digest(args.run / "predictions.json"), "cohorts": []}
    for cohort in protocol["cohorts"]:
        items = []
        for scene in cohort["object_ids"]:
            r = rows[scene]
            target = weights.get(scene, weights.get(scene.split("_")[0]))
            if target is None:
                raise ValueError(f"Missing target for {scene}")
            items.append(
                dict(
                    scene_id=scene,
                    target_mass_kg=target,
                    direct_mass_kg=r["direct_mass_kg"],
                    ensemble_mass_kg=r["ensemble_mass_kg"],
                    visible_mass_kg=r.get("visible_mass_kg"),
                    component_mass_kg=r.get("component_mass_kg"),
                    nerf_mass_kg=nerf.get(scene),
                    prior_mass_kg=old.get(scene, {}).get("construction_mass_kg"),
                )
            )
        stats = {}
        for method in items[0].keys() - {"scene_id", "target_mass_kg"}:
            pairs = [
                (r["target_mass_kg"], r[method], r["direct_mass_kg"])
                for r in items
                if r[method] is not None and r[method] > 0
            ]
            if pairs:
                target, pred, _ = np.array(pairs).T
                stats[method] = dict(
                    n=len(pairs),
                    metrics=mass_metrics(target, pred),
                    paired_vs_direct=paired_comparison(pairs),
                )
        report["cohorts"].append(dict(n=len(items), stats=stats, per_scene=items))
    write_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    infer = sub.add_parser("predict")
    infer.add_argument("--observations", type=Path, nargs="+", required=True)
    infer.add_argument("--model", type=Path, required=True)
    infer.add_argument("--bf16", action="store_true")
    evaluate = sub.add_parser("score")
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--weights", type=Path, required=True)
    evaluate.add_argument("--nerf", type=Path, required=True)
    for command in (infer, evaluate):
        command.add_argument("--archived-direct", type=Path)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (predict if args.action == "predict" else score)(args)


if __name__ == "__main__":
    main()
