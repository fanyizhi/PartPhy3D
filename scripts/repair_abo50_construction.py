"""Re-query failed construction hypotheses without reading the mass labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh
from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo50_mass_hypotheses import (
    ROOT,
    SYSTEM,
    VIEWS,
    LocalVLM,
    _parse_response_content,
    integrate,
    load_material_priors,
    positive,
)

CONTRACT = """Re-evaluate the previous construction proposal using the three images.
Return JSON with max_length_m and parts. List one to six components. Each component
has name, evidence, material, construction, volume_share and area_share.
Allowed material names: {materials}.
construction must be exactly solid, shell or partial. Voids belong inside a shell
or a partly occupied component. volume_share allocates the OUTER extent of each
component, even for thin walls. Every component needs positive volume_share and
the shares must sum to 1. area_share allocates visible surface area and must sum
to 1. Shell also needs positive thickness_mm. Partial also needs occupied_fraction
strictly above zero and at most one. Give a positive maximum length in meters.
Return only JSON. Do not derive or estimate mass."""

WEIGHT_CONTRACT = """Describe the real object seen in these images. Return JSON with
max_length_m and parts. Use one to five material-bearing components. Each part has
name, evidence, material_id, construction_id, v, a and parameter.
Choose material_id by number from this list: {materials}.
Use construction_id 1 for filled material, 2 for a hollow thin shell and 3 for an
unfilled lattice or assembly. For each component, give a positive relative size
weight v (1–100) and a visible surface weight a (0–100), with at least one positive
a. The weights need not sum to 100. Never list a void as a part. Set parameter to
1 for filled material, wall thickness in millimeters for a shell, or the material
fill fraction strictly between 0 and 1 for an unfilled assembly. Give a positive
max_length_m. Return JSON only. Do not estimate mass."""


def decode_weights(payload: dict, material_names: list[str]) -> dict:
    parts = payload["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 5:
        raise ValueError("Expected one to five material components")
    volumes = [positive(part["v"], "relative component size") for part in parts]
    areas = [float(part["a"]) for part in parts]
    area_total = positive(sum(areas), "total visible area weight")
    kinds = {1: "solid", 2: "shell", 3: "partial"}
    result = []
    for part, volume, area in zip(parts, volumes, areas, strict=True):
        material_id = part["material_id"]
        kind_id = part["construction_id"]
        if type(material_id) is not int or not 1 <= material_id <= len(material_names):
            raise ValueError("Choose an available integer material_id")
        if type(kind_id) is not int or kind_id not in kinds:
            raise ValueError("Choose construction_id 1, 2 or 3")
        if not 0 <= area <= 100 or not 1 <= volume <= 100:
            raise ValueError("Weight outside stated range")
        row = {
            "name": part["name"],
            "evidence": part["evidence"],
            "material": material_names[material_id - 1],
            "construction": kinds[kind_id],
            "volume_share": volume / sum(volumes),
            "area_share": area / area_total,
        }
        if kind_id == 2:
            row["thickness_mm"] = part["parameter"]
        elif kind_id == 3:
            row["occupied_fraction"] = part["parameter"]
        result.append(row)
    return {"max_length_m": payload["max_length_m"], "parts": result}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--weighted-followup", action="store_true")
    args = parser.parse_args()
    previous = args.benchmark / (
        "construction_retry" if args.weighted_followup else "winding_only"
    )
    original = args.benchmark / "mass_hypotheses"
    for directory in (previous, original):
        if (
            digest(directory / "predictions.json")
            != read_json(directory / "prediction_seal.json")["sha256"]
        ):
            raise ValueError(f"Prediction seal mismatch in {directory.name}")
    previous_rows = read_json(previous / "predictions.json")
    original_rows = {
        row["scene_id"]: row for row in read_json(original / "predictions.json")
    }
    obs = args.benchmark / "observations"
    manifest = read_json(obs / "cohort.json")
    protocol = read_json(previous / "protocol.json")
    prior = ROOT / "configs/material_priors.yaml"
    if (
        digest(obs / "cohort.json") != protocol["cohort_sha256"]
        or digest(prior) != protocol["material_priors_sha256"]
    ):
        raise ValueError("Repair must use the sealed cohort and priors")
    if [row["scene_id"] for row in previous_rows] != manifest["object_ids"]:
        raise ValueError("Prediction order differs from the frozen cohort")
    densities = {p.name: p.density_kg_m3.mean for p in load_material_priors(prior)}
    args.output.mkdir()
    model = LocalVLM(args.model, max_tokens=1200, max_pixels=448**2, quantized=True)
    contract = CONTRACT.format(materials=", ".join(densities))
    if args.weighted_followup:
        contract = WEIGHT_CONTRACT.format(
            materials=", ".join(f"{i}: {name}" for i, name in enumerate(densities, 1))
        )
    write_json(
        args.output / "protocol.json",
        {
            **protocol,
            "scope": "target-free validation-guided construction re-query",
            "repair_source_sha256": digest(previous / "predictions.json"),
            "repair_code_sha256": digest(Path(__file__)),
            "prompt": contract,
            "maximum_attempts": 2,
            "weighted_followup": args.weighted_followup,
            "model": model.provenance,
            "mass_labels_used_for_inference": False,
        },
    )
    rows = []
    for old in previous_rows:
        if "construction_mass_kg" in old:
            rows.append(old)
            continue
        scene = old["scene_id"]
        row = dict(old)
        run = args.benchmark / "reconstructions" / scene
        mesh_path = run / "camera_space.ply"
        mesh_hash = digest(mesh_path)
        if mesh_hash != read_json(run / "report.json")["mesh_sha256"]:
            raise ValueError(f"Mesh changed for {scene}")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        if mesh.is_watertight and not mesh.is_winding_consistent:
            trimesh.repair.fix_winding(mesh)
        images = []
        for view in VIEWS:
            relative = f"scenes/{scene}/images/{scene}_{view:02d}.png"
            path = obs / relative
            if digest(path) != manifest["files_sha256"][relative]:
                raise ValueError(f"Observation changed for {scene}")
            with Image.open(path) as image:
                images.append(image.copy())
        response_path = original / scene / "construction.txt"
        prior_response = response_path.read_text(encoding="utf-8")
        row["repair_original_response_sha256"] = digest(response_path)
        error = original_rows[scene]["errors"].get("construction", row.get("error", ""))
        child = args.output / scene
        child.mkdir()
        attempts = []
        for step in (1, 2):
            if args.weighted_followup:
                prompt = (
                    contract if step == 1 else f"{contract}\nValidation error: {error}"
                )
            else:
                prompt = f"{contract}\nMost recent validation error: {error}\nPrevious proposal:\n{prior_response}"
            response = model.generate(prompt, images, SYSTEM)
            prior_response = response.text
            (child / f"attempt_{step}.txt").write_text(prior_response, encoding="utf-8")
            try:
                payload = _parse_response_content(prior_response)
                if args.weighted_followup:
                    payload = decode_weights(payload, list(densities))
                row.update(integrate(payload, mesh, densities))
            except (KeyError, TypeError, ValueError) as failure:
                error = str(failure)
                attempts.append({"step": step, "error": error})
            else:
                row.pop("error", None)
                row["mesh_sha256_before_winding_repair"] = mesh_hash
                attempts.append({"step": step, "recovered": True})
                break
        row["repair_attempts"] = attempts
        rows.append(row)
        write_json(child / "result.json", row)
        print(scene, attempts, flush=True)
    path = args.output / "predictions.json"
    write_json(path, rows)
    write_json(args.output / "prediction_seal.json", {"sha256": digest(path)})


if __name__ == "__main__":
    main()
