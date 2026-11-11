"""Component-supported VLM construction, non-overlapping domains and rigid moments.

The earlier geometric-cluster driver is archived at commit 1a0559f. This entry
uses an existing VLM/geometry part state and keeps numeric material priors fixed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from tactile3d.experiments.reference_study import load_prepared
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.physics.support_domains import (
    SupportDomain,
    assemble_properties,
    convex_envelope,
    eroded_convex,
    partition_domains,
    solid_from_mesh,
)
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.view_selection import select_azimuth_views
from tactile3d.vision.vlm_interior import _parse_response_content


def validate_hypothesis(answer, roles):
    rows = answer["upholstery"]
    if len(rows) != len(roles) or {r["role"] for r in rows} != set(roles):
        raise ValueError("Every upholstery role must occur exactly once")
    for row in rows:
        fractions = np.asarray(
            [row["foam_fraction"], row["hidden_wood_fraction"]], dtype=float
        )
        if (
            not np.isfinite(fractions).all()
            or np.any(fractions < 0)
            or fractions.sum() > 1 + 1e-8
        ):
            raise ValueError(
                "Core fractions must be nonnegative and sum to at most one"
            )
        row["foam_fraction"], row["hidden_wood_fraction"] = fractions.tolist()
    panel = answer["underside"]
    thickness = float(panel["thickness_mm"])
    if panel["kind"] not in ("manufactured_panel", "visual_cap") or not np.isfinite(
        thickness
    ):
        raise ValueError("Invalid planar-component hypothesis")
    if (panel["kind"] == "visual_cap" and thickness != 0) or (
        panel["kind"] == "manufactured_panel" and thickness <= 0
    ):
        raise ValueError(
            "A panel requires thickness; a cap hypothesis must explicitly use zero"
        )
    answer["underside"]["thickness_mm"] = thickness
    return answer


def construct(pieces, parts, answer, thickness, *, allow_removed=False):
    role_phases = {r["role"]: r for r in answer["upholstery"]}
    bounds = [p.bounds for p in pieces if len(p.faces)]
    if not bounds:
        raise ValueError("No surviving assembly geometry")
    assembly_center = np.mean(bounds, axis=(0, 1))
    domains, excluded = [], []
    for part in parts:
        index = part["component_id"]
        mesh = pieces[index]
        material = part["surface_material"]
        if not len(mesh.faces):
            if not allow_removed:
                raise ValueError(f"Missing reference support for component {index}")
            excluded.append(
                {
                    "component_id": index,
                    "reason": "component geometry removed by simplification",
                    "material_volume_m3": 0.0,
                }
            )
            continue
        if material not in ("wood", "fabric"):
            raise ValueError(
                "This component-state pilot supports upholstered wood-frame assets"
            )
        if material == "fabric":
            if part["functional_role"] not in role_phases:
                raise ValueError("Missing upholstery-role hypothesis")
            hull, source = convex_envelope(mesh, thickness, assembly_center)
            domain = SupportDomain(
                index, material, hull, eroded_convex(hull, thickness), 2, source
            )
        else:
            singular = np.linalg.svd(
                mesh.vertices - mesh.vertices.mean(0), compute_uv=False
            )
            planar = singular[-1] < singular[0] * 1e-7
            if planar and part["functional_role"] != "underside":
                if not allow_removed:
                    raise ValueError("A solid wood support collapsed to a planar patch")
                excluded.append(
                    {
                        "component_id": index,
                        "reason": "solid support lost its three-dimensional extent during simplification",
                        "material_volume_m3": 0.0,
                    }
                )
                continue
            if planar and answer["underside"]["kind"] == "visual_cap":
                excluded.append(
                    {
                        "component_id": index,
                        "reason": "explicit visual-cap hypothesis",
                        "material_volume_m3": 0.0,
                    }
                )
                continue
            panel_thickness = (
                answer["underside"]["thickness_mm"] / 1000 if planar else 0
            )
            if planar:
                spans = np.sort(mesh.extents)
                if panel_thickness > 0.1 * spans[-2]:
                    raise ValueError(
                        "Panel thickness violates the planar-aspect hypothesis"
                    )
            if mesh.is_volume and not planar:
                hull, source = mesh, "provided_closed_boundary"
            else:
                hull, source = convex_envelope(mesh, panel_thickness, assembly_center)
            domain = SupportDomain(
                index, material, hull, solid_from_mesh(hull), 1 if planar else 0, source
            )
        domains.append(domain)
    result = partition_domains(domains)
    result["excluded_geometry_hypotheses"] = excluded
    return result


def properties(
    partition, parts, answer, densities, edited_component=None, edited_density=None
):
    state = {p["component_id"]: p for p in parts}
    phases = {r["role"]: r for r in answer["upholstery"]}
    weighted, allocations = [], []
    for domain in partition["parts"]:
        index = domain["component_id"]
        part = state[index]
        if part["surface_material"] == "wood":
            rho = edited_density if index == edited_component else densities["wood"]
            weighted.append((domain["assigned"], rho))
            allocations.append(
                {
                    "component_id": index,
                    "wood_volume_m3": domain["assigned"]["volume_m3"],
                    "mass_kg": domain["assigned"]["volume_m3"] * rho,
                }
            )
        else:
            phase = phases[part["functional_role"]]
            rho = (
                phase["foam_fraction"] * densities["foam"]
                + phase["hidden_wood_fraction"] * densities["wood"]
            )
            weighted.extend(
                [(domain["shell"], densities["fabric"]), (domain["core"], rho)]
            )
            allocations.append(
                {
                    "component_id": index,
                    "fabric_volume_m3": domain["shell"]["volume_m3"],
                    "foam_volume_m3": domain["core"]["volume_m3"]
                    * phase["foam_fraction"],
                    "hidden_wood_volume_m3": domain["core"]["volume_m3"]
                    * phase["hidden_wood_fraction"],
                    "mass_kg": domain["shell"]["volume_m3"] * densities["fabric"]
                    + domain["core"]["volume_m3"] * rho,
                }
            )
    result = assemble_properties(weighted)
    if not np.isclose(
        sum(p["mass_kg"] for p in allocations), result["mass_kg"], rtol=1e-10
    ):
        raise ValueError("Part masses do not sum to whole-object mass")
    result["parts"] = allocations
    result["core_model"] = (
        "homogenized phase density; hidden spatial arrangement remains unresolved"
    )
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    scene = json.loads((args.reference / "input_manifest.json").read_text())["scene"]
    _mesh, _labels, pieces, _colors = load_prepared(args.reference)
    state_path = args.reference / "part_state_geometry_constrained.json"
    state = json.loads(state_path.read_text())
    if state["geometry_sha256"] != file_sha256(args.reference / "reference_arrays.npz"):
        raise ValueError("Part IDs are bound to different reference geometry")
    parts = state["parts"]
    roles = sorted(
        {p["functional_role"] for p in parts if p["surface_material"] == "fabric"}
    )
    materials = json.loads((args.root / "scenes" / scene / "info.json").read_text())
    # Use the same released density/thickness priors as the existing development controls.
    import yaml

    from tactile3d.experiments.ten_objects import materials_from_info

    cfg = yaml.safe_load(Path("configs/ten_object_pilot.yaml").read_text())
    priors, _ = materials_from_info(materials, cfg)
    densities = {
        kind: float(np.mean(next(m["density"] for m in priors if m["kind"] == kind)))
        for kind in ("wood", "fabric", "foam")
    }
    thickness = float(
        np.mean(next(m["thickness_m"] for m in priors if m["kind"] == "fabric"))
    )
    transforms = json.loads(
        (args.root / "scenes" / scene / "transforms.json").read_text()
    )
    views = select_azimuth_views(
        np.asarray([f["transform_matrix"] for f in transforms["frames"]]), 4
    )
    paths = [
        args.root
        / "scenes"
        / scene
        / transforms["frames"][v["frame_index"]]["file_path"]
        for v in views
    ]
    paths += [
        args.reference / "component_sheet.png",
        args.reference / "views_contacts/global_component_positions.png",
    ]
    features = [
        {
            "component_id": p["component_id"],
            "material": p["surface_material"],
            "role": p["functional_role"],
            "extent_m": pieces[p["component_id"]].extents.tolist(),
            "closed_as_loaded": bool(pieces[p["component_id"]].is_volume),
        }
        for p in parts
    ]
    protocol = {
        "scene": scene,
        "scope": "exposed reference-geometry development case",
        "part_state_sha256": file_sha256(state_path),
        "source_sha256": file_sha256(Path(__file__)),
        "geometry_module_sha256": file_sha256(
            Path(__file__).parents[1] / "physics/support_domains.py"
        ),
        "input_hashes": {str(p): file_sha256(p) for p in paths},
        "selected_views": views,
        "densities_kg_m3": densities,
        "fabric_thickness_m": thickness,
        "components": features,
        "target_mass_used_for_inference": False,
        "rounds": 3,
        "geometry": "provided closed wood solids; per-component convex/prism completion elsewhere; disjoint CSG",
        "aggregation": "equal mean of three independent-question construction hypotheses; not calibrated probabilities",
    }
    write_json(args.output / "protocol.json", protocol)
    model = LocalVLM(args.model, max_tokens=1000, max_pixels=768**2, quantized=True)
    write_json(args.output / "model.json", model.provenance)
    images = [Image.open(p).copy() for p in paths]
    prompt = (
        "Analyze one upholstered wood-frame object from four original views and two numbered mesh diagrams. "
        "Surface materials and functional roles were inferred previously and are fixed in this experiment: "
        + json.dumps(features)
        + ". Do NOT estimate mass or densities. Numeric material priors and cloth "
        "thickness are supplied separately. Give core construction hypotheses for each of these upholstery "
        f"roles: {json.dumps(roles)}. foam_fraction and hidden_wood_fraction are MACROSCOPIC CORE VOLUME "
        "fractions, summing <=1. Remaining fraction is macroscopic empty space. Foam density ALREADY "
        "includes microscopic pores: do not reduce foam_fraction again for micro-porosity. Exclude the "
        "visible wooden legs/rails from hidden_wood_fraction; they already have separate support domains. "
        "For the planar wood-labelled underside, infer whether it represents a manufactured thin panel "
        "or a rendering cap, and if a panel estimate its thickness in MILLIMETERS. A visual_cap hypothesis "
        "uses thickness_mm=0 explicitly; it is not measured absence of a panel. Return JSON with "
        "upholstery:[{role:string,foam_fraction:number,hidden_wood_fraction:number}], "
        "underside:{kind:manufactured_panel|visual_cap,thickness_mm:number}, evidence:string. "
        "Cover every role once, with no extra roles. Avoid physical certainty for hidden structure."
    )
    focuses = [
        "Infer the assembly construction from all evidence.",
        "Independently inspect support versus padding, and distinguish bulk foam from macroscopic air.",
        "Independently inspect the underside and whether any hidden frame is being counted twice.",
    ]
    results = []
    for trial, focus in enumerate(focuses):
        response = model.generate(
            prompt + "\n" + focus,
            images,
            "Return physically consistent construction JSON.",
            seed=2410 + trial,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        record = {"prompt": prompt + "\n" + focus, "response": asdict(response)}
        write_json(args.output / f"raw_{trial}.json", record)
        answer = validate_hypothesis(_parse_response_content(response.text), roles)
        partition = construct(pieces, parts, answer, thickness)
        result = properties(partition, parts, answer, densities)
        wood_id = next(
            p["component_id"] for p in parts if p["functional_role"] == "leg"
        )
        edited = properties(partition, parts, answer, densities, wood_id, 2700.0)
        expected_delta = next(
            p["assigned"]["volume_m3"]
            for p in partition["parts"]
            if p["component_id"] == wood_id
        ) * (2700 - densities["wood"])
        if not np.isclose(
            edited["mass_kg"] - result["mass_kg"], expected_delta, rtol=1e-7
        ):
            raise ValueError("Density edit violates fixed-domain mass conservation")
        result.update(
            {
                "trial": trial,
                "answer": answer,
                "partition": partition,
                "edited_component_id": wood_id,
                "edited_mass_kg": edited["mass_kg"],
                "edit_mass_delta_kg": expected_delta,
            }
        )
        write_json(args.output / f"hypothesis_{trial}.json", result)
        results.append(result)
        print(
            "COMPONENT_MASS",
            trial,
            result["mass_kg"],
            "overlap_removed_m3",
            partition["overlap_removed_m3"],
            flush=True,
        )
    prediction = {
        "scene_id": scene,
        "mass_kg": float(np.mean([r["mass_kg"] for r in results])),
        "hypothesis_masses_kg": [r["mass_kg"] for r in results],
    }
    write_json(args.output / "predictions.json", [prediction])
    files = list(args.output.glob("*.json"))
    write_json(
        args.output / "prediction_seal.json", {p.name: file_sha256(p) for p in files}
    )
    weights = json.loads((args.root / "filtered_product_weights.json").read_text())
    target = float(weights.get(scene, weights.get(scene.split("_")[0])))
    report = {
        "prediction": prediction,
        "target_mass_kg": target,
        "metrics": mass_metrics([target], [prediction["mass_kg"]]),
        "scope": "single exposed development case; not a whole-cohort comparison",
    }
    write_json(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reference", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(json.dumps(run(parser.parse_args())), flush=True)


if __name__ == "__main__":
    main()
