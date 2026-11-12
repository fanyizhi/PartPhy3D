"""Component-supported VLM construction, non-overlapping domains and rigid moments.

The earlier geometric-cluster driver is archived at commit 1a0559f. This entry
uses an existing VLM/geometry part state and keeps numeric material priors fixed.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
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


def validate_grounded_roles(pieces, parts):
    """Reject an impossible leg binding; never infer material from a role."""
    if len(parts) != len(pieces) or {p["component_id"] for p in parts} != set(
        range(len(pieces))
    ):
        raise ValueError("Part state must cover every native component exactly once")
    bounds = np.asarray([mesh.bounds for mesh in pieces])
    object_low, object_high = bounds[:, 0].min(0), bounds[:, 1].max(0)
    tolerance = np.linalg.norm(object_high - object_low) * 1e-3
    conflicts = []
    for part in parts:
        if part["functional_role"] != "leg":
            continue
        box = pieces[part["component_id"]].bounds
        size = box[1] - box[0]
        if abs(box[0, 2] - object_low[2]) > tolerance or size[2] <= 1.5 * max(
            size[0], size[1]
        ):
            conflicts.append(part["component_id"])
    if conflicts:
        raise ValueError(
            f"Leg role conflicts with z-up ground support and extent for native IDs {conflicts}; rebind components before physical inference"
        )


def evaluate_saved_prediction(output, root):
    """Open catalogue targets only after verifying every sealed prediction input."""
    seal = json.loads((output / "prediction_seal.json").read_text())
    if "predictions.json" not in seal or any(
        file_sha256(output / name) != value for name, value in seal.items()
    ):
        raise ValueError("Prediction seal mismatch")
    rows = json.loads((output / "predictions.json").read_text())
    if len(rows) != 1:
        raise ValueError("This development runner expects one complete object")
    prediction = rows[0]
    scene = prediction["scene_id"]
    weights = json.loads((root / "filtered_product_weights.json").read_text())
    value = weights.get(scene, weights.get(scene.split("_")[0]))
    if value is None:
        raise ValueError(f"No catalogue mass for {scene} or its ASIN")
    target = float(value)
    report = {
        "prediction": prediction,
        "target_mass_kg": target,
        "metrics": mass_metrics([target], [prediction["mass_kg"]]),
        "scope": "single exposed development object; consult protocol for query budgets",
    }
    write_json(output / "report.json", report)
    return report


def seal_prediction_files(output):
    paths = [
        p
        for p in output.glob("*.json")
        if p.name in {"protocol.json", "model.json", "predictions.json"}
        or p.stem.startswith(("raw_", "hypothesis_"))
    ]
    write_json(output / "prediction_seal.json", {p.name: file_sha256(p) for p in paths})


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


def apply_beam_inventory(base, proposal, role_geometry, sections):
    """Convert a bounded beam inventory to volume fractions, without clipping.

    Beam position remains unresolved. This is an auditable volume prior, not
    recovered internal CAD or proof that a hidden frame exists.
    """
    rows = proposal["upholstery"]
    if len(rows) != len(role_geometry) or {r["role"] for r in rows} != set(
        role_geometry
    ):
        raise ValueError("Every upholstery role needs exactly one beam inventory")
    answer = deepcopy(base)
    for phase in answer["upholstery"]:
        role = phase["role"]
        beams = next(r["beams"] for r in rows if r["role"] == role)
        volume = 0.0
        for beam in beams:
            count, length = beam["count"], float(beam["length_m"])
            section = beam["section_component_id"]
            if (
                type(count) is not int
                or count <= 0
                or type(section) is not int
                or section not in sections
                or not np.isfinite(length)
                or not 0 < length <= role_geometry[role]["diagonal_m"]
            ):
                raise ValueError("Invalid beam count, section or length")
            volume += count * length * sections[section]["area_m2"]
        core = role_geometry[role]["core_volume_m3"]
        if core <= 0 or volume > core * (1 - phase["foam_fraction"]) + 1e-12:
            raise ValueError("Beam and foam volumes exceed the available core")
        phase["hidden_wood_fraction"] = volume / core
        phase["hidden_beams"] = deepcopy(beams)
    return validate_hypothesis(answer, sorted(role_geometry))


def refine_construction(args):
    """Paired second-round construction queries; freeze all non-frame parameters."""
    source = args.refine_from
    seal = json.loads((source / "prediction_seal.json").read_text())
    if any(file_sha256(source / name) != value for name, value in seal.items()):
        raise ValueError("Prior prediction seal mismatch")
    protocol = json.loads((source / "protocol.json").read_text())
    state_path = args.reference / "part_state_geometry_constrained.json"
    state = json.loads(state_path.read_text())
    if file_sha256(state_path) != protocol["part_state_sha256"] or state[
        "geometry_sha256"
    ] != file_sha256(args.reference / "reference_arrays.npz"):
        raise ValueError("Geometry and prior construction state mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    _, _, pieces, _ = load_prepared(args.reference)
    parts = state["parts"]
    validate_grounded_roles(pieces, parts)
    sections = {}
    for part in parts:
        if (
            part["surface_material"] == "wood"
            and part["functional_role"] == "frame_rail"
        ):
            spans = np.sort(
                pieces[part["component_id"]].bounding_box_oriented.primitive.extents
            )
            sections[part["component_id"]] = {
                "width_m": float(spans[0]),
                "depth_m": float(spans[1]),
                "area_m2": float(spans[0] * spans[1]),
            }
    if not sections and args.refine_representation == "beams":
        raise ValueError("Beam refinement requires observed frame-rail cross sections")
    paths = [Path(p) for p in protocol["input_hashes"]]
    if any(file_sha256(p) != protocol["input_hashes"][str(p)] for p in paths):
        raise ValueError("Original image evidence changed")
    protocol.update(
        {
            "source_sha256": file_sha256(Path(__file__)),
            "refined_from_seal_sha256": file_sha256(source / "prediction_seal.json"),
            "construction_model": args.refine_representation,
            "section_source": "visible frame-rail oriented bounding-box section proxy",
            "sections": sections,
            "additional_queries": protocol["rounds"],
            "total_queries_including_prior": 2 * protocol["rounds"],
            "non_frame_parameters": "foam fractions, underside hypothesis and numeric material priors frozen",
        }
    )
    write_json(args.output / "protocol.json", protocol)
    model = LocalVLM(args.model, max_tokens=1000, max_pixels=768**2, quantized=True)
    write_json(args.output / "model.json", model.provenance)
    images = [Image.open(p).copy() for p in paths]
    results = []
    for trial in range(protocol["rounds"]):
        prior = json.loads((source / f"hypothesis_{trial}.json").read_text())
        roles = {}
        for phase in prior["answer"]["upholstery"]:
            indices = [
                p["component_id"]
                for p in parts
                if p["functional_role"] == phase["role"]
            ]
            bounds = np.asarray([pieces[i].bounds for i in indices])
            roles[phase["role"]] = {
                "diagonal_m": float(
                    np.linalg.norm(bounds[:, 1].max(0) - bounds[:, 0].min(0))
                ),
                "core_volume_m3": sum(
                    p["core"]["volume_m3"]
                    for p in prior["partition"]["parts"]
                    if p["component_id"] in indices
                ),
            }
        prompt = (
            "Audit hidden wooden framing in this upholstered object using the same four original views "
            "and numbered mesh diagrams. Visible rails, legs and the bottom panel already have separate "
            "material domains: DO NOT count them again. Propose only additional hidden beams. "
            "Do not estimate object mass, change foam or bottom-panel assumptions, or force agreement "
            "with a previous answer. A hidden beam is a construction hypothesis, not visible evidence. "
            "Use an empty beams list if no additional beam is supported by your construction hypothesis. "
            "Observed rail sections (component ID -> dimensions): "
            + json.dumps(sections)
            + ". Available role geometry: "
            + json.dumps(roles)
            + '. Return JSON {"upholstery":[{"role":string,"beams":[{"section_component_id":integer,'
            '"count":positive_integer,"length_m":number}]}],"evidence":string}. '
            "Cover every listed role exactly once. Beam length cannot exceed the role diagonal. "
            "Consider construction alternatives; do not assume all padding contains wood."
        )
        if args.refine_representation != "beams":
            context = prompt.split(" Return JSON ")[0]
            if args.refine_representation == "fractions-no-geometry":
                context = (
                    context.split("Observed rail sections")[0]
                    + "Upholstery roles: "
                    + json.dumps(sorted(roles))
                    + "."
                )
            prompt = context + (
                ' Return JSON {"upholstery":[{"role":string,"hidden_wood_fraction":number}],'
                '"evidence":string}. Cover every role once. Estimate additional hidden wood as '
                "a fraction of its core volume using the supplied evidence. "
                "Do not change foam or bottom-panel assumptions."
            )
        response = model.generate(
            prompt,
            images,
            "Return construction JSON grounded in the images and metric geometry.",
            seed=3510 + trial,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        write_json(
            args.output / f"raw_{trial}.json",
            {"prompt": prompt, "response": asdict(response)},
        )
        proposal = _parse_response_content(response.text)
        if args.refine_representation == "beams":
            answer = apply_beam_inventory(prior["answer"], proposal, roles, sections)
        else:
            rows = proposal["upholstery"]
            if len(rows) != len(roles) or {r["role"] for r in rows} != set(roles):
                raise ValueError("Every role needs exactly one fraction proposal")
            answer = deepcopy(prior["answer"])
            for phase in answer["upholstery"]:
                phase["hidden_wood_fraction"] = next(
                    r["hidden_wood_fraction"]
                    for r in rows
                    if r["role"] == phase["role"]
                )
            answer = validate_hypothesis(answer, sorted(roles))
        partition = construct(pieces, parts, answer, protocol["fabric_thickness_m"])
        result = properties(partition, parts, answer, protocol["densities_kg_m3"])
        result.update(
            {
                "trial": trial,
                "answer": answer,
                "partition": partition,
                "role_geometry": roles,
            }
        )
        write_json(args.output / f"hypothesis_{trial}.json", result)
        results.append(result["mass_kg"])
        print(
            "REFINED_MASS",
            args.refine_representation,
            trial,
            result["mass_kg"],
            flush=True,
        )
    prediction = {
        "scene_id": protocol["scene"],
        "mass_kg": float(np.mean(results)),
        "hypothesis_masses_kg": results,
    }
    write_json(args.output / "predictions.json", [prediction])
    seal_prediction_files(args.output)
    return evaluate_saved_prediction(args.output, args.root)


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
        if material not in ("wood", "fabric", "metal", "plastic", "rubber"):
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
        if part["surface_material"] != "fabric":
            material = part["surface_material"]
            rho = edited_density if index == edited_component else densities[material]
            weighted.append((domain["assigned"], rho))
            allocations.append(
                {
                    "component_id": index,
                    f"{material}_volume_m3": domain["assigned"]["volume_m3"],
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


def density_edit_check(partition, parts, answer, densities, before):
    state = {p["component_id"]: p for p in parts}
    solids = [
        p
        for p in partition["parts"]
        if p["material"] != "fabric" and p["assigned"]["volume_m3"] > 0
    ]
    if not solids:
        return {"edit_status": "not applicable: no independent solid domain"}
    chosen = min(
        solids,
        key=lambda p: (
            state[p["component_id"]]["functional_role"] != "leg",
            p["material"] != "wood",
            p["component_id"],
        ),
    )
    index = chosen["component_id"]
    edited = properties(partition, parts, answer, densities, index, 2700.0)
    delta = chosen["assigned"]["volume_m3"] * (2700 - densities[chosen["material"]])
    if not np.isclose(
        edited["mass_kg"] - before["mass_kg"], delta, rtol=1e-7, atol=1e-12
    ):
        raise ValueError("Density edit violates fixed-domain mass conservation")
    return {
        "edit_status": "passed",
        "edited_component_id": index,
        "edited_mass_kg": edited["mass_kg"],
        "edit_mass_delta_kg": delta,
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=getattr(args, "resume", False))
    scene = json.loads((args.reference / "input_manifest.json").read_text())["scene"]
    _mesh, _labels, pieces, _colors = load_prepared(args.reference)
    state_path = args.reference / "part_state_geometry_constrained.json"
    state = json.loads(state_path.read_text())
    if state["geometry_sha256"] != file_sha256(args.reference / "reference_arrays.npz"):
        raise ValueError("Part IDs are bound to different reference geometry")
    parts = state["parts"]
    validate_grounded_roles(pieces, parts)
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
        for kind in sorted(
            {"wood", "fabric", "foam"} | {p["surface_material"] for p in parts}
        )
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
    if len(features) > 30:
        grouped = {}
        for part in parts:
            grouped.setdefault(part["evidence_group_id"], []).append(part)
        features = [
            {
                "evidence_group_id": group_id,
                "native_components": len(rows),
                "material": rows[0]["surface_material"],
                "role": rows[0]["functional_role"],
                "extent_m": (
                    np.max([pieces[p["component_id"]].bounds[1] for p in rows], axis=0)
                    - np.min(
                        [pieces[p["component_id"]].bounds[0] for p in rows], axis=0
                    )
                ).tolist(),
            }
            for group_id, rows in sorted(grouped.items())
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
    if getattr(args, "resume", False) and (args.output / "protocol.json").is_file():
        previous = json.loads((args.output / "protocol.json").read_text())
        if any(
            previous[key] != protocol[key]
            for key in (
                "part_state_sha256",
                "input_hashes",
                "densities_kg_m3",
                "fabric_thickness_m",
            )
        ):
            raise ValueError("Resume inputs differ from the frozen protocol")
        protocol["resumed_from_source_sha256"] = previous["source_sha256"]
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
        raw_path = args.output / f"raw_{trial}.json"
        if getattr(args, "resume", False) and raw_path.is_file():
            record = json.loads(raw_path.read_text())
            if record["prompt"] != prompt + "\n" + focus:
                raise ValueError("Cached query prompt differs from the frozen prompt")
            response_text = record["response"]["text"]
        else:
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
            write_json(raw_path, record)
            response_text = response.text
        answer = validate_hypothesis(_parse_response_content(response_text), roles)
        partition = construct(pieces, parts, answer, thickness)
        result = properties(partition, parts, answer, densities)
        edit = density_edit_check(partition, parts, answer, densities, result)
        result.update(
            {
                "trial": trial,
                "answer": answer,
                "partition": partition,
                **edit,
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
    seal_prediction_files(args.output)
    return evaluate_saved_prediction(args.output, args.root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reference", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=name in {"root", "output"})
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse raw responses only when prompts and input hashes match",
    )
    parser.add_argument("--refine-from", type=Path)
    parser.add_argument(
        "--refine-representation",
        choices=["beams", "fractions", "fractions-no-geometry"],
        default="fractions",
    )
    args = parser.parse_args()
    if args.evaluate_only:
        print(json.dumps(evaluate_saved_prediction(args.output, args.root)), flush=True)
        return
    if args.reference is None or args.model is None:
        parser.error("Inference requires --reference and --model")
    print(
        json.dumps(refine_construction(args) if args.refine_from else run(args)),
        flush=True,
    )


if __name__ == "__main__":
    main()
