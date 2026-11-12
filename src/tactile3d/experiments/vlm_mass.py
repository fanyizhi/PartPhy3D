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
    eroded_closed,
    eroded_convex,
    partition_domains,
    solid_from_mesh,
)
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.view_selection import select_azimuth_views
from tactile3d.vision.vlm_interior import _parse_response_content


def leg_role_conflicts(pieces, parts):
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
    return conflicts


def validate_grounded_roles(pieces, parts):
    unresolved = [
        p["component_id"]
        for p in parts
        if p.get("surface_material") in {"mixed", "unknown"}
    ]
    if unresolved:
        raise ValueError(
            f"Unresolved surface material on {len(unresolved)} native components (first IDs {unresolved[:8]}); subdivide or rebind before physical inference"
        )
    conflicts = leg_role_conflicts(pieces, parts)
    if conflicts:
        raise ValueError(
            f"Leg role conflicts with z-up ground support and extent for native IDs {conflicts}; rebind components before physical inference"
        )


def resolve_native_priors(info, cfg, required):
    """Strict cached-name binding, with explicit existing-library fallbacks."""
    from tactile3d.experiments.ten_objects import materials_from_info
    from tactile3d.physics.materials import load_material_priors

    priors, _ = materials_from_info(info, cfg, strict=True)
    library_path = Path("configs/material_priors.yaml")
    library = {p.name: p for p in load_material_priors(library_path)}
    densities, sources = {}, {}
    for kind in sorted(required):
        candidates = [p for p in priors if p["kind"] == kind]
        if candidates:
            chosen = candidates[0]
            densities[kind] = float(np.mean(chosen["density"]))
            sources[kind] = chosen
        elif kind in library:
            densities[kind] = library[kind].density_kg_m3.mean
            sources[kind] = {
                "source": "existing_declared_material_library_mean",
                "path": str(library_path),
                "sha256": file_sha256(library_path),
            }
        else:
            raise ValueError(
                f"No explicit density prior for {kind}; refine the material hypothesis"
            )
    thickness = float(
        np.mean(next(p["thickness_m"] for p in priors if p["kind"] == "fabric"))
    )
    return densities, thickness, sources


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


def apply_fraction_map(base, proposal):
    roles = sorted(row["role"] for row in base["upholstery"])
    mapping = {f"R{i}": role for i, role in enumerate(roles)}
    values = proposal["wood_fraction_by_id"]
    if not isinstance(values, dict) or set(values) != set(mapping):
        raise ValueError("Fraction map must cover every fixed role ID exactly once")
    if any(type(value) not in (int, float) for value in values.values()):
        raise ValueError("Fraction map values must be numeric fractions")
    answer = deepcopy(base)
    inverse = {role: key for key, role in mapping.items()}
    for phase in answer["upholstery"]:
        phase["hidden_wood_fraction"] = values[inverse[phase["role"]]]
    return validate_hypothesis(answer, roles)


def refine_construction(args):
    """Paired second-round construction queries; freeze all non-frame parameters."""
    source = args.refine_from
    seal = json.loads((source / "prediction_seal.json").read_text())
    if any(file_sha256(source / name) != value for name, value in seal.items()):
        raise ValueError("Prior prediction seal mismatch")
    protocol = json.loads((source / "protocol.json").read_text())
    state_path = args.reference / protocol.get(
        "part_state_name", "part_state_geometry_constrained.json"
    )
    state = json.loads(state_path.read_text())
    if file_sha256(state_path) != protocol["part_state_sha256"] or state[
        "geometry_sha256"
    ] != file_sha256(args.reference / "reference_arrays.npz"):
        raise ValueError("Geometry and prior construction state mismatch")
    args.output.mkdir(parents=True, exist_ok=getattr(args, "resume", False))
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
            "fraction_role_schema": getattr(args, "role_schema", "map"),
            "fraction_bounds_exposed": getattr(args, "expose_fraction_bounds", False),
            "section_source": "visible frame-rail oriented bounding-box section proxy",
            "sections": sections,
            "additional_queries": protocol["rounds"],
            "total_queries_including_prior": 2 * protocol["rounds"],
            "non_frame_parameters": "foam fractions, underside hypothesis and numeric material priors frozen",
        }
    )
    # Refinement changes core densities, not geometry. The source prediction
    # seal fixes its per-trial domains, even if the builder is later updated.
    # A new geometry policy requires rebuilding the prior explicitly.
    protocol["partition_source"] = "sealed prior moments; geometry unchanged"
    if getattr(args, "resume", False) and (args.output / "protocol.json").is_file():
        previous = json.loads((args.output / "protocol.json").read_text())
        keys = (
            "part_state_sha256",
            "input_hashes",
            "refined_from_seal_sha256",
            "construction_model",
            "fraction_role_schema",
            "fraction_bounds_exposed",
        )
        if any(previous.get(key) != protocol.get(key) for key in keys):
            raise ValueError("Refinement resume protocol differs")
        protocol["resumed_from_source_sha256"] = previous["source_sha256"]
    write_json(args.output / "protocol.json", protocol)
    model = None
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
        if (
            args.refine_representation != "beams"
            and getattr(args, "role_schema", "map") == "map"
        ):
            foam_by_role = {
                r["role"]: r["foam_fraction"] for r in prior["answer"]["upholstery"]
            }
            mapping = {
                f"R{i}": {
                    "role": role,
                    **(
                        {"max_wood_fraction": 1 - foam_by_role[role]}
                        if getattr(args, "expose_fraction_bounds", False)
                        else {}
                    ),
                    **(
                        roles[role] if args.refine_representation == "fractions" else {}
                    ),
                }
                for i, role in enumerate(sorted(roles))
            }
            prompt = (
                "Infer only ADDITIONAL hidden wood in this upholstered object from the four original views "
                "and two diagrams. Visible supports already have separate domains; do not count them twice. "
                "Keep existing foam and bottom-panel assumptions fixed. Remaining volume can be macroscopic air. "
                "Geometric core volume is NOT wood volume. Unused core may be macroscopic air. "
                "No mass or density estimates. Required region IDs and evidence: "
                + json.dumps(mapping)
                + '. Return JSON {"wood_fraction_by_id":'
                + json.dumps({key: "number" for key in mapping})
                + ',"evidence":"string"}. Replace number placeholders with plausible finite volume fractions. '
                "Do not fill the remaining core with wood merely to reach a sum of one. "
                "Include EVERY region ID, even a zero-volume or broad upholstery role. "
                "Do not infer extra wood merely because a component is upholstered."
            )
        raw_path = args.output / f"raw_{trial}.json"
        if getattr(args, "resume", False) and raw_path.is_file():
            record = json.loads(raw_path.read_text())
            if record["prompt"] != prompt:
                raise ValueError("Cached refinement prompt differs")
            response_text = record["response"]["text"]
        else:
            if model is None:
                model = LocalVLM(
                    args.model, max_tokens=1000, max_pixels=768**2, quantized=True
                )
                write_json(args.output / "model.json", model.provenance)
            response = model.generate(
                prompt,
                images,
                "Return construction JSON grounded in the images and metric geometry.",
                seed=3510 + trial,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            write_json(raw_path, {"prompt": prompt, "response": asdict(response)})
            response_text = response.text
        proposal = _parse_response_content(response_text)
        if args.refine_representation == "beams":
            answer = apply_beam_inventory(prior["answer"], proposal, roles, sections)
        elif getattr(args, "role_schema", "map") == "map":
            answer = apply_fraction_map(prior["answer"], proposal)
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
        partition = prior["partition"]
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


def construct(
    pieces, parts, answer, thickness, *, allow_removed=False, geometry_policy="convex"
):
    role_phases = {r["role"]: r for r in answer["upholstery"]}
    if geometry_policy not in {"convex", "closed"}:
        raise ValueError("Unknown geometry policy")
    bounds = [p.bounds for p in pieces if len(p.faces)]
    if not bounds:
        raise ValueError("No surviving assembly geometry")
    assembly_center = np.mean(bounds, axis=(0, 1))
    domains, excluded = [], []
    for part in parts:
        index = part["component_id"]
        mesh = pieces[index]
        orientation_repaired = False
        if (
            geometry_policy == "closed"
            and mesh.is_watertight
            and mesh.is_winding_consistent
            and mesh.volume < 0
        ):
            mesh = mesh.copy()
            mesh.invert()
            orientation_repaired = True
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
            if geometry_policy == "closed" and mesh.is_volume:
                hull, source = (
                    mesh,
                    "provided_closed_boundary; exact convex erosion or inscribed_ball16_erosion",
                )
                inner = eroded_closed(hull, thickness)
            else:
                hull, source = convex_envelope(mesh, thickness, assembly_center)
                inner = eroded_convex(hull, thickness)
            domain = SupportDomain(index, material, hull, inner, 2, source)
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
                index,
                material,
                hull,
                solid_from_mesh(hull),
                1 if planar else 0,
                source,
                full_core=True,
            )
        if orientation_repaired:
            domain.source += (
                "; reversed globally inward winding; vertex positions unchanged"
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
    state_path = (
        getattr(args, "part_state", None)
        or args.reference / "part_state_geometry_constrained.json"
    )
    if state_path.resolve().parent != args.reference.resolve():
        raise ValueError("Part state must be inside its bound reference directory")
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

    cfg = yaml.safe_load(Path("configs/ten_object_pilot.yaml").read_text())
    densities, thickness, density_sources = resolve_native_priors(
        materials,
        cfg,
        {"wood", "fabric", "foam"} | {p["surface_material"] for p in parts},
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
        "part_state_name": state_path.name,
        "geometry_policy": getattr(args, "geometry_policy", "convex"),
        "source_sha256": file_sha256(Path(__file__)),
        "geometry_module_sha256": file_sha256(
            Path(__file__).parents[1] / "physics/support_domains.py"
        ),
        "input_hashes": {str(p): file_sha256(p) for p in paths},
        "selected_views": views,
        "densities_kg_m3": densities,
        "density_prior_sources": density_sources,
        "material_name_policy": "strict; unknown names never become plastic",
        "fabric_thickness_m": thickness,
        "components": features,
        "target_mass_used_for_inference": False,
        "rounds": 3,
        "geometry": (
            "provided closed solids and closed upholstery; inscribed-ball16 erosion for concave upholstery; open-component convex/prism completion; disjoint CSG"
            if getattr(args, "geometry_policy", "convex") == "closed"
            else "provided closed wood solids; per-component convex/prism completion elsewhere; disjoint CSG"
        ),
        "aggregation": "equal mean of three independent-question construction hypotheses; not calibrated probabilities",
    }
    if getattr(args, "resume", False) and (args.output / "protocol.json").is_file():
        previous = json.loads((args.output / "protocol.json").read_text())
        if previous.get("geometry_policy", "convex") != protocol["geometry_policy"]:
            raise ValueError("Resume geometry policy differs from the frozen protocol")
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
        partition = construct(
            pieces,
            parts,
            answer,
            thickness,
            geometry_policy=protocol["geometry_policy"],
        )
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


def audit_binding_parameters(args):
    """Two-by-two diagnostic: binding versus closed-boundary preservation."""
    import yaml

    args.output.mkdir(parents=True, exist_ok=False)
    _, _, pieces, _ = load_prepared(args.reference)
    base = args.reference / "part_state_geometry_constrained.json"
    if args.part_state.resolve().parent != args.reference.resolve():
        raise ValueError("Audited state must share its reference directory")
    protocol = json.loads((args.reference / "construction/protocol.json").read_text())
    cfg = yaml.safe_load(Path("configs/ten_object_pilot.yaml").read_text())
    foam = float(cfg["construction"]["soft_core_occupancy"][1])
    wood = float(cfg["construction"]["hidden_wood_fraction"][1])
    original = json.loads(
        (args.reference / "construction/hypothesis_0.json").read_text()
    )["answer"]
    output = {
        "scope": "fixed-parameter development diagnostic; not a VLM-method ranking",
        "foam_fraction": foam,
        "hidden_wood_fraction": wood,
        "densities_kg_m3": protocol["densities_kg_m3"],
        "fabric_thickness_m": protocol["fabric_thickness_m"],
        "source_sha256": file_sha256(Path(__file__)),
        "prior_protocol_sha256": file_sha256(
            args.reference / "construction/protocol.json"
        ),
        "prior_construction_sha256": file_sha256(
            args.reference / "construction/hypothesis_0.json"
        ),
        "geometry_source_sha256": file_sha256(
            Path(__file__).parents[1] / "physics/support_domains.py"
        ),
        "arms": [],
    }
    for binding, path in [("original", base), ("local", args.part_state)]:
        state = json.loads(path.read_text())
        if state["geometry_sha256"] != file_sha256(
            args.reference / "reference_arrays.npz"
        ):
            raise ValueError("State geometry mismatch")
        conflicts = leg_role_conflicts(pieces, state["parts"])
        if binding == "local" and conflicts:
            raise ValueError("Local binding still contains geometric leg conflicts")
        roles = sorted(
            {
                p["functional_role"]
                for p in state["parts"]
                if p["surface_material"] == "fabric"
            }
        )
        answer = {
            "upholstery": [
                {"role": r, "foam_fraction": foam, "hidden_wood_fraction": wood}
                for r in roles
            ],
            "underside": original["underside"],
        }
        validate_hypothesis(answer, roles)
        for policy in ["convex", "closed"]:
            partition = construct(
                pieces,
                state["parts"],
                answer,
                protocol["fabric_thickness_m"],
                geometry_policy=policy,
            )
            result = properties(
                partition, state["parts"], answer, protocol["densities_kg_m3"]
            )
            row = {
                "binding": binding,
                "geometry_policy": policy,
                "state_sha256": file_sha256(path),
                "geometric_role_conflicts": conflicts,
                "answer": answer,
                "properties": result,
                "union_volume_m3": partition["union_volume_m3"],
            }
            output["arms"].append(row)
            write_json(args.output / "predictions.json", output)
            print(
                "BINDING_GEOMETRY_AUDIT", binding, policy, result["mass_kg"], flush=True
            )
    seal_prediction_files(args.output)
    scene = protocol["scene"]
    weights = json.loads((args.root / "filtered_product_weights.json").read_text())
    target = float(weights.get(scene, weights.get(scene.split("_")[0])))
    report = {
        "target_mass_kg": target,
        "arms": [
            {
                "binding": r["binding"],
                "geometry_policy": r["geometry_policy"],
                "mass_kg": r["properties"]["mass_kg"],
                "metrics": mass_metrics([target], [r["properties"]["mass_kg"]]),
            }
            for r in output["arms"]
        ],
    }
    write_json(args.output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "reference", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=name in {"root", "output"})
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--part-state", type=Path)
    parser.add_argument(
        "--geometry-policy", choices=["convex", "closed"], default="convex"
    )
    parser.add_argument("--audit-binding", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse raw responses only when prompts and input hashes match",
    )
    parser.add_argument("--refine-from", type=Path)
    parser.add_argument("--role-schema", choices=["map", "list"], default="map")
    parser.add_argument(
        "--expose-fraction-bounds",
        action="store_true",
        help="Diagnostic prompt ablation; physical bounds are always checked externally",
    )
    parser.add_argument(
        "--refine-representation",
        choices=["beams", "fractions", "fractions-no-geometry"],
        default="fractions",
    )
    args = parser.parse_args()
    if args.evaluate_only:
        print(json.dumps(evaluate_saved_prediction(args.output, args.root)), flush=True)
        return
    if args.audit_binding:
        if args.reference is None or args.part_state is None:
            parser.error("Binding audit requires --reference and --part-state")
        print(json.dumps(audit_binding_parameters(args)), flush=True)
        return
    if args.reference is None or args.model is None:
        parser.error("Inference requires --reference and --model")
    print(
        json.dumps(refine_construction(args) if args.refine_from else run(args)),
        flush=True,
    )


if __name__ == "__main__":
    main()
