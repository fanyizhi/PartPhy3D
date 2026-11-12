"""Shared material queries and calibrated-view evidence preparation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from tactile3d.experiments.reference_study import (
    SCENE,
    load_prepared,
)
from tactile3d.figures.mesh_fields import project_points as project
from tactile3d.figures.mesh_fields import render_mesh
from tactile3d.physics.part_io import write_json
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.view_selection import select_azimuth_views
from tactile3d.vision.vlm_interior import _parse_response_content


def refine_local_binding(args):
    """Audit one target at a time, preserving native geometry and prior states."""
    from tactile3d.experiments.vlm_mass import leg_role_conflicts

    out = args.output
    mesh, labels, pieces, texture = load_prepared(out)
    mode = getattr(args, "binding_evidence", "global-context")
    if mode not in {"global-context", "isolated"}:
        raise ValueError("Unknown binding evidence mode")
    path = (
        getattr(args, "part_state", None)
        or out / "part_state_geometry_constrained.json"
    )
    if path.resolve().parent != out.resolve():
        raise ValueError("Binding state must share its reference directory")
    state = json.loads(path.read_text())
    if (
        state["geometry_sha256"]
        != hashlib.sha256((out / "reference_arrays.npz").read_bytes()).hexdigest()
    ):
        raise ValueError("Binding state geometry mismatch")
    conflict_ids = set(leg_role_conflicts(pieces, state["parts"]))
    manifest = json.loads((out / "input_manifest.json").read_text())
    groups = manifest.get("evidence_groups", [[i] for i in range(len(pieces))])
    priorities = []
    for index, children in enumerate(groups):
        rows = [p for p in state["parts"] if p["component_id"] in children]
        conflict = bool(conflict_ids.intersection(children))
        uncertain = any(
            p.get("support", 0) < 3 or p["surface_material"] in {"mixed", "unknown"}
            for p in rows
        )
        if conflict or uncertain:
            priorities.append(
                (not conflict, -sum(pieces[i].area for i in children), index)
            )
    selected = [row[2] for row in sorted(priorities)[:3]]
    directory = out / (
        "local_binding_isolated_audit" if mode == "isolated" else "local_binding_audit"
    )
    directory.mkdir(exist_ok=False)
    views = json.loads((out / "views_contacts/view_selection.json").read_text())["4"]
    paths = [out / f"views_contacts/view_{v['frame_index']:02d}.png" for v in views]
    originals = [Image.open(p).copy() for p in paths]
    protocol = {
        "selected_groups": selected,
        "evidence_mode": mode,
        "images_per_query": 1 if mode == "isolated" else 6,
        "total_queries": 3 * len(selected),
        "selection": "geometric leg conflicts first, then uncertain groups by area; max 3 groups",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prior_state_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "mass_labels_used": False,
        "trials_per_group": 3,
        "image_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }
    write_json(directory / "protocol.json", protocol)
    records = []
    model = (
        LocalVLM(args.model, max_tokens=450, max_pixels=768**2, quantized=True)
        if selected
        else None
    )
    if model:
        write_json(directory / "model.json", model.provenance)
    for group in selected:
        children = groups[group]
        target_faces = np.flatnonzero(np.isin(labels, children))
        target = mesh.submesh([target_faces], append=True, repair=False)
        representative = max(children, key=lambda i: pieces[i].area)
        # A small repeated-detail group gets a magnified representative, not a
        # convex hull bridging its disconnected instances.
        show = pieces[representative] if len(children) > 1 else target
        colors = (
            texture[labels == representative]
            if len(children) > 1
            else texture[target_faces]
        )
        isolated = Image.fromarray(
            render_mesh(show.vertices, show.faces, colors, resolution=700)
        )
        highlight = np.full((len(mesh.faces), 3), 0.78)
        highlight[target_faces] = [0.14, 0.57, 0.78]
        context = Image.fromarray(
            render_mesh(mesh.vertices, mesh.faces, highlight, resolution=700)
        )
        isolated.save(directory / f"group_{group}_isolated.png")
        context.save(directory / f"group_{group}_context.png")
        size = target.extents
        ground = bool(
            abs(target.bounds[0, 2] - mesh.bounds[0, 2])
            <= np.linalg.norm(mesh.extents) * 1e-3
        )
        prompt = (
            "Inspect ONE target, with four original views, its isolated textured rendering, and "
            "a context rendering where ONLY the target is blue and the rest is gray. "
            "The blue highlight is artificial and is NOT its material color. Do not map any numeric IDs. "
            f"Target extent xyz in meters: {size.tolist()}; touches common ground: {ground}; "
            f"native instances: {len(children)}. Isolated image shows {'one representative' if len(children)>1 else 'the complete target'}. "
            "Infer visible OUTERMOST surface, independently of hidden structure or functional role. "
            "Fabric-covered wood is surface fabric. Closed geometry does not imply solid wood. "
            "A whole body spanning the chair width is not a single leg. "
            "Return JSON {surface_material:wood|fabric|metal|plastic|rubber|mixed|unknown, "
            "functional_role:leg|frame_rail|frame|underside|seat|backrest|cushion|armrest|upholstery|fastener|mixed|unknown, "
            "evidence:string}. If ambiguous use unknown. Do not estimate mass or density."
        )
        query_images = originals + [isolated, context]
        if mode == "isolated":
            prompt = (
                "Inspect ONLY the physical component rendered in the attached image on white background. "
                "It is an isolated mesh component, not a complete object. No other components are shown. "
                "Classify its visible outermost surface and its likely component role. "
                "Do not include materials belonging to absent components or imagine the whole object. "
                "A painted or finished surface can still be wood, metal or plastic; finish alone does not mean mixed. "
                "Use mixed only if distinct material regions are visible ON THIS isolated target. "
                f"Its full target extent xyz in meters is {size.tolist()}; native instances {len(children)}. "
                f"The image shows {'one representative' if len(children)>1 else 'the complete target'}. "
                "Return JSON {surface_material:wood|fabric|metal|plastic|rubber|mixed|unknown, "
                "functional_role:leg|frame_rail|frame|underside|seat|backrest|cushion|armrest|upholstery|fastener|mixed|unknown, "
                "evidence:string}. Use unknown if the rendering is insufficient. Do not estimate mass or density."
            )
            query_images = [isolated]
        query_paths = (
            [directory / f"group_{group}_isolated.png"]
            if mode == "isolated"
            else paths
            + [
                directory / f"group_{group}_isolated.png",
                directory / f"group_{group}_context.png",
            ]
        )
        votes = []
        for trial in range(3):
            response = model.generate(
                prompt,
                query_images,
                "Return one target's material and role JSON.",
                seed=5710 + trial,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
            )
            record = {
                "group": group,
                "trial": trial,
                "prompt": prompt,
                "input_image_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in query_paths
                },
                "response": asdict(response),
            }
            write_json(directory / f"raw_{group}_{trial}.json", record)
            answer = _parse_response_content(response.text)
            material, role = answer["surface_material"], answer["functional_role"]
            if material not in {
                "wood",
                "fabric",
                "metal",
                "plastic",
                "rubber",
                "mixed",
                "unknown",
            } or role not in {
                "leg",
                "frame_rail",
                "frame",
                "underside",
                "seat",
                "backrest",
                "cushion",
                "armrest",
                "upholstery",
                "fastener",
                "mixed",
                "unknown",
            }:
                raise ValueError("Invalid local-binding vocabulary")
            votes.append((material, role))
            records.append(record)
        pair, support = Counter(votes).most_common(1)[0]
        if support < 2:
            pair = ("unknown", "unknown")
        for part in state["parts"]:
            if part["component_id"] in children:
                part.update(
                    {
                        "surface_material": pair[0],
                        "functional_role": pair[1],
                        "support": support,
                        "samples": 3,
                        "binding_source": (
                            "isolated target only plus metric extent"
                            if mode == "isolated"
                            else "isolated target plus geometry and highlighted context"
                        ),
                    }
                )
        print("LOCAL_BINDING", group, pair, support, flush=True)
    state["prior_state_sha256"] = protocol["prior_state_sha256"]
    state["local_binding_protocol"] = protocol
    state_name = (
        "part_state_isolated_audit.json"
        if mode == "isolated"
        else "part_state_local_audit.json"
    )
    write_json(out / state_name, state)
    write_json(directory / state_name, state)
    write_json(
        directory / "seal.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.iterdir()
            if p.is_file()
        },
    )


def infer_native_parts(args):
    """Bind native candidates to joint material/role votes without review labels."""
    out = args.output
    manifest = json.loads((out / "input_manifest.json").read_text())
    groups = manifest.get(
        "evidence_groups", [[i] for i in range(len(manifest["components"]))]
    )
    count = len(groups)
    if sorted(i for group in groups for i in group) != list(
        range(len(manifest["components"]))
    ):
        raise ValueError(
            "Evidence groups must partition native component IDs exactly once"
        )
    if count > 30:
        raise ValueError(
            f"{count} native candidates exceed the declared 30-ID evidence budget; hierarchical grouping required"
        )
    audit = getattr(args, "surface_audit", False)
    directory = out / ("native_binding_surface_audit" if audit else "native_binding")
    directory.mkdir(exist_ok=False)
    views = json.loads((out / "views_contacts/view_selection.json").read_text())["4"]
    paths = [out / f"views_contacts/view_{v['frame_index']:02d}.png" for v in views]
    paths += [
        out / "component_sheet.png",
        out / "views_contacts/global_component_positions.png",
    ]
    images = [Image.open(p).copy() for p in paths]
    prompt = (
        "Bind numbered native mesh components to surface material and functional role. "
        "Four original views and two numbered diagrams show the same object. "
        "Native connectivity is NOT guaranteed to be a material or manufactured-part partition. "
        "If a component contains multiple surface materials, use mixed, not a forced majority material. "
        "Materials: wood,fabric,metal,plastic,rubber,mixed,unknown. "
        "Roles: leg,frame_rail,frame,underside,seat,backrest,cushion,armrest,upholstery,fastener,mixed,unknown. "
        "Numbered groups may contain repeated small native pieces; assign one role only if coherent. "
        f"Return JSON {{components:[{{component_id:integer,surface_material:string,functional_role:string}}]}} "
        f"covering IDs 0 through {count-1} exactly once. Do not estimate mass or hidden properties."
    )
    if audit:
        prompt += (
            " Audit visible OUTERMOST material independently of structural role. "
            "A fabric-covered wooden frame has FABRIC on its surface, not wood. "
            "A closed mesh does not imply solid wood. Examine the texture and seams; "
            "use upholstery for a connected fabric cover spanning multiple functional parts. "
            "Classify repeated small trim from its own appearance, not the large neighboring cushion. "
            "If visual evidence cannot resolve a material, use unknown rather than infer it from the role."
        )
    model = LocalVLM(args.model, max_tokens=2200, max_pixels=768**2, quantized=True)
    write_json(directory / "model.json", model.provenance)
    records = []
    for trial in range(3):
        response = model.generate(
            prompt,
            images,
            "Return component binding JSON.",
            seed=4610 + trial,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        record = {"prompt": prompt, "response": asdict(response)}
        write_json(directory / f"raw_{trial}.json", record)
        rows = _parse_response_content(response.text)["components"]
        if len(rows) != count or {r["component_id"] for r in rows} != set(range(count)):
            raise ValueError("Component binding must cover every native candidate once")
        if any(
            r["surface_material"]
            not in {"wood", "fabric", "metal", "plastic", "rubber", "mixed", "unknown"}
            or r["functional_role"]
            not in {
                "leg",
                "frame_rail",
                "underside",
                "seat",
                "backrest",
                "cushion",
                "armrest",
                "frame",
                "upholstery",
                "fastener",
                "mixed",
                "unknown",
            }
            for r in rows
        ):
            raise ValueError("Unsupported component-binding vocabulary")
        records.append(rows)
    parts = []
    for index in range(count):
        votes = Counter(
            (r["surface_material"], r["functional_role"])
            for rows in records
            for r in rows
            if r["component_id"] == index
        )
        pair, support = votes.most_common(1)[0]
        if support < 2:
            pair = ("unknown", "unknown")
        for child in groups[index]:
            parts.append(
                {
                    "component_id": child,
                    "evidence_group_id": index,
                    "surface_material": pair[0],
                    "functional_role": pair[1],
                    "support": support,
                    "samples": 3,
                }
            )
    write_json(
        out
        / (
            "part_state_surface_audit.json"
            if audit
            else "part_state_geometry_constrained.json"
        ),
        {
            "schema": "object/part/construction_region/mesh_support",
            "version": 2,
            "geometry_sha256": hashlib.sha256(
                (out / "reference_arrays.npz").read_bytes()
            ).hexdigest(),
            "parts": parts,
            "scope": "joint VLM votes on native mesh candidates; no manual-role transfer",
            "input_manifest_sha256": hashlib.sha256(
                (out / "input_manifest.json").read_bytes()
            ).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "evidence_hashes": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
            },
        },
    )
    write_json(
        directory / "seal.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.glob("*.json")
        },
    )
    print(
        "NATIVE_BINDING",
        manifest["scene"],
        "groups",
        count,
        "native",
        len(parts),
        flush=True,
    )


def query_properties(args):
    out = args.output
    states = json.loads((out / "consensus_states.json").read_text())
    names = sorted(
        {c["surface_material"] for c in states["progressive"]["components"]}
        - {"unknown"}
    )
    directory = out / "property_queries"
    directory.mkdir(exist_ok=False)
    model = LocalVLM(args.model, max_tokens=1100, max_pixels=384**2, quantized=True)
    write_json(directory / "model.json", model.provenance)
    image = Image.open(out / "observation_00.png").convert("RGB")
    prompt = (
        "Estimate typical material-property PRIORS for these already identified surface "
        f"materials: {json.dumps(names)}. These are conditional nominal estimates, not "
        "measurements and not whole-upholstery effective properties. Give reasonable "
        "numeric central values and broad ranges for density and Young's modulus when "
        "a material-level value is meaningful. For wood use longitudinal modulus, unknown "
        "species. For textile use an explicitly assumed effective in-plane tensile modulus. "
        "Define friction as static friction for a clean DRY stainless-steel slider against "
        "this material under light contact at room temperature. State that exact surface "
        "finish is unresolved. Define hardness using Shore D only when meaningful; "
        "textile has no useful bulk Shore D hardness and must be null. Do not copy null "
        "schema examples for all other properties. Return JSON {materials:[{name:string, "
        "density_kg_m3:number, density_range:[low,high], youngs_modulus_pa:number or null, "
        "modulus_direction:string, friction_coefficient:number or null, "
        "friction_contact_pair:'dry stainless steel slider / material', "
        "hardness_value:number or null, hardness_scale:'Shore D' or null, "
        "assumptions:string}]}. Do not fabricate a literature citation."
    )
    records = []
    for index in range(3):
        response = model.generate(
            prompt,
            [image],
            "Return conditional material priors as JSON.",
            seed=891 + index,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        record = {"prompt": prompt, "response": asdict(response), "valid": False}
        try:
            answer = _parse_response_content(response.text)
            rows = answer["materials"]
            if not isinstance(rows, list) or {r["name"] for r in rows} != set(names):
                raise ValueError("Property material names do not match fixed state")
            record["answer"] = answer
            record["valid"] = True
        except (KeyError, ValueError, TypeError) as error:
            record["error"] = str(error)
        records.append(record)
        write_json(directory / f"query_{index}.json", record)
        print(
            f"Property query {index}: valid={record['valid']}, {response.seconds:.1f}s",
            flush=True,
        )
    for arm in states:
        for name in names:
            rows = [
                r
                for record in records
                if record["valid"]
                for r in record["answer"]["materials"]
                if r["name"] == name
            ]
            bound = states[arm]["materials"].setdefault(name, {})
            for key in ("density_kg_m3", "youngs_modulus_pa", "friction_coefficient"):
                values = [
                    r[key]
                    for r in rows
                    if type(r.get(key)) in (int, float)
                    and np.isfinite(r[key])
                    and r[key] > 0
                    and (
                        key != "friction_coefficient"
                        or isinstance(r.get("friction_contact_pair"), str)
                    )
                ]
                if values:
                    bound[key] = float(np.median(values))
                    bound[f"{key}_source"] = (
                        "separate shared VLM conditional material-prior query"
                    )
            bound["hardness_records"] = [
                {"value": r["hardness_value"], "scale": "Shore D"}
                for r in rows
                if name != "fabric"
                and type(r.get("hardness_value")) in (int, float)
                and 0 <= r["hardness_value"] <= 100
                and str(r.get("hardness_scale", "")).lower() == "shore d"
            ]
            bound["friction_contact_pair"] = (
                "clean dry stainless-steel slider / material, light contact, room temperature; finish unresolved"
            )
            bound["assumptions"] = [r.get("assumptions") for r in rows]
    write_json(directory / "records.json", records)
    write_json(
        directory / "seal.json",
        {
            "sha256": hashlib.sha256(
                (directory / "records.json").read_bytes()
            ).hexdigest(),
            "shared_between_structural_arms": True,
            "separate_from_18_call_structure_comparison": True,
        },
    )
    state_path = out / "consensus_states_with_properties.json"
    write_json(state_path, states)
    write_json(
        out / "active_state.json",
        {
            "path": state_path.name,
            "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        },
    )


def prepare_views(args):
    scene = json.loads((args.output / "input_manifest.json").read_text())["scene"]
    out = args.output / "views_contacts"
    out.mkdir(exist_ok=False)
    mesh, _labels, pieces, texture = load_prepared(args.output)
    manifest = json.loads((args.output / "input_manifest.json").read_text())
    groups = manifest.get("evidence_groups", [[i] for i in range(len(pieces))])
    pieces = [
        mesh.submesh(
            [np.flatnonzero(np.isin(_labels, children))], append=True, repair=False
        )
        for children in groups
    ]
    centers = np.array(
        [np.average(p.triangles_center, weights=p.area_faces, axis=0) for p in pieces]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), layout="constrained")
    for ax, az in zip(axes, [55, -125]):
        ax.imshow(
            render_mesh(mesh.vertices, mesh.faces, texture, azimuth=az, resolution=600)
        )
        uv = project(mesh.vertices, centers, az)
        for side in (0, 1):
            ids = np.flatnonzero((uv[:, 0] >= 300) == bool(side))
            ids = ids[np.argsort(uv[ids, 1])]
            for i, y in zip(ids, np.linspace(70, 530, len(ids))):
                ax.annotate(
                    str(i),
                    xy=uv[i],
                    xytext=(20 if side == 0 else 580, y),
                    fontsize=12,
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "circle,pad=0.2", "fc": "white", "ec": "#426980"},
                    arrowprops={"arrowstyle": "-", "color": "#426980", "lw": 0.8},
                )
        ax.axis("off")
        ax.set_title(f"Reference component positions · azimuth {az}°")
    fig.savefig(out / "global_component_positions.png", dpi=110, facecolor="white")
    plt.close(fig)
    camera = json.loads((args.root / "scenes" / scene / "transforms.json").read_text())
    poses = np.array([f["transform_matrix"] for f in camera["frames"]])
    selections = {}
    for count in (3, 4):
        selection = select_azimuth_views(poses, count)
        selections[str(count)] = selection
        fig, axes = plt.subplots(
            1, count, figsize=(3 * count, 3.2), layout="constrained"
        )
        for row, ax in zip(selection, axes):
            frame = camera["frames"][row["frame_index"]]
            image = Image.open(
                args.root / "scenes" / scene / frame["file_path"]
            ).convert("RGBA")
            rgb = Image.alpha_composite(
                Image.new("RGBA", image.size, "white"), image
            ).convert("RGB")
            rgb.save(out / f"view_{row['frame_index']:02d}.png")
            ax.imshow(rgb)
            ax.set_title(
                f"View {row['frame_index']:02d} · az {row['actual_azimuth_degrees']:.0f}°"
            )
            ax.axis("off")
        fig.savefig(out / f"selected_{count}_views.png", dpi=140)
        plt.close(fig)
    write_json(out / "view_selection.json", selections)
    if scene != SCENE:
        return  # Historical hand-reviewed chaise roles are not labels for other assets.
    write_json(
        out / "role_review.json",
        {
            "scope": "single-reviewer reference-component functional roles",
            "roles": {
                str(i): role
                for i, role in enumerate(
                    [
                        "cushion",
                        "seat",
                        "backrest",
                        "armrest",
                        "armrest",
                        "backrest",
                        "leg",
                        "leg",
                        "leg",
                        "leg",
                        "frame_rail",
                        "frame_rail",
                        "frame_rail",
                        "frame_rail",
                        "underside",
                    ]
                )
            },
            "material_review": "configs/evidence/chaise_surface_material_review.json",
            "interior_ground_truth": False,
        },
    )


def infer_views(args):
    from tactile3d.vision.local_vlm import LocalVLM
    from tactile3d.vision.vlm_interior import _parse_response_content

    out = args.output / "views_contacts"
    selections = json.loads((out / "view_selection.json").read_text())
    roles = json.loads((out / "role_review.json").read_text())["roles"]
    material_review = json.loads(
        Path("configs/evidence/chaise_surface_material_review.json").read_text()
    )["component_materials"]
    model = LocalVLM(args.model, max_tokens=1400, max_pixels=768**2, quantized=True)
    write_json(out / "view_model.json", model.provenance)
    prompt = (
        "Identify numbered components of this chaise. The first images are actual calibrated "
        "observations. The final two images are an isolated numbered-component sheet and a "
        "whole-reference-mesh position diagram; numbers in these diagrams denote the SAME "
        "components. Use spatial position and shape to distinguish the loose back cushion "
        "from the long horizontal seat. Do not name all fabric panels headrests. "
        "Classify all 15 IDs exactly once. Functional role vocabulary is "
        "cushion, seat, backrest, armrest, leg, frame_rail, underside, unknown. "
        "Material vocabulary is wood, fabric, metal, plastic, unknown. "
        "Return strict JSON {components:[{component_id:integer,functional_role:string,"
        "surface_material:string,evidence:string}]}. Describe visible evidence briefly."
    )
    records = []
    for trial in range(3):
        for count in (3, 4):
            paths = [
                out / f"view_{row['frame_index']:02d}.png"
                for row in selections[str(count)]
            ]
            paths += [
                args.output / "component_sheet.png",
                out / "global_component_positions.png",
            ]
            response = model.generate(
                prompt,
                [Image.open(p).convert("RGB") for p in paths],
                "Assign component roles from observed geometry and context; output JSON.",
                seed=1201 + trial,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            record = {
                "trial": trial,
                "view_count": count,
                "prompt": prompt,
                "images": [p.name for p in paths],
                "response": asdict(response),
                "valid": False,
            }
            try:
                answer = _parse_response_content(response.text)
                rows = answer["components"]
                if sorted(c["component_id"] for c in rows) != list(range(15)):
                    raise ValueError("Missing or duplicated component IDs")
                prediction = {str(c["component_id"]): c for c in rows}
                record["answer"] = answer
                record["valid"] = True
                record["role_accuracy"] = sum(
                    prediction[i]["functional_role"] == role
                    for i, role in roles.items()
                ) / len(roles)
                known = {i: m for i, m in material_review.items() if m != "unknown"}
                record["material_accuracy"] = sum(
                    prediction[i]["surface_material"] == mat for i, mat in known.items()
                ) / len(known)
                record["wrong_role_ids"] = [
                    i
                    for i, role in roles.items()
                    if prediction[i]["functional_role"] != role
                ]
            except (KeyError, ValueError, TypeError) as error:
                record["error"] = str(error)
            write_json(out / f"view_count_{count}_trial_{trial}.json", record)
            records.append(record)
            print(
                f"{count} views trial {trial}: valid={record['valid']}, role={record.get('role_accuracy')}, {response.seconds:.1f}s",
                flush=True,
            )
    summary = []
    for count in (3, 4):
        selected = [r for r in records if r["view_count"] == count]
        summary.append(
            {
                "views": count,
                "trials": 3,
                "valid": sum(r["valid"] for r in selected),
                "role_accuracy": float(
                    np.mean([r.get("role_accuracy", 0) for r in selected])
                ),
                "material_accuracy": float(
                    np.mean([r.get("material_accuracy", 0) for r in selected])
                ),
                "mean_seconds": float(
                    np.mean([r["response"]["seconds"] for r in selected])
                ),
            }
        )
    winner = min(
        summary,
        key=lambda r: (
            -r["role_accuracy"],
            -r["material_accuracy"],
            -r["valid"],
            r["mean_seconds"],
        ),
    )
    write_json(
        out / "view_count_results.json",
        {
            "scope": "development role/material comparison with shared context diagrams",
            "summary": summary,
            "selected_views": winner["views"],
            "selection_rule": "role accuracy, material accuracy, validity, then runtime",
            "records": records,
        },
    )


def infer_color(args):
    out = args.output / "views_contacts"
    directory = out / "color_control"
    directory.mkdir(exist_ok=False)
    selected = json.loads((out / "view_count_results.json").read_text())[
        "selected_views"
    ]
    views = json.loads((out / "view_selection.json").read_text())[str(selected)]
    manifest = json.loads((args.output / "input_manifest.json").read_text())
    # Only sampled RGB is added here; no geometry dimensions or evaluation labels.
    table = [
        {"component_id": r["component_id"], "mean_texture_rgb": r["mean_texture_rgb"]}
        for r in manifest["components"]
    ]
    prompt = (
        "Assign all 15 numbered regions. Per-component sampled RGB below is visual "
        "evidence from the reference texture and helps bind the correct isolated component "
        "to its number. It is not a material label. Use images and color jointly. "
        "Return JSON {components:[{component_id:integer,functional_role:string,surface_material:string}]}. "
        "Role vocabulary: cushion, seat, backrest, armrest, leg, frame_rail, underside, unknown. "
        "Material vocabulary: wood, fabric, metal, plastic, unknown. Use each ID exactly once. "
        "Visual color observations: " + json.dumps(table)
    )
    paths = [out / f"view_{r['frame_index']:02d}.png" for r in views]
    paths += [
        args.output / "component_sheet.png",
        out / "global_component_positions.png",
    ]
    images = [Image.open(p).convert("RGB") for p in paths]
    model = LocalVLM(args.model, max_tokens=1400, max_pixels=768**2, quantized=True)
    write_json(directory / "model.json", model.provenance)
    records = []
    for trial in range(3):
        response = model.generate(
            prompt,
            images,
            "Bind component identity to its own visual evidence; JSON only.",
            seed=1201 + trial,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        record = {
            "trial": trial,
            "view_count": selected,
            "prompt": prompt,
            "images": [p.name for p in paths],
            "response": asdict(response),
            "valid": False,
        }
        try:
            answer = _parse_response_content(response.text)
            if sorted(r["component_id"] for r in answer["components"]) != list(
                range(15)
            ):
                raise ValueError("Incomplete component identities")
            record.update(answer=answer, valid=True)
        except (KeyError, ValueError, TypeError) as error:
            record["error"] = str(error)
        records.append(record)
        write_json(directory / f"trial_{trial}.json", record)
        print(
            f"Color evidence trial {trial}: valid={record['valid']}, {response.seconds:.1f}s",
            flush=True,
        )
    write_json(
        directory / "answers.json",
        {
            "selected_views": selected,
            "records": records,
            "scope": "additional visual-evidence control; geometry-only descriptors excluded from VLM prompt",
        },
    )
