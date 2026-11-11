"""Shared material queries and calibrated-view evidence preparation."""

from __future__ import annotations

import hashlib
import json
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
    out = args.output / "views_contacts"
    out.mkdir(exist_ok=False)
    mesh, _labels, pieces, texture = load_prepared(args.output)
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
    camera = json.loads((args.root / "scenes" / SCENE / "transforms.json").read_text())
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
                args.root / "scenes" / SCENE / frame["file_path"]
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
