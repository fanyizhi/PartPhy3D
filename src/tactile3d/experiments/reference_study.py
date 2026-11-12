"""Reference inputs, inference state and mesh-physics benchmarks."""

from __future__ import annotations

import hashlib
import json
import struct
from collections import Counter
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from PIL import Image

from tactile3d.figures.mesh_fields import render_mesh
from tactile3d.mesh.assembly import evidence_groups, texture_face_rgb
from tactile3d.mesh.task_lod import simplify_components, simplify_with_lineage
from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.part_io import write_json
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.progressive import make_prompt, validate_answer
from tactile3d.vision.vlm_interior import _parse_response_content

SCENE = "B07B4G5RBN_ATVPDKIKX0DER"

MATERIAL_COLORS = {
    "wood": "#bd8656",
    "fabric": "#558eae",
    "metal": "#a0a8b2",
    "plastic": "#b494be",
    "unknown": "#d1d4da",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def active_states(output):
    marker = output / "active_state.json"
    if marker.is_file():
        reference = read(marker)
        path = output / reference["path"]
        if digest(path) != reference["sha256"]:
            raise ValueError("Active state hash mismatch")
        return read(path)
    return read(output / "consensus_states.json")


def load_prepared(output):
    data = np.load(output / "reference_arrays.npz", allow_pickle=False)
    mesh = trimesh.Trimesh(data["vertices"], data["faces"], process=False)
    labels = data["component_ids"]
    pieces = [
        mesh.submesh([np.flatnonzero(labels == i)], append=True, repair=False)
        for i in range(int(labels.max()) + 1)
    ]
    return mesh, labels, pieces, data["face_rgb"]


def prepare(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    scene = getattr(args, "scene", None) or SCENE
    path = args.root / "meshes_reference" / scene / "model.glb"
    with path.open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12 or struct.unpack("<4sII", header) != (
        b"glTF",
        2,
        path.stat().st_size,
    ):
        raise ValueError("Reference GLB is incomplete or has an invalid header")
    mesh = trimesh.load_scene(path, process=False).to_geometry()
    colors = texture_face_rgb(mesh)
    # UV seams must not masquerade as physical disconnections. Preserve face
    # order and sampled texture separately before welding coincident vertices.
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh)
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]
    mesh.faces = mesh.faces[:, [0, 2, 1]]  # axis swap reverses handedness
    groups = trimesh.graph.connected_components(
        mesh.face_adjacency, nodes=np.arange(len(mesh.faces))
    )
    groups = sorted(
        groups, key=lambda g: (-len(g), tuple(mesh.triangles_center[g].mean(0)))
    )
    labels = np.empty(len(mesh.faces), dtype=int)
    observations = []
    if scene == SCENE and len(groups) != 15:
        raise ValueError("Frozen chaise reference expected 15 connected candidates")
    for index, group in enumerate(groups):
        labels[group] = index
        part = mesh.submesh([group], append=True, repair=False)
        observations.append(
            {
                "component_id": index,
                "faces": len(group),
                "area_m2": float(part.area),
                "bounds_m": part.bounds.tolist(),
                "watertight": bool(part.is_watertight),
                "mean_texture_rgb": colors[group].mean(0).tolist(),
                "volume_m3": float(part.volume) if part.is_volume else None,
            }
        )
    parents = (
        [[i] for i in range(len(groups))]
        if len(groups) <= 30
        else evidence_groups(
            [r["area_m2"] for r in observations],
            [r["mean_texture_rgb"] for r in observations],
        )
    )
    rows = (len(parents) + 4) // 5
    fig, axes = plt.subplots(
        rows, 5, figsize=(12, rows * 2.4), layout="constrained", squeeze=False
    )
    for index, (children, ax) in enumerate(zip(parents, axes.flat)):
        ids = np.flatnonzero(np.isin(labels, children))
        part = mesh.submesh([ids], append=True, repair=False)
        ax.imshow(render_mesh(part.vertices, part.faces, colors[ids], resolution=220))
        ax.set_title(f"Group {index} ({len(children)} native)", fontsize=10)
        ax.axis("off")
    for ax in axes.flat[len(parents) :]:
        ax.axis("off")
    fig.savefig(output / "component_sheet.png", dpi=110, facecolor="white")
    plt.close(fig)
    np.savez_compressed(
        output / "reference_arrays.npz",
        vertices=mesh.vertices,
        faces=mesh.faces,
        component_ids=labels,
        face_rgb=colors,
    )
    Image.fromarray(
        render_mesh(mesh.vertices, mesh.faces, colors, resolution=700)
    ).save(output / "reference_render.png")
    for view in (0, 10, 20):
        source = args.root / "scenes" / scene / "images" / f"{scene}_{view:02d}.png"
        with Image.open(source) as im:
            rgba = im.convert("RGBA")
            Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert(
                "RGB"
            ).save(output / f"observation_{view:02d}.png")
    write_json(
        output / "input_manifest.json",
        {
            "scene": scene,
            "scope": "reference_geometry_assisted_development",
            "original_split": (
                "test; previously exposed development object, not held-out"
                if scene == SCENE
                else "declared cohort; see selection manifest"
            ),
            "reference_sha256": digest(path),
            "reference_geometry_used_for_inference": True,
            "target_mass_used_for_inference": False,
            "views": [0, 10, 20],
            "geometry_unit": "glTF meter convention; not independently measured",
            "components": observations,
            "evidence_groups": parents,
            "face_budget_targets": [2400, 1200, 600],
            "methods": ["global_qem", "component_qem", "sensitivity_qem"],
            "vlm_trials": args.trials,
            "vlm_calls_per_arm_per_trial": 3,
            "input_files": {
                p.name: digest(p) for p in output.glob("observation_*.png")
            },
        },
    )
    print(f"Prepared {len(groups)} components for {scene}", flush=True)


def run_vlm(args):

    manifest = read(args.output / "input_manifest.json")
    trials = manifest["vlm_trials"]
    directory = args.output / "vlm"
    directory.mkdir(exist_ok=False)
    model = LocalVLM(args.model, max_tokens=2100, max_pixels=768 * 768, quantized=True)
    write_json(directory / "model.json", model.provenance)
    images = [
        Image.open(args.output / f"observation_{v:02d}.png").convert("RGB")
        for v in manifest["views"]
    ]
    images.append(Image.open(args.output / "component_sheet.png").convert("RGB"))
    records = []
    for trial in range(trials):
        for arm in ("repeated", "progressive"):
            previous = None
            for stage in range(3):
                seed = 401 + 10 * trial + stage
                stage_name = (
                    "baseline"
                    if arm == "repeated"
                    else ("overview", "geometry_check", "final")[stage]
                )
                prompt = make_prompt(
                    manifest["components"],
                    previous if arm == "progressive" else None,
                    stage_name,
                )
                response = model.generate(
                    prompt,
                    images,
                    "Analyze observed parts and physical construction. Return JSON only.",
                    seed=seed,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
                record = {
                    "arm": arm,
                    "trial": trial,
                    "stage": stage,
                    "seed": seed,
                    "prompt": prompt,
                    "response": asdict(response),
                    "valid": False,
                }
                try:
                    answer = validate_answer(
                        _parse_response_content(response.text),
                        len(manifest["components"]),
                    )
                    record["answer"] = answer
                    record["valid"] = True
                    previous = answer
                except (ValueError, KeyError, TypeError) as error:
                    record["error"] = str(error)
                write_json(directory / f"{arm}_{trial}_{stage}.json", record)
                records.append(record)
                print(
                    f"{arm} trial={trial} stage={stage} valid={record['valid']} "
                    f"seconds={response.seconds:.1f} peak_GiB={response.peak_gpu_bytes/2**30:.2f}",
                    flush=True,
                )
    write_json(directory / "records.json", records)
    finalize_vlm(args)


def finalize_vlm(args):
    """Aggregate sealed predictions on CPU, retaining property-source provenance."""
    directory = args.output / "vlm"
    manifest = read(args.output / "input_manifest.json")
    records = read(directory / "records.json")
    # Seal every answer before subsequent geometry/edit evaluation.
    write_json(
        directory / "seal.json",
        {"sha256": digest(directory / "records.json"), "mass_labels_read": False},
    )
    states = {}
    library = {
        p.name: p for p in load_material_priors(Path("configs/material_priors.yaml"))
    }
    for arm in ("repeated", "progressive"):
        valid = [
            r
            for r in records
            if r["arm"] == arm and r["valid"] and (arm == "repeated" or r["stage"] == 2)
        ]
        if not valid:
            raise ValueError(f"No valid final state for {arm}")
        assignments = []
        for component in range(len(manifest["components"])):
            candidates = [
                next(
                    c
                    for c in r["answer"]["components"]
                    if c["component_id"] == component
                )
                for r in valid
            ]
            counts = Counter(c["surface_material"] for c in candidates)
            winner, support = counts.most_common(1)[0]
            selected = next(c for c in candidates if c["surface_material"] == winner)
            construction_votes = Counter(
                c["construction"] for c in candidates if c["surface_material"] == winner
            )
            construction_order = construction_votes.most_common()
            construction = construction_order[0][0]
            if (
                len(construction_order) > 1
                and construction_order[0][1] == construction_order[1][1]
            ):
                construction = "unknown"
            assignments.append(
                {
                    **selected,
                    "construction": construction,
                    "construction_votes": dict(construction_votes),
                    "support": support,
                    "samples": len(candidates),
                    "material_votes": dict(counts),
                }
            )
        materials = {}
        for record in valid:
            for material in record["answer"]["materials"]:
                name = material.get("name", "unknown")
                if name not in MATERIAL_COLORS:
                    continue
                materials.setdefault(name, []).append(material)
        priors = {}
        for name, rows in materials.items():
            priors[name] = {"source": "VLM typical-material prior, not measured"}
            for key in ("density_kg_m3", "youngs_modulus_pa", "friction_coefficient"):
                values = [
                    r.get(key)
                    for r in rows
                    if type(r.get(key)) in (int, float)
                    and np.isfinite(r[key])
                    and r[key] > 0
                ]
                priors[name][key] = float(np.median(values)) if values else None
                if (
                    key in ("density_kg_m3", "youngs_modulus_pa")
                    and not values
                    and name in library
                ):
                    prior = getattr(library[name], key)
                    priors[name][key] = prior.mean
                    priors[name][
                        f"{key}_source"
                    ] = "existing material library nominal prior"
                    priors[name][f"{key}_range"] = [prior.min, prior.max]
                else:
                    priors[name][f"{key}_source"] = (
                        "VLM prior" if values else "unresolved"
                    )
            priors[name]["hardness_records"] = [
                {"value": r.get("hardness_value"), "scale": r.get("hardness_scale")}
                for r in rows
                if r.get("hardness_value") is not None
            ]
        states[arm] = {
            "components": assignments,
            "materials": priors,
            "valid_outputs_for_aggregation": len(valid),
        }
    write_json(args.output / "consensus_states.json", states)
    write_json(
        args.output / "parameter_binding.json",
        {
            "material_library_sha256": digest(Path("configs/material_priors.yaml")),
            "policy": "keep valid VLM priors; bind missing density/modulus to existing library; no invented contact/hardness values",
            "states": states,
        },
    )


def closed_properties(pieces, ids, density):
    mass, first, inertia_origin = 0.0, np.zeros(3), np.zeros((3, 3))
    for i in ids:
        part = pieces[i]
        if not part.is_volume:
            return None
        props = part.mass_properties
        value = float(props.mass * density)
        center = np.asarray(props.center_mass)
        mass += value
        first += value * center
        inertia_origin += props.inertia * density + value * (
            np.dot(center, center) * np.eye(3) - np.outer(center, center)
        )
    if mass <= 0:
        return None
    center = first / mass
    inertia = inertia_origin - mass * (
        np.dot(center, center) * np.eye(3) - np.outer(center, center)
    )
    return {
        "mass_kg": mass,
        "center_of_mass_m": center.tolist(),
        "inertia_kg_m2": inertia.tolist(),
    }


def benchmark(args):
    output = args.output
    write_json(
        output / "benchmark_environment.json",
        {
            name: version(name)
            for name in ("numpy", "scipy", "trimesh", "fast-simplification", "rtree")
        },
    )
    mesh, labels, pieces, _colors = load_prepared(output)
    state = active_states(output)["progressive"]
    names = {c["component_id"]: c["surface_material"] for c in state["components"]}
    wood = [i for i, p in enumerate(pieces) if names[i] == "wood" and p.is_volume]
    if not wood:
        raise ValueError("No VLM-assigned closed wood region available")
    rho = state["materials"].get("wood", {}).get("density_kg_m3")
    if rho is None:
        raise ValueError("No wood density prior was produced")
    reference = closed_properties(pieces, wood, rho)
    bulk_density = np.array([rho if i in wood else np.nan for i in range(len(pieces))])
    directory = output / "lod"
    directory.mkdir(exist_ok=False)
    surface_points, reference_sample_faces = trimesh.sample.sample_surface(
        mesh, 12000, seed=711
    )
    rows = []
    # Warm up the QEM extension before timed runs.
    pieces[0].simplify_quadric_decimation(face_count=100)
    targets = read(output / "input_manifest.json")["face_budget_targets"]
    for target in targets:
        for method in ("global_qem", "component_qem", "sensitivity_qem"):
            timings = []
            for repetition in range(3):
                start = perf_counter()
                if method == "global_qem":
                    result = simplify_with_lineage(mesh, labels, target)
                    reduced, mapped = result.mesh, result.face_components
                    budgets = None
                else:
                    result = simplify_components(
                        pieces,
                        target,
                        bulk_density if method == "sensitivity_qem" else None,
                    )
                    reduced, mapped, budgets = (
                        result.mesh,
                        result.face_components,
                        result.budgets,
                    )
                timings.append(perf_counter() - start)
            split = [
                (
                    reduced.submesh(
                        [np.flatnonzero(mapped == i)], append=True, repair=False
                    )
                    if np.any(mapped == i)
                    else trimesh.Trimesh()
                )
                for i in range(len(pieces))
            ]
            props = closed_properties(split, wood, rho)
            sample_start = perf_counter()
            # Exact point-to-triangle distances if trimesh's spatial index is available.
            _, distances, hit_faces = trimesh.proximity.closest_point(
                reduced, surface_points
            )
            reverse_points, _ = trimesh.sample.sample_surface(reduced, 12000, seed=712)
            _, reverse_distances, _ = trimesh.proximity.closest_point(
                mesh, reverse_points
            )
            cosine = np.einsum(
                "ij,ij->i",
                mesh.face_normals[reference_sample_faces],
                reduced.face_normals[hit_faces],
            )
            normal_error = np.rad2deg(np.arccos(np.clip(cosine, -1, 1)))
            errors = None
            if props:
                errors = {
                    "mass_relative": abs(props["mass_kg"] / reference["mass_kg"] - 1),
                    "com_error_mm": 1000
                    * float(
                        np.linalg.norm(
                            np.array(props["center_of_mass_m"])
                            - reference["center_of_mass_m"]
                        )
                    ),
                    "inertia_relative": float(
                        np.linalg.norm(
                            np.array(props["inertia_kg_m2"])
                            - reference["inertia_kg_m2"]
                        )
                        / np.linalg.norm(reference["inertia_kg_m2"])
                    ),
                }
            row = {
                "method": method,
                "target_faces": target,
                "actual_faces": len(reduced.faces),
                "seconds_samples": timings,
                "median_seconds": float(np.median(timings)),
                "component_budgets": budgets,
                "surface_mean_mm": float(distances.mean() * 1000),
                "surface_p95_mm": float(np.quantile(distances, 0.95) * 1000),
                "reverse_surface_mean_mm": float(reverse_distances.mean() * 1000),
                "reverse_surface_p95_mm": float(
                    np.quantile(reverse_distances, 0.95) * 1000
                ),
                "symmetric_mean_distance_mm": float(
                    (distances.mean() + reverse_distances.mean()) * 500
                ),
                "mean_normal_error_degrees": float(normal_error.mean()),
                "closed_wood_regions": int(sum(split[i].is_volume for i in wood)),
                "wood_region_count": len(wood),
                "wood_physics": props,
                "wood_errors": errors,
                "distance_evaluation_seconds": perf_counter() - sample_start,
                "global_mapping": (
                    "verified edge-collapse ancestry"
                    if method == "global_qem"
                    else "preserved component identity"
                ),
                "physical_scope": "conditional VLM wood density on reference closed regions",
            }
            rows.append(row)
            np.savez_compressed(
                directory / f"{method}_{target}.npz",
                vertices=reduced.vertices,
                faces=reduced.faces,
                component_ids=mapped,
            )
            write_json(directory / "metrics.json", rows)
            print(
                f"{method}: {len(reduced.faces)} faces, surface P95={row['surface_p95_mm']:.3f} mm",
                flush=True,
            )
    # Independent per-component full recomputation vs one-component cached edit.
    chosen = wood[0]
    before = reference
    after_density = 2700.0  # explicit user-style aluminum edit, not a model prediction
    old_part = closed_properties(pieces, [chosen], rho)
    new_part = closed_properties(pieces, [chosen], after_density)

    def origin(p):
        c = np.array(p["center_of_mass_m"])
        return np.array(p["inertia_kg_m2"]) + p["mass_kg"] * (
            np.dot(c, c) * np.eye(3) - np.outer(c, c)
        )

    mass = before["mass_kg"] - old_part["mass_kg"] + new_part["mass_kg"]
    first = (
        before["mass_kg"] * np.array(before["center_of_mass_m"])
        - old_part["mass_kg"] * np.array(old_part["center_of_mass_m"])
        + new_part["mass_kg"] * np.array(new_part["center_of_mass_m"])
    )
    center = first / mass
    inertia = (
        origin(before)
        - origin(old_part)
        + origin(new_part)
        - mass * (np.dot(center, center) * np.eye(3) - np.outer(center, center))
    )
    # Full recomputation from every component's geometry and edited density.
    full_mass, full_first, full_origin = 0.0, np.zeros(3), np.zeros((3, 3))
    for i in wood:
        p = closed_properties(pieces, [i], after_density if i == chosen else rho)
        full_mass += p["mass_kg"]
        full_first += p["mass_kg"] * np.array(p["center_of_mass_m"])
        full_origin += origin(p)
    full_center = full_first / full_mass
    full_inertia = full_origin - full_mass * (
        np.dot(full_center, full_center) * np.eye(3)
        - np.outer(full_center, full_center)
    )
    write_json(
        output / "edit.json",
        {
            "scope": "four closed wooden legs only; open rails and upholstery excluded",
            "component_id": chosen,
            "wood_components": wood,
            "before_density_kg_m3": rho,
            "edited_density_kg_m3": after_density,
            "edit_material": "aluminum, user-specified nominal prior",
            "before": before,
            "after": {
                "mass_kg": mass,
                "center_of_mass_m": center.tolist(),
                "inertia_kg_m2": inertia.tolist(),
            },
            "full_recompute_residual": {
                "mass_kg": abs(mass - full_mass),
                "center_m": float(np.linalg.norm(center - full_center)),
                "inertia_kg_m2": float(np.linalg.norm(inertia - full_inertia)),
            },
        },
    )
    write_json(
        output / "part_state.json",
        {
            "schema": "object/part/construction_region/closed_mesh_domain",
            "version": 1,
            "reference_array_sha256": digest(output / "reference_arrays.npz"),
            "components": [
                {
                    **c,
                    "part_id": f"part_{c['component_id']:03d}",
                    "instance_version": 1,
                    "geometry_support": f"reference_arrays.component_ids=={c['component_id']}",
                    "physical_domain": (
                        "closed domain with conditional homogeneous density"
                        if c["component_id"] in wood
                        else "surface only; internal support unresolved"
                    ),
                    "evidence_source": "original RGB + artist reference mesh + progressive VLM",
                }
                for c in state["components"]
            ],
        },
    )
