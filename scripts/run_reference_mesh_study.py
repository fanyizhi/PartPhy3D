"""Reference-geometry development study: bounded VLM, task QEM and physical edits.

The artist mesh is an oracle geometry input. Property maps show model priors.
Closed wooden components support conditional physical-integral comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import LogNorm, Normalize
from matplotlib.patches import Patch
from PIL import Image
from scipy.spatial import cKDTree

from tactile3d.figures.mesh_fields import render_mesh
from tactile3d.mesh.assembly import texture_face_rgb
from tactile3d.mesh.task_lod import simplify_components
from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.part_io import write_json
from tactile3d.vision.progressive import make_prompt, validate_answer

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


def savefig(fig, path):
    for suffix in ("png", "svg", "pdf", "tiff"):
        fig.savefig(
            path.with_suffix("." + suffix),
            dpi=600 if suffix == "tiff" else 250,
            facecolor="white",
            bbox_inches="tight",
            **({"pil_kwargs": {"compression": "tiff_lzw"}} if suffix == "tiff" else {}),
        )
    plt.close(fig)


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
    path = args.root / "meshes_reference" / SCENE / "model.glb"
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
    fig, axes = plt.subplots(3, 5, figsize=(12, 7.2), layout="constrained")
    if len(groups) != 15:
        raise ValueError("Frozen chaise reference expected 15 connected candidates")
    for index, (group, ax) in enumerate(zip(groups, axes.flat)):
        labels[group] = index
        part = mesh.submesh([group], append=True, repair=False)
        observations.append(
            {
                "component_id": index,
                "faces": len(group),
                "bounds_m": part.bounds.tolist(),
                "watertight": bool(part.is_watertight),
                "mean_texture_rgb": colors[group].mean(0).tolist(),
                "volume_m3": float(part.volume) if part.is_volume else None,
            }
        )
        ax.imshow(render_mesh(part.vertices, part.faces, colors[group], resolution=220))
        ax.set_title(f"Component {index}", fontsize=12)
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
        source = args.root / "scenes" / SCENE / "images" / f"{SCENE}_{view:02d}.png"
        with Image.open(source) as im:
            rgba = im.convert("RGBA")
            Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert(
                "RGB"
            ).save(output / f"observation_{view:02d}.png")
    write_json(
        output / "input_manifest.json",
        {
            "scene": SCENE,
            "scope": "reference_geometry_assisted_development",
            "original_split": "test; previously exposed development object, not held-out",
            "reference_sha256": digest(path),
            "reference_geometry_used_for_inference": True,
            "target_mass_used_for_inference": False,
            "views": [0, 10, 20],
            "geometry_unit": "glTF meter convention; not independently measured",
            "components": observations,
            "face_budget_targets": [2400, 1200, 600],
            "methods": ["global_qem", "component_qem", "sensitivity_qem"],
            "vlm_trials": args.trials,
            "vlm_calls_per_arm_per_trial": 3,
            "input_files": {
                p.name: digest(p) for p in output.glob("observation_*.png")
            },
        },
    )
    print("Prepared 15 components and fixed study protocol", flush=True)


def run_vlm(args):
    from tactile3d.vision.local_vlm import LocalVLM
    from tactile3d.vision.vlm_interior import _parse_response_content

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
    tree = cKDTree(mesh.triangles_center)
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
                    reduced = mesh.simplify_quadric_decimation(face_count=target)
                    # Explicit approximate mapping, included in measured cost.
                    _, nearest = tree.query(reduced.triangles_center)
                    mapped = labels[nearest]
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
                    "nearest original triangle centroid"
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
            print(json.dumps(row), flush=True)
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
            "scope": "closed wooden frame only, upholstery excluded",
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


def figures(args):
    from matplotlib.colors import to_rgb

    out = args.output
    dest = args.figures
    dest.mkdir(parents=True, exist_ok=False)
    mesh, labels, _pieces, _texture = load_prepared(out)
    states = active_states(out)
    state = states["progressive"]
    lookup = {c["component_id"]: c["surface_material"] for c in state["components"]}
    material = np.array([to_rgb(MATERIAL_COLORS[lookup[int(i)]]) for i in labels])
    unknown = np.array(to_rgb(MATERIAL_COLORS["unknown"]))
    rendered = render_mesh(mesh.vertices, mesh.faces, material, resolution=600)
    fields = {}
    for key in ("density_kg_m3", "youngs_modulus_pa", "friction_coefficient"):
        fields[key] = np.array(
            [
                state["materials"].get(lookup[int(i)], {}).get(key) or np.nan
                for i in labels
            ]
        )
    # Hardness is comparable only on an explicitly shared scale.
    hardness_records = [
        (name, r)
        for name, values in state["materials"].items()
        for r in values.get("hardness_records", [])
        if type(r.get("value")) in (int, float)
        and np.isfinite(r["value"])
        and r["value"] >= 0
        and isinstance(r.get("scale"), str)
        and r["scale"].strip()
    ]
    scales = Counter(r["scale"].strip().lower() for _, r in hardness_records)
    hardness_scale = scales.most_common(1)[0][0] if scales else None
    hardness = {}
    for name in state["materials"]:
        values = [
            r["value"]
            for mat, r in hardness_records
            if mat == name and r["scale"].strip().lower() == hardness_scale
        ]
        hardness[name] = float(np.median(values)) if values else np.nan
    fields["hardness"] = np.array(
        [hardness.get(lookup[int(i)], np.nan) for i in labels]
    )
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.2), layout="constrained")
    axes[0, 0].imshow(Image.open(out / "observation_00.png"))
    axes[0, 0].set_title("a  Original observation")
    axes[0, 1].imshow(rendered)
    axes[0, 1].set_title("b  Material grounding")
    axes[0, 1].legend(
        handles=[
            Patch(color=color, label=name)
            for name, color in MATERIAL_COLORS.items()
            if name in lookup.values()
        ],
        loc="lower center",
        ncol=3,
        fontsize=7,
    )
    for ax, key, title, unit, factor in zip(
        (axes[0, 2], *axes[1]),
        fields,
        (
            "c  Surface density prior",
            "d  Material modulus prior",
            "e  Friction prior",
            "f  Hardness prior",
        ),
        ("kg m$^{-3}$", "GPa", "dimensionless", hardness_scale or "unresolved scale"),
        (1, 1e-9, 1, 1),
    ):
        values = fields[key] * factor
        finite = np.isfinite(values)
        rgb = np.tile(unknown, (len(mesh.faces), 1))
        if finite.any():
            lo, hi = values[finite].min(), values[finite].max()
            if hi <= lo:
                lo, hi = max(lo * 0.8, 1e-9), hi * 1.2
            norm = LogNorm(lo, hi) if key == "youngs_modulus_pa" else Normalize(lo, hi)
            rgb[finite] = plt.get_cmap("viridis")(norm(values[finite]))[:, :3]
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap="viridis"),
                ax=ax,
                shrink=0.65,
                label=unit,
                pad=0.01,
            )
        ax.imshow(render_mesh(mesh.vertices, mesh.faces, rgb, resolution=600))
        ax.set_title(title)
        if not finite.any():
            ax.text(
                0.5,
                0.02,
                (
                    "Unresolved contact pair"
                    if key == "friction_coefficient"
                    else "No supported hardness scale"
                ),
                transform=ax.transAxes,
                ha="center",
                fontsize=8,
            )
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(
        "Reference geometry · VLM material labels + explicit property priors",
        fontsize=9,
    )
    savefig(fig, dest / "dense_property_maps")
    rows = read(out / "lod" / "metrics.json")
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.2), layout="constrained")
    for ax, row in zip(axes.flat, rows):
        data = np.load(out / "lod" / f"{row['method']}_{row['target_faces']}.npz")
        cols = np.array(
            [to_rgb(MATERIAL_COLORS[lookup[int(i)]]) for i in data["component_ids"]]
        )
        ax.imshow(render_mesh(data["vertices"], data["faces"], cols, resolution=500))
        ax.set_title(
            f"{row['method']} · {row['actual_faces']:,} faces\n"
            f"surface P95 {row['surface_p95_mm']:.2f} mm · wood closed {row['closed_wood_regions']}/{row['wood_region_count']}",
            fontsize=8,
        )
        ax.axis("off")
    savefig(fig, dest / "qem_mesh_comparison")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), layout="constrained")
    for method, color in zip(
        ("global_qem", "component_qem", "sensitivity_qem"),
        ("#8e969e", "#588cb2", "#bd805c"),
    ):
        subset = [r for r in rows if r["method"] == method]
        x = [r["actual_faces"] for r in subset]
        axes[0].plot(
            x, [r["surface_p95_mm"] for r in subset], "o-", label=method, color=color
        )
        axes[1].plot(
            x,
            [
                (
                    100 * r["wood_errors"]["inertia_relative"]
                    if r["wood_errors"]
                    else np.nan
                )
                for r in subset
            ],
            "o-",
            color=color,
        )
        axes[2].plot(x, [1000 * r["median_seconds"] for r in subset], "o-", color=color)
    for ax, title, ylabel in zip(
        axes,
        ("Surface fidelity", "Closed wood-frame fidelity", "QEM + mapping cost"),
        ("P95 distance (mm)", "Inertia tensor error (%)", "Median time (ms)"),
    ):
        ax.set(title=title, xlabel="Actual triangle count", ylabel=ylabel)
    axes[0].legend(fontsize=7)
    savefig(fig, dest / "qem_accuracy_cost")
    edit = read(out / "edit.json")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5), layout="constrained")
    axes = axes.ravel()
    edited = material.copy()
    edited[labels == edit["component_id"]] = to_rgb("#c0c6ce")
    axes[0].imshow(rendered)
    axes[0].set_title("Before · wooden frame")
    axes[0].axis("off")
    axes[1].imshow(render_mesh(mesh.vertices, mesh.faces, edited, resolution=600))
    axes[1].set_title(f"Edit component {edit['component_id']} to aluminum")
    axes[1].axis("off")
    axes[2].bar(
        ["Before", "After"],
        [edit[k]["mass_kg"] for k in ("before", "after")],
        color=["#bd8656", "#a0a8b2"],
    )
    axes[2].set(title="Wood-frame domain mass", ylabel="Mass (kg)")
    moments = [
        np.linalg.eigvalsh(edit[k]["inertia_kg_m2"]) for k in ("before", "after")
    ]
    for i, (name, vals) in enumerate(zip(("Before", "After"), moments)):
        axes[3].bar(np.arange(3) + (i - 0.5) * 0.35, vals, 0.35, label=name)
    axes[3].set(
        title="Frame principal moments",
        ylabel="kg m$^2$",
        xticks=range(3),
        xticklabels=["I1", "I2", "I3"],
    )
    axes[3].legend(fontsize=7)
    centers = (
        np.array([edit[k]["center_of_mass_m"] for k in ("before", "after")]) * 1000
    )
    for i, name in enumerate(("Before", "After")):
        axes[4].bar(np.arange(3) + (i - 0.5) * 0.35, centers[i], 0.35, label=name)
    axes[4].set(
        title="Frame center of mass",
        ylabel="Coordinate (mm)",
        xticks=range(3),
        xticklabels=["x", "y", "z"],
    )
    displacement = centers[1] - centers[0]
    axes[5].bar(range(3), displacement, color="#bd805c")
    axes[5].set(
        title="Center-of-mass displacement",
        ylabel="After − before (mm)",
        xticks=range(3),
        xticklabels=["x", "y", "z"],
    )
    savefig(fig, dest / "local_material_edit")
    np.savez_compressed(dest / "surface_fields.npz", component_ids=labels, **fields)
    write_json(
        dest / "source_data.json",
        {
            "input": read(out / "input_manifest.json"),
            "consensus": states,
            "qem": rows,
            "edit": edit,
            "vlm_seal": read(out / "vlm" / "seal.json"),
            "geometry_arrays_sha256": digest(out / "reference_arrays.npz"),
        },
    )
    (dest / "CAPTIONS.md").write_text(
        "# Chaise reference-geometry study\n\n"
        "Artist reference geometry is an additional input, not reconstructed geometry. "
        "Component assignments and surface property fields are VLM predictions/priors. "
        "Density maps describe surface material and do not assert homogeneous upholstery. "
        "Missing VLM density/modulus values bind to the existing configured material library; "
        "per-parameter provenance is recorded in source_data.json. "
        "Unknown properties remain gray. Hardness is not pooled across incompatible scales.\n\n"
        "QEM comparison uses common requested budgets and reports actual face counts. "
        "Distances are one-way original-surface samples to reduced triangles. "
        "Physical errors concern only VLM-assigned closed wooden regions with fixed nominal density. "
        "Invalid domains have no physical-error value. Times include reduction and component mapping, "
        "exclude VLM, and summarize three warmed runs.\n\n"
        "Local edit changes one closed wooden component to specified aluminum density. "
        "Mass and inertia refer to the frame domain, excluding upholstery and hidden supports.\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "vlm", "finalize", "benchmark", "figures", "all"]
    )
    parser.add_argument("--root", type=Path, default=Path("data/abo500_five"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--figures", type=Path, default=Path("paper_assets/chaise_reference_20261111")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"),
    )
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    for name, function in (
        ("prepare", prepare),
        ("vlm", run_vlm),
        ("benchmark", benchmark),
        ("figures", figures),
    ):
        if args.stage in (name, "all"):
            function(args)
    if args.stage == "finalize":
        finalize_vlm(args)


if __name__ == "__main__":
    main()
