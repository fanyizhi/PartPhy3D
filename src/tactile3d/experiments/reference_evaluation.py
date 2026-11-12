"""Frozen-output evaluation, contact geometry and matched-budget controls."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh
from scipy.spatial import ConvexHull

from tactile3d.experiments.reference_study import (
    active_states,
    closed_properties,
    load_prepared,
)
from tactile3d.figures.reference_study import savefig
from tactile3d.mesh.contacts import closest_surface_pair
from tactile3d.mesh.task_lod import simplify_components, simplify_with_lineage
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.role_constraints import furniture_role_candidates


def evaluate_support_domains(args):
    """Replay fixed VLM constructions over matched QEM and structural ablations.

    Physical errors measure retention of a conditional domain model. They are
    separate from absolute mass accuracy and do not read product mass labels.
    """
    from tactile3d.experiments.vlm_mass import construct, properties

    source = args.construction
    seal = json.loads((source / "prediction_seal.json").read_text())
    if any(file_sha256(source / name) != value for name, value in seal.items()):
        raise ValueError("Construction prediction seal mismatch")
    protocol = json.loads((source / "protocol.json").read_text())
    state_path = args.output / protocol.get(
        "part_state_name", "part_state_geometry_constrained.json"
    )
    state = json.loads(state_path.read_text())
    if file_sha256(state_path) != protocol["part_state_sha256"] or state[
        "geometry_sha256"
    ] != file_sha256(args.output / "reference_arrays.npz"):
        raise ValueError("Construction state and reference geometry do not match")
    output = source / "domain_evaluation"
    output.mkdir(exist_ok=False)
    write_json(
        output / "protocol.json",
        {
            "source_sha256": file_sha256(Path(__file__)),
            "construction_seal_sha256": file_sha256(source / "prediction_seal.json"),
            "geometry_module_sha256": file_sha256(
                Path(__file__).parents[1] / "physics/support_domains.py"
            ),
            "construction_module_sha256": file_sha256(
                Path(__file__).with_name("vlm_mass.py")
            ),
            "mass_labels_read": False,
            "scope": "single exposed development object; conditional whole-body retention",
            "samples": 12000,
            "sample_seed": 711,
            "sensitivity_density": "mean conditional assigned mass divided by assigned volume; homogenized ranking proxy",
            "timing_scope": "single simplification call; no comparative speed claim",
        },
    )
    mesh, labels, pieces, _ = load_prepared(args.output)
    parts = state["parts"]
    hypotheses = [
        json.loads((source / f"hypothesis_{i}.json").read_text())
        for i in range(protocol["rounds"])
    ]
    densities, thickness = protocol["densities_kg_m3"], protocol["fabric_thickness_m"]

    def integrate(subsets, answer):
        return properties(
            construct(
                subsets,
                parts,
                answer,
                thickness,
                allow_removed=True,
                geometry_policy=protocol.get("geometry_policy", "convex"),
            ),
            parts,
            answer,
            densities,
        )

    # Catch implementation or geometry drift before using a frozen answer.
    for row in hypotheses:
        replay = integrate(pieces, row["answer"])
        for key in ("mass_kg", "center_of_mass_m", "inertia_kg_m2"):
            if not np.allclose(replay[key], row[key], rtol=1e-8, atol=1e-10):
                raise ValueError("Frozen domain prediction cannot be reproduced")
    ablations = []
    for row in hypotheses:
        no_panel, no_frame = deepcopy(row["answer"]), deepcopy(row["answer"])
        no_panel["underside"] = {"kind": "visual_cap", "thickness_mm": 0.0}
        for phase in no_frame["upholstery"]:
            phase["hidden_wood_fraction"] = 0.0
        ablations.append(
            {
                "trial": row["trial"],
                "original_mass_kg": row["mass_kg"],
                "panel_as_visual_cap_mass_kg": integrate(pieces, no_panel)["mass_kg"],
                "hidden_wood_as_air_mass_kg": integrate(pieces, no_frame)["mass_kg"],
                "interpretation": "counterfactual sensitivity; alternatives are not ground truth or selected predictions",
            }
        )
    write_json(output / "structural_sensitivity.json", ablations)
    effective = np.zeros(len(pieces))
    for row in hypotheses:
        volumes = {
            p["component_id"]: p["assigned"]["volume_m3"]
            for p in row["partition"]["parts"]
        }
        for part in row["parts"]:
            index = part["component_id"]
            if volumes[index] > 0:
                effective[index] += part["mass_kg"] / volumes[index] / len(hypotheses)
    points, _ = trimesh.sample.sample_surface(mesh, 12000, seed=711)

    def measure(candidate):
        asset_hash = hashlib.sha256(
            b"".join(
                np.ascontiguousarray(value).tobytes()
                for value in (
                    candidate.mesh.vertices,
                    candidate.mesh.faces,
                    candidate.face_components,
                )
            )
        ).hexdigest()
        asset = output / f"mesh_{asset_hash}.npz"
        if not asset.is_file():
            np.savez_compressed(
                asset,
                vertices=candidate.mesh.vertices,
                faces=candidate.mesh.faces,
                component_ids=candidate.face_components,
            )
        subsets = [
            (
                candidate.mesh.submesh(
                    [np.flatnonzero(candidate.face_components == i)],
                    append=True,
                    repair=False,
                )
                if np.any(candidate.face_components == i)
                else trimesh.Trimesh()
            )
            for i in range(len(pieces))
        ]
        _, distances, _ = trimesh.proximity.closest_point(candidate.mesh, points)
        result = {
            "actual_faces": len(candidate.mesh.faces),
            "mesh_asset": asset.name,
            "mesh_asset_sha256": file_sha256(asset),
            "seconds": candidate.seconds,
            "surface_p95_mm": float(np.quantile(distances, 0.95) * 1000),
            "budgets": candidate.budgets,
            "removed_component_ids": [
                i for i, p in enumerate(subsets) if not len(p.faces)
            ],
            "physical_errors": [],
            "valid": False,
        }
        try:
            for row in hypotheses:
                physics = integrate(subsets, row["answer"])
                result["physical_errors"].append(
                    {
                        "trial": row["trial"],
                        "mass_relative": abs(physics["mass_kg"] / row["mass_kg"] - 1),
                        "com_error_mm": float(
                            np.linalg.norm(
                                np.asarray(physics["center_of_mass_m"])
                                - row["center_of_mass_m"]
                            )
                            * 1000
                        ),
                        "inertia_relative": float(
                            np.linalg.norm(
                                np.asarray(physics["inertia_kg_m2"])
                                - row["inertia_kg_m2"]
                            )
                            / np.linalg.norm(row["inertia_kg_m2"])
                        ),
                    }
                )
            result["valid"] = True
        except (ValueError, IndexError) as error:
            result["validation_error"] = str(error)
        return result

    targets = json.loads((args.output / "input_manifest.json").read_text())[
        "face_budget_targets"
    ]
    results = []
    for target in targets:
        for method in ("component_qem", "domain_density_qem"):
            component = simplify_components(
                pieces, target, effective if method == "domain_density_qem" else None
            )
            global_result = simplify_with_lineage(
                mesh, labels, len(component.mesh.faces)
            )
            pair = {
                "target_faces": target,
                "method": method,
                "component": measure(component),
                "matched_global": measure(global_result),
                "absolute_face_count_difference": abs(
                    len(component.mesh.faces) - len(global_result.mesh.faces)
                ),
            }
            results.append(pair)
            write_json(output / "matched_qem.json", results)
            print(
                "DOMAIN_QEM",
                target,
                method,
                pair["component"]["valid"],
                pair["matched_global"]["valid"],
                flush=True,
            )
    return results


def evaluate_vlm(args):
    source = args.output
    raw_path = source / "vlm" / "records.json"
    records = json.loads(raw_path.read_text())
    seal = json.loads((source / "vlm" / "seal.json").read_text())
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != seal["sha256"]:
        raise ValueError("VLM seal mismatch")
    review = json.loads(args.review.read_text())
    known = {
        int(i): value
        for i, value in review["component_materials"].items()
        if value != "unknown"
    }
    rows = []
    for arm in ("repeated", "progressive"):
        for trial in sorted({r["trial"] for r in records}):
            for stage in range(3):
                budgeted = [
                    r
                    for r in records
                    if r["arm"] == arm and r["trial"] == trial and r["stage"] <= stage
                ]
                valid = [r for r in budgeted if r["valid"]]
                used = (
                    valid
                    if arm == "repeated"
                    else [r for r in valid if r["stage"] == stage]
                )
                predictions = {}
                for i in known:
                    votes = [
                        next(
                            c
                            for c in r["answer"]["components"]
                            if c["component_id"] == i
                        )["surface_material"]
                        for r in used
                    ]
                    predictions[i] = (
                        Counter(votes).most_common(1)[0][0] if votes else "unknown"
                    )
                correct = sum(predictions[i] == truth for i, truth in known.items())
                rows.append(
                    {
                        "arm": arm,
                        "trial": trial,
                        "calls": stage + 1,
                        "known_components": len(known),
                        "correct": correct,
                        "accuracy": correct / len(known),
                        "coverage": sum(x != "unknown" for x in predictions.values())
                        / len(known),
                        "predictions": predictions,
                        "total_seconds": sum(
                            r["response"]["seconds"] for r in budgeted
                        ),
                        "total_output_tokens": sum(
                            r["response"]["output_tokens"] for r in budgeted
                        ),
                        "peak_gpu_bytes": max(
                            r["response"]["peak_gpu_bytes"] for r in budgeted
                        ),
                    }
                )
    report = {
        "scope": review["scope"],
        "reference_sha256": hashlib.sha256(args.review.read_bytes()).hexdigest(),
        "interior_accuracy_evaluated": False,
        "raw_call_count": len(records),
        "valid_call_count": sum(r["valid"] for r in records),
        "records": rows,
    }
    write_json(source / "vlm_evaluation.json", report)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for arm, color, marker in (
        ("repeated", "#7b8b9b", "o"),
        ("progressive", "#bd805c", "s"),
    ):
        selected = [r for r in rows if r["arm"] == arm]
        means = [
            np.mean([r["accuracy"] for r in selected if r["calls"] == n])
            for n in (1, 2, 3)
        ]
        axes[0].plot(
            [1, 2, 3],
            np.array(means) * 100,
            marker=marker,
            color=color,
            label=arm,
            fillstyle="none" if arm == "progressive" else "full",
            markersize=7,
        )
        axes[1].plot(
            [1, 2, 3],
            [
                np.mean([r["total_seconds"] for r in selected if r["calls"] == n])
                for n in (1, 2, 3)
            ],
            marker=marker,
            color=color,
        )
    axes[0].set(
        title="Visible-material review", ylabel="Component accuracy (%)", ylim=(0, 105)
    )
    axes[1].set(title="Measured VLM generation cost", ylabel="Cumulative seconds")
    for ax in axes:
        ax.set(xlabel="Calls per trial", xticks=[1, 2, 3])
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.savefig(source / "vlm_rounds.png", dpi=250)
    fig.savefig(source / "vlm_rounds.svg")
    plt.close(fig)
    print(f"VLM records evaluated: {len(records)}")


def evaluate_roles(args):
    mesh, _labels, pieces, _texture = load_prepared(args.output)
    state = active_states(args.output)["progressive"]
    materials = {c["component_id"]: c["surface_material"] for c in state["components"]}
    records = []
    for i, part in enumerate(pieces):
        proposal = furniture_role_candidates(
            part.bounds, mesh.bounds, materials[i], part.is_watertight
        )
        records.append({"component_id": i, "material": materials[i], **proposal})
    parts = []
    for record, original, part in zip(records, state["components"], pieces):
        role = record["allowed_roles"][0]
        parts.append(
            {
                **original,
                "part_id": f"part_{record['component_id']:03d}",
                "instance_version": 2,
                "functional_role": role,
                "part_name": f"{role}_{record['component_id']}",
                "vlm_original_part_name": original.get("part_name"),
                "role_evidence": record["evidence"],
                "role_source": "category-conditioned geometric constraints",
                "anchor_m": part.bounds.mean(0).tolist(),
                "material_parameters": state["materials"].get(record["material"], {}),
            }
        )
    connections = []
    legs = [r["component_id"] for r in records if r["allowed_roles"] == ["leg"]]
    rails = [r["component_id"] for r in records if r["allowed_roles"] == ["frame_rail"]]
    for leg in legs:
        for rail in rails:
            pair = closest_surface_pair(pieces[leg], pieces[rail])
            if pair["distance_m"] <= 0.001:
                connections.append(
                    {
                        "part_ids": [f"part_{leg:03d}", f"part_{rail:03d}"],
                        "relation": "geometric interface within 1 mm",
                        "surface_pair": pair,
                        "mechanical_connection_verified": False,
                    }
                )
    write_json(
        args.output / "part_state_geometry_constrained.json",
        {
            "schema": "object/part/construction_region/mesh_support",
            "version": 2,
            "geometry_sha256": hashlib.sha256(
                (args.output / "reference_arrays.npz").read_bytes()
            ).hexdigest(),
            "parts": parts,
            "connections": connections,
            "role_predictions_use_evaluation_annotations": False,
        },
    )
    # Annotation access occurs after every prediction has been formed.
    review = json.loads(
        (args.output / "views_contacts" / "role_review.json").read_text()
    )["roles"]
    correct = sum(
        r["allowed_roles"][0] == review[str(r["component_id"])] for r in records
    )
    report = {
        "scope": "single development reference with category-conditioned furniture rules",
        "role_accuracy": correct / len(records),
        "correct": correct,
        "components": len(records),
        "rules_have_not_been_validated_on_independent_objects": True,
        "records": records,
    }
    write_json(args.output / "views_contacts" / "geometry_role_results.json", report)
    print(f"Geometric-role check: {correct}/{len(records)}")


def evaluate_matched_qem(args):
    out = args.output
    mesh, labels, pieces, _texture = load_prepared(out)
    edit = json.loads((out / "edit.json").read_text())
    reference = edit["before"]
    original = json.loads((out / "lod" / "metrics.json").read_text())
    points, _ = trimesh.sample.sample_surface(mesh, 12000, seed=711)
    results = []
    for row in original:
        if row["method"] == "global_qem":
            continue
        bulk_density = np.array(
            [
                edit["before_density_kg_m3"] if i in edit["wood_components"] else np.nan
                for i in range(len(pieces))
            ]
        )

        def global_run(row=row):
            return simplify_with_lineage(mesh, labels, row["actual_faces"])

        def component_run(row=row, bulk_density=bulk_density):
            return simplify_components(
                pieces,
                row["target_faces"],
                bulk_density if row["method"] == "sensitivity_qem" else None,
            )

        global_run()
        component_run()
        times, component_times = [], []
        result = None
        for repetition in range(15):
            order = [(global_run, times), (component_run, component_times)]
            if repetition % 2:
                order.reverse()
            for operation, samples in order:
                start = perf_counter()
                candidate = operation()
                samples.append(perf_counter() - start)
                if operation is global_run:
                    result = candidate
        row = {
            **row,
            "median_seconds": float(np.median(component_times)),
            "seconds_samples": component_times,
            "timing_iqr_seconds": np.quantile(component_times, [0.25, 0.75]).tolist(),
        }
        reduced = result.mesh
        subset = [
            (
                reduced.submesh(
                    [np.flatnonzero(result.face_components == i)],
                    append=True,
                    repair=False,
                )
                if np.any(result.face_components == i)
                else trimesh.Trimesh()
            )
            for i in range(len(pieces))
        ]
        physics = closed_properties(
            subset, edit["wood_components"], edit["before_density_kg_m3"]
        )
        _, distances, _ = trimesh.proximity.closest_point(reduced, points)
        errors = None
        if physics is not None:
            errors = {
                "mass_relative": abs(physics["mass_kg"] / reference["mass_kg"] - 1),
                "com_error_mm": float(
                    np.linalg.norm(
                        np.array(physics["center_of_mass_m"])
                        - reference["center_of_mass_m"]
                    )
                    * 1000
                ),
                "inertia_relative": float(
                    np.linalg.norm(
                        np.array(physics["inertia_kg_m2"]) - reference["inertia_kg_m2"]
                    )
                    / np.linalg.norm(reference["inertia_kg_m2"])
                ),
            }
        baseline = {
            "method": "matched_global_qem",
            "actual_faces": len(reduced.faces),
            "median_seconds": float(np.median(times)),
            "seconds_samples": times,
            "timing_iqr_seconds": np.quantile(times, [0.25, 0.75]).tolist(),
            "surface_p95_mm": float(np.quantile(distances, 0.95) * 1000),
            "surface_mean_mm": float(distances.mean() * 1000),
            "wood_errors": errors,
        }
        pair = {
            "component_method": row,
            "matched_global": baseline,
            "absolute_face_count_difference": abs(
                len(reduced.faces) - row["actual_faces"]
            ),
        }
        results.append(pair)
        print(
            f"Matched {row['method']}: {row['actual_faces']} / {len(reduced.faces)} faces",
            flush=True,
        )
    write_json(
        out / "matched_qem.json",
        {
            "scope": "matched achieved face counts; same reference and sample points",
            "timing_protocol": "one warmup per method, 15 alternating paired repeats, VLM processes finished",
            "pairs": results,
        },
    )


def evaluate_grounding(args):
    out = args.output / "views_contacts"
    payload = json.loads(
        (args.answers or (out / "view_count_results.json")).read_text()
    )
    selected = payload["selected_views"]
    calls = [
        r for r in payload["records"] if r["view_count"] == selected and r["valid"]
    ]
    mesh, _labels, pieces, _texture = load_prepared(args.output)
    predictions = {"mesh_geometry": [], "vlm": [], "fusion": []}
    for i, part in enumerate(pieces):
        rows = [
            next(c for c in r["answer"]["components"] if c["component_id"] == i)
            for r in calls
        ]
        material = Counter(c["surface_material"] for c in rows).most_common(1)[0][0]
        role = Counter(c["functional_role"] for c in rows).most_common(1)[0][0]
        geometry = furniture_role_candidates(
            part.bounds, mesh.bounds, "geometry_only", part.is_watertight
        )
        fused = furniture_role_candidates(
            part.bounds, mesh.bounds, material, part.is_watertight
        )
        predictions["mesh_geometry"].append(
            {
                "component_id": i,
                "role": geometry["allowed_roles"][0],
                "surface_material": None,
                "evidence": geometry["evidence"],
            }
        )
        predictions["vlm"].append(
            {"component_id": i, "role": role, "surface_material": material}
        )
        predictions["fusion"].append(
            {
                "component_id": i,
                "role": fused["allowed_roles"][0],
                "surface_material": material,
                "evidence": fused["evidence"],
            }
        )
    # Evaluation labels are loaded only after all three predictions are fixed.
    truth = json.loads((out / "role_review.json").read_text())["roles"]
    metrics = []
    for method, rows in predictions.items():
        correct = sum(r["role"] == truth[str(r["component_id"])] for r in rows)
        metrics.append(
            {
                "method": method,
                "role_accuracy": correct / len(truth),
                "component_count": len(truth),
                "correct_components": correct,
            }
        )
    report = {
        "scope": "one previously exposed reference object; development ablation",
        "original_views": selected,
        "vlm_calls": len(calls),
        "vlm_and_fusion_reuse_identical_raw_answers": True,
        "mesh_geometry_receives_rgb_or_vlm_materials": False,
        "geometry_baseline": "same geometry rules with appearance gating removed",
        "metrics": metrics,
        "predictions": predictions,
    }
    write_json(out / "grounding_ablation.json", report)
    args.figures.mkdir(parents=True, exist_ok=True)
    write_json(args.figures / "grounding_ablation.json", report)
    fig, ax = plt.subplots(figsize=(3.6, 2.8), layout="constrained")
    bars = ax.bar(
        ["Mesh geometry", "VLM", "Fusion"],
        [100 * m["role_accuracy"] for m in metrics],
        color=["#8e969e", "#588cb2", "#bd805c"],
        width=0.6,
    )
    for bar, row in zip(bars, metrics):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2,
            f"{row['correct_components']}/{row['component_count']}",
            ha="center",
            fontsize=8,
        )
    ax.set(
        ylabel="Functional-role accuracy (%)",
        ylim=(0, 112),
        title="Reference-component grounding",
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    savefig(fig, args.figures / "grounding_ablation")
    print(
        "Role ablation: "
        + ", ".join(
            f"{r['method']}={r['correct_components']}/{r['component_count']}"
            for r in metrics
        )
    )


def evaluate_contacts(args):
    out = args.output / "views_contacts"
    _mesh, _labels, pieces, _texture = load_prepared(args.output)
    leg_ids = [6, 7, 8, 9]
    rail_ids = [10, 11, 12, 13]
    rails = trimesh.util.concatenate([pieces[i] for i in rail_ids])
    points, summaries, arrays = [], [], {}
    for index in leg_ids:
        leg = pieces[index]
        ground = leg.vertices[np.abs(leg.vertices[:, 2] - leg.bounds[0, 2]) < 1e-7]
        hull = ConvexHull(ground[:, :2])
        footprint = ground[hull.vertices]
        center = footprint.mean(0)
        points.append(center)
        band = leg.slice_plane([0, 0, leg.bounds[1, 2] - 0.05], [0, 0, 1], cap=False)
        samples, _ = trimesh.sample.sample_surface(band, 4096, seed=1400 + index)
        paired, distance, _ = trimesh.proximity.closest_point(rails, samples)
        nearest = int(np.argmin(distance))
        exact = closest_surface_pair(leg, rails)
        summaries.append(
            {
                "leg_component_id": index,
                "ground_contact_center_m": center.tolist(),
                "ground_footprint_area_m2": float(hull.volume),
                "ground_footprint_vertices_m": footprint.tolist(),
                "interface_sample_count": len(samples),
                "exact_surface_pair": exact,
                "interface_min_sample_distance_mm": float(distance.min() * 1000),
                "interface_p05_distance_mm": float(np.quantile(distance, 0.05) * 1000),
                "near_surface_fraction_1mm": float(np.mean(distance <= 0.001)),
                "near_surface_fraction_3mm": float(np.mean(distance <= 0.003)),
                "closest_sample_on_leg_m": samples[nearest].tolist(),
                "paired_point_on_frame_m": paired[nearest].tolist(),
            }
        )
        arrays[f"leg_{index}_samples"] = samples
        arrays[f"leg_{index}_paired_points"] = paired
        arrays[f"leg_{index}_distances_m"] = distance
    points = np.array(points)
    np.savez_compressed(out / "contact_points.npz", **arrays)
    write_json(
        out / "contact_geometry.json",
        {
            "scope": "reference-mesh geometric proximity and horizontal-ground footprints",
            "reference_unit": "glTF meters, not laboratory calibration",
            "sampled_min_is_not_exact_surface_min": True,
            "frame_interfaces_force_or_bond_strength_evaluated": False,
            "ground_plane_z_m": 0.0,
            "legs": summaries,
        },
    )
    print(
        "Saved ground footprints and exact interface coordinates",
        flush=True,
    )
