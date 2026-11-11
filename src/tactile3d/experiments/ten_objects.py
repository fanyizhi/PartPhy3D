"""Fixed ten-object development pilot using shared, untrained physical rules.

Stages are isolated: predict reads images/geometry/cached language proposals;
evaluate opens mass labels only after every prediction is saved and sealed.
No chaise-specific geometry or target-mass fitting is reused.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import traceback
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
import yaml
from PIL import Image
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable

from tactile3d.experiments.single_object import (
    crop_to_object,
    digest,
    raster_face_ids,
    silhouette_audit,
    stats,
    visual_hull_partition,
    write_csv,
)
from tactile3d.figures.style import configure_style
from tactile3d.mesh.assembly import face_components, texture_face_rgb
from tactile3d.mesh.segmentation import _kmeans, _smooth_face_labels, nearest_surface_labels
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.fields import (
    depth_within_region,
    integrate_density_cells,
    material_conditioned_shell,
)
from tactile3d.utils.io import ensure_dir, read_json, write_json

COLORS = ("#5A76A8", "#C96A6A", "#4FA38F", "#8573B5", "#D1A14B", "#747474",
          "#74A8C6", "#B78378", "#98AA67", "#C088A7")
# Illustrative isotropic phase constants, not measured local-property labels.
YOUNGS = {"wood": 1e10, "foam": 2e6, "fabric": 1e8, "metal": 2e11,
          "aluminum": 6.9e10, "plastic": 2.2e9, "glass": 7e10,
          "rubber": 8e6, "board": 1e10, "mineral": 3e10, "composite": 2e10}


def canonical_material(name):
    name = name.lower()
    if any(word in name for word in ("foam", "batting")):
        return "foam"
    if any(word in name for word in ("wood", "mdf")):
        return "wood"
    if any(word in name for word in ("fabric", "felt", "wool", "cotton", "polyester", "nylon")):
        return "fabric"
    if "circuit" in name:
        return "board"
    if "alumin" in name:
        return "aluminum"
    if any(word in name for word in ("steel", "metal", "copper", "solder")):
        return "metal"
    if any(word in name for word in ("glass", "fiber")):
        return "composite" if "fiber" in name else "glass"
    if any(word in name for word in ("rubber", "silicone")):
        return "rubber"
    if any(word in name for word in ("concrete", "ceramic", "stone")):
        return "mineral"
    return "plastic"


def parse_ranges(text):
    result = {}
    for name, low, high in re.findall(
            r"\(([^:]+):\s*([0-9.]+)(?:\s*-\s*([0-9.]+))?[^)]*\)", text):
        pair = (float(low), float(high or low))
        if pair[0] < 0 or pair[1] < pair[0]:
            raise ValueError("Invalid cached material range")
        result[name.strip().lower()] = pair
    if not result:
        raise ValueError("No parsable cached material proposals")
    return result


def construction_profile(caption):
    text = caption.lower()
    for words, profile in (
        (("dumbbell",), "coated_dense"),
        (("usb", "hub", "ports", "circuit"), "electronics"),
        (("chaise", "ottoman", "sofa", "stool", "couch"), "seating"),
        (("cabinet", "wardrobe", "cupboard"), "panels"),
        (("sleeve", "pouch", "bag"), "sleeve"),
        (("ladder",), "ladder"),
        (("mirror",), "mirror"),
    ):
        if any(word in text for word in words):
            return profile
    return "generic"


def materials_from_info(info, cfg):
    ranges = parse_ranges(info["candidate_materials_density"])
    thickness = parse_ranges(info["thickness"])
    result = [{"name": name, "kind": canonical_material(name), "density": list(pair),
               "thickness_m": [v/100 for v in thickness.get(name, (.1, .5))],
               "source": "released_cached_language_proposal"}
              for name, pair in ranges.items()]
    profile = construction_profile(info["caption"])
    required = {"seating": ("foam", "wood", "fabric"),
                "electronics": ("circuit_board",), "coated_dense": ("steel",)}.get(profile, ())
    for name in required:
        kind = canonical_material(name)
        if any(m["kind"] == kind for m in result):
            continue
        values = cfg["fallback_materials"][name]
        result.append({"name": name, "kind": kind, **values,
                       "source": "declared_global_fallback_prior"})
    return result, profile


def region_labels(mesh, rgb, cfg):
    """Shared geometry/color clustering; region IDs carry no semantic claim."""
    position = mesh.triangles_center
    position = (position-mesh.bounds.mean(axis=0))/max(float(mesh.extents.max()), 1e-9)
    features = np.c_[position, .35*mesh.face_normals, 1.2*rgb]
    rng = np.random.default_rng(cfg["seed"])
    cap = min(len(features), cfg["max_clustering_faces"])
    selected = rng.choice(len(features), cap, replace=False)
    clusters = _kmeans(features[selected], min(cfg["regions"], cap), rng, 35)
    centroids = np.asarray([features[selected][clusters == i].mean(axis=0)
                            for i in np.unique(clusters)])
    labels = cKDTree(centroids).query(features)[1].astype(np.int64)
    labels = _smooth_face_labels(labels, mesh.face_adjacency, 2)
    _, labels = np.unique(labels, return_inverse=True)
    return labels


def material_probabilities(materials, rgb, profile):
    red, green, blue = rgb
    saturation = float(max(rgb)-min(rgb))
    brown = float(red > green*1.08 and green > blue*1.08)
    preferred = {
        "seating": {"fabric": 2.4, "wood": .3, "foam": -.7},
        "electronics": {"plastic": 2.5, "board": -1.0, "metal": -.5},
        "ladder": {"aluminum": 2.5, "plastic": .2, "wood": -1},
        "mirror": {"glass": 2.2, "metal": .5},
        "sleeve": {"fabric": 3}, "panels": {"wood": 2.5},
        "coated_dense": {"rubber": 2.5, "plastic": 1.5, "metal": -1.5},
    }.get(profile, {})
    logits = []
    for i, material in enumerate(materials):
        kind, name = material["kind"], material["name"]
        score = -.45*np.log1p(i)+preferred.get(kind, 0)
        if kind == "wood":
            score += 3*brown - (1-brown)*.6
        elif kind in ("metal", "aluminum"):
            score += .9*(saturation < .08)
        elif kind in ("plastic", "fabric", "rubber"):
            score += 1.4*(saturation > .12)
        if any(word in name for word in ("screw", "solder", "copper")):
            score -= 2.5
        logits.append(score)
    logits = np.asarray(logits)
    probabilities = np.exp(logits-logits.max())
    return probabilities/probabilities.sum()


def geometric_support(mesh, points, pitch, cfg):
    """Conservative native-component box support, intersected with visual hull.

    Boxes are geometric priors, not manufactured-part volumes. Concave pieces
    and connected assemblies can remain overfilled; report this explicitly.
    """
    count, component = face_components(mesh)
    keep = np.zeros(len(points), bool)
    padding = cfg["component_gate_padding_in_voxels"]*pitch
    for index in range(count):
        faces = np.flatnonzero(component == index)
        vertices = mesh.vertices[np.unique(mesh.faces[faces])]
        low, high = vertices.min(axis=0)-padding, vertices.max(axis=0)+padding
        keep |= np.all((points >= low) & (points <= high), axis=1)
    return keep, count


def predict_physics(points, ids, region_rgb, materials, profile, pitch, cfg):
    """Mass-blind model; categorical material hypotheses are not volume fractions."""
    rng = np.random.default_rng(cfg["seed"])
    shell_model = cfg.get("shell_model", "legacy_mean_thickness")
    if shell_model not in ("legacy_mean_thickness", "material_conditioned"):
        raise ValueError("Unknown shell integration model")
    n = cfg["samples"]
    rho = np.stack([rng.uniform(*m["density"], n) for m in materials])
    e = np.asarray([YOUNGS[m["kind"]] for m in materials])
    nu = .3
    bulk, shear = e/(3*(1-2*nu)), e/(2*(1+nu))
    kinds = np.asarray([m["kind"] for m in materials])
    first = {kind: int(np.flatnonzero(kinds == kind)[0]) for kind in set(kinds)}
    density = np.zeros(len(points))
    sigma, occupied, bulk_field, shear_field = [np.zeros(len(points)) for _ in range(4)]
    part_samples, solid_samples, shell_samples, rows = [], [], [], []

    def draw(key):
        low, mode, high = cfg["construction"][key]
        return rng.triangular(low, mode, high, n)

    # Automatic regions are not physical pieces: only the global exterior
    # creates a shell. Artificial clustering boundaries must not create coats.
    global_depth = depth_within_region(points, pitch)
    for part in range(len(region_rgb)):
        selected = ids == part
        if not selected.any():
            part_samples.append(np.zeros(n))
            solid_samples.append(np.zeros(n))
            shell_samples.append(np.zeros(n))
            continue
        probabilities = material_probabilities(materials, region_rgb[part], profile)
        choices = rng.choice(len(materials), size=n, p=probabilities)
        outer = rho[choices, np.arange(n)]
        outer_k, outer_g = bulk[choices], shear[choices]
        phi = draw("generic_core_occupancy")
        core, core_k, core_g = outer*phi, outer_k*phi, outer_g*phi
        hollow = np.isin(kinds[choices], ["metal", "aluminum", "plastic", "composite"])
        hollow_phi = draw("hollow_core_occupancy")
        core[hollow] = outer[hollow]*hollow_phi[hollow]
        core_k[hollow], core_g[hollow] = outer_k[hollow]*hollow_phi[hollow], outer_g[hollow]*hollow_phi[hollow]
        phi[hollow] = hollow_phi[hollow]
        if profile == "seating":
            soft = np.isin(kinds[choices], ["fabric", "foam", "plastic", "rubber"])
            fill, frame = draw("soft_core_occupancy"), draw("hidden_wood_fraction")
            foam_id, wood_id = first["foam"], first["wood"]
            rho_core = rho[wood_id]*frame + rho[foam_id]*(1-frame)*fill
            k_core = bulk[wood_id]*frame + bulk[foam_id]*(1-frame)*fill
            g_core = shear[wood_id]*frame + shear[foam_id]*(1-frame)*fill
            core[soft], core_k[soft], core_g[soft] = rho_core[soft], k_core[soft], g_core[soft]
            phi[soft] = (frame+(1-frame)*fill)[soft]
        elif profile == "electronics":
            phi = draw("electronics_core_occupancy")
            board = first["board"]
            core, core_k, core_g = rho[board]*phi, bulk[board]*phi, shear[board]*phi
        elif profile == "coated_dense":
            steel = first["metal"]
            fraction = draw("dumbbell_steel_fraction")
            core = rho[steel]*fraction + outer*(1-fraction)
            core_k, core_g = bulk[steel]*fraction+outer_k*(1-fraction), shear[steel]*fraction+outer_g*(1-fraction)
            phi = np.ones(n)
        elif profile in ("sleeve", "panels"):
            phi = draw("hollow_core_occupancy" if profile == "sleeve" else "furniture_panel_fill")
            core, core_k, core_g = outer*phi, outer_k*phi, outer_g*phi
        thickness = float(sum(p*np.mean(m["thickness_m"]) for p, m in zip(probabilities, materials, strict=True)))
        depth = global_depth[selected]
        volume = float(selected.sum()*pitch**3)
        if shell_model == "material_conditioned":
            mixture = material_conditioned_shell(
                depth, pitch, [np.mean(m["thickness_m"]) for m in materials], choices,
                {"density": outer, "bulk": outer_k, "shear": outer_g, "occupied": np.ones(n)},
                {"density": core, "bulk": core_k, "shear": core_g, "occupied": phi})
            outer_volume = mixture["shell_volume_samples_m3"]
            density[selected] = mixture["mean"]["density"]
            sigma[selected] = mixture["std"]["density"]
            occupied[selected] = mixture["mean"]["occupied"]
            bulk_field[selected] = mixture["mean"]["bulk"]
            shear_field[selected] = mixture["mean"]["shear"]
        else:
            shell = np.clip((thickness-np.maximum(depth-pitch, 0))/pitch, 0, 1)
            outer_volume = float(shell.sum()*pitch**3)
            density[selected] = shell*outer.mean()+(1-shell)*core.mean()
            covariance = np.cov(np.stack([outer, core]))
            variance = shell**2*covariance[0, 0] + (1-shell)**2*covariance[1, 1] + 2*shell*(1-shell)*covariance[0, 1]
            sigma[selected] = np.sqrt(np.maximum(variance, 0))
            occupied[selected] = shell+(1-shell)*phi.mean()
            bulk_field[selected] = shell*outer_k.mean()+(1-shell)*core_k.mean()
            shear_field[selected] = shell*outer_g.mean()+(1-shell)*core_g.mean()
        samples = outer_volume*outer + (volume-outer_volume)*core
        part_samples.append(samples)
        solid_samples.append(volume*outer)
        shell_samples.append(outer_volume*outer)
        rows.append({"part_id": part, "stable_id": f"region_{part:02d}",
            "name": f"Geometric region {part:02d}", "volume_m3": volume,
            "density_mean_kg_m3": float(density[selected].mean()),
            "occupied_fraction": float(occupied[selected].mean()),
            "shell_model": shell_model,
            "shell_thickness_m_fixed": thickness if shell_model == "legacy_mean_thickness" else None,
            "shell_thickness_m_by_material": {m["name"]: float(np.mean(m["thickness_m"])) for m in materials},
            "mass_kg": stats(samples),
            "material_hypothesis_probabilities_uncalibrated": {
                m["name"]: float(p) for m, p in zip(materials, probabilities, strict=True)},
            "local_property_accuracy": None})
    part_samples = np.asarray(part_samples)
    upper = 9*bulk_field*shear_field/np.maximum(3*bulk_field+shear_field, 1e-12)
    moments = integrate_density_cells(points, density, pitch, ids)
    if not np.isclose(moments["mass_kg"], part_samples.sum(axis=0).mean()):
        raise RuntimeError("Mass samples and finite-cell integration disagree")
    return {"density": density, "sigma": sigma, "occupied": occupied, "E_upper": upper,
            "part_samples": part_samples, "moments": moments, "parts": rows,
            "scenarios": {"Surface solid": np.sum(solid_samples, axis=0),
                          "Shell only": np.sum(shell_samples, axis=0),
                          "Construction prior": part_samples.sum(axis=0)}}


def load_mesh(path):
    loaded = trimesh.load(path, process=True)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("No valid triangle mesh")
    return mesh


def predict_one(root, asset, cfg, output):
    started = time.perf_counter()
    scene_id = asset["scene_id"]
    scene = root/"scenes"/scene_id
    timings = {}
    mesh = load_mesh(asset["mesh_path"])
    info, transforms = read_json(scene/"info.json"), read_json(scene/"transforms.json")
    tick = time.perf_counter()
    align = silhouette_audit(mesh, scene, transforms)
    timings["alignment_seconds"] = time.perf_counter()-tick
    write_csv(output/"alignment.csv", align)
    mean_iou = float(np.mean([r["silhouette_IoU"] for r in align]))
    base = {"scene_id": scene_id, "caption": info["caption"], "faces": len(mesh.faces),
            "mesh_watertight": bool(mesh.is_watertight), "mean_silhouette_iou": mean_iou,
            "mesh_sha256": digest(asset["mesh_path"]),
            "camera_sha256": digest(scene/"transforms.json"),
            "proposal_sha256": digest(scene/"info.json"), "status_flags": cfg["status"]}
    if mean_iou < cfg["alignment_mean_iou_min"]:
        return {**base, "status": "alignment_failed", "timings": timings,
                "reason": "reference mesh/camera correspondence failed fixed IoU gate"}
    tick = time.perf_counter()
    rgb = texture_face_rgb(mesh)
    labels = region_labels(mesh, rgb, cfg)
    n_parts = int(labels.max())+1
    region_rgb = np.asarray([np.average(rgb[labels == i], weights=mesh.area_faces[labels == i], axis=0)
                             for i in range(n_parts)])
    materials, profile = materials_from_info(info, cfg)
    timings["region_proposal_seconds"] = time.perf_counter()-tick
    np.save(output/"face_labels.npy", labels)
    colored = mesh.copy()
    colored.visual = trimesh.visual.ColorVisuals(mesh=colored,
        face_colors=colors.to_rgba_array(COLORS)[labels])
    colored.export(output/"regions.ply")
    convergences, main = [], None
    for resolution in cfg["voxel_resolutions"]:
        tick = time.perf_counter()
        pitch = float(mesh.extents.max())/resolution
        bbox_count = np.prod(np.ceil(mesh.extents/pitch)+3)
        if bbox_count > cfg["max_bbox_cells"]:
            pitch *= (bbox_count/cfg["max_bbox_cells"])**(1/3)*1.02
        points, ids, _audit = visual_hull_partition(mesh, labels, scene, transforms, pitch,
                                                   cfg["alpha_threshold"])
        keep, components = geometric_support(mesh, points, pitch, cfg)
        before = len(points)
        points, ids = points[keep], ids[keep]
        if not len(points):
            raise ValueError("No cells left after geometric support")
        fields = predict_physics(points, ids, region_rgb, materials, profile, pitch, cfg)
        seconds = time.perf_counter()-tick
        row = {"requested_resolution": resolution, "pitch_m": pitch, "cells": len(points),
               "envelope_volume_m3": len(points)*pitch**3,
               "mass_kg": fields["moments"]["mass_kg"], "seconds": seconds,
               "visual_hull_cells": before, "native_components": components}
        convergences.append(row)
        print(json.dumps({"scene_id": scene_id, "progress": row}), flush=True)
        if resolution == cfg["main_resolution"]:
            main = points, ids, pitch, fields
    if main is None:
        raise ValueError("Main resolution not included")
    points, ids, pitch, fields = main
    timings["volume_and_physics_seconds"] = sum(r["seconds"] for r in convergences)
    write_csv(output/"resolution_convergence.csv", convergences)
    np.savez_compressed(output/"physical_field.npz", centers_m=points, part_ids=ids,
        pitch_m=pitch, density_mean_kg_m3=fields["density"], density_std_kg_m3=fields["sigma"],
        occupied_fraction=fields["occupied"], E_voigt_upper_pa=fields["E_upper"],
        part_mass_samples_kg=fields["part_samples"])
    write_json(output/"part_states.json", fields["parts"])
    write_csv(output/"part_parameters.csv", [{"part_id": r["part_id"], "stable_id": r["stable_id"],
        "volume_m3": r["volume_m3"], "density_mean_kg_m3": r["density_mean_kg_m3"],
        "occupied_fraction": r["occupied_fraction"],
        **{f"mass_{k}_kg": v for k, v in r["mass_kg"].items()}} for r in fields["parts"]])
    tick = time.perf_counter()
    lod_rows = []
    for fraction in cfg["lod_face_fractions"]:
        target = max(4, int(len(mesh.faces)*fraction))
        lod = mesh.simplify_quadric_decimation(face_count=target)
        mapped = labels[cKDTree(mesh.triangles_center).query(lod.triangles_center)[1]]
        lod_rows.append({"target_fraction": fraction, "faces": len(lod.faces),
            "surface_area_drift_fraction": abs(lod.area-mesh.area)/mesh.area,
            "mapping": "nearest original face; appearance only",
            "canonical_mass_kg": fields["moments"]["mass_kg"],
            "mass_semantics": "same original field reused, not independent LOD accuracy"})
        if fraction == cfg["lod_face_fractions"][-1]:
            lod.export(output/"mesh_25pct.ply")
            np.save(output/"mesh_25pct_labels.npy", mapped)
    timings["qem_and_mapping_seconds"] = time.perf_counter()-tick
    write_csv(output/"lod_diagnostics.csv", lod_rows)
    return {**base, "status": "predicted", "construction_profile": profile,
            "num_regions": n_parts, "main_pitch_m": pitch,
            "main_volume_m3": len(points)*pitch**3, "materials": materials,
            "parts": fields["parts"], "moments": fields["moments"], "timings": timings,
            "scenarios": {k: stats(v) for k, v in fields["scenarios"].items()},
            "seconds": time.perf_counter()-started,
            "limits": ["native boxes and regions are geometric proxies",
                       "all material probabilities and construction fractions are uncalibrated",
                       "fixed shell geometry excludes thickness uncertainty",
                       "E is an idealized upper bound from assumed isotropic constants",
                       "local property ground truth unavailable"]}


def predict(root, assets, config_path, output):
    started = time.perf_counter()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    selection = read_json(output/"selection_before_evaluation.json")
    expected = read_json(root/"splits.json")["test"][:10]
    if selection["object_ids"] != expected:
        raise ValueError("Fixed selection mismatch")
    spec = {"config": cfg, "config_sha256": digest(config_path), "code_sha256": digest(__file__),
            "selection": selection, "prediction_uses_target_mass": False,
            "source_modules": {name: digest(Path(__file__).parents[1]/name) for name in
                ("experiments/single_object.py", "physics/fields.py", "mesh/assembly.py")},
            "versions": {"numpy": np.__version__, "trimesh": trimesh.__version__},
            "runtime_device": "local CPU; not a two-4090 benchmark"}
    write_json(output/"frozen_spec.json", spec)
    (output/"inference_source_snapshot.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
    records = []
    assets_by_id = {r["scene_id"]: r for r in read_json(output/"asset_downloads.json")}
    for scene_id in expected:
        local = ensure_dir(output/"objects"/scene_id)
        asset = assets_by_id[scene_id]
        try:
            if asset["status"] != "available":
                raise FileNotFoundError(asset.get("error", "asset unavailable"))
            result = predict_one(root, asset, cfg, local)
        except Exception as exc:  # noqa: BLE001 - batch boundary records every failure
            result = {"scene_id": scene_id, "status": "failed", "error": str(exc)}
            (local/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(local/"prediction.json", result)
        records.append(result)
        print(json.dumps({"completed": scene_id, "status": result["status"],
                          "seconds": result.get("seconds")}), flush=True)
    summary = {"requested": 10, "predictions": records, "seconds": time.perf_counter()-started,
               "target_mass_read": False, "status_flags": cfg["status"]}
    write_json(output/"predictions.json", summary)
    files = [output/"predictions.json", output/"frozen_spec.json"]
    files.extend((output/"objects").glob("*/prediction.json"))
    files.extend((output/"objects").glob("*/physical_field.npz"))
    files.extend((output/"objects").glob("*/part_states.json"))
    files.extend((output/"objects").glob("*/face_labels.npy"))
    write_json(output/"prediction_seal.json", {str(p.relative_to(output)): digest(p) for p in files})
    return summary


def metrics(rows):
    if not rows:
        return None
    pred = np.array([r["mass_kg"] for r in rows])
    target = np.array([r["target_mass_kg"] for r in rows])
    return {"n": len(rows), "ADE_kg": float(np.abs(pred-target).mean()),
            "ALDE": float(np.abs(np.log(pred/target)).mean()),
            "APE_percent": float((100*np.abs(pred-target)/target).mean()),
            "MnRE": float(np.minimum(pred/target, target/pred).mean()),
            "RMSE_kg": float(np.sqrt(np.mean((pred-target)**2))),
            "prior_interval_hit_fraction": float(np.mean([r["interval_hit"] for r in rows]))}


def export_diagnostics(output, predictions, rows):
    """Post-evaluation diagnostics; no parameters are fitted or updated."""
    from tactile3d.eval.properties import PROPERTIES, evaluate_property_rows

    templates, diagnostics = [], []
    primary = {r["scene_id"]: r for r in rows if r["method"] == "Construction prior"}
    for prediction in predictions:
        if prediction["status"] != "predicted":
            continue
        scene_id = prediction["scene_id"]
        with (output/"objects"/scene_id/"resolution_convergence.csv").open() as stream:
            convergence = list(csv.DictReader(stream))
        main_mass = prediction["moments"]["mass_kg"]
        fine_mass = float(convergence[-1]["mass_kg"])
        drift = 100*abs(fine_mass/main_mass-1)
        record = primary.get(scene_id)
        diagnostics.append({"scene_id": scene_id,
            "mean_silhouette_iou": prediction["mean_silhouette_iou"],
            "main_volume_m3": prediction["main_volume_m3"],
            "main_predicted_effective_density_kg_m3": main_mass/prediction["main_volume_m3"],
            "mass_drift_main_to_finest_percent": drift,
            "numerical_sensitivity_above_5percent": drift > 5,
            "mass_APE_percent": record["APE_percent"] if record else "",
            "released_label_inside_prior_interval": record["interval_hit"] if record else "",
            "construction_profile": prediction["construction_profile"]})
        for part in prediction["parts"]:
            for prop, (unit, _, _) in PROPERTIES.items():
                if prop == "hardness_n2p_encoded":
                    continue
                templates.append({"method": "shared_untrained_construction_prior",
                    "object_id": scene_id, "sample_id": part["stable_id"], "property": prop,
                    "unit": unit, "protocol": "region_envelope_effective_density" if prop == "density"
                                               else "measurement_protocol_to_be_defined",
                    "predicted": part["density_mean_kg_m3"] if prop == "density" else "",
                    "target": "", "reference_kind": "", "reference_source": ""})
    if diagnostics:
        write_csv(output/"diagnostic_summary.csv", diagnostics)
    if templates:
        write_csv(output/"local_property_reference_template.csv", templates)
        write_json(output/"local_property_evaluation.json", evaluate_property_rows(templates))


def evaluate(root, output):
    seal = read_json(output/"prediction_seal.json")
    if any(digest(output/name) != checksum for name, checksum in seal.items()):
        raise ValueError("Sealed prediction changed")
    batch = read_json(output/"predictions.json")
    weights = read_json(root/"filtered_product_weights.json")
    known = read_json(output/"selection_before_evaluation.json")["known_development_object"]
    per_object, rows = [], []
    for prediction in batch["predictions"]:
        scene_id = prediction["scene_id"]
        target = weights.get(scene_id, weights.get(scene_id.split("_")[0]))
        record = {"scene_id": scene_id, "status": prediction["status"],
                  "previous_development_case": scene_id == known,
                  "caption": prediction.get("caption", ""), "target_mass_kg": target,
                  "mean_silhouette_iou": prediction.get("mean_silhouette_iou"),
                  "seconds": prediction.get("seconds")}
        if prediction["status"] == "predicted" and isinstance(target, (float, int)) and target > 0:
            for method, value in prediction["scenarios"].items():
                mass = value["mean"]
                rows.append({**record, "method": method, "mass_kg": mass,
                    "q05_kg": value["q05"], "q95_kg": value["q95"],
                    "ADE_kg": abs(mass-target), "ALDE": abs(float(np.log(mass/target))),
                    "APE_percent": 100*abs(mass-target)/target,
                    "MnRE": min(mass/target, target/mass),
                    "interval_hit": value["q05"] <= target <= value["q95"]})
        else:
            record["failure_reason"] = prediction.get("error", prediction.get("reason", "invalid mass reference"))
        per_object.append(record)
    write_csv(output/"per_object_status.csv", [{**r, "failure_reason": r.get("failure_reason", "")} for r in per_object])
    if rows:
        write_csv(output/"mass_metrics_per_object.csv", rows)
    methods = sorted({r["method"] for r in rows})
    aggregate = {"requested_objects": 10,
        "successful_predictions": sum(p["status"] == "predicted" for p in per_object),
        "failed_or_rejected_objects": [p for p in per_object if p["status"] != "predicted"],
        "all_ten_development_summary": {m: metrics([r for r in rows if r["method"] == m]) for m in methods},
        "excluding_previously_seen_chaise": {m: metrics([r for r in rows if r["method"] == m
                                                    and not r["previous_development_case"]]) for m in methods},
        "not_a_heldout_final_benchmark": True, "label_file_sha256": digest(root/"filtered_product_weights.json"),
        "target_mass_used_for_prediction": False, "prediction_seal_verified": True,
        "prediction_wall_seconds_cpu": batch["seconds"],
        "published_baseline_status": {"PUGS": "not run", "NeRF2Physics": "not run", "PhysVGGT": "not run"},
        "local_property_metrics": None, "local_property_reference_status": "unavailable",
        "intervals": "uncalibrated prior intervals, hit fraction is descriptive only"}
    write_json(output/"aggregate_metrics.json", aggregate)
    export_diagnostics(output, batch["predictions"], rows)
    return aggregate


def save(fig, output, stem):
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output/f"{stem}.{suffix}", dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def plot(root, output):
    configure_style()
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8})
    batch = read_json(output/"predictions.json")
    assets = {a["scene_id"]: a for a in read_json(output/"asset_downloads.json")}
    cfg = read_json(output/"frozen_spec.json")["config"]
    overview, property_images = [], []
    render_started = time.perf_counter()
    for index, prediction in enumerate(batch["predictions"]):
        scene_id = prediction["scene_id"]
        local = output/"objects"/scene_id
        scene = root/"scenes"/scene_id
        transforms, info = read_json(scene/"transforms.json"), read_json(scene/"info.json")
        frame = transforms["frames"][int(info["idx_to_caption"])]
        image = Image.open(scene/frame["file_path"]).convert("RGBA")
        reference_mask = np.array(image)[:, :, 3] >= cfg["alpha_threshold"]
        white = Image.new("RGBA", image.size, "white")
        white.alpha_composite(image)
        rgb_image = crop_to_object(np.asarray(white.convert("RGB")), reference_mask)
        if prediction["status"] != "predicted":
            overview.append((scene_id, rgb_image, None, prediction["status"]))
            continue
        mesh = load_mesh(assets[scene_id]["mesh_path"])
        labels = np.load(local/"face_labels.npy")
        field = np.load(local/"physical_field.npz")
        render = raster_face_ids(mesh, np.array(frame["transform_matrix"]), transforms,
                                  cfg["display_resolution"])
        visible = render >= 0
        region_image = np.ones((*render.shape, 3))
        region_image[visible] = colors.to_rgba_array(COLORS)[labels[render[visible]], :3]
        region_image = crop_to_object(region_image, reference_mask)
        nearest = cKDTree(field["centers_m"]).query(mesh.triangles_center)[1]
        density = field["density_mean_kg_m3"]
        occupancy = field["occupied_fraction"]
        sigma = field["density_std_kg_m3"]
        values = (density, occupancy, sigma/np.maximum(density, 1e-9))
        norms = (colors.LogNorm(10, 8000), colors.Normalize(0, 1), colors.Normalize(0, 1))
        cmaps = ("viridis", "cividis", "magma")
        maps = []
        for values_one, norm, cmap in zip(values, norms, cmaps, strict=True):
            canvas = np.ones((*render.shape, 3))
            face_colors = plt.get_cmap(cmap)(norm(values_one[nearest]))[:, :3]
            canvas[visible] = face_colors[render[visible]]
            maps.append(crop_to_object(canvas, reference_mask))
        overview.append((scene_id, rgb_image, region_image, "predicted"))
        property_images.append((scene_id, maps))
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.7))
        for ax, picture, title in zip(axes, [rgb_image, region_image, maps[0], maps[1]],
                                     ["Dataset RGB", "Geometric regions", "Density prior", "Occupancy hypothesis"], strict=True):
            ax.imshow(picture)
            ax.set_title(title, loc="left")
            ax.axis("off")
        fig.colorbar(ScalarMappable(norm=norms[0], cmap=cmaps[0]), ax=axes[2],
                     fraction=.045, pad=.02, shrink=.75, label="kg m$^{-3}$")
        fig.colorbar(ScalarMappable(norm=norms[1], cmap=cmaps[1]), ax=axes[3],
                     fraction=.045, pad=.02, shrink=.75, label="fraction")
        fig.suptitle(f"{index+1:02d} | {scene_id}", fontsize=11)
        fig.subplots_adjust(bottom=.16, top=.85)
        fig.text(.5, .02, "Untrained shared rules; original mesh + dataset masks. Regions and interior are hypotheses; local physical truth unavailable.",
                 ha="center", fontsize=8)
        save(fig, local, "object_fields")
        print(json.dumps({"figure_completed": scene_id}), flush=True)
    fig, axes = plt.subplots(5, 4, figsize=(13, 14))
    for index, (scene_id, image, regions, status) in enumerate(overview):
        row, column = index//2, (index % 2)*2
        axes[row, column].imshow(image)
        axes[row, column].set_title(f"{index+1:02d} {scene_id[:10]} | RGB", loc="left", fontsize=9)
        if regions is None:
            axes[row, column+1].text(.5, .5, status, ha="center", va="center", transform=axes[row, column+1].transAxes)
        else:
            axes[row, column+1].imshow(regions)
        axes[row, column+1].set_title("Region hypotheses", loc="left", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.subplots_adjust(left=.02, right=.98, top=.97, bottom=.05, wspace=.08, hspace=.24)
    fig.text(.5, .015, "First ten published test IDs; no substitution. One shared geometry/color rule; not learned or ground-truth part segmentation.",
             ha="center", fontsize=8)
    save(fig, output, "01_ten_objects_regions")

    fig, axes = plt.subplots(5, 4, figsize=(13, 14))
    for index, (scene_id, maps) in enumerate(property_images):
        row, column = index//2, (index % 2)*2
        original_index = next(i for i, p in enumerate(batch["predictions"]) if p["scene_id"] == scene_id)+1
        axes[row, column].imshow(maps[0])
        axes[row, column+1].imshow(maps[1])
        axes[row, column].set_title(f"{original_index:02d} {scene_id[:10]} | density", loc="left", fontsize=9)
        axes[row, column+1].set_title("Material occupancy", loc="left", fontsize=9)
    for ax in axes.flat:
        ax.axis("off")
    fig.subplots_adjust(left=.02, right=.98, top=.97, bottom=.095, wspace=.08, hspace=.24)
    fig.colorbar(ScalarMappable(norm=colors.LogNorm(10, 8000), cmap="viridis"),
                 cax=fig.add_axes((.08, .045, .35, .012)), orientation="horizontal", label="Effective density prior (kg m$^{-3}$)")
    fig.colorbar(ScalarMappable(norm=colors.Normalize(0, 1), cmap="cividis"),
                 cax=fig.add_axes((.57, .045, .35, .012)), orientation="horizontal", label="Material occupancy hypothesis (fraction)")
    fig.text(.5, .008, "Shared color scales across objects. Untrained material/construction assumptions; no measured local-property ground truth.",
             ha="center", fontsize=8)
    save(fig, output, "04_ten_objects_density_occupancy")

    summary = read_json(output/"aggregate_metrics.json")
    with (output/"mass_metrics_per_object.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    methods = ("Surface solid", "Shell only", "Construction prior")
    by_method = {m: [r for r in rows if r["method"] == m] for m in methods}
    chosen = by_method["Construction prior"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for method, color in zip(methods, ("#888888", "#5A76A8", "#4FA38F"), strict=True):
        data = by_method[method]
        axes[0, 0].scatter([float(r["target_mass_kg"]) for r in data],
                            [float(r["mass_kg"]) for r in data], label=method, color=color, s=30, alpha=.8)
    all_masses = [float(r[k]) for r in rows for k in ("mass_kg", "target_mass_kg")]
    low, high = min(all_masses)*.7, max(all_masses)*1.4
    axes[0, 0].plot([low, high], [low, high], color="#222222", linewidth=1)
    axes[0, 0].set(xscale="log", yscale="log", xlabel="Released mass label (kg)", ylabel="Predicted mean mass (kg)")
    axes[0, 0].set_title("a  Mass prediction | logarithmic axes", loc="left")
    axes[0, 0].legend(loc="best")
    x = np.arange(len(chosen))
    means = np.array([float(r["mass_kg"]) for r in chosen])
    low = np.array([float(r["q05_kg"]) for r in chosen])
    high = np.array([float(r["q95_kg"]) for r in chosen])
    centers = (low+high)/2
    axes[0, 1].errorbar(x, centers, yerr=[centers-low, high-centers], fmt="none",
                        color="#4FA38F", capsize=3, label="90% prior interval")
    axes[0, 1].scatter(x, means, color="#4FA38F", s=24, label="Prior mean")
    axes[0, 1].scatter(x, [float(r["target_mass_kg"]) for r in chosen], marker="x", color="#222222", label="Released label")
    axes[0, 1].set(yscale="log", xlabel="Fixed object index", ylabel="Mass (kg)")
    order = read_json(output/"selection_before_evaluation.json")["object_ids"]
    axes[0, 1].set_xticks(x, [str(order.index(r["scene_id"])+1) for r in chosen])
    axes[0, 1].set_title("b  Uncalibrated prior intervals", loc="left")
    axes[0, 1].legend(loc="best")
    for method, offset, color in zip(methods, (-.25, 0, .25), ("#888888", "#5A76A8", "#4FA38F"), strict=True):
        data = {r["scene_id"]: r for r in by_method[method]}
        axes[1, 0].bar(x+offset, [float(data[r["scene_id"]]["APE_percent"]) for r in chosen], width=.24, color=color, label=method)
    axes[1, 0].set(yscale="symlog", xlabel="Fixed object index", ylabel="Absolute percentage error (%)")
    axes[1, 0].set_xticks(x, [str(order.index(r["scene_id"])+1) for r in chosen])
    axes[1, 0].set_title("c  All individual errors retained", loc="left")
    axes[1, 1].axis("off")
    table_rows = []
    for method in methods:
        value = summary["all_ten_development_summary"][method]
        table_rows.append([method, f"{value['ADE_kg']:.2f}", f"{value['ALDE']:.3f}",
                           f"{value['APE_percent']:.1f}", f"{value['MnRE']:.3f}"])
    table = axes[1, 1].table(cellText=table_rows, colLabels=["Assumption", "ADE kg", "ALDE", "APE %", "MnRE"],
        colWidths=[.36, .16, .16, .16, .16], cellLoc="center", bbox=(0, .42, 1, .43))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DDE2")
        if r == 0:
            cell.set_facecolor("#EEF2F6")
    axes[1, 1].set_title("d  Development summary, valid objects only", loc="left")
    axes[1, 1].text(0, .16, f"Attempted 10; predicted {summary['successful_predictions']}.\n"
        "Published paper baselines not executed.\nPreviously used chaise is also reported separately.", fontsize=8)
    fig.subplots_adjust(left=.08, right=.98, top=.94, bottom=.14, hspace=.43, wspace=.32)
    fig.text(.5, .025, "Predictions sealed before evaluation; no target-mass calibration. Shared untrained rules, reference meshes and dataset alpha masks.\n"
             "This is a development diagnostic, not a held-out benchmark or an accuracy claim for local physical properties.", ha="center", fontsize=8)
    save(fig, output, "02_ten_objects_mass")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    valid = [p for p in batch["predictions"] if p["status"] == "predicted"]
    stages = ("alignment_seconds", "region_proposal_seconds", "volume_and_physics_seconds", "qem_and_mapping_seconds")
    bottom = np.zeros(len(valid))
    for stage, color in zip(stages, COLORS, strict=False):
        values = np.array([p["timings"][stage] for p in valid])
        axes[0].bar(np.arange(len(valid)), values, bottom=bottom, color=color, label=stage.replace("_seconds", "").replace("_", " "))
        bottom += values
    axes[0].set_xticks(np.arange(len(valid)), [str(order.index(p["scene_id"])+1) for p in valid])
    axes[0].set(xlabel="Fixed object index", ylabel="Local CPU time (s)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("a  Measured prediction stage times", loc="left")
    for p in valid:
        with (output/"objects"/p["scene_id"]/"resolution_convergence.csv").open() as stream:
            data = list(csv.DictReader(stream))
        masses = np.array([float(r["mass_kg"]) for r in data])
        reference = masses[cfg["voxel_resolutions"].index(cfg["main_resolution"])]
        axes[1].plot([int(r["requested_resolution"]) for r in data], masses/reference,
                      "o-", markersize=3, linewidth=1, label=str(order.index(p["scene_id"])+1))
    axes[1].set(xlabel="Requested grid resolution", ylabel="Mass / main-resolution mass")
    axes[1].set_title("b  Discretization sensitivity", loc="left")
    axes[1].legend(ncol=5, title="Object index", fontsize=7, title_fontsize=8)
    fig.subplots_adjust(left=.07, right=.98, top=.9, bottom=.22, wspace=.3)
    fig.text(.5, .035, "Local CPU measurements; GPU reconstruction, learned segmentation and Qwen are not included. QEM reuses the original physical field.",
             ha="center", fontsize=8)
    save(fig, output, "03_timing_and_resolution")
    write_json(output/"figure_manifest.json", {"plot_seconds": time.perf_counter()-render_started,
        "formats": ["svg", "pdf", "png"], "main_figures": ["01_ten_objects_regions", "02_ten_objects_mass", "03_timing_and_resolution", "04_ten_objects_density_occupancy"],
        "per_object_figure_count": len(property_images), "local_properties_are_unmeasured": True})


def compare_shell_models(root, assets, cohort_path, config_path, output):
    """Paired development audit: reuse exactly the same geometry and material draws."""
    if (output/"predictions.json").exists():
        raise FileExistsError("Use a fresh output directory for the paired shell audit")
    cohort = read_json(cohort_path)
    order = cohort["object_ids"]
    if not order or len(set(order)) != len(order):
        raise ValueError("Require a fixed nonempty cohort without duplicates")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cfg["shell_model"] = "legacy_mean_thickness"
    cfg["voxel_resolutions"] = [cfg["main_resolution"]]
    corrected = {**cfg, "shell_model": "material_conditioned"}
    write_json(output/"frozen_spec.json", {
        "cohort": cohort, "legacy_config": cfg, "conditional_config": corrected,
        "code_sha256": digest(__file__),
        "integration_sha256": digest(Path(__file__).parents[1]/"physics/fields.py"),
        "surface_assignment_sha256": digest(Path(__file__).parents[1]/"mesh/segmentation.py"),
        "surface_assignment_ablation": ["nearest_triangle_centroid", "nearest_triangle_surface"],
        "target_mass_used_for_inference": False,
        "scope": "paired development audit of an untrained rule prototype",
    })
    records = []
    for scene in order:
        directory = ensure_dir(output/"objects"/scene)
        asset = {"scene_id": scene, "mesh_path": str(assets/scene/"model.glb")}
        try:
            before = predict_one(root, asset, cfg, directory)
            if before["status"] != "predicted":
                raise ValueError(before.get("reason", "Geometry alignment failed"))
            write_json(directory/"legacy_prediction.json", before)
            with np.load(directory/"physical_field.npz") as field:
                points, ids = field["centers_m"], field["part_ids"]
                pitch = float(field["pitch_m"])
            mesh = load_mesh(asset["mesh_path"])
            rgb = texture_face_rgb(mesh)
            labels = np.load(directory/"face_labels.npy")
            colors_rgb = np.asarray([
                np.average(rgb[labels == i], weights=mesh.area_faces[labels == i], axis=0)
                for i in range(int(labels.max())+1)])
            after = predict_physics(points, ids, colors_rgb, before["materials"],
                                    before["construction_profile"], pitch, corrected)
            np.savez_compressed(directory/"conditional_field.npz", centers_m=points,
                part_ids=ids, pitch_m=pitch, density_mean_kg_m3=after["density"],
                density_std_kg_m3=after["sigma"], occupied_fraction=after["occupied"],
                part_mass_samples_kg=after["part_samples"])
            write_json(directory/"conditional_parts.json", after["parts"])
            nearest_ids, distances = nearest_surface_labels(mesh, points, labels)
            nearest = predict_physics(points, nearest_ids, colors_rgb, before["materials"],
                                      before["construction_profile"], pitch, corrected)
            np.savez_compressed(directory/"nearest_surface_field.npz", centers_m=points,
                part_ids=nearest_ids, pitch_m=pitch, density_mean_kg_m3=nearest["density"],
                density_std_kg_m3=nearest["sigma"], occupied_fraction=nearest["occupied"],
                part_mass_samples_kg=nearest["part_samples"])
            write_json(directory/"nearest_surface_parts.json", nearest["parts"])
            record = {"scene_id": scene, "status": "predicted",
                "legacy_mass_kg": before["moments"]["mass_kg"],
                "conditional_mass_kg": after["moments"]["mass_kg"],
                "conditional_nearest_surface_mass_kg": nearest["moments"]["mass_kg"],
                "changed_owner_fraction": float(np.mean(ids != nearest_ids)),
                "surface_distance_p95_m": float(np.quantile(distances, .95)),
                "volume_m3": before["main_volume_m3"], "pitch_m": pitch,
                "part_mass_change_kg": [a["mass_kg"]["mean"]-b["mass_kg"]["mean"]
                    for a, b in zip(after["parts"], before["parts"], strict=True)]}
        except Exception as exc:  # noqa: BLE001 - retain every requested development object
            record = {"scene_id": scene, "status": "failed", "error": str(exc)}
            (directory/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        records.append(record)
        print(json.dumps(record), flush=True)
    write_json(output/"predictions.json", records)
    sealed = [output/"predictions.json", output/"frozen_spec.json"]
    sealed.extend((output/"objects").glob("*/*.npz"))
    sealed.extend((output/"objects").glob("*/*.json"))
    write_json(output/"prediction_seal.json", {str(p.relative_to(output)): digest(p) for p in sealed})
    # Labels are opened only after all paired predictions have been sealed.
    report = {"requested": len(order), "completed": sum(r["status"] == "predicted" for r in records)}
    if report["completed"] == len(order):
        weights = read_json(root/"filtered_product_weights.json")
        target = [float(weights.get(s, weights.get(s.split("_")[0]))) for s in order]
        for name, key in (("legacy", "legacy_mass_kg"), ("conditional", "conditional_mass_kg"),
                          ("conditional_nearest_surface", "conditional_nearest_surface_mass_kg")):
            report[name] = mass_metrics(target, [r[key] for r in records])
        report["targets_kg"] = dict(zip(order, target, strict=True))
    write_json(output/"report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("data/abo500_ten"))
    parser.add_argument("--config", type=Path, default=Path("configs/ten_object_pilot.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/abo500_ten"))
    parser.add_argument("--stage", choices=("predict", "evaluate", "plot", "all", "audit-shell"), default="all")
    parser.add_argument("--cohort", type=Path, default=Path("configs/five_object_pilot.json"))
    args = parser.parse_args()
    output = ensure_dir(args.output)
    if args.stage == "audit-shell":
        print(json.dumps(compare_shell_models(args.root, args.assets, args.cohort,
                                              args.config, output)), flush=True)
        return
    if args.stage in ("predict", "all"):
        predict(args.root, args.assets, args.config, output)
    if args.stage in ("evaluate", "all"):
        print(json.dumps(evaluate(args.root, output)), flush=True)
    if args.stage in ("plot", "all"):
        plot(args.root, output)


if __name__ == "__main__":
    main()
