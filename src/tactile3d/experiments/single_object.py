"""ABO-500 one-object diagnostic: predict, seal, evaluate, then plot.

No measured mass is accessible to predict_fields. This is an untrained
construction-prior baseline using an original mesh, not a trained final model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
import yaml
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch

from tactile3d.figures.style import configure_style
from tactile3d.mesh.assembly import face_components, texture_face_rgb
from tactile3d.physics.fields import integrate_density_cells, isotropic_mixture_bounds
from tactile3d.utils.io import ensure_dir, read_json, write_json

NAMES = ("Wood base rails", "Rear leg L", "Rear leg R", "Front leg L", "Front leg R",
         "Upholstered base", "Seat top", "Backrest", "Loose cushion", "Arm L", "Arm R")
COLORS = ("#85705A", "#C96A6A", "#D1A14B", "#4FA38F", "#8573B5", "#74A8C6",
          "#5A76A8", "#B78378", "#A2AF69", "#B98DB4", "#D39363")
GROUPS = ("base", "back_and_arms", "cushions")


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def label_chaise(mesh):
    """Object-specific geometric naming of native pieces, not learned semantics."""
    count, ids = face_components(mesh)
    labels = np.zeros(len(mesh.faces), dtype=np.int64)
    pieces = []
    for component in range(count):
        faces = np.flatnonzero(ids == component)
        points = mesh.vertices[np.unique(mesh.faces[faces])]
        low, high = points.min(axis=0), points.max(axis=0)
        center = (low + high) / 2
        if high[1] < 0.24:
            part = 1 + int(center[0] > 0) + 2*int(center[2] > 0) if low[1] < 0.05 else 0
        elif high[1] < 0.40:
            part = 5
        elif high[1] < 0.48:
            part = 6
        elif high[1] < 0.66 and abs(center[0]) > 0.29:
            part = 9 + int(center[0] > 0)
        elif low[2] > -0.79:
            part = 8
        else:
            part = 7
        labels[faces] = part
        pieces.append({"component": component, "part_id": part, "faces": len(faces),
                       "bounds_m": [low.tolist(), high.tolist()]})
    return labels, pieces


def voxel_partition(mesh, labels, pitch, closing=1):
    """Close/fill each semantic envelope; assign overlaps once, by surface proximity.

    Rounded lattice coordinates are shared across all parts. A voxel closure is
    an envelope approximation and does not establish true watertight anatomy.
    """
    candidates, owners, rows = [], [], []
    for part in np.unique(labels):
        local = mesh.submesh([np.flatnonzero(labels == part)], append=True, repair=False)
        surface = np.rint(local.voxelized(pitch).points / pitch).astype(np.int64)
        origin = surface.min(axis=0) - closing - 2
        shape = surface.max(axis=0) - origin + closing + 3
        mask = np.zeros(shape, dtype=bool)
        mask[tuple((surface-origin).T)] = True
        sealed = ndimage.binary_closing(mask, iterations=closing) if closing else mask
        filled = ndimage.binary_fill_holes(sealed)
        # Closing must not silently remove the observed surface cells.
        filled |= mask
        indices = np.argwhere(filled) + origin
        candidates.append(indices)
        owners.append(np.full(len(indices), part, dtype=np.int64))
        rows.append({"part_id": int(part), "observed_surface_cells": len(surface),
                     "envelope_cells_before_overlap_resolution": len(indices),
                     "mesh_watertight": bool(local.is_watertight)})
    all_indices, all_owners = np.concatenate(candidates), np.concatenate(owners)
    unique, inverse, counts = np.unique(all_indices, axis=0, return_inverse=True,
                                         return_counts=True)
    # Compare each candidate to its own part's surface before resolving overlap.
    distance = np.zeros(len(all_indices))
    for part in np.unique(labels):
        selected = all_owners == part
        tree = cKDTree(mesh.triangles_center[labels == part])
        distance[selected] = tree.query(all_indices[selected]*pitch)[0]
    order = np.lexsort((all_owners, distance, inverse))
    _, first = np.unique(inverse[order], return_index=True)
    assigned = all_owners[order[first]]
    audit = {"method": "partwise surface voxelization + closing + hole fill + unique ownership",
             "pitch_m": pitch, "closing_iterations": closing,
             "unique_cells": len(unique), "overlap_cells": int((counts > 1).sum()),
             "disjoint_after_assignment": True, "is_true_interior": False, "parts": rows}
    return unique*pitch, assigned, audit


def visual_hull_partition(mesh, labels, scene, transforms, pitch, alpha_threshold=128,
                          geometric_trim=False):
    """Carve metric cells by all original silhouettes; never infer hidden filling.

    This geometric envelope can overfill unseen concavities. Surface mesh
    anchors assign part hypotheses; all retained cells have a unique owner.
    """
    low = np.floor(mesh.bounds[0]/pitch).astype(int)
    high = np.ceil(mesh.bounds[1]/pitch).astype(int)
    lattice = np.stack(np.meshgrid(*[np.arange(a, b+1) for a, b in zip(low, high, strict=True)],
                                   indexing="ij"), axis=-1).reshape(-1, 3)
    total_candidates = len(lattice)
    active = lattice*pitch
    rows = []
    for index, frame in enumerate(transforms["frames"]):
        world = active[:, [0, 2, 1]] * [1, -1, 1]
        matrix = np.linalg.inv(np.asarray(frame["transform_matrix"]))
        camera = world @ matrix[:3, :3].T + matrix[:3, 3]
        z = -camera[:, 2]
        u = np.rint(transforms["fl_x"]*camera[:, 0]/z + transforms["cx"]).astype(int)
        v = np.rint(-transforms["fl_y"]*camera[:, 1]/z + transforms["cy"]).astype(int)
        alpha = np.asarray(Image.open(scene/frame["file_path"]).convert("RGBA"))[:, :, 3]
        in_view = (z > 0) & (u >= 0) & (u < alpha.shape[1]) & (v >= 0) & (v < alpha.shape[0])
        keep = np.zeros(len(active), dtype=bool)
        keep[in_view] = alpha[v[in_view], u[in_view]] >= alpha_threshold
        active = active[keep]
        rows.append({"view_index": index, "remaining_cells": len(active)})
    if not len(active):
        raise ValueError("No visual-hull cells: audit camera, scale and alpha masks")
    before_trim = len(active)
    if geometric_trim:
        # A visual hull from upper-hemisphere views can fill free space below a
        # chair. Restrict exposed wood to native component boxes; do not extend
        # a high-density rail throughout its unconstrained nearest-face region.
        # This is an explicit object-specific geometric baseline, not a learned
        # manufacturing-part model or an exact arbitrary-shape interior test.
        _, components = face_components(mesh)
        wood_owner = np.full(len(active), -1, dtype=np.int64)
        for component in np.unique(components[labels < 5]):
            face_ids = np.flatnonzero(components == component)
            part = int(np.bincount(labels[face_ids]).argmax())
            if part >= 5:
                continue
            vertices = mesh.vertices[np.unique(mesh.faces[face_ids])]
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            inside = np.all((active >= low) & (active <= high), axis=1)
            wood_owner[inside] = part
        body_low = mesh.triangles[labels >= 5, :, 1].min()
        keep = (active[:, 1] >= body_low) | (wood_owner >= 0)
        active, wood_owner = active[keep], wood_owner[keep]
        soft_faces = np.flatnonzero(labels >= 5)
        distance, nearest = cKDTree(mesh.triangles_center[soft_faces]).query(active)
        ids = labels[soft_faces[nearest]]
        known_wood = wood_owner >= 0
        ids[known_wood] = wood_owner[known_wood]
    else:
        distance, nearest = cKDTree(mesh.triangles_center).query(active)
        ids = labels[nearest]
    return active, ids, {
        "method": "calibrated original multi-view alpha silhouette intersection",
        "pitch_m": pitch, "alpha_threshold": alpha_threshold,
        "initial_bbox_cells": total_candidates, "unique_cells": len(active),
        "input_views": len(rows), "per_view_carving": rows,
        "part_assignment": "native wood component boxes + nearest upholstery surface"
                           if geometric_trim else "nearest original mesh triangle center",
        "geometric_trim": geometric_trim, "cells_before_geometric_trim": before_trim,
        "removed_underseat_cells": before_trim-len(active),
        "surface_distance_q95_m": float(np.quantile(distance, .95)),
        "disjoint_after_assignment": True, "is_true_interior": False,
        "limitation": "visual-hull envelope may retain unseen concavities; not true material volume",
    }


def field_basis(centers, part_ids, pitch, cfg):
    """Linear physical field basis; thin shells use partial cell occupancy."""
    # Coefficients: solid wood, fabric, batting, then wood/foam for 3 groups.
    basis = np.zeros((len(centers), 9), dtype=np.float64)
    basis[part_ids < 5, 0] = 1
    shell = cfg["shell"]
    fabric_t, batting_t = shell["fabric_thickness_m"], shell["batting_thickness_m"]
    group_ids = np.full(len(centers), -1, dtype=np.int64)
    for part in range(5, len(NAMES)):
        selected = np.flatnonzero(part_ids == part)
        if not len(selected):
            continue
        points = centers[selected]
        ijk = np.rint(points/pitch).astype(np.int64)
        origin = ijk.min(axis=0)-2
        mask = np.zeros(ijk.max(axis=0)-origin+3, dtype=bool)
        mask[tuple((ijk-origin).T)] = True
        depth = ndimage.distance_transform_edt(mask)[tuple((ijk-origin).T)] * pitch
        inner_edge = np.maximum(depth-pitch, 0)
        fabric = np.clip((fabric_t-inner_edge)/pitch, 0, 1)
        batting = np.clip((fabric_t+batting_t-inner_edge)/pitch, 0, 1)-fabric
        core = 1-fabric-batting
        low, high = points.min(axis=0), points.max(axis=0)
        axes = np.argsort(high-low)[-2:]
        edge_distance = np.minimum(points[:, axes]-low[axes], high[axes]-points[:, axes])
        frame_shape = 0.5 + np.exp(-edge_distance.min(axis=1)
                                  / shell["frame_localization_width_m"])
        group = 0 if part == 5 else (2 if part in (6, 8) else 1)
        group_ids[selected] = group
        basis[selected, 1], basis[selected, 2] = fabric, batting
        basis[selected, 3+2*group] = core*frame_shape
        basis[selected, 4+2*group] = core
    return basis, group_ids


def construction_samples(cfg):
    rng = np.random.default_rng(cfg["seed"])
    n = cfg["samples"]
    material_rho = np.stack([rng.uniform(*cfg["materials"][name]["density_range"], n)
                             for name in cfg["material_order"]], axis=1)
    wood, foam = [], []
    for group in GROUPS:
        definition = cfg["construction"][group]
        values = []
        for key in ("wood_fraction_triangular", "foam_fraction_triangular"):
            low, mode, high = definition[key]
            values.append(np.full(n, mode) if low == high else rng.triangular(low, mode, high, n))
        wood.append(values[0])
        foam.append(values[1])
    wood, foam = np.asarray(wood).T, np.asarray(foam).T
    if np.any(1.5*wood+foam > 1+1e-10):
        raise ValueError("Construction prior can exceed unit occupancy")
    coefficients = np.column_stack([material_rho[:, 0], material_rho[:, 2],
                                    material_rho[:, 3]])
    for group in range(3):
        coefficients = np.column_stack([coefficients, material_rho[:, 0]*wood[:, group],
                                        material_rho[:, 1]*foam[:, group]])
    return material_rho, wood, foam, coefficients


def predict_fields(centers, part_ids, pitch, cfg):
    """Predict without any target mass argument or dataset-root access."""
    basis, _ = field_basis(centers, part_ids, pitch, cfg)
    rho, wood, foam, coefficients = construction_samples(cfg)
    means = coefficients.mean(axis=0)
    density = basis @ means
    covariance = np.cov(coefficients, rowvar=False)
    sigma = np.sqrt(np.maximum(np.sum((basis @ covariance)*basis, axis=1), 0))
    volumes = np.zeros((len(NAMES), basis.shape[1]))
    np.add.at(volumes, part_ids, basis*pitch**3)
    part_samples = volumes @ coefficients.T
    alpha = np.zeros((len(centers), 5))
    alpha[:, 0], alpha[:, 2], alpha[:, 3] = basis[:, 0], basis[:, 1], basis[:, 2]
    for group in range(3):
        alpha[:, 0] += basis[:, 3+2*group]*wood[:, group].mean()
        alpha[:, 1] += basis[:, 4+2*group]*foam[:, group].mean()
    alpha[:, 4] = 1-alpha[:, :4].sum(axis=1)
    if alpha.min() < -1e-10:
        raise RuntimeError("Negative material occupancy")
    e = [cfg["materials"][k]["youngs_pa"] for k in cfg["material_order"]]
    nu = [cfg["materials"][k]["poisson"] for k in cfg["material_order"]]
    lower, upper = isotropic_mixture_bounds(alpha, e, nu)
    moments = integrate_density_cells(centers, density, pitch, part_ids)
    if not np.isclose(moments["mass_kg"], part_samples.sum(axis=0).mean()):
        raise RuntimeError("Sampled and integrated mass do not agree")
    # Controlled assumptions on the exact same geometry, not published baselines.
    wood_volume = float((part_ids < 5).sum()*pitch**3)
    soft_volume = float((part_ids >= 5).sum()*pitch**3)
    scenarios = {
        "Uniform fabric solid": len(centers)*pitch**3*rho[:, 2],
        "Wood + foam solid": wood_volume*rho[:, 0] + soft_volume*rho[:, 1],
        "Layered construction prior": part_samples.sum(axis=0),
    }
    return {"density": density, "sigma": sigma, "fractions": alpha, "E_lower": lower,
            "E_upper": upper, "part_samples": part_samples, "moments": moments,
            "scenarios": scenarios, "material_density_samples": rho}


def stats(values):
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95))}


def project_vertices(mesh, c2w, transforms, resolution):
    # Fixed glTF Y-up -> ABO-500 Z-up rotation; no fitted scale or mass correction.
    world = mesh.vertices[:, [0, 2, 1]] * [1, -1, 1]
    camera = np.c_[world, np.ones(len(world))] @ np.linalg.inv(c2w).T
    z = -camera[:, 2]
    sx, sy = resolution/transforms["w"], resolution/transforms["h"]
    uv = np.column_stack([(transforms["fl_x"]*camera[:, 0]/z+transforms["cx"])*sx,
                          (-transforms["fl_y"]*camera[:, 1]/z+transforms["cy"])*sy])
    return uv, z


def silhouette_audit(mesh, scene, transforms, resolution=256):
    rows = []
    for index, frame in enumerate(transforms["frames"]):
        uv, depth = project_vertices(mesh, np.array(frame["transform_matrix"]),
                                     transforms, resolution)
        canvas = Image.new("L", (resolution, resolution))
        draw = ImageDraw.Draw(canvas)
        for face in mesh.faces[np.all(depth[mesh.faces] > 0, axis=1)]:
            draw.polygon([tuple(v) for v in uv[face]], fill=255)
        image = Image.open(scene/frame["file_path"]).convert("RGBA")
        target = np.asarray(image.getchannel("A").resize((resolution, resolution),
                                                        Image.Resampling.NEAREST)) > 0
        predicted = np.asarray(canvas) > 0
        union = (target | predicted).sum()
        rows.append({"view_index": index, "image": frame["file_path"],
                     "silhouette_IoU": float((target & predicted).sum()/max(union, 1)),
                     "image_sha256": digest(scene/frame["file_path"]),
                     "width_px": image.width, "height_px": image.height})
    return rows


def raster_face_ids(mesh, c2w, transforms, resolution=650):
    """Perspective-correct depth test; integer face IDs for scientific maps."""
    uv, depth = project_vertices(mesh, c2w, transforms, resolution)
    ids = np.full((resolution, resolution), -1, dtype=np.int32)
    zbuffer = np.full((resolution, resolution), np.inf)
    for face_id, face in enumerate(mesh.faces):
        if np.any(depth[face] <= 0):
            continue
        tri = uv[face]
        low = np.maximum(np.floor(tri.min(axis=0)).astype(int), 0)
        high = np.minimum(np.ceil(tri.max(axis=0)).astype(int)+1, resolution)
        if np.any(high <= low):
            continue
        x, y = np.meshgrid(np.arange(low[0], high[0])+.5, np.arange(low[1], high[1])+.5)
        a, b, c = tri
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-9:
            continue
        w0 = ((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den
        w1 = ((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den
        w2 = 1-w0-w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        reciprocal = w0/depth[face[0]]+w1/depth[face[1]]+w2/depth[face[2]]
        z = np.divide(1, reciprocal, out=np.full_like(reciprocal, np.inf), where=reciprocal > 0)
        current = zbuffer[low[1]:high[1], low[0]:high[0]]
        update = inside & (z < current)
        current[update] = z[update]
        ids[low[1]:high[1], low[0]:high[0]][update] = face_id
    return ids


def predict(root, mesh_path, config_path, output):
    started = time.perf_counter()
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    scene_id = cfg["scene_id"]
    split_path = root/"splits.json"
    splits = read_json(split_path)
    if scene_id not in splits["test"] or splits["test"][0] != scene_id:
        raise ValueError("This preregistered pilot requires the first released test object")
    scene = root/"scenes"/scene_id
    transforms = read_json(scene/"transforms.json")
    info = read_json(scene/"info.json")
    loaded = trimesh.load(mesh_path, process=True)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    mesh.apply_scale(cfg["mesh_unit_to_meter"])
    labels, pieces = label_chaise(mesh)
    spec = {"config": cfg, "config_sha256": digest(config_path),
            "mesh_sha256": digest(mesh_path), "split_sha256": digest(split_path),
            "camera_sha256": digest(scene/"transforms.json"),
            "cached_language_proposal_sha256": digest(scene/"info.json"),
            "code_sha256": digest(__file__), "dataset_root": str(root.resolve()),
            "mesh_path": str(mesh_path.resolve()), "prediction_uses_target_mass": False,
            "selection_status": "first test ID; no mass-based sample selection",
            "language_source": "released NeRF2Physics info.json, not a fresh Qwen inference",
            "caption": info["caption"], "platform": platform.platform(),
            "versions": {"numpy": np.__version__, "trimesh": trimesh.__version__,
                         "matplotlib": matplotlib.__version__}}
    write_json(output/"frozen_experiment_spec.json", spec)
    geometry = {"faces": len(mesh.faces), "vertices": len(mesh.vertices),
                "watertight": bool(mesh.is_watertight), "is_volume": bool(mesh.is_volume),
                "bounds_m": mesh.bounds.tolist(), "native_pieces": len(pieces),
                "part_label_source": "object-specific geometric rules on native pieces",
                "exact_enclosed_volume_m3": float(mesh.volume) if mesh.is_volume else None,
                "coordinate_frame": "original glTF asset; Y up, length unit metre",
                "projection_transform": "(x,y,z) -> (x,-z,y), fixed unit scale"}
    write_json(output/"geometry_and_parts.json", {**geometry, "native_pieces": pieces})
    np.save(output/"face_labels.npy", labels)
    colored = mesh.copy()
    colored.visual = trimesh.visual.ColorVisuals(mesh=colored,
        face_colors=colors.to_rgba_array(COLORS)[labels])
    colored.export(output/"part_labels.ply")
    views = silhouette_audit(mesh, scene, transforms)
    write_csv(output/"multiview_alignment.csv", views)
    convergences, main = [], None
    for pitch in cfg["convergence_pitches_m"]:
        tick = time.perf_counter()
        if cfg.get("volume_method", "").endswith("multiview_visual_hull"):
            centers, ids, audit = visual_hull_partition(
                mesh, labels, scene, transforms, pitch, cfg["alpha_threshold"],
                geometric_trim=cfg["volume_method"].startswith("geometry_trimmed"))
        else:
            centers, ids, audit = voxel_partition(mesh, labels, pitch, cfg["closing_iterations"])
        fields = predict_fields(centers, ids, pitch, cfg)
        moments = fields["moments"]
        convergences.append({"pitch_m": pitch, "unique_cells": len(centers),
                             "volume_m3": len(centers)*pitch**3,
                             "mass_kg": moments["mass_kg"],
                             "com_x_m": moments["center_of_mass_m"][0],
                             "com_y_m": moments["center_of_mass_m"][1],
                             "com_z_m": moments["center_of_mass_m"][2],
                             "seconds": time.perf_counter()-tick})
        print(json.dumps({"prediction_progress": convergences[-1]}), flush=True)
        if np.isclose(pitch, cfg["main_pitch_m"]):
            main = (centers, ids, audit, fields)
    if main is None:
        raise ValueError("Main pitch must be among convergence pitches")
    centers, ids, audit, fields = main
    np.savez_compressed(output/"physical_field.npz", centers_m=centers, part_ids=ids,
        pitch_m=cfg["main_pitch_m"], density_mean_kg_m3=fields["density"],
        density_std_kg_m3=fields["sigma"], material_fractions=fields["fractions"],
        E_reuss_lower_pa=fields["E_lower"], E_voigt_upper_pa=fields["E_upper"],
        part_mass_samples_kg=fields["part_samples"])
    write_json(output/"voxel_envelope_audit.json", audit)
    write_csv(output/"resolution_convergence.csv", convergences)
    part_rows = []
    for part, name in enumerate(NAMES):
        selected = ids == part
        if not selected.any():
            continue
        part_rows.append({"part_id": part, "part": name,
            "envelope_volume_m3": selected.sum()*cfg["main_pitch_m"]**3,
            "density_mean_kg_m3": float(fields["density"][selected].mean()),
            "occupied_fraction": float(1-fields["fractions"][selected, 4].mean()),
            **{f"mass_{key}_kg": val for key, val in stats(fields["part_samples"][part]).items()},
            "friction": "not predicted: contact pair unmeasured",
            "E_status": "idealized bounds from assumed phase constants; no measured E",
            "part_ground_truth": "unavailable"})
    write_csv(output/"part_physical_parameters.csv", part_rows)
    scenario_stats = {key: stats(value) for key, value in fields["scenarios"].items()}
    prediction = {"scene_id": scene_id, "split": "test", "n_objects": 1,
        "geometry": geometry, "status": cfg["status"], "scenarios": scenario_stats,
        "physical_moments": fields["moments"],
        "mean_silhouette_IoU": float(np.mean([r["silhouette_IoU"] for r in views])),
        "uncertainty": "uncalibrated prior predictive, fixed shell thickness and geometry",
        "local_accuracy_metrics": None,
        "local_accuracy_unavailable_reason": "ABO-500 has no local measured physical labels",
        "prediction_seconds": time.perf_counter()-started}
    write_json(output/"prediction.json", prediction)
    # Seal every input to evaluation. Reading mass labels happens only afterwards.
    sealed_names = ["prediction.json", "physical_field.npz", "part_physical_parameters.csv",
                    "resolution_convergence.csv", "frozen_experiment_spec.json", "face_labels.npy"]
    write_json(output/"prediction_seal.json", {name: digest(output/name) for name in sealed_names})
    return prediction


def evaluate(root, output):
    seal = read_json(output/"prediction_seal.json")
    if any(digest(output/name) != checksum for name, checksum in seal.items()):
        raise ValueError("A sealed prediction artifact was modified")
    prediction = read_json(output/"prediction.json")
    weights = read_json(root/"filtered_product_weights.json")
    scene_id = prediction["scene_id"]
    target = weights.get(scene_id, weights.get(scene_id.split("_")[0]))
    if not isinstance(target, (int, float)) or target <= 0:
        raise ValueError("Pilot requires a positive scalar released kg mass label")
    rows = []
    for method, values in prediction["scenarios"].items():
        mass = values["mean"]
        rows.append({"method": method, "n_objects": 1, "predicted_mass_kg": mass,
            "target_mass_kg": target, "q05_kg": values["q05"], "q95_kg": values["q95"],
            "ADE_kg": abs(mass-target), "ALDE": abs(float(np.log(mass/target))),
            "APE_percent": 100*abs(mass-target)/target,
            "MnRE": min(mass/target, target/mass),
            "target_inside_prior_interval": values["q05"] <= target <= values["q95"]})
    write_csv(output/"mass_metrics.csv", rows)
    write_json(output/"evaluation.json", {"target_mass_kg": target,
        "target_source": "released ABO-500 filtered_product_weights.json, catalogue-derived",
        "target_file_sha256": digest(root/"filtered_product_weights.json"), "metrics": rows,
        "target_used_during_prediction": False, "prediction_seal_verified": True,
        "published_baselines": {"NeRF2Physics": "not executed",
                                "PUGS": "not executed", "PhysVGGT": "not executed"},
        "single_case_interval_hit_is_not_calibration": True,
        "claim": "single-object numerical diagnostic; no benchmark superiority established"})
    return rows


def save(fig, output, stem):
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output/f"{stem}.{suffix}", dpi=240, facecolor="white")
    plt.close(fig)


def shade_image(ids, face_colors):
    result = np.ones((*ids.shape, 3))
    visible = ids >= 0
    result[visible] = np.asarray(face_colors)[ids[visible], :3]
    return result


def crop_to_object(array, reference_mask, padding=.07):
    yy, xx = np.nonzero(reference_mask)
    low = np.array([xx.min(), yy.min()], float)
    high = np.array([xx.max()+1, yy.max()+1], float)
    margin = (high-low)*padding
    low, high = (low-margin)/reference_mask.shape[1::-1], (high+margin)/reference_mask.shape[1::-1]
    scale = np.array(array.shape[1::-1])
    lo = np.maximum((low*scale).astype(int), 0)
    hi = np.minimum(np.ceil(high*scale).astype(int), scale)
    return array[lo[1]:hi[1], lo[0]:hi[0]]


def sample_display_field(ids, mesh, c2w, transforms, labels, field, values):
    """Sample volume values at ray/triangle intersections, for display only.

    Four-neighbour interpolation within the same hypothesized part avoids
    assigning a single color to a large triangle. Numerical integration uses
    the unmodified volume cells, not these display samples.
    """
    yy, xx = np.nonzero(ids >= 0)
    face_ids = ids[yy, xx]
    camera_rays = np.column_stack([
        ((xx+.5)*transforms["w"]/ids.shape[1]-transforms["cx"])/transforms["fl_x"],
        -((yy+.5)*transforms["h"]/ids.shape[0]-transforms["cy"])/transforms["fl_y"],
        -np.ones(len(xx)),
    ])
    world_rays = camera_rays @ c2w[:3, :3].T
    rays = world_rays[:, [0, 2, 1]] * [1, 1, -1]
    camera = c2w[:3, 3][[0, 2, 1]] * [1, 1, -1]
    normals = mesh.face_normals[face_ids]
    numerator = np.einsum("ij,ij->i", normals, mesh.triangles[face_ids, 0]-camera)
    denominator = np.einsum("ij,ij->i", normals, rays)
    positions = camera + rays*(numerator/denominator)[:, None]
    sampled = np.zeros((len(xx), values.shape[1]))
    for part in np.unique(labels[face_ids]):
        selected = labels[face_ids] == part
        valid_cells = field["part_ids"] == part
        if not valid_cells.any():
            raise ValueError(f"No volume support for displayed part {part}")
        distance, nearest = cKDTree(field["centers_m"][valid_cells]).query(positions[selected], k=4)
        weights = 1/np.maximum(distance, 1e-8)**2
        weights /= weights.sum(axis=1, keepdims=True)
        sampled[selected] = np.einsum("ij,ijk->ik", weights, values[valid_cells][nearest])
    return yy, xx, sampled


def plot(root, mesh_path, output):
    configure_style()
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})
    spec = read_json(output/"frozen_experiment_spec.json")
    cfg = spec["config"]
    scene = root/"scenes"/cfg["scene_id"]
    transforms = read_json(scene/"transforms.json")
    loaded = trimesh.load(mesh_path, process=True)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    mesh.apply_scale(cfg["mesh_unit_to_meter"])
    labels = np.load(output/"face_labels.npy")
    field = np.load(output/"physical_field.npz")
    selected_views = (cfg["display_view_index"], 0)
    maps = [raster_face_ids(mesh, np.array(transforms["frames"][i]["transform_matrix"]),
                            transforms) for i in selected_views]
    face_rgb = colors.to_rgba_array(COLORS)[labels, :3]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for row, (index, ids) in enumerate(zip(selected_views, maps, strict=True)):
        image = Image.open(scene/transforms["frames"][index]["file_path"]).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        reference_mask = np.asarray(image)[:, :, 3] >= cfg["alpha_threshold"]
        axes[row, 0].imshow(crop_to_object(np.asarray(background.convert("RGB")), reference_mask))
        axes[row, 0].set_title(f"{'a' if row == 0 else 'd'}  Dataset RGB | view {index:02d}", loc="left")
        axes[row, 1].imshow(crop_to_object(shade_image(ids, texture_face_rgb(mesh)), reference_mask))
        axes[row, 1].set_title(f"{'b' if row == 0 else 'e'}  Matched original mesh", loc="left")
        axes[row, 2].imshow(crop_to_object(shade_image(ids, face_rgb), reference_mask))
        axes[row, 2].set_title(f"{'c' if row == 0 else 'f'}  Geometric part hypotheses", loc="left")
    for ax in axes.flat:
        ax.axis("off")
    handles = [Patch(color=COLORS[i], label=name) for i, name in enumerate(NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(.5, .055), fontsize=8)
    fig.subplots_adjust(left=.02, right=.98, top=.94, bottom=.20, wspace=.04, hspace=.05)
    fig.text(.5, .024, f"ABO-500 test: {cfg['scene_id']} | Native-piece geometric rules; no part ground truth or learned segmentation.",
             ha="center", fontsize=8, color="#555555")
    save(fig, output, "01_multiview_parts")

    density = field["density_mean_kg_m3"]
    cv = field["density_std_kg_m3"] / np.maximum(density, 1e-9)
    all_values = np.column_stack([density, 1-field["material_fractions"][:, 4],
                                 field["material_fractions"][:, 0], cv, field["E_voigt_upper_pa"]])
    frame = transforms["frames"][selected_views[0]]
    yy, xx, sampled = sample_display_field(maps[0], mesh, np.array(frame["transform_matrix"]),
                                           transforms, labels, field, all_values)

    def field_image(column, cmap, norm):
        rgb = np.ones((*maps[0].shape, 3))
        rgb[yy, xx] = plt.get_cmap(cmap)(norm(sampled[:, column]))[:, :3]
        return crop_to_object(rgb, maps[0] >= 0)

    items = [
        (density, "a  Effective density (kg m$^{-3}$)", "viridis", colors.Normalize(0, 700)),
        (1-field["material_fractions"][:, 4], "b  Material occupancy (fraction)", "cividis", colors.Normalize(0, 1)),
        (field["material_fractions"][:, 0], "c  Wood occupancy hypothesis (fraction)", "YlOrBr", colors.Normalize(0, 1)),
        (cv, "d  Density prior CV (unitless)", "magma", colors.Normalize(0, .5)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.8))
    for column, (ax, (_, title, cmap, norm)) in enumerate(zip(axes.flat, items, strict=False)):
        ax.imshow(field_image(column, cmap, norm))
        ax.axis("off")
        ax.set_title(title, loc="left")
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=.035, pad=.015,
                     shrink=.72)
    norm = colors.LogNorm(1e5, 1e10)
    axes[1, 1].imshow(field_image(4, "plasma", norm))
    axes[1, 1].axis("off")
    axes[1, 1].set_title("e  Idealized Voigt E upper bound (Pa)", loc="left")
    fig.colorbar(ScalarMappable(norm=norm, cmap="plasma"), ax=axes[1, 1], fraction=.035,
                 pad=.015, shrink=.72)
    points = field["centers_m"]
    sagittal = np.abs(points[:, 0]) <= cfg["main_pitch_m"]*.51
    axes[1, 2].scatter(points[sagittal, 2], points[sagittal, 1],
                       c=density[sagittal], s=5, marker="s", cmap="viridis", vmin=0, vmax=700,
                       linewidths=0, rasterized=True)
    axes[1, 2].set(xlabel="Length z (m)", ylabel="Height y (m)", aspect="equal")
    axes[1, 2].set_title("f  Hypothesized interior | x = 0", loc="left")
    fig.subplots_adjust(left=.04, right=.98, top=.93, bottom=.12, hspace=.2, wspace=.22)
    fig.text(.5, .025, "Material/structure priors are untrained and uncalibrated. Interior is a hypothesis. E bounds assume ideal isotropic phases;\n"
             "they are not measured effective modulus or gripper contact stiffness. Friction/hardness accuracy is unavailable.",
             ha="center", fontsize=8, color="#555555")
    save(fig, output, "02_physical_fields")

    evaluation = read_json(output/"evaluation.json")
    metrics = {row["method"]: row for row in evaluation["metrics"]}
    names = ["Uniform fabric solid", "Wood + foam solid", "Layered construction prior"]
    values = [metrics[name] for name in names]
    samples = field["part_mass_samples_kg"]
    target = evaluation["target_mass_kg"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    means = np.array([row["predicted_mass_kg"] for row in values])
    low = means-np.array([row["q05_kg"] for row in values])
    high = np.array([row["q95_kg"] for row in values])-means
    axes[0, 0].bar(np.arange(3), means, color=["#AAAAAA", "#5A76A8", "#4FA38F"],
                    yerr=np.stack([low, high]), capsize=4, alpha=.8)
    axes[0, 0].set_xticks(np.arange(3), ["Fabric solid", "Wood + foam", "Layered prior"])
    axes[0, 0].axhline(target, color="#C96A6A", label=f"Released label: {target:.2f} kg")
    axes[0, 0].set_ylabel("Object mass (kg)")
    axes[0, 0].set_title("a  Controlled construction assumptions", loc="left")
    axes[0, 0].legend(loc="upper right")
    axes[0, 1].hist(samples.sum(axis=0), bins=40, density=True, color="#4FA38F", alpha=.8)
    axes[0, 1].axvline(target, color="#C96A6A", label="Released label")
    axes[0, 1].set(xlabel="Object mass (kg)", ylabel="Prior density (kg$^{-1}$)")
    axes[0, 1].set_title("b  Uncalibrated prior-predictive distribution", loc="left")
    axes[0, 1].legend()
    means = samples.mean(axis=1)
    low, high = np.quantile(samples, [.05, .95], axis=1)
    axes[1, 0].errorbar(means, np.arange(len(NAMES)), xerr=[means-low, high-means],
                        fmt="o", color="#5A76A8", capsize=3, markersize=4)
    axes[1, 0].set_yticks(np.arange(len(NAMES)), NAMES, fontsize=8)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Part mass (kg), mean and 90% prior interval")
    axes[1, 0].set_title("c  Mass decomposition without part mass labels", loc="left")
    axes[1, 1].axis("off")
    rows = [[label, f"{r['ADE_kg']:.2f}", f"{r['ALDE']:.3f}",
              f"{r['APE_percent']:.1f}", f"{r['MnRE']:.3f}"]
             for label, r in zip(["Fabric solid", "Wood + foam", "Layered prior"], values, strict=True)]
    table = axes[1, 1].table(cellText=rows,
        colLabels=["Assumption", "ADE kg", "ALDE", "APE %", "MnRE"],
        colWidths=[.32, .17, .17, .17, .17], cellLoc="center", bbox=(0, .43, 1, .42))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#D8DDE2")
        if row == 0:
            cell.set_facecolor("#EEF2F6")
    axes[1, 1].set_title("d  One-object evaluation | N = 1", loc="left")
    axes[1, 1].text(0, .18, "NeRF2Physics / PUGS / PhysVGGT: not executed.\n"
                            "No published aggregate scores are substituted here.", fontsize=9)
    fig.subplots_adjust(left=.14, right=.98, top=.93, bottom=.13, hspace=.43, wspace=.32)
    fig.text(.5, .025, "Predictions sealed before loading the released mass label. No target-mass scaling or sample selection by error.\n"
             "All comparators above are controlled assumptions, not paper reproductions. One object cannot establish benchmark superiority.",
             ha="center", fontsize=8, color="#555555")
    save(fig, output, "03_mass_evaluation")

    with (output/"resolution_convergence.csv").open(encoding="utf-8") as stream:
        resolution = sorted(csv.DictReader(stream), key=lambda r: float(r["pitch_m"]))
    with (output/"multiview_alignment.csv").open(encoding="utf-8") as stream:
        align = list(csv.DictReader(stream))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    pitches = [1000*float(r["pitch_m"]) for r in resolution]
    axes[0].plot(pitches, [float(r["volume_m3"]) for r in resolution], "o-", color="#5A76A8")
    axes[0].set(xlabel="Voxel pitch (mm)", ylabel="Geometric envelope volume (m$^3$)")
    axes[0].set_title("a  Discretization sensitivity", loc="left")
    axes[1].plot(pitches, [float(r["mass_kg"]) for r in resolution], "o-", color="#4FA38F")
    axes[1].axhline(target, color="#C96A6A", label="Released mass label")
    axes[1].set(xlabel="Voxel pitch (mm)", ylabel="Prior mean mass (kg)")
    axes[1].set_title("b  Geometry-to-mass sensitivity", loc="left")
    axes[1].legend()
    axes[2].plot([int(r["view_index"]) for r in align],
                  [float(r["silhouette_IoU"]) for r in align], "o-", markersize=3, color="#8573B5")
    axes[2].set(xlabel="Dataset view index", ylabel="Silhouette IoU", ylim=(0, 1.05))
    axes[2].set_title("c  Fixed-transform correspondence", loc="left")
    fig.subplots_adjust(left=.07, right=.98, bottom=.26, top=.88, wspace=.37)
    fig.text(.5, .045, "Envelope closure is approximate; this tests numerical sensitivity, not true hidden volume.\n"
             "Silhouette IoU tests geometric correspondence only, not part segmentation or local physical accuracy.",
             ha="center", fontsize=8, color="#555555")
    save(fig, output, "04_numerical_audit")
    return {"figure_stems": ["01_multiview_parts", "02_physical_fields",
                              "03_mass_evaluation", "04_numerical_audit"],
            "formats": ["svg", "pdf", "png"], "n_objects": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/single_object_chaise.yaml"))
    parser.add_argument("--output", type=Path, default=Path("outputs/abo500_single_object"))
    parser.add_argument("--stage", choices=("predict", "evaluate", "plot", "all"), default="all")
    args = parser.parse_args()
    output = ensure_dir(args.output)
    if args.stage in ("predict", "all"):
        print(json.dumps(predict(args.root, args.mesh, args.config, output)), flush=True)
    if args.stage in ("evaluate", "all"):
        print(json.dumps(evaluate(args.root, output)), flush=True)
    if args.stage in ("plot", "all"):
        print(json.dumps(plot(args.root, args.mesh, output)), flush=True)


if __name__ == "__main__":
    main()
