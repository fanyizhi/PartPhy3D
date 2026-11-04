"""Single-object, untrained Part Token and canonical mapping experiment."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from PIL import Image
from scipy.sparse import coo_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tactile3d.figures.abo_pilot import (
    ABO_SOFA,
    PARTS,
    _construction_posterior,
    _geometric_part_labels,
    _load_mesh,
    _stats,
)
from tactile3d.figures.style import PALETTE, configure_style
from tactile3d.mesh.assembly import texture_face_rgb
from tactile3d.physics.unified import CanonicalPartState, moment_disagreement
from tactile3d.utils.io import ensure_dir, write_json


def refine_geometry_labels(mesh: trimesh.Trimesh) -> tuple[np.ndarray, dict]:
    """Geometry-seeded normal-aware diffusion, with small-component voting."""
    initial = _geometric_part_labels(mesh)
    adjacent = mesh.face_adjacency
    angles = mesh.face_adjacency_angles
    weights = np.exp(-(angles / 0.35) ** 2) + 0.01
    rows = np.r_[adjacent[:, 0], adjacent[:, 1]]
    cols = np.r_[adjacent[:, 1], adjacent[:, 0]]
    graph = coo_matrix((np.r_[weights, weights], (rows, cols)),
                       shape=(len(mesh.faces), len(mesh.faces))).tocsr()
    transition = diags(1 / np.maximum(np.asarray(graph.sum(axis=1)).ravel(), 1e-12)) @ graph
    unary = np.eye(len(PARTS))[initial]
    scores = unary.copy()
    for _ in range(25):
        scores = 0.35 * unary + 0.65 * (transition @ scores)
    result = scores.argmax(axis=1)
    count, components = connected_components(graph, directed=False)
    sizes = np.bincount(components)
    # Small disconnected ornaments and cushion patches vote as intact pieces.
    votes = np.zeros((count, len(PARTS)))
    np.add.at(votes, (components, initial), mesh.area_faces)
    for component in np.flatnonzero(sizes <= 1500):
        mask = components == component
        result[mask] = votes[component].argmax()
    return result, {
        "source": "geometry seeds + normal-weighted graph diffusion + component voting",
        "connected_components": count,
        "changed_area_fraction": float(mesh.area_faces[result != initial].sum() / mesh.area),
        "ground_truth_segmentation_available": False,
    }


def _state_from_mesh(mesh, reference_tree, reference_labels, densities, pitch):
    voxels = mesh.voxelized(pitch=pitch).fill()
    points = np.asarray(voxels.points)
    _, index = reference_tree.query(points)
    return CanonicalPartState(points, reference_labels[index], pitch, densities)


def run_token_pilot(asset_dir, output_dir, *, samples=4000, seed=17, pitch=0.04):
    output = ensure_dir(output_dir)
    assets = Path(asset_dir)
    mesh = _load_mesh(assets / "model.glb")
    labels, segmentation = refine_geometry_labels(mesh)
    tree = cKDTree(mesh.triangles_center)
    volume_proxy = abs(float(mesh.volume))
    prior, raw = _construction_posterior(volume_proxy, samples=samples, seed=seed)
    assumed_envelopes = volume_proxy * np.asarray([p["envelope_share"] for p in PARTS])
    density_samples = raw["part_mass"] / assumed_envelopes[:, None]
    densities = density_samples.mean(axis=1)
    state = _state_from_mesh(mesh, tree, labels, densities, pitch)
    part_volumes = state.part_volumes_m3
    actual_samples = part_volumes[:, None] * density_samples
    object_samples = actual_samples.sum(axis=0)
    reference = state.integrate()
    tokens = _make_tokens(mesh, labels, state, prior, actual_samples)
    np.save(output / "reference_face_labels.npy", labels)
    np.savez_compressed(output / "canonical_state.npz", centers_m=state.centers_m,
                        part_ids=state.part_ids, pitch_m=pitch,
                        density_kg_m3=densities, part_mass_samples_kg=actual_samples)
    rows, lods = [], []
    diagonal = float(np.linalg.norm(mesh.extents))
    for target in (25000, 12500, 5000, 2500):
        lod = mesh.simplify_quadric_decimation(face_count=target)
        independent = _state_from_mesh(lod, tree, labels, densities, pitch)
        # The canonical quadrature is shared, so invariance is architectural.
        # This is a data-flow check, not an independently predicted volume field.
        mapped_labels, distances = state.face_lookup(lod.triangles_center)
        naive = independent.integrate()
        shared = state.integrate()
        shared_errors = moment_disagreement(reference, shared, diagonal)
        rows.append({
            "faces": len(lod.faces),
            "independent_volume_m3": float(independent.part_volumes_m3.sum()),
            "independent_mass_kg": naive["mass_kg"],
            "canonical_mass_kg": shared["mass_kg"],
            "lookup_distance_q95_m": float(np.quantile(distances, 0.95)),
            **moment_disagreement(reference, naive, diagonal),
            "canonical_mass_drift_percent": shared_errors["mass_error_percent"],
            "canonical_com_error_mm": shared_errors["com_error_mm"],
            "canonical_inertia_tensor_error_percent": shared_errors["inertia_tensor_error_percent"],
        })
        lods.append((lod, mapped_labels))
    lods[2][0].export(output / "sofa_5k.ply")
    np.save(output / "sofa_5k_labels.npy", lods[2][1])
    solid_densities = densities / np.asarray([p["fill_ratio"]["mean"] for p in prior["parts"]])
    reference_mass = float(ABO_SOFA["catalog_mass_kg"])
    comparisons = [
        ("Uniform fabric / solid", float(part_volumes.sum() * 350)),
        ("Part mixture / solid", float(part_volumes @ solid_densities)),
        ("Layered / canonical", reference["mass_kg"]),
    ]
    mass_rows = [{"method": name, "mass_kg": mass,
                  "ADE_kg": abs(mass - reference_mass),
                  "ALDE": abs(float(np.log(mass / reference_mass))),
                  "APE_percent": 100 * abs(mass - reference_mass) / reference_mass,
                  "MnRE": min(mass / reference_mass, reference_mass / mass)}
                 for name, mass in comparisons]
    report = {
        "experiment": "untrained unified part state and canonical LOD lookup",
        "object_id": ABO_SOFA["object_id"], "seed": seed, "samples": samples,
        "source": "Official ABO example; membership in ABO-500 is not established",
        "trained_latent_projection": False, "vlm_or_sam_executed": False,
        "geometry": {"watertight": bool(mesh.is_watertight),
                     "pitch_m": pitch, "cells": len(state.centers_m),
                     "voxel_envelope_proxy_m3": float(part_volumes.sum()),
                     "signed_volume_proxy_m3": volume_proxy},
        "segmentation": segmentation,
        "canonical_moments": reference,
        "mass_prior_predictive_kg": _stats(object_samples),
        "catalog_mass_kg": reference_mass, "target_used_for_inference": False,
        "tokens": tokens, "lod_comparison": rows, "mass_comparison": mass_rows,
        "limitations": [
            "Geometry-seeded segmentation has no part ground truth; not SAM/VLM inference.",
            "Open source mesh: filled voxel envelope is an approximation, not measured volume.",
            "Internal fractions and physics are inherited hand-specified sofa priors.",
            "E and nu are illustrative mixture descriptors, not measured homogenized elasticity.",
            "Intervals are uncalibrated prior-predictive intervals.",
            "Zero canonical LOD drift follows from reusing the same volume state by design.",
            "No learned RGB/geometry latent alignment or boundary-locked QEM is implemented.",
        ],
    }
    write_json(output / "report.json", report)
    write_json(output / "part_tokens.json", tokens)
    _csv(output / "lod_metrics.csv", rows)
    _csv(output / "mass_metrics.csv", mass_rows)
    _csv(output / "part_parameters.csv", [_parameter_row(t) for t in tokens])
    _fields(mesh, labels, assets / "product.jpg", tokens, output)
    _mass(object_samples, actual_samples, mass_rows, reference_mass, output)
    _mapping(mesh, labels, lods[2], rows, output)
    return report


def _make_tokens(mesh, labels, state, prior, samples):
    tokens = []
    try:
        rgb = texture_face_rgb(mesh)
    except (AttributeError, ValueError):
        rgb = None
    for i, part in enumerate(PARTS):
        face_ids = np.flatnonzero(labels == i)
        points = mesh.triangles_center[face_ids]
        row = prior["parts"][i]
        tokens.append({
            "part_id": i, "name": part["name"], "stable_id": f"{ABO_SOFA['object_id']}:part:{i}",
            "visual": {"mean_mesh_texture_rgb": (
                np.average(rgb[face_ids], weights=mesh.area_faces[face_ids], axis=0).tolist()
                if rgb is not None else None), "learned_encoder": None},
            "semantic": {"name_source": "manual sofa template", "role": part["structure"]},
            "geometry": {"surface_area_m2": float(mesh.area_faces[face_ids].sum()),
                         "surface_centroid_m": points.mean(axis=0).tolist(),
                         "envelope_volume_proxy_m3": float(state.part_volumes_m3[i]),
                         "source_faces": len(face_ids)},
            "structure": {"type": part["structure"], "source": "hand-specified sofa prior",
                          "fill_ratio": row["fill_ratio"]},
            "material": {"surface": row["surface_material"],
                         "interior_candidates": row["interior_materials"]},
            "physics": {"effective_density_kg_m3": row["effective_density_kg_m3"],
                        "youngs_proxy_pa": row["youngs_modulus_pa"],
                        "poisson_proxy": row["poisson_ratio"], "mass_kg": _stats(samples[i])},
        })
    return tokens


def _parameter_row(t):
    return {"part": t["name"], "faces": t["geometry"]["source_faces"],
            "envelope_m3": t["geometry"]["envelope_volume_proxy_m3"],
            "fill": t["structure"]["fill_ratio"]["mean"],
            "density_kg_m3": t["physics"]["effective_density_kg_m3"]["mean"],
            "youngs_proxy_pa": t["physics"]["youngs_proxy_pa"]["mean"],
            "poisson_proxy": t["physics"]["poisson_proxy"]["mean"],
            **{f"mass_{k}_kg": v for k, v in t["physics"]["mass_kg"].items()}}


def _csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save(fig, output, name):
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def _draw(ax, mesh, rgba, *, wire=False):
    triangles = (np.asarray(mesh.triangles) - mesh.bounds.mean(axis=0)) / mesh.extents.max()
    triangles = triangles[:, :, [0, 2, 1]]
    normals = np.asarray(mesh.face_normals)[:, [0, 2, 1]]
    illumination = 0.65 + 0.35 * np.clip(normals @ np.array([-0.3, -0.4, 0.85]), 0, 1)
    rgba = colors.to_rgba_array(rgba)
    if len(rgba) == 1:
        rgba = np.repeat(rgba, len(mesh.faces), axis=0)
    rgba[:, :3] *= illumination[:, None]
    poly = Poly3DCollection(triangles, facecolors=rgba, rasterized=True,
                           edgecolors="#333333" if wire else "none",
                           linewidths=0.09 if wire else 0)
    ax.add_collection3d(poly)
    ax.set(xlim=(-0.55, 0.55), ylim=(-0.26, 0.26), zlim=(-0.24, 0.24))
    ax.set_box_aspect((2.2, 0.9, 0.9))
    ax.view_init(elev=23, azim=-65)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def _fields(mesh, labels, image_path, tokens, output):
    configure_style()
    fig = plt.figure(figsize=(13, 7.1))
    gs = fig.add_gridspec(2, 4, left=0.025, right=0.98, top=0.89, bottom=0.14,
                          wspace=0.14, hspace=0.22)
    ax = fig.add_subplot(gs[0, 0])
    product = Image.open(image_path).convert("RGB")
    a = np.asarray(product)
    yy, xx = np.where(np.any(a < 235, axis=2))
    ax.imshow(product.crop((max(0, xx.min()-20), max(0, yy.min()-20),
                            min(a.shape[1], xx.max()+20), min(a.shape[0], yy.max()+20))))
    ax.axis("off")
    ax.set_title("a  Official ABO photograph", loc="left", weight="bold")
    ax = fig.add_subplot(gs[0, 1], projection="3d")
    _draw(ax, mesh, np.asarray(PALETTE)[labels])
    ax.set_title("b  Geometry-seeded parts", loc="left", weight="bold")
    specifications = [
        ("c  Effective density", "effective_density_kg_m3", "viridis", False, "kg m$^{-3}$"),
        ("d  Young's modulus proxy", "youngs_proxy_pa", "magma", True, "Pa"),
        ("e  Poisson ratio proxy", "poisson_proxy", "cividis", False, "dimensionless"),
        ("f  Material fill ratio", "fill_ratio", "YlGnBu", False, "fraction"),
        ("g  Part mass", "mass_kg", "plasma", False, "kg / part"),
        ("h  Density uncertainty", "density_cv", "inferno", False, "SD / mean"),
    ]
    for slot, (title, key, cmap, log, unit) in enumerate(specifications, start=2):
        values = []
        for t in tokens:
            if key == "fill_ratio":
                value = t["structure"][key]["mean"]
            elif key == "density_cv":
                stats = t["physics"]["effective_density_kg_m3"]
                value = stats["std"] / stats["mean"]
            else:
                value = t["physics"][key]["mean"]
            values.append(value)
        low, high = (0, 1) if key == "fill_ratio" else (min(values), max(values))
        norm = (colors.LogNorm if log else colors.Normalize)(vmin=low, vmax=high)
        ax = fig.add_subplot(gs[slot // 4, slot % 4], projection="3d")
        _draw(ax, mesh, plt.colormaps[cmap](norm(np.asarray(values)[labels])))
        ax.set_title(title, loc="left", weight="bold")
        fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.52,
                     pad=0.01, label=unit)
    fig.suptitle("Unified Part-Level Physical Representation | ABO sofa pilot", y=0.98,
                 fontsize=14, weight="bold")
    fig.text(0.5, 0.92, "Texture + geometry + semantics + construction + physical distributions, indexed by stable part ID",
             ha="center", fontsize=8)
    fig.legend(handles=[Patch(color=PALETTE[i], label=p["name"]) for i, p in enumerate(PARTS)],
               loc="lower center", bbox_to_anchor=(0.5, 0.065), ncol=4, fontsize=8)
    fig.text(0.5, 0.025, "Geometry labels and hand-specified material priors; no learned latent alignment. E/nu are illustrative descriptors.\nMass colors encode total part mass, not a measured surface field. Intervals and uncertainty are uncalibrated.",
             ha="center", fontsize=7, color="#555555")
    _save(fig, output, "01_part_token_fields")


def _mass(samples, part_samples, rows, target, output):
    configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.subplots_adjust(left=0.085, right=0.98, top=0.87, bottom=0.14,
                        hspace=0.48, wspace=0.32)
    ax = axes[0, 0]
    names = ["Fabric solid", "Part solid", "Layered state"]
    bars = ax.bar(names, [r["mass_kg"] for r in rows], color=["#999999", "#D1A14B", "#4FA38F"])
    ax.axhline(target, color="#C96A6A", linewidth=1.4, label=f"Catalog {target:.2f} kg")
    for bar, row in zip(bars, rows, strict=True):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+5, f"{row['mass_kg']:.1f}", ha="center", fontsize=8)
    ax.set(ylabel="Mass (kg)", ylim=(0, max(r["mass_kg"] for r in rows)*1.19))
    ax.set_title("a  Controlled assumptions, same voxel envelope", loc="left", weight="bold")
    ax.legend(loc="upper right")
    ax = axes[0, 1]
    ax.hist(samples, bins=45, density=True, color="#5A76A8", alpha=0.8)
    ax.axvline(target, color="#C96A6A", label="Catalog")
    ax.axvline(samples.mean(), color="#222222", linestyle="--", label="Prior mean")
    q05, q95 = np.quantile(samples, (0.05, 0.95))
    ax.axvspan(q05, q95, color="#5A76A8", alpha=0.10)
    ax.set(xlabel="Mass (kg)", ylabel="Probability density (kg$^{-1}$)")
    ax.set_title("b  Uncalibrated prior-predictive interval", loc="left", weight="bold")
    ax.legend(loc="upper right")
    ax = axes[1, 0]
    mean = part_samples.mean(axis=1)
    low, high = np.quantile(part_samples, (0.05, 0.95), axis=1)
    ax.errorbar(mean, np.arange(4), xerr=np.stack((mean-low, high-mean)), fmt="o",
                color="#4FA38F", capsize=3)
    ax.set_yticks(np.arange(4), [p["name"] for p in PARTS])
    ax.invert_yaxis()
    ax.set_xlabel("Part mass (kg), mean and 90% interval")
    ax.set_title("c  Material occupancy integrated by part", loc="left", weight="bold")
    ax = axes[1, 1]
    ax.axis("off")
    data = [[row["method"].split(" / ")[0], f"{row['ADE_kg']:.2f}",
             f"{row['APE_percent']:.1f}", f"{row['MnRE']:.3f}"] for row in rows]
    table = ax.table(cellText=data, colLabels=["Assumption", "ADE (kg)", "APE (%)", "MnRE"],
                     colWidths=[0.46, 0.18, 0.18, 0.18], cellLoc="center", bbox=(0, 0.37, 1, 0.53))
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (r, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if r == 0:
            cell.set_facecolor("#E8EDF3")
    ax.set_title("d  One-object numeric comparison", loc="left", weight="bold")
    ax.text(0, 0.14, f"Mean: {samples.mean():.2f} kg | catalog: {target:.2f} kg\n90% prior interval: [{q05:.2f}, {q95:.2f}] kg", fontsize=8)
    fig.suptitle("Mass comparison | one real ABO asset, controlled construction assumptions",
                 fontsize=12, weight="bold")
    fig.text(0.5, 0.035, "Catalog mass is used only for evaluation. Comparators are simple assumptions, not PUGS/VoMP reproductions.\nOpen-mesh voxel fill is an envelope proxy; one object does not establish predictive accuracy.",
             ha="center", fontsize=7, color="#555555")
    _save(fig, output, "02_mass_comparison")


def _mapping(mesh, labels, lod_pair, rows, output):
    configure_style()
    fig = plt.figure(figsize=(11.5, 6.8))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.97, top=0.88, bottom=0.17,
                          hspace=0.30, wspace=0.32)
    lod, mapped = lod_pair
    for col, (model, lab, title) in enumerate((
        (mesh, _geometric_part_labels(mesh), "a  Coordinate-rule initialization"),
        (mesh, labels, "b  Geometry graph refinement"),
        (lod, mapped, f"c  Shared-state lookup: {len(lod.faces):,} faces"),
    )):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        _draw(ax, model, np.asarray(PALETTE)[lab], wire=col == 2)
        ax.set_title(title, loc="left", weight="bold")
    ordered = sorted(rows, key=lambda r: r["faces"])
    x = [r["faces"] for r in ordered]
    for col, (key, shared_key, title, unit) in enumerate((
        ("mass_error_percent", "canonical_mass_drift_percent", "d  Mass disagreement", "Mass drift (%)"),
        ("com_error_mm", "canonical_com_error_mm", "e  Center-of-mass disagreement", "COM shift (mm)"),
        ("inertia_tensor_error_percent", "canonical_inertia_tensor_error_percent",
         "f  Full inertia-tensor disagreement", "Inertia drift (%)"),
    )):
        ax = fig.add_subplot(gs[1, col])
        ax.plot(x, [r[key] for r in ordered], color="#C96A6A", marker="o", label="Revoxelize each LOD")
        ax.plot(x, [r[shared_key] for r in ordered], color="#4FA38F", marker="o", label="Reuse canonical state")
        ax.set(xscale="log", xlabel="Mesh triangles", ylabel=unit)
        ax.set_title(title, loc="left", weight="bold")
        if col == 0:
            ax.legend(loc="best", fontsize=6)
    fig.suptitle("Unified data-layer check | stable part IDs and physical state across mesh LODs",
                 fontsize=12, weight="bold")
    fig.text(0.5, 0.055, "Reference = original-mesh voxel state. All densities are held fixed across LODs.\nZero canonical drift is expected by design: the same cells are reused. This is a consistency check, not a learned mapping benchmark.\nTop-row labels are geometry-seeded, without part ground truth or SAM/VLM inference.",
             ha="center", fontsize=7, color="#555555")
    _save(fig, output, "03_mapping_consistency")
