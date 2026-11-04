from __future__ import annotations

import csv
import urllib.request
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import trimesh
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tactile3d.figures.style import PALETTE, configure_style, save_figure
from tactile3d.mesh.physics_lod import LODConstraints, select_physics_constrained_lod
from tactile3d.utils.io import ensure_dir, write_json

ABO_SOFA = {
    "object_id": "B075X4QMX3",
    "name": "Stone & Beam Westport upholstered sofa",
    "category": "sofa",
    "image_url": "https://m.media-amazon.com/images/I/814aS-KaUBL.jpg",
    "mesh_url": (
        "https://amazon-berkeley-objects.s3.amazonaws.com/"
        "3dmodels/original/3/B075X4QMX3.glb"
    ),
    "catalog_dimensions_in": [87.0, 37.75, 33.0],
    "catalog_mass_kg": 177.76 * 0.45359237,
    "surface_evidence": "60% cotton / 40% linen upholstery; wood legs; metal trim",
    "license": "Amazon Berkeley Objects, CC BY 4.0",
}

PARTS = (
    {
        "id": 0,
        "name": "Frame / base",
        "structure": "frame_supported",
        "envelope_share": 0.24,
        "components": {
            "wood": (0.035, 0.045, 0.060),
            "steel": (0.0008, 0.0013, 0.0020),
            "foam": (0.010, 0.020, 0.030),
            "fabric": (0.002, 0.004, 0.006),
        },
        "structure_posterior": {"frame_supported": 0.72, "foam_filled": 0.20, "solid": 0.08},
    },
    {
        "id": 1,
        "name": "Seat cushion",
        "structure": "foam_filled",
        "envelope_share": 0.27,
        "components": {"foam": (0.16, 0.20, 0.25), "fabric": (0.002, 0.003, 0.005)},
        "structure_posterior": {"foam_filled": 0.86, "layered": 0.11, "solid": 0.03},
    },
    {
        "id": 2,
        "name": "Back cushions",
        "structure": "foam_filled",
        "envelope_share": 0.30,
        "components": {"foam": (0.09, 0.13, 0.18), "fabric": (0.002, 0.004, 0.006)},
        "structure_posterior": {"foam_filled": 0.79, "layered": 0.17, "solid": 0.04},
    },
    {
        "id": 3,
        "name": "Arms",
        "structure": "layered_frame",
        "envelope_share": 0.19,
        "components": {
            "wood": (0.010, 0.015, 0.022),
            "foam": (0.025, 0.035, 0.050),
            "fabric": (0.001, 0.002, 0.004),
        },
        "structure_posterior": {"frame_supported": 0.58, "foam_filled": 0.34, "solid": 0.08},
    },
)

MATERIALS = {
    "foam": {"density": (65.0, 30.0, 15.0, 180.0), "youngs": (2e6, 1e6, 1e5, 8e6), "poisson": (0.20, 0.30, 0.42)},
    "fabric": {"density": (350.0, 220.0, 60.0, 1200.0), "youngs": (1e8, 1.2e8, 1e6, 8e8), "poisson": (0.20, 0.32, 0.45)},
    "wood": {"density": (650.0, 180.0, 300.0, 950.0), "youngs": (1e10, 4e9, 2e9, 1.8e10), "poisson": (0.25, 0.34, 0.45)},
    "steel": {"density": (7850.0, 180.0, 7400.0, 8200.0), "youngs": (2e11, 1.5e10, 1.6e11, 2.2e11), "poisson": (0.27, 0.30, 0.33)},
}


def generate_abo_sofa_pilot(
    asset_dir: str | Path,
    output_dir: str | Path,
    *,
    download_missing: bool = True,
    samples: int = 8000,
    seed: int = 17,
    formats: tuple[str, ...] = ("png", "svg", "pdf"),
) -> dict[str, Any]:
    """Generate a transparent one-object research pilot and paper figures."""

    asset_dir = ensure_dir(asset_dir)
    output_dir = ensure_dir(output_dir)
    image_path = asset_dir / "product.jpg"
    mesh_path = asset_dir / "model.glb"
    if download_missing:
        _download_if_missing(ABO_SOFA["image_url"], image_path)
        _download_if_missing(ABO_SOFA["mesh_url"], mesh_path)
    if not image_path.is_file() or not mesh_path.is_file():
        raise FileNotFoundError("Pilot requires product.jpg and model.glb in the asset directory")

    mesh = _load_mesh(mesh_path)
    lod_mesh, lod_report = select_physics_constrained_lod(
        mesh,
        constraints=LODConstraints(
            volume_relative=0.04,
            center_of_mass_relative=0.0075,
            inertia_relative=0.05,
            surface_area_relative=0.01,
        ),
    )
    lod_path = output_dir / "sofa_physics_lod.ply"
    lod_mesh.export(lod_path)
    labels = _geometric_part_labels(mesh)
    lod_labels = _geometric_part_labels(lod_mesh)
    posterior, raw_samples = _construction_posterior(
        abs(float(mesh.volume)), samples=samples, seed=seed
    )
    report = {
        "schema_version": 1,
        "status": "single-object construction-prior pilot; not a trained-VLM benchmark result",
        "target_used_for_inference": False,
        "object": ABO_SOFA,
        "mesh": {
            "path": str(mesh_path.resolve()),
            "faces": len(mesh.faces),
            "vertices": len(mesh.vertices),
            "watertight": bool(mesh.is_watertight),
            "dimensions_m": mesh.extents.tolist(),
            "signed_volume_proxy_m3": abs(float(mesh.volume)),
        },
        "lod": lod_report,
        "posterior": posterior,
    }
    write_json(output_dir / "pilot_report.json", report)
    _write_source_csv(output_dir / "pilot_source_data.csv", posterior)

    files: list[str] = []
    image = Image.open(image_path).convert("RGB")
    files += _plot_pipeline(image, mesh, lod_mesh, output_dir, formats)
    files += _plot_physics_maps(image, mesh, labels, posterior, output_dir, formats)
    files += _plot_mass_and_lod(raw_samples, posterior, lod_report, output_dir, formats)
    files += _plot_lod_comparison(mesh, lod_mesh, lod_labels, output_dir, formats)
    report["figure_files"] = files
    report["lod_mesh"] = str(lod_path)
    write_json(output_dir / "pilot_report.json", report)
    return report


def _construction_posterior(
    volume_proxy_m3: float, *, samples: int, seed: int
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if samples < 100:
        raise ValueError("At least 100 Monte Carlo samples are required")
    rng = np.random.default_rng(seed)
    volume = volume_proxy_m3 * np.clip(rng.normal(1.0, 0.03, samples), 0.88, 1.12)
    part_mass = np.zeros((len(PARTS), samples))
    part_occupied = np.zeros_like(part_mass)
    part_youngs = np.zeros_like(part_mass)
    part_poisson = np.zeros_like(part_mass)
    material_mass = {name: np.zeros(samples) for name in MATERIALS}

    for part_index, part in enumerate(PARTS):
        log_youngs_sum = np.zeros(samples)
        poisson_sum = np.zeros(samples)
        for material, fraction_range in part["components"].items():
            fraction = rng.triangular(*fraction_range, size=samples)
            component_volume = volume * fraction
            prior = MATERIALS[material]
            density = _sample_clipped_normal(rng, prior["density"], samples)
            youngs = _sample_clipped_normal(rng, prior["youngs"], samples)
            poisson = rng.triangular(*prior["poisson"], size=samples)
            mass = component_volume * density
            part_mass[part_index] += mass
            part_occupied[part_index] += component_volume
            log_youngs_sum += component_volume * np.log(youngs)
            poisson_sum += component_volume * poisson
            material_mass[material] += mass
        occupied = np.maximum(part_occupied[part_index], 1e-12)
        part_youngs[part_index] = np.exp(log_youngs_sum / occupied)
        part_poisson[part_index] = poisson_sum / occupied

    object_mass = part_mass.sum(axis=0)
    object_occupied = part_occupied.sum(axis=0)
    parts = []
    for index, part in enumerate(PARTS):
        envelope = volume * float(part["envelope_share"])
        parts.append(
            {
                "part_id": part["id"],
                "name": part["name"],
                "structure": part["structure"],
                "surface_material": "fabric" if part["id"] else "fabric + wood trim",
                "interior_materials": list(part["components"]),
                "structure_posterior": part["structure_posterior"],
                "mass_kg": _stats(part_mass[index]),
                "occupied_volume_m3": _stats(part_occupied[index]),
                "effective_density_kg_m3": _stats(part_mass[index] / envelope),
                "youngs_modulus_pa": _stats(part_youngs[index]),
                "poisson_ratio": _stats(part_poisson[index]),
                "fill_ratio": _stats(np.clip(part_occupied[index] / envelope, 0.0, 1.0)),
            }
        )
    posterior = {
        "object_mass_kg": _stats(object_mass),
        "catalog_mass_kg": ABO_SOFA["catalog_mass_kg"],
        "absolute_percentage_error_of_mean": abs(
            float(object_mass.mean()) - ABO_SOFA["catalog_mass_kg"]
        )
        / ABO_SOFA["catalog_mass_kg"],
        "occupied_volume_m3": _stats(object_occupied),
        "global_fill_ratio": _stats(np.clip(object_occupied / volume, 0.0, 1.0)),
        "parts": parts,
        "material_mass_kg": {name: _stats(values) for name, values in material_mass.items()},
    }
    raw = {"object_mass": object_mass, "part_mass": part_mass, **material_mass}
    return posterior, raw


def _stats(values: np.ndarray) -> dict[str, float]:
    q05, q50, q95 = np.quantile(values, (0.05, 0.5, 0.95))
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
    }


def _geometric_part_labels(mesh: trimesh.Trimesh) -> np.ndarray:
    centers = np.asarray(mesh.triangles_center)
    unit = (centers - mesh.bounds[0]) / np.maximum(mesh.extents, 1e-12)
    labels = np.ones(len(mesh.faces), dtype=np.int64)
    labels[unit[:, 1] < 0.29] = 0
    labels[unit[:, 1] > 0.57] = 2
    side = np.abs(unit[:, 0] - 0.5) > 0.34
    labels[side & (unit[:, 1] >= 0.27)] = 3
    return labels


def _plot_pipeline(
    product: Image.Image,
    mesh: trimesh.Trimesh,
    lod_mesh: trimesh.Trimesh,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    configure_style()
    figure = plt.figure(figsize=(12.0, 5.0))
    figure.text(0.03, 0.955, "PartPhy3D: mesh-first probabilistic physical grounding", fontsize=13, weight="bold")
    figure.text(0.03, 0.912, "Single/multi-view reconstruction, construction reasoning, calibrated physics and closed-loop validation", fontsize=7.5, color="#555555")

    top_y, box_w, box_h = 0.61, 0.135, 0.19
    xs = [0.03, 0.19, 0.35, 0.51, 0.67, 0.83]
    titles = [
        "1  Visual evidence",
        "2  Mesh entry",
        "3  Physics-Risk LOD",
        "4  Part proposals",
        "5  Construction graph",
        "6  Physical posterior",
    ]
    descriptions = [
        "ABO views + scale cues",
        "InstantMesh / reference GLB",
        "Preserve V, COM, inertia",
        "SAM + VLM + mesh cues",
        "shell / core / frame / void",
        "rho, E, nu, fill, mass",
    ]
    box_colors = ["#EAF2F8", "#E8F3F1", "#FFF5D9", "#FCECE6", "#F1EAF3", "#E9EEF4"]
    for index, x in enumerate(xs):
        patch = FancyBboxPatch((x, top_y), box_w, box_h, boxstyle="round,pad=0.008,rounding_size=0.01", transform=figure.transFigure, facecolor=box_colors[index], edgecolor="#4A4A4A", linewidth=0.8, zorder=-2)
        figure.patches.append(patch)
        figure.text(x + 0.01, top_y + 0.142, titles[index], fontsize=7.6, weight="bold")
        figure.text(x + 0.01, top_y + 0.026, descriptions[index], fontsize=6.6, color="#444444")
        if index < len(xs) - 1:
            arrow = FancyArrowPatch((x + box_w + 0.004, top_y + box_h / 2), (xs[index + 1] - 0.004, top_y + box_h / 2), transform=figure.transFigure, arrowstyle="-|>", mutation_scale=9, linewidth=0.9, color="#555555")
            figure.patches.append(arrow)

    image_ax = figure.add_axes((0.058, 0.675, 0.08, 0.06))
    image_ax.set_zorder(5)
    image_ax.imshow(product)
    image_ax.axis("off")
    mesh_ax = figure.add_axes((0.217, 0.67, 0.08, 0.07), projection="3d")
    mesh_ax.set_zorder(5)
    _draw_mesh(mesh_ax, mesh, np.full(len(mesh.faces), "#A8B5C0"), max_faces=4500)
    lod_ax = figure.add_axes((0.377, 0.67, 0.08, 0.07), projection="3d")
    lod_ax.set_zorder(5)
    _draw_mesh(lod_ax, lod_mesh, np.full(len(lod_mesh.faces), "#E0A458"), max_faces=5000, edges=True)

    parts_ax = figure.add_axes((0.535, 0.68, 0.085, 0.045))
    parts_ax.set_zorder(5)
    parts_ax.barh(np.arange(4), (0.86, 0.73, 0.92, 0.61), color=PALETTE[:4], height=0.7)
    parts_ax.set(xlim=(0, 1), ylim=(-0.7, 3.7))
    parts_ax.axis("off")

    graph_ax = figure.add_axes((0.695, 0.67, 0.085, 0.065))
    graph_ax.set_zorder(5)
    nodes = np.asarray(((0.08, 0.5), (0.38, 0.78), (0.38, 0.22), (0.72, 0.5), (0.94, 0.5)))
    for start, end in ((0, 1), (0, 2), (1, 3), (2, 3), (3, 4)):
        graph_ax.plot(nodes[[start, end], 0], nodes[[start, end], 1], color="#747474", linewidth=1)
    graph_ax.scatter(nodes[:, 0], nodes[:, 1], s=(18, 22, 22, 25, 18), c=("#4C78A8", "#72B7B2", "#8B6F47", "#E07B54", "#D7D7D7"), edgecolors="white", linewidths=0.5, zorder=2)
    graph_ax.set(xlim=(0, 1), ylim=(0, 1))
    graph_ax.axis("off")

    posterior_ax = figure.add_axes((0.855, 0.67, 0.085, 0.065))
    posterior_ax.set_zorder(5)
    x_curve = np.linspace(-3, 3, 80)
    posterior_ax.fill_between(x_curve, np.exp(-(x_curve**2) / 2), color="#4C78A8", alpha=0.35)
    posterior_ax.plot(x_curve, np.exp(-(x_curve**2) / 2), color="#4C78A8", linewidth=1.2)
    posterior_ax.axvline(0, color="#C94C3B", linewidth=0.9)
    posterior_ax.axis("off")

    lower = [
        (0.06, "A  Evidence fusion", "Multiple views vote on visible materials; inconsistent views increase uncertainty."),
        (0.29, "B  Hidden construction", "The VLM predicts hypotheses, not a single guess: fabric shell, foam, wood frame, steel and void."),
        (0.52, "C  Metric integration", "Watertight diagnostics and layered occupancy convert geometry into mass, COM and inertia distributions."),
        (0.75, "D  Consistency loop", "Simulation residuals and cross-LOD disagreement request more views, faces or voxels where needed."),
    ]
    for x, title, body in lower:
        patch = FancyBboxPatch((x, 0.18), 0.19, 0.25, boxstyle="round,pad=0.012,rounding_size=0.012", transform=figure.transFigure, facecolor="white", edgecolor="#B9B9B9", linewidth=0.8, zorder=-2)
        figure.patches.append(patch)
        figure.text(x + 0.012, 0.37, title, fontsize=8, weight="bold")
        figure.text(x + 0.012, 0.225, _wrap(body, 36), fontsize=6.8, color="#444444", linespacing=1.45)
    feedback = FancyArrowPatch((0.93, 0.16), (0.14, 0.16), transform=figure.transFigure, connectionstyle="arc3,rad=-0.12", arrowstyle="-|>", mutation_scale=10, linewidth=1.2, color="#D06B4E")
    figure.patches.append(feedback)
    figure.text(0.45, 0.035, "active compute allocation: uncertainty -> new view / local remesh / finer occupancy", color="#B95036", fontsize=7.2, weight="bold")
    return save_figure(figure, output_dir / "partphy3d_pipeline", formats)


def _plot_physics_maps(
    product: Image.Image,
    mesh: trimesh.Trimesh,
    labels: np.ndarray,
    posterior: dict[str, Any],
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    configure_style()
    figure = plt.figure(figsize=(11.5, 6.0), constrained_layout=True)
    axes = [figure.add_subplot(2, 3, index + 1, projection="3d" if index else None) for index in range(6)]
    axes[0].imshow(product)
    axes[0].axis("off")
    axes[0].set_title("a  ABO catalog observation", loc="left", weight="bold")
    parts = posterior["parts"]
    part_colors = np.asarray([colors.to_rgba(PALETTE[int(label)]) for label in labels])
    values = {
        "density": np.asarray([parts[int(label)]["effective_density_kg_m3"]["mean"] for label in labels]),
        "youngs": np.asarray([parts[int(label)]["youngs_modulus_pa"]["mean"] for label in labels]),
        "poisson": np.asarray([parts[int(label)]["poisson_ratio"]["mean"] for label in labels]),
        "fill": np.asarray([parts[int(label)]["fill_ratio"]["mean"] for label in labels]),
    }
    panels = [
        (axes[1], part_colors, "b  Part / construction proposal", None, None),
        (axes[2], plt.colormaps["viridis"](colors.Normalize()(values["density"])), "c  Effective density", colors.Normalize(vmin=values["density"].min(), vmax=values["density"].max()), "kg m$^{-3}$"),
        (axes[3], plt.colormaps["magma"](colors.LogNorm()(values["youngs"])), "d  Young's modulus proxy", colors.LogNorm(vmin=values["youngs"].min(), vmax=values["youngs"].max()), "Pa"),
        (axes[4], plt.colormaps["cividis"](colors.Normalize()(values["poisson"])), "e  Poisson ratio", colors.Normalize(vmin=values["poisson"].min(), vmax=values["poisson"].max()), "$\\nu$"),
        (axes[5], plt.colormaps["YlGnBu"](colors.Normalize(vmin=0, vmax=1)(values["fill"])), "f  Solid occupancy", colors.Normalize(vmin=0, vmax=1), "fill ratio"),
    ]
    cmaps = [None, "viridis", "magma", "cividis", "YlGnBu"]
    for index, (axis, face_colors, title, norm, label) in enumerate(panels):
        _draw_mesh(axis, mesh, face_colors, max_faces=50000)
        axis.set_title(title, loc="left", weight="bold")
        if norm is not None:
            figure.colorbar(ScalarMappable(norm=norm, cmap=cmaps[index]), ax=axis, shrink=0.55, pad=0.01, label=label)
    axes[1].legend(handles=[Patch(facecolor=PALETTE[p["part_id"]], label=p["name"]) for p in parts], loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    figure.suptitle("Single-object physical fields from a layered construction posterior", fontsize=11, weight="bold")
    return save_figure(figure, output_dir / "abo_sofa_physical_fields", formats)


def _plot_mass_and_lod(
    raw: dict[str, np.ndarray],
    posterior: dict[str, Any],
    lod_report: dict,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.2), constrained_layout=True)
    mass = raw["object_mass"]
    ground_truth = float(posterior["catalog_mass_kg"])
    axes[0, 0].hist(mass, bins=55, density=True, color="#4C78A8", alpha=0.85)
    axes[0, 0].axvline(ground_truth, color="#C94C3B", linewidth=1.5, label=f"catalog: {ground_truth:.1f} kg")
    axes[0, 0].axvline(mass.mean(), color="#222222", linestyle="--", linewidth=1.2, label=f"mean: {mass.mean():.1f} kg")
    axes[0, 0].set(xlabel="Object mass (kg)", ylabel="Posterior density")
    axes[0, 0].set_title("a  Total mass posterior", loc="left", weight="bold")
    axes[0, 0].legend()

    part_rows = posterior["parts"]
    means = np.asarray([row["mass_kg"]["mean"] for row in part_rows])
    low = np.asarray([row["mass_kg"]["q05"] for row in part_rows])
    high = np.asarray([row["mass_kg"]["q95"] for row in part_rows])
    y = np.arange(len(part_rows))
    axes[0, 1].errorbar(means, y, xerr=np.vstack((means - low, high - means)), fmt="o", color="#3B7C78", capsize=3)
    axes[0, 1].set_yticks(y, [row["name"] for row in part_rows])
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlabel("Mass (kg), mean and 90% interval")
    axes[0, 1].set_title("b  Part mass uncertainty", loc="left", weight="bold")
    axes[0, 1].grid(axis="x", color="#E2E2E2", linewidth=0.5)

    materials = list(MATERIALS)
    material_means = [posterior["material_mass_kg"][name]["mean"] for name in materials]
    axes[1, 0].bar(materials, material_means, color=["#72B7B2", "#E6C45A", "#8B6F47", "#7F8790"])
    axes[1, 0].set(ylabel="Expected mass (kg)")
    axes[1, 0].set_title("c  Hidden-material mass decomposition", loc="left", weight="bold")
    for index, value in enumerate(material_means):
        axes[1, 0].text(index, value + max(material_means) * 0.02, f"{value:.1f}", ha="center", fontsize=6.5)

    rows = sorted(lod_report["candidates"], key=lambda row: row["faces"])
    faces = np.asarray([row["faces"] for row in rows])
    for key, label, color in (
        ("volume_relative", "Volume proxy", "#4C78A8"),
        ("center_of_mass_relative", "COM / bbox", "#72B7B2"),
        ("inertia_relative", "Inertia", "#E07B54"),
        ("surface_area_relative", "Surface area", "#B279A2"),
    ):
        axes[1, 1].plot(faces, 100 * np.asarray([row["errors"][key] for row in rows]), marker="o", label=label, color=color)
    selected_faces = lod_report["selected"]["faces"]
    axes[1, 1].axvline(selected_faces, color="#222222", linestyle="--", linewidth=1, label=f"selected: {selected_faces:,}")
    axes[1, 1].set(xscale="log", xlabel="Triangles", ylabel="Relative error (%)")
    axes[1, 1].set_title("d  Cross-resolution physical consistency", loc="left", weight="bold")
    axes[1, 1].legend(ncol=2)
    axes[1, 1].grid(color="#E8E8E8", linewidth=0.45)
    figure.suptitle("Mass inference and physics-preserving mesh reduction", fontsize=11, weight="bold")
    return save_figure(figure, output_dir / "abo_sofa_mass_and_lod", formats)


def _plot_lod_comparison(
    mesh: trimesh.Trimesh,
    lod_mesh: trimesh.Trimesh,
    lod_labels: np.ndarray,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[str]:
    configure_style()
    figure = plt.figure(figsize=(7.2, 3.0), constrained_layout=True)
    left = figure.add_subplot(1, 2, 1, projection="3d")
    right = figure.add_subplot(1, 2, 2, projection="3d")
    _draw_mesh(left, mesh, np.full(len(mesh.faces), "#AAB6C2"), max_faces=50000)
    _draw_mesh(right, lod_mesh, np.asarray([PALETTE[int(label)] for label in lod_labels]), max_faces=10000, edges=True)
    left.set_title(f"a  Reference GLB ({len(mesh.faces):,} triangles)", loc="left", weight="bold")
    right.set_title(f"b  Physics-constrained LOD ({len(lod_mesh.faces):,} triangles)", loc="left", weight="bold")
    return save_figure(figure, output_dir / "abo_sofa_mesh_lod", formats)


def _draw_mesh(
    axis: plt.Axes,
    mesh: trimesh.Trimesh,
    face_colors: np.ndarray,
    *,
    max_faces: int,
    edges: bool = False,
) -> None:
    triangles = np.asarray(mesh.triangles)
    face_colors = np.asarray(face_colors)
    if len(triangles) > max_faces:
        selected = np.linspace(0, len(triangles) - 1, max_faces, dtype=np.int64)
        triangles = triangles[selected]
        face_colors = face_colors[selected]
    center = mesh.bounds.mean(axis=0)
    scale = max(float(mesh.extents.max()), 1e-12)
    triangles = (triangles - center) / scale
    triangles = triangles[:, :, [0, 2, 1]]
    collection = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors="#48535D" if edges else "none",
        linewidths=0.08 if edges else 0,
        rasterized=True,
    )
    axis.add_collection3d(collection)
    axis.set(xlim=(-0.55, 0.55), ylim=(-0.27, 0.27), zlim=(-0.24, 0.24))
    axis.set_box_aspect((2.2, 0.9, 0.9))
    axis.view_init(elev=14, azim=-72)
    axis.set_proj_type("ortho")
    axis.set_axis_off()


def _write_source_csv(path: Path, posterior: dict[str, Any]) -> None:
    fields = ["part", "mass_mean_kg", "mass_q05_kg", "mass_q95_kg", "density_mean_kg_m3", "youngs_mean_pa", "poisson_mean", "fill_ratio_mean"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for part in posterior["parts"]:
            writer.writerow(
                {
                    "part": part["name"],
                    "mass_mean_kg": part["mass_kg"]["mean"],
                    "mass_q05_kg": part["mass_kg"]["q05"],
                    "mass_q95_kg": part["mass_kg"]["q95"],
                    "density_mean_kg_m3": part["effective_density_kg_m3"]["mean"],
                    "youngs_mean_pa": part["youngs_modulus_pa"]["mean"],
                    "poisson_mean": part["poisson_ratio"]["mean"],
                    "fill_ratio_mean": part["fill_ratio"]["mean"],
                }
            )


def _sample_clipped_normal(
    rng: np.random.Generator,
    parameters: tuple[float, float, float, float],
    samples: int,
) -> np.ndarray:
    mean, std, low, high = parameters
    return np.clip(rng.normal(mean, std, samples), low, high)


def _download_if_missing(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(destination)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=True)
    if isinstance(loaded, trimesh.Scene):
        mesh = loaded.to_geometry()
    else:
        mesh = loaded
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"No triangular mesh found in {path}")
    return mesh


def _wrap(text: str, width: int) -> str:
    words, lines, current = text.split(), [], []
    for word in words:
        if current and len(" ".join(current + [word])) > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)
