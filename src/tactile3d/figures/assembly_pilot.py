"""Inspectable native-component sofa parsing and local connection diagnostics."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np
import trimesh
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from tactile3d.figures.abo_pilot import ABO_SOFA, _load_mesh
from tactile3d.figures.style import configure_style
from tactile3d.mesh.assembly import crop_mesh, face_components, interface_geometry, texture_face_rgb
from tactile3d.utils.io import ensure_dir, write_json

NAMES = (
    "Body shell (internal frame unresolved)", "Long seat cushion",
    "Back cushion X-", "Back cushion center", "Back cushion X+",
    "Leg X-/Z-", "Leg X-/Z+", "Leg X+/Z-", "Leg X+/Z+",
    "Nailhead trim", "Exposed base rail",
)
COLORS = (
    "#B0B8C0", "#54A89C", "#D4AF47", "#6585BD", "#B182B0",
    "#BF655D", "#DA9A65", "#719F55", "#6CA2C4", "#6E6260", "#856D4A",
)


def native_sofa_parts(mesh):
    """Template naming of native mesh pieces, without claiming semantic inference."""
    count, components = face_components(mesh)
    largest = np.bincount(components).argmax()
    labels = np.zeros(len(mesh.faces), dtype=np.int64)
    pieces = []
    width = mesh.extents[0]
    for component in range(count):
        ids = np.flatnonzero(components == component)
        vertices = mesh.vertices[np.unique(mesh.faces[ids])]
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        ext, center = hi-lo, (hi+lo)/2
        area = float(mesh.area_faces[ids].sum())
        if component == largest:
            part = 0
        elif ext.max() < 0.035 and area < 0.002:
            part = 9
        elif hi[1] < 0.25 and ext[0] > width * 0.65:
            part = 10
        elif hi[1] < 0.25 and ext.max() < 0.3:
            part = 5 + int(center[0] > 0)*2 + int(center[2] > 0)
        elif lo[1] > 0.4 and ext[0] < width * 0.4 and center[2] < 0:
            part = 2 + int(center[0] > -width*0.15) + int(center[0] > width*0.15)
        elif ext[0] > width*0.65 and center[1] < 0.6:
            part = 1
        else:
            part = 0
        labels[ids] = part
        pieces.append({"native_component": component, "part_id": part,
                       "faces": len(ids), "surface_area_m2": area,
                       "bounds_m": [lo.tolist(), hi.tolist()]})
    return labels, pieces


def run_assembly_pilot(asset_dir, output_dir):
    output = ensure_dir(output_dir)
    assets = Path(asset_dir)
    mesh = _load_mesh(assets / "model.glb")
    labels, pieces = native_sofa_parts(mesh)
    rgb = texture_face_rgb(mesh)
    np.save(output / "fine_face_labels.npy", labels)
    colored = mesh.copy()
    colored.visual = trimesh.visual.ColorVisuals(
        mesh=colored, face_colors=plt.matplotlib.colors.to_rgba_array(COLORS)[labels]
    )
    colored.export(output / "fine_parts.ply")
    parts = []
    for i, name in enumerate(NAMES):
        ids = np.flatnonzero(labels == i)
        local = mesh.submesh([ids], append=True, repair=False)
        parts.append({
            "part_id": i, "name": name, "faces": len(ids),
            "native_pieces": sum(p["part_id"] == i for p in pieces),
            "area_m2": float(mesh.area_faces[ids].sum()),
            "dimensions_m": local.extents.tolist(),
            "mean_texture_rgb": np.average(rgb[ids], weights=mesh.area_faces[ids], axis=0).tolist(),
            "naming_source": "sofa-specific template on native connected components",
            "material_status": "appearance only; composition unverified",
            "part_is_watertight": bool(local.is_watertight),
        })
    support = mesh.submesh([np.flatnonzero(labels == 10)], append=True, repair=False)
    connections, details = [], []
    for part in range(5, 9):
        leg = mesh.submesh([np.flatnonzero(labels == part)], append=True, repair=False)
        stats, points, distances = interface_geometry(leg, support, seed=part+17)
        stats.update({"connection_id": f"J{part-4}", "source_part": part,
                      "target_part": 10, "source_name": NAMES[part],
                      "target_name": NAMES[10]})
        connections.append(stats)
        center = leg.bounds.mean(axis=0)
        center[1] = leg.bounds[1, 1] - 0.025
        details.append((leg, center, points, distances))
    report = {
        "object_id": ABO_SOFA["object_id"], "parts": parts,
        "native_components": pieces, "connections": connections,
        "segmentation_method": "native connectivity + explicit sofa naming template",
        "learned_segmentation_executed": False,
        "mechanical_strength_simulation_executed": False,
        "limitations": [
            "Native pieces reflect asset construction, not verified manufacturing parts.",
            "Main upholstered body and hidden frame remain unresolved.",
            "Closeness is an unsigned surface measurement, not penetration or bond strength.",
            "Near-sample percentage refers to upper-band area samples, not real contact area.",
            "Interior joints, fasteners, adhesive, load paths and material strength are unknown.",
        ],
    }
    write_json(output / "assembly_report.json", report)
    write_json(output / "part_spec.json", {str(i): n for i, n in enumerate(NAMES)})
    with (output / "joint_geometry.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["connection_id", "source_name", "target_name", "sample_min_distance_mm",
                  "sample_q05_distance_mm", "sample_median_distance_mm", "near_sample_fraction",
                  "near_threshold_m", "mechanical_quality", "connection_type"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(connections)
    configure_style()
    _overview(mesh, labels, rgb, assets / "product.jpg", parts, output)
    _joints(support, connections, details, output)
    _details(mesh, labels, rgb, output)
    return report


def _mesh_plot(ax, triangles, rgb, *, alpha=1.0):
    if len(triangles) == 0:
        return
    tris = np.asarray(triangles)[:, :, [0, 2, 1]]
    normal = np.cross(tris[:, 1]-tris[:, 0], tris[:, 2]-tris[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    light = np.array([-0.3, 0.4, 0.85])
    shade = 0.75 + 0.25*np.clip(normal @ light, 0, 1)
    rgba = plt.matplotlib.colors.to_rgba_array(rgb)
    if len(rgba) == 1:
        rgba = np.repeat(rgba, len(tris), axis=0)
    rgba[:, :3] *= shade[:, None]
    rgba[:, 3] = alpha
    ax.add_collection3d(Poly3DCollection(tris, facecolors=rgba, edgecolors="none",
                                        linewidths=0, rasterized=True))


def _camera(ax, center, half, *, azim=60, elev=20):
    ax.set(xlim=(center[0]-half[0], center[0]+half[0]),
           ylim=(center[2]-half[2], center[2]+half[2]),
           zlim=(center[1]-half[1], center[1]+half[1]))
    ax.set_box_aspect(np.asarray(half)[[0, 2, 1]])
    ax.view_init(elev=elev, azim=azim)
    ax.set_proj_type("ortho")
    ax.set_axis_off()


def _save(fig, output, name):
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(fig)


def _overview(mesh, labels, rgb, image_path, parts, output):
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 3, left=0.025, right=0.98, bottom=0.15, top=0.87,
                          hspace=0.15, wspace=0.06)
    ax = fig.add_subplot(gs[0, 0])
    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image)
    y, x = np.where(np.any(array < 235, axis=2))
    ax.imshow(image.crop((x.min(), y.min(), x.max(), y.max())))
    ax.axis("off")
    ax.set_title("a  Catalog photo", loc="left", weight="bold")
    center = mesh.bounds.mean(axis=0)
    half = mesh.extents*0.55
    for slot, title, face_colors in (
        (gs[0, 1], "b  Textured GLB (front view)", rgb),
        (gs[0, 2], "c  Native pieces + template names", np.asarray(COLORS)[labels]),
    ):
        ax = fig.add_subplot(slot, projection="3d")
        _mesh_plot(ax, mesh.triangles, face_colors)
        _camera(ax, center, half)
        ax.set_title(title, loc="left", weight="bold")
    ax = fig.add_subplot(gs[1, :2], projection="3d")
    triangles = np.array(mesh.triangles, copy=True)
    for part in range(len(NAMES)):
        offset = np.zeros(3)
        if 2 <= part <= 4:
            offset[1] = 0.24
        elif part == 1:
            offset[1], offset[2] = 0.09, 0.14
        elif 5 <= part <= 8:
            offset[1] = -0.15
        elif part == 9:
            offset[0] = 0.07
        triangles[labels == part] += offset
    _mesh_plot(ax, triangles, np.asarray(COLORS)[labels])
    _camera(ax, center+np.array([0, 0.03, 0]), half+np.array([0.04, 0.22, 0.05]))
    ax.set_title("d  Exploded view: one seat, three backs, four legs, trim and rail", loc="left", weight="bold")
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    cells = [[p["name"].replace(" (internal frame unresolved)", "*"),
              str(p["native_pieces"])] for p in parts]
    table = ax.table(cellText=cells, colLabels=["Assembly group", "Pieces"],
                     colWidths=[0.78, 0.22], cellLoc="left", bbox=(0, 0.03, 1, 0.95))
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#DDDDDD")
        if row == 0:
            cell.set_facecolor("#E9EDF2")
    fig.suptitle("Assembly detail pilot | ABO sofa B075X4QMX3", fontsize=14, weight="bold")
    fig.text(0.5, 0.91, "Native mesh connectivity + texture inspection; naming is sofa-specific and not a learned segmentation result.", ha="center", fontsize=8)
    fig.legend(handles=[Patch(color=COLORS[i], label=n.replace(" (internal frame unresolved)", "*"))
                        for i, n in enumerate(NAMES)], ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.035), fontsize=7)
    fig.text(0.5, 0.01, "* Main body remains a connected shell; hidden frame and manufacturing joints are not resolved by this asset.",
             ha="center", fontsize=7, color="#555555")
    _save(fig, output, "01_fine_assembly")


def _joints(support, connections, details, output):
    fig = plt.figure(figsize=(11, 7.5))
    gs = fig.add_gridspec(2, 2, left=0.06, right=0.96, bottom=0.19, top=0.84,
                          wspace=0.22, hspace=0.23)
    for i, (stats, (leg, center, points, distances)) in enumerate(zip(connections, details, strict=True)):
        ax = fig.add_subplot(gs[i//2, i % 2], projection="3d")
        half = np.array([0.07, 0.07, 0.07])
        rail_crop = crop_mesh(support, center, half)
        _mesh_plot(ax, rail_crop.triangles, ["#B2BEC9"])
        leg_crop = crop_mesh(leg, center, half)
        _mesh_plot(ax, leg_crop.triangles, [COLORS[i+5]])
        near = distances <= stats["near_threshold_m"]
        ax.scatter(points[near, 0], points[near, 2], points[near, 1],
                    color="#D55745", s=0.6, depthshade=False)
        _camera(ax, center, half, azim=45 if center[0] > 0 else 135, elev=22)
        ax.set_title(f"{stats['connection_id']}  {stats['source_name']} -> base rail", loc="left", weight="bold", fontsize=8)
        ax.text2D(0.02, -0.04,
                  f"5th-percentile distance: {stats['sample_q05_distance_mm']:.2f} mm\n"
                  f"Samples within 3 mm: {100*stats['near_sample_fraction']:.1f}% | strength: unverified",
                  transform=ax.transAxes, fontsize=7)
    fig.suptitle("Connection close-ups | four leg-to-rail candidate interfaces", fontsize=13, weight="bold")
    fig.text(0.5, 0.89, "Gray: rail surface   Color: leg   Red: near-surface samples in the upper 50 mm band", ha="center", fontsize=8)
    fig.text(0.5, 0.065, "Measured: unsigned surface proximity from 2,048 area-uniform samples per leg, to exact rail triangles.\nThis percentage is not contact area. Surface closeness does not identify penetration, glue, tenons or bolts.\nMechanical quality and safety factor require joint construction, material data, loads and supports; none are available here.",
             ha="center", fontsize=8, color="#555555")
    _save(fig, output, "02_joint_closeups")


def _details(mesh, labels, rgb, output):
    fig = plt.figure(figsize=(11.5, 6.8))
    gs = fig.add_gridspec(2, 3, left=0.025, right=0.98, bottom=0.18, top=0.85,
                          wspace=0.10, hspace=0.20)
    # These ROIs are explicitly selected on the example, not automatically detected defects.
    regions = [
        ("Back-cushion seam", np.array([-0.322, 0.70, -0.18]), np.array([0.12, 0.18, 0.16]), 60),
        ("Seat / arm interface", np.array([-0.94, 0.44, 0.19]), np.array([0.18, 0.14, 0.24]), 125),
        ("Nailhead trim", np.array([-1.075, 0.49, 0.17]), np.array([0.04, 0.16, 0.16]), 165),
    ]
    for i, (name, center, half, azim) in enumerate(regions):
        local = crop_mesh(mesh, center, half)
        if not len(local.faces):
            continue
        local_rgb = texture_face_rgb(local)
        for row in range(2):
            ax = fig.add_subplot(gs[row, i], projection="3d")
            if row == 0:
                color = local_rgb
                caption = f"{chr(97+i)}  {name}: texture"
            else:
                _, index = _nearest_faces(mesh, local.triangles_center)
                color = np.asarray(COLORS)[labels[index]]
                caption = f"{chr(100+i)}  {name}: assembly labels"
            _mesh_plot(ax, local.triangles, color)
            _camera(ax, center, half, azim=azim, elev=18)
            ax.set_title(caption, loc="left", weight="bold", fontsize=8)
    fig.suptitle("Local evidence | seams, cushion contact and decorative fasteners", fontsize=13, weight="bold")
    fig.text(0.5, 0.90, "Detail views preserve the original 49,850-face mesh; no early decimation.", ha="center", fontsize=8)
    fig.text(0.5, 0.065, "Example-specific regions selected for inspection. Decorative nailheads are not assumed to be structural fasteners.\nSeams, cushion contacts and load-bearing joints are different relations and must be classified separately.\nInternal foam, structural fasteners and bond quality cannot be read from the exterior texture alone.",
             ha="center", fontsize=8, color="#555555")
    _save(fig, output, "03_local_details")


def _nearest_faces(mesh, points):
    from scipy.spatial import cKDTree

    return cKDTree(mesh.triangles_center).query(points)
