"""ABO参考Mesh的材料颜色预览，不运行重建或推断真实内部。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import trimesh
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch

from tactile3d.experiments.single_object import raster_face_ids
from tactile3d.mesh.assembly import texture_face_rgb
from tactile3d.physics.part_io import file_sha256, write_json

OBJECT_ID = "B082VML3QJ_ATVPDKIKX0DER"
DEFAULT_MESH = Path("data/abo500_ten") / OBJECT_ID / "model.glb"
DEFAULT_IMAGE = Path("data/material_color_demo/source.png")
DEFAULT_CAMERA = Path("data/material_color_demo/camera.json")
DEFAULT_OUTPUT = Path("outputs/material_color_preview")
MAX_SUPPORT_HEIGHT_M = 0.25
RED_BLUE_RATIO = 1.15
MIN_RED_VALUE = 0.2
PREVIEW_DPI = 180
RENDER_RESOLUTION = 700
MATERIALS = (
    ("wood", "Original mesh colors", "#B68B57", 650.0),
    ("aluminum", "Aluminum: silver gray", "#A5AFBA", 2700.0),
    ("abs", "ABS plastic: orange", "#EB8B35", 1050.0),
)
GLTF_TO_Z_UP = np.array(
    [
        [1.0, 0, 0, 0],
        [0, 0, -1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
    ]
)


def select_support(mesh: trimesh.Trimesh, rgb: np.ndarray) -> np.ndarray:
    """用本物体的显式颜色/高度规则选择木质底架和腿，不冒充通用分割。

    Args:
        mesh: 原生glTF坐标系中、单位为米的参考Mesh，y向上。
        rgb: 每个面的RGB颜色，范围为0到1。

    Returns:
        可保存和人工检查的面选择掩码。
    """
    if (
        rgb.shape != (len(mesh.faces), 3)
        or not np.isfinite(rgb).all()
        or np.any((rgb < 0) | (rgb > 1))
    ):
        raise ValueError("需要与三角面对应的有限RGB颜色")
    mask = (
        (rgb[:, 0] > rgb[:, 2] * RED_BLUE_RATIO)
        & (rgb[:, 0] > MIN_RED_VALUE)
        & (mesh.triangles_center[:, 1] < MAX_SUPPORT_HEIGHT_M)
    )
    if not np.any(mask) or np.all(mask):
        raise ValueError("未得到有效局部选择，请检查原图、坐标和选择规则")
    return mask


def rendered_colors(
    mesh: trimesh.Trimesh, rgb: np.ndarray, face_ids: np.ndarray
) -> np.ndarray:
    """按深度测试后的面编号着色，避免画家排序造成背面颜色穿透。"""
    displayed = mesh.copy()
    displayed.apply_transform(GLTF_TO_Z_UP)
    light = np.array([0.4, -0.6, 1.0])
    light /= np.linalg.norm(light)
    brightness = 0.65 + 0.35 * np.abs(displayed.face_normals @ light)
    shaded = np.clip(rgb * brightness[:, None], 0, 1)
    canvas = np.ones((*face_ids.shape, 3))
    visible = face_ids >= 0
    canvas[visible] = shaded[face_ids[visible]]
    return canvas


def plot_preview(
    image_path: Path,
    mesh: trimesh.Trimesh,
    variants: list[np.ndarray],
    face_ids: np.ndarray,
    output: Path,
) -> None:
    """绘制真实数据集图像与三维材质颜色示意，不重绘或修改输入照片。"""
    figure = plt.figure(figsize=(17, 5.7))
    figure.subplots_adjust(left=0.03, right=0.99, top=0.80, bottom=0.27, wspace=0.17)
    y, x = np.nonzero(face_ids >= 0)
    margin = 20
    x0, x1 = max(0, x.min() - margin), min(face_ids.shape[1], x.max() + margin + 1)
    y0, y1 = max(0, y.min() - margin), min(face_ids.shape[0], y.max() + margin + 1)
    axis = figure.add_subplot(1, 4, 1)
    with Image.open(image_path) as source:
        pixels = np.asarray(source.convert("RGBA").resize(face_ids.shape[::-1]))
        axis.imshow(pixels[y0:y1, x0:x1])
    axis.set_title("ABO view (display crop)", fontsize=12)
    axis.set_axis_off()
    for index, ((_, title, color, density), rgb) in enumerate(
        zip(MATERIALS, variants, strict=True), start=2
    ):
        axis = figure.add_subplot(1, 4, index)
        rendered = rendered_colors(mesh, rgb, face_ids)
        axis.imshow(rendered[y0:y1, x0:x1], extent=(x0, x1, y1, y0))
        axis.set(title=title, xlabel="u (pixel)", ylabel="v (pixel)")
        axis.set_xticks([x0, x1])
        axis.set_yticks([y0, y1])
        axis.text(
            0.05,
            -0.34,
            f"Density preset: {density:.0f} kg/m³\n"
            f"Target mass ratio: {density / MATERIALS[0][3]:.2f}x*",
            transform=axis.transAxes,
            fontsize=10,
        )
    figure.suptitle(
        "Material-edit preview: keep the upholstered seat, change its support",
        fontsize=15,
        y=0.97,
    )
    figure.legend(
        handles=[Patch(color=color, label=name) for name, _, color, _ in MATERIALS],
        loc="upper center",
        bbox_to_anchor=(0.62, 0.90),
        ncol=3,
        frameon=False,
    )
    figure.text(
        0.03,
        0.025,
        "*Equal solid-volume design assumption; selected-region ratio, "
        "not total-object mass.\n"
        "Reference mesh + object-specific selection. Colors indicate material choices; "
        "no reconstructed mesh, measured density or absolute mass is claimed.",
        fontsize=9,
        color="#444444",
    )
    for suffix in ("svg", "pdf", "png"):
        figure.savefig(output / f"material_preview.{suffix}", dpi=PREVIEW_DPI)
    plt.close(figure)


def run_preview(
    mesh_path: Path, image_path: Path, output: Path, camera_path: Path = DEFAULT_CAMERA
) -> dict[str, Any]:
    """导出颜色预览及面掩码，只改变颜色，显式保留物理推断缺失状态。

    Args:
        mesh_path: 带颜色/纹理的参考网格。
        image_path: 用于说明样例的原始数据集视图。
        output: 必须尚不存在的结果目录。
        camera_path: 与输入图像对应的ABO相机内外参。

    Returns:
        含颜色选择、条件质量比及来源哈希的报告。
    """
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有结果：{output}")
    if not all(path.is_file() for path in (mesh_path, image_path, camera_path)):
        raise FileNotFoundError("需要参考Mesh、数据集图像和相机，请检查输入路径")
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("输入没有可显示的三角面")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("网格坐标必须有限")
    rgb = texture_face_rgb(mesh)
    selected = select_support(mesh, rgb)
    camera = json.loads(camera_path.read_text(encoding="utf-8"))
    required = {"w", "h", "fl_x", "fl_y", "cx", "cy", "camera_to_world"}
    if not isinstance(camera, dict) or not required <= set(camera):
        raise ValueError("相机文件缺少图像尺寸、内参或camera_to_world")
    pose = np.asarray(camera["camera_to_world"], dtype=float)
    scalars = np.asarray(
        [camera[key] for key in ("w", "h", "fl_x", "fl_y", "cx", "cy")]
    )
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.isfinite(scalars).all()
        or np.any(scalars[:4] <= 0)
    ):
        raise ValueError("相机形状或数值无效")
    if not np.allclose(pose[3], [0, 0, 0, 1]):
        raise ValueError("相机位姿不是齐次变换")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-5
    ):
        raise ValueError("相机位姿必须是有效刚体变换")
    face_ids = raster_face_ids(mesh, pose, camera, RENDER_RESOLUTION)
    with Image.open(image_path) as image:
        alpha = (
            np.asarray(
                image.convert("RGBA")
                .getchannel("A")
                .resize(face_ids.shape[::-1], Image.Resampling.NEAREST)
            )
            > 0
        )
    visible = face_ids >= 0
    union = np.count_nonzero(alpha | visible)
    if union == 0 or not np.any(visible):
        raise ValueError("相机未观察到有效网格")
    silhouette_iou = float(np.count_nonzero(alpha & visible) / union)
    allowed_pixels = np.zeros_like(visible)
    allowed_pixels[visible] = selected[face_ids[visible]]
    original_render = rendered_colors(mesh, rgb, face_ids)
    variants, rows = [], []
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "selected_faces.npy", selected)
    for index, (name, _, color, density) in enumerate(MATERIALS):
        edited_rgb = rgb.copy()
        if index:
            edited_rgb[selected] = colors.to_rgb(color)
        if not np.array_equal(edited_rgb[~selected], rgb[~selected]):
            raise ValueError("非目标区域的颜色意外变化")
        edited_render = rendered_colors(mesh, edited_rgb, face_ids)
        changed = np.any(edited_render != original_render, axis=2)
        if np.any(changed & ~allowed_pixels):
            raise ValueError("预览改变了非目标可见区域")
        edited = mesh.copy()
        edited.visual = trimesh.visual.ColorVisuals(
            mesh=edited, face_colors=np.column_stack([edited_rgb, np.ones(len(rgb))])
        )
        if not (
            np.array_equal(edited.vertices, mesh.vertices)
            and np.array_equal(edited.faces, mesh.faces)
        ):
            raise ValueError("颜色预览意外改变了网格几何")
        edited.export(output / f"{name}.ply")
        variants.append(edited_rgb)
        rows.append(
            {
                "material_preset": name,
                "display_color": color,
                "density_preset_kg_m3": density,
                "conditional_target_mass_ratio": density / MATERIALS[0][3],
                "absolute_object_mass_kg": None,
            }
        )
    plot_preview(image_path, mesh, variants, face_ids, output)
    report = {
        "status": "color_preview_complete",
        "mesh_source": "reference_asset_not_InstantMesh_output",
        "mesh_path": str(mesh_path.resolve()),
        "image_path": str(image_path.resolve()),
        "mesh_sha256": file_sha256(mesh_path),
        "image_sha256": file_sha256(image_path),
        "code_sha256": file_sha256(Path(__file__)),
        "camera_sha256": file_sha256(camera_path),
        "silhouette_iou": silhouette_iou,
        "renderer": "perspective face-ID depth buffer; fixed dataset camera",
        "renderer_sha256": file_sha256(Path(__file__).with_name("single_object.py")),
        "selected_faces": int(selected.sum()),
        "total_faces": len(mesh.faces),
        "selection": "object-specific color and height rule; not learned segmentation",
        "target": "wooden base and legs, grouped for this appearance preview",
        "geometry_unchanged": True,
        "unselected_face_colors_unchanged": True,
        "visible_pixel_edit_support_verified": True,
        "watertight_source_mesh": bool(mesh.is_watertight),
        "physics_committed": False,
        "mass_ratio_condition": "same solid material volume; preset densities only",
        "absolute_mass_status": (
            "not estimated: internal volume/construction unverified"
        ),
        "variants": rows,
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--camera", type=Path, default=DEFAULT_CAMERA)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUT / f"run_{time.time_ns()}"
    try:
        report = run_preview(args.mesh, args.image, output, args.camera)
    except (OSError, ValueError) as error:
        parser.exit(1, f"预览失败：{error}\n")
    print(f"预览完成：{output.resolve()}；目标三角面：{report['selected_faces']}")


if __name__ == "__main__":
    main()
