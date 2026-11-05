"""从本地ABO-500构建固定五物体子集，保留图像、相机和参考Mesh来源。"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
from importlib.resources import files as resource_files
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw

from tactile3d.abo500.dataset import ABO500Dataset, load_camera_views
from tactile3d.experiments.five_object_pilot import read_cohort
from tactile3d.mesh.io import load_triangle_mesh
from tactile3d.physics.part_io import file_sha256, write_json

DEFAULT_SOURCE = Path("data/abo_500")
DEFAULT_MESH_SOURCE = Path("data/abo500_ten")
DEFAULT_OUTPUT = Path("data/abo500_five")
DEFAULT_COHORT = Path("configs/five_object_pilot.json")
MANIFEST_NAME = "subset_manifest.json"
SCHEMA = "abo500_fixed_five_v1"
PREVIEW_CELL_SIZE = (300, 320)
CAMERA_ATOL = 1e-4


def write_attribution(directory: Path) -> None:
    """随子集保留上游署名及许可，读取安装包内资源而不依赖本机绝对路径。"""
    directory.mkdir(parents=True, exist_ok=True)
    resources = resource_files("tactile3d.abo500").joinpath("licenses")
    for resource in resources.iterdir():
        if resource.is_file():
            (directory / resource.name).write_bytes(resource.read_bytes())


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise TypeError(f"需要JSON对象：{path}")
    return value


def scene_files(scene: Path) -> list[Path]:
    """按相机清单收集所有观测，禁止缺图后静默减少视角。"""
    metadata = read_object(scene / "transforms.json")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"相机清单没有有效frames：{scene.name}")
    views = load_camera_views(scene)
    if len(views) != len(frames):
        raise ValueError(f"图像缺失或相机无效：{scene.name}")
    for view in views:
        if not view.image_path.is_relative_to(scene.resolve()):
            raise ValueError("图像路径越出对应物体目录")
        matrix = view.c2w
        if (
            not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=CAMERA_ATOL)
            or not np.allclose(
                matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=CAMERA_ATOL
            )
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=CAMERA_ATOL)
            or not np.isfinite([view.fx, view.fy, view.cx, view.cy]).all()
            or min(view.fx, view.fy, view.width, view.height) <= 0
        ):
            raise ValueError(f"相机参数不是有效刚体相机：{scene.name}")
        with Image.open(view.image_path) as image:
            if image.size != (view.width, view.height):
                raise ValueError(f"相机尺寸与图像不一致：{view.image_path}")
            image.verify()
    files = [scene / "transforms.json", *(view.image_path for view in views)]
    if (scene / "info.json").is_file():
        files.append(scene / "info.json")
    return sorted({path.resolve() for path in files})


def reference_mesh_report(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        header = stream.read(12)
    if len(header) != 12 or struct.unpack("<4sII", header) != (
        b"glTF",
        2,
        path.stat().st_size,
    ):
        raise ValueError(f"参考Mesh不是完整GLB 2文件：{path}")
    mesh = load_triangle_mesh(path)
    return {
        "vertices": len(mesh.vertices),
        "triangles": len(mesh.faces),
        "watertight": bool(mesh.is_watertight),
        "extent_in_file_units": mesh.extents.tolist(),
        "geometry_scale_verified": False,
        "image_mesh_registration_verified": False,
    }


def create_preview(root: Path, ids: list[str]) -> None:
    width, height = PREVIEW_CELL_SIZE
    canvas = Image.new("RGB", (len(ids) * width, height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, scene_id in enumerate(ids):
        scene = root / "scenes" / scene_id
        frames = read_object(scene / "transforms.json")["frames"]
        info_path = scene / "info.json"
        info = read_object(info_path) if info_path.is_file() else {}
        selected = int(info.get("idx_to_caption", 0))
        if not 0 <= selected < len(frames):
            raise ValueError(f"代表视角索引越界：{scene_id}")
        views = load_camera_views(scene)
        with Image.open(views[selected].image_path) as image:
            rgba = image.convert("RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            white.alpha_composite(rgba)
            box = rgba.getchannel("A").getbbox()
            thumbnail = white.crop(box).convert("RGB") if box else white.convert("RGB")
        thumbnail.thumbnail((width - 20, height - 95))
        left = index * width
        canvas.paste(thumbnail, (left + (width - thumbnail.width) // 2, 30))
        draw.text(
            (left + 8, 8), f"{index + 1:02d} {scene_id.split('_')[0]}", fill="black"
        )
        caption = str(info.get("caption", scene_id))
        draw.text((left + 8, height - 55), caption[:42], fill="black")
        draw.text((left + 8, height - 38), caption[42:84], fill="black")
        draw.text(
            (left + 8, height - 20), f"{len(frames)} original views", fill="black"
        )
    canvas.save(root / "selected_objects.jpg", quality=92)


def build_subset(
    source: Path,
    mesh_source: Path,
    output: Path = DEFAULT_OUTPUT,
    cohort: Path = DEFAULT_COHORT,
) -> dict[str, Any]:
    """复制固定五物体所需文件，显式映射质量标签并生成可核验清单。

    Args:
        source: 已解压ABO-500根目录。
        mesh_source: 已下载的原生参考GLB目录，不是生成Mesh。
        output: 不存在的子集目录。
        cohort: 已冻结的五物体配置，不按标签或成功率换样本。

    Returns:
        对象、视角、标签键映射和文件SHA-256清单；不执行推理或训练。
    """
    source, mesh_source, output = (
        path.expanduser().resolve() for path in (source, mesh_source, output)
    )
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有数据子集：{output}")
    if output.is_relative_to(source) or output.is_relative_to(mesh_source):
        raise ValueError("子集输出不能位于原始数据或参考Mesh目录内部")
    selection = read_cohort(cohort)
    ids = selection["object_ids"]
    dataset = ABO500Dataset(source)
    rows, files, weights = [], [], {}
    for scene_id in ids:
        scene = (source / "scenes" / scene_id).resolve()
        if not scene.is_relative_to(source / "scenes"):
            raise ValueError("非法物体ID：场景路径越出数据目录")
        asin = scene_id.split("_")[0]
        keys = [
            key for key in dict.fromkeys((scene_id, asin)) if key in dataset.weights
        ]
        if not keys:
            raise ValueError(f"缺少固定物体的质量标签：{scene_id}")
        if len(keys) == 2 and not np.isclose(
            dataset.weights[keys[0]], dataset.weights[keys[1]], rtol=1e-12, atol=0
        ):
            raise ValueError(f"场景ID与产品ID质量标签冲突：{scene_id}")
        key = keys[0]
        weights[scene_id] = dataset.weights[key]
        if not np.isfinite(weights[scene_id]) or weights[scene_id] <= 0:
            raise ValueError(f"质量标签不是有限正数：{scene_id}")
        original_split = [
            name for name, members in dataset.splits.items() if scene_id in members
        ]
        if len(original_split) != 1:
            raise ValueError(f"物体缺少唯一的原始数据分割：{scene_id}")
        selected_files = scene_files(scene)
        mesh = mesh_source / scene_id / "model.glb"
        report = reference_mesh_report(mesh)
        files.extend(
            (path, Path("scenes") / scene_id / path.relative_to(scene))
            for path in selected_files
        )
        files.append((mesh, Path("meshes_reference") / scene_id / "model.glb"))
        rows.append(
            {
                "scene_id": scene_id,
                "source_weight_key": key,
                "original_split": original_split[0],
                "views": len(load_camera_views(scene)),
                "reference_mesh": report,
            }
        )
    output.mkdir(parents=True)
    write_json(output / "build_status.json", {"status": "copying"})
    try:
        for original, relative in files:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, destination)
            if file_sha256(original) != file_sha256(destination):
                raise ValueError(f"复制校验失败：{relative}")
        write_json(output / "filtered_product_weights.json", weights)
        splits = {
            name: [row["scene_id"] for row in rows if row["original_split"] == name]
            for name in dataset.splits
        }
        write_json(output / "splits.json", splits)
        write_json(output / "selected_objects.json", selection)
        write_attribution(output)
        create_preview(output, ids)
        (output / "README.md").write_text(
            "# ABO-500 five-object development subset\n\n"
            f"{sum(row['views'] for row in rows)} original views; "
            "counts are in the manifest.\n"
            "scenes/: original observations, cameras and cached upstream info.json.\n"
            "meshes_reference/: native reference GLBs, not InstantMesh outputs.\n"
            "filtered_product_weights.json: evaluation labels, excluded from prompts.\n"
            "Scale and camera-to-GLB registration are not verified by copying files.\n"
            "No features/, ns/ caches or inferred part/physical labels are included.\n"
            "This is a development cohort, not an untouched test benchmark.\n"
            "Upstream data licenses apply; the project MIT license covers code only.\n",
            encoding="utf-8",
        )
        write_json(output / "build_status.json", {"status": "complete"})
        manifest = {
            "schema": SCHEMA,
            "status": "complete",
            "object_ids": ids,
            "num_objects": len(ids),
            "num_views": sum(row["views"] for row in rows),
            "held_out_benchmark": False,
            "inference_or_training_executed": False,
            "source_weights_sha256": file_sha256(
                source / "filtered_product_weights.json"
            ),
            "source_splits_sha256": file_sha256(source / "splits.json"),
            "cohort_sha256": selection["cohort_sha256"],
            "objects": rows,
            "files": {
                path.relative_to(output).as_posix(): file_sha256(path)
                for path in sorted(output.rglob("*"))
                if path.is_file()
            },
        }
        write_json(output / MANIFEST_NAME, manifest)
        return verify_subset(output)
    except Exception as error:
        write_json(
            output / "build_status.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def verify_subset(root: Path) -> dict[str, Any]:
    """核对可移植子集的完整文件清单、哈希和五个场景读取能力。"""
    root = root.expanduser().resolve()
    manifest = read_object(root / MANIFEST_NAME)
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("数据子集未完成或格式不支持")
    ids, files = manifest["object_ids"], manifest["files"]
    if len(ids) != 5 or len(set(ids)) != 5:
        raise ValueError("需要恰好五个唯一对象")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != set(files) | {MANIFEST_NAME}:
        raise ValueError("数据文件清单不一致，存在缺失或额外文件")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or file_sha256(path) != expected:
            raise ValueError(f"数据文件SHA256校验失败：{relative}")
    # 1仅使读取器能检查文件结构，不作为几何单位核验或写入尺度声明。
    dataset = ABO500Dataset(root, root / "meshes_reference", 1.0)
    if set(dataset.weights) != set(ids):
        raise ValueError("质量标签未与固定五物体一一对应")
    total = sum(len(dataset.get_scene(scene_id).views) for scene_id in ids)
    if total != manifest["num_views"]:
        raise ValueError("相机视角计数与清单不一致")
    return manifest


def archive_subset(root: Path, archive: Path) -> None:
    verify_subset(root)
    root, archive = root.resolve(), archive.expanduser().resolve()
    if archive.is_relative_to(root):
        raise ValueError("数据压缩包必须放在子集目录之外")
    if archive.exists():
        raise FileExistsError(f"拒绝覆盖已有压缩包：{archive}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive, "x", ZIP_DEFLATED) as bundle:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(root.name) / path.relative_to(root))
    with ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("数据压缩包完整性校验失败")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mesh-source", type=Path, default=DEFAULT_MESH_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    try:
        result = (
            verify_subset(args.output)
            if args.verify_only
            else build_subset(
                args.source_root, args.mesh_source, args.output, args.cohort
            )
        )
        if args.archive is not None:
            archive_subset(args.output, args.archive)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"五物体数据准备失败：{error}\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output.resolve()),
                "num_objects": result["num_objects"],
                "num_views": result["num_views"],
                "reference_meshes": len(result["objects"]),
                "inference_executed": False,
                "archive": str(args.archive.resolve()) if args.archive else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
