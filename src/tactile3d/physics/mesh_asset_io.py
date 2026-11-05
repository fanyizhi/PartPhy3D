"""Mesh原生区域状态的保存、加载、来源记录与显示几何导出。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from tactile3d.mesh.io import load_triangle_mesh
from tactile3d.physics.mesh_native import (
    ClosedMesh,
    MeshPhysicalObject,
    MeshRegion,
    closed_shell,
    combine_regions,
    positive_scalar,
)
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.physics.part_state import Material

SCHEMA = "mesh_native_regions_v1"
DEFAULT_RGBA = (160, 160, 160, 255)


@dataclass(frozen=True)
class MeshDisplay:
    vertices_m: np.ndarray
    faces: np.ndarray
    face_regions: np.ndarray


def display_mesh(state: MeshPhysicalObject) -> MeshDisplay:
    """为渲染合并区域边界，标签保留原区域索引而不靠材质名称匹配。"""
    vertices, faces, labels = [], [], []
    offset = 0
    for index, region in enumerate(state.regions):
        vertices.append(region.domain.vertices_m)
        faces.append(region.domain.faces + offset)
        labels.append(np.full(len(region.domain.faces), index, dtype=np.int64))
        offset += len(region.domain.vertices_m)
    return MeshDisplay(
        np.concatenate(vertices), np.concatenate(faces), np.concatenate(labels)
    )


def face_colors(state: MeshPhysicalObject, surface: MeshDisplay) -> np.ndarray:
    rgba = np.array(
        [r.material.display_rgba or DEFAULT_RGBA for r in state.regions], dtype=np.uint8
    )
    if np.any(rgba[:, 3] != 255):
        raise ValueError("当前Mesh示意渲染只支持不透明材料颜色")
    return rgba[surface.face_regions]


def save_mesh_asset(state: MeshPhysicalObject, directory: Path) -> None:
    """导出无需体素文件的区域资产，重载时重新检查几何和相交。

    Args:
        state: 区域互斥、单位明确的Mesh原生状态。
        directory: 尚不存在的目标目录，不覆盖历史资产。
    """
    if directory.exists():
        raise FileExistsError(f"拒绝覆盖已有Mesh资产：{directory}")
    surface = display_mesh(state)
    colors = face_colors(state, surface)
    directory.mkdir(parents=True)
    regions = []
    for index, region in enumerate(state.regions):
        filename = f"region_{index:03d}.npz"
        np.savez_compressed(
            directory / filename,
            vertices_m=region.domain.vertices_m,
            faces=region.domain.faces,
        )
        regions.append(
            {
                "part_id": region.part_id,
                "region_id": region.region_id,
                "region_version": region.version,
                "boundary": filename,
                "sha256": file_sha256(directory / filename),
                "provenance": region.domain.provenance,
                "fill_fraction": region.fill_fraction,
                "material": region.material.to_dict(),
            }
        )
    write_json(
        directory / "asset.json",
        {
            "schema": SCHEMA,
            "units": {"length": "m", "mass": "kg"},
            "object_id": state.object_id,
            "version": state.version,
            "regions": regions,
            "physics": state.properties,
            "voxelization_used": False,
        },
    )
    grouped: dict[str, list[MeshRegion]] = {}
    for region in state.regions:
        grouped.setdefault(region.part_id, []).append(region)
    write_json(
        directory / "part_tokens.json",
        {
            "representation": "executable_schema_not_learned_latent",
            "parts": [
                {
                    "part_id": part_id,
                    "region_ids": [r.region_id for r in group],
                    "geometry": {
                        "region_volumes_m3": {
                            r.region_id: r.domain.moments.volume_m3 for r in group
                        },
                    },
                    "structure": "explicit_closed_material_regions",
                    "materials": {r.region_id: r.material.to_dict() for r in group},
                    "macroscopic_fill": {r.region_id: r.fill_fraction for r in group},
                    "physics": combine_regions(tuple(group)),
                }
                for part_id, group in grouped.items()
            ],
        },
    )
    trimesh.Trimesh(
        surface.vertices_m,
        surface.faces,
        face_colors=colors,
        process=False,
    ).export(directory / "display.ply")


def load_mesh_asset(directory: Path) -> MeshPhysicalObject:
    """校验区域文件哈希后重算物理量，不信任保存的physics字段。"""
    payload = json.loads((directory / "asset.json").read_text(encoding="utf-8-sig"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != SCHEMA
        or payload.get("units")
        != {
            "length": "m",
            "mass": "kg",
        }
    ):
        raise ValueError("不支持的Mesh资产格式或单位")
    if not {"object_id", "version", "regions"} <= set(payload):
        raise ValueError("Mesh资产缺少物体ID、版本或区域字段")
    if not isinstance(payload["regions"], list) or not payload["regions"]:
        raise ValueError("Mesh资产需要非空区域列表")
    regions = []
    for row in payload["regions"]:
        required = {
            "part_id",
            "region_id",
            "boundary",
            "sha256",
            "provenance",
            "material",
            "fill_fraction",
            "region_version",
        }
        if not isinstance(row, dict) or not required <= set(row):
            raise ValueError("Mesh资产区域字段缺失")
        if not isinstance(row["boundary"], str) or not row["boundary"]:
            raise ValueError("区域boundary需要非空文件路径")
        path = (directory / row["boundary"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("区域文件必须位于资产目录内")
        if file_sha256(path) != row["sha256"]:
            raise ValueError("Mesh区域文件SHA256不匹配")
        with np.load(path, allow_pickle=False) as value:
            domain = ClosedMesh(value["vertices_m"], value["faces"], row["provenance"])
        regions.append(
            MeshRegion(
                row["part_id"],
                row["region_id"],
                domain,
                Material.from_dict(row["material"]),
                row["fill_fraction"],
                row["region_version"],
            )
        )
    return MeshPhysicalObject(payload["object_id"], tuple(regions), payload["version"])


def import_mesh_regions(path: Path) -> MeshPhysicalObject:
    """导入独立标注/生成的封闭区域，允许外边界减去已声明空腔。

    Args:
        path: 明确单位、区域文件、材料和来源的JSON；不能只有表面面片标签。

    Returns:
        经拓扑、体积和区域相交检查的物理对象。不能证明隐藏构造是真实的。
    """
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    allowed = {"object_id", "mesh_unit_to_meter", "regions"}
    if not isinstance(payload, dict) or set(payload) != allowed:
        raise ValueError("输入JSON需要object_id、mesh_unit_to_meter和regions")
    scale = positive_scalar(payload["mesh_unit_to_meter"], "长度单位到米的比例")
    rows = payload["regions"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("regions必须为非空列表")
    regions = []
    for row in rows:
        required = {"part_id", "region_id", "outer_mesh", "material", "provenance"}
        if (
            not isinstance(row, dict)
            or not required <= set(row)
            or set(row) - required - {"cavity_mesh", "fill_fraction"}
        ):
            raise ValueError("区域字段错误，表面面片需先建立有依据的闭合体域")

        def read_domain(filename: str, provenance: str) -> ClosedMesh:
            mesh = load_triangle_mesh(path.parent / filename, process=False)
            return ClosedMesh(mesh.vertices * scale, mesh.faces, provenance)

        domain = read_domain(row["outer_mesh"], row["provenance"])
        if row.get("cavity_mesh"):
            domain = closed_shell(
                domain, read_domain(row["cavity_mesh"], row["provenance"])
            )
        regions.append(
            MeshRegion(
                row["part_id"],
                row["region_id"],
                domain,
                Material.from_dict(row["material"]),
                row.get("fill_fraction", 1.0),
            )
        )
    return MeshPhysicalObject(payload["object_id"], tuple(regions))
