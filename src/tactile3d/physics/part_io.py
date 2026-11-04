"""显式部件状态的JSON/NPZ存储及旧开发字段导入。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tactile3d.physics.part_state import (
    Connection,
    Interface,
    Material,
    PartPhysicalState,
    PhysicalObject,
)

SCHEMA_VERSION = 1
HASH_CHUNK_BYTES = 1024 * 1024


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(HASH_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_object(state: PhysicalObject, directory: str | Path) -> Path:
    """导出独立可加载的状态，拒绝覆盖已有状态目录。

    Args:
        state: 明确坐标、占用及材料来源的物体状态。
        directory: 新输出目录，防止覆盖源资产或旧实验。

    Returns:
        JSON清单路径，清单绑定NPZ哈希。
    """
    folder = Path(directory).expanduser().resolve()
    if folder.exists():
        raise FileExistsError(f"状态目录已存在：{folder}")
    folder.mkdir(parents=True)
    arrays, parts = {}, []
    for index, part in enumerate(state.parts):
        prefix = f"part_{index}"
        arrays[f"{prefix}_centers"] = part.centers_local_m
        arrays[f"{prefix}_fractions"] = part.fractions
        materials = []
        for phase, material in enumerate(part.materials):
            key = f"{prefix}_density_{phase}"
            arrays[key] = np.asarray(material.density_kg_m3)
            materials.append(
                {"name": material.name, "source": material.source, "density_key": key}
            )
        parts.append(
            {
                "part_id": part.part_id,
                "instance_id": part.instance_id,
                "version": part.version,
                "pitch_m": part.pitch_m,
                "local_to_world": part.local_to_world.tolist(),
                "construction": part.construction,
                "evidence": list(part.evidence),
                "mesh_ref": part.mesh_ref,
                "feature_refs": list(part.feature_refs),
                "centers_key": f"{prefix}_centers",
                "fractions_key": f"{prefix}_fractions",
                "materials": materials,
                "interfaces": [
                    {
                        "name": item.name,
                        "frame_local": item.frame_local.tolist(),
                        "size_m": item.size_m.tolist(),
                    }
                    for item in part.interfaces
                ],
            }
        )
    npz = folder / "cells.npz"
    np.savez_compressed(npz, **arrays)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "object_id": state.object_id,
        "version": state.version,
        "grid_origin_m": state.grid_origin_m.tolist(),
        "units": {"length": "m", "mass": "kg", "density": "kg/m^3"},
        "cells_sha256": file_sha256(npz),
        "parts": parts,
        "connections": [vars(edge) for edge in state.connections],
        "physics": state.total.properties(),
    }
    write_json(folder / "state.json", payload)
    return folder / "state.json"


def load_object(directory: str | Path) -> PhysicalObject:
    """校验存储完整性后重建状态，不信任旧物理缓存。"""
    folder = Path(directory).expanduser().resolve()
    payload = json.loads((folder / "state.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("units") != {
        "length": "m",
        "mass": "kg",
        "density": "kg/m^3",
    }:
        raise ValueError("不支持的状态版本或单位")
    file = folder / "cells.npz"
    if payload["cells_sha256"] != file_sha256(file):
        raise ValueError("体素文件校验失败，不能复用不一致的状态")
    parts = []
    with np.load(file, allow_pickle=False) as arrays:
        for row in payload["parts"]:
            materials = tuple(
                Material(item["name"], arrays[item["density_key"]], item["source"])
                for item in row["materials"]
            )
            parts.append(
                PartPhysicalState(
                    part_id=row["part_id"],
                    instance_id=row["instance_id"],
                    centers_local_m=arrays[row["centers_key"]],
                    pitch_m=row["pitch_m"],
                    materials=materials,
                    fractions=arrays[row["fractions_key"]],
                    local_to_world=row["local_to_world"],
                    interfaces=tuple(Interface(**item) for item in row["interfaces"]),
                    version=row["version"],
                    construction=row["construction"],
                    evidence=tuple(row["evidence"]),
                    mesh_ref=row["mesh_ref"],
                    feature_refs=tuple(row["feature_refs"]),
                )
            )
    return PhysicalObject(
        payload["object_id"],
        tuple(parts),
        payload["grid_origin_m"],
        tuple(Connection(**item) for item in payload["connections"]),
        version=payload["version"],
    )


def import_development_field(path: str | Path) -> PhysicalObject:
    """导入旧十物体密度均值，不伪造材料、接口或内部真值。

    Args:
        path: 含中心、部件标签、边长及有效密度的旧NPZ文件。

    Returns:
        保留空间密度积分的确定性状态；不迁移原样本相关性。
    """
    file = Path(path).expanduser().resolve()
    required = {"centers_m", "part_ids", "pitch_m", "density_mean_kg_m3"}
    with np.load(file, allow_pickle=False) as arrays:
        if not required <= set(arrays.files):
            raise ValueError("旧字段缺少中心、标签、边长或密度均值")
        centers = np.array(arrays["centers_m"], dtype=float)
        ids = np.array(arrays["part_ids"])
        density = np.array(arrays["density_mean_kg_m3"], dtype=float)
        pitch = float(arrays["pitch_m"])
    if centers.ndim != 2 or centers.shape[1:] != (3,) or not len(centers):
        raise ValueError("旧字段体素中心必须为非空(N,3)")
    if ids.shape != (len(centers),) or ids.dtype.kind not in "iu" or np.any(ids < 0):
        raise ValueError("旧字段部件标签必须为非负整数")
    if density.shape != (len(centers),):
        raise ValueError("旧字段密度长度不一致")
    parts = []
    for number in np.unique(ids):
        selected = ids == number
        # 有效混合密度已含占用影响，不再乘旧occupied_fraction。
        parts.append(
            PartPhysicalState(
                part_id=f"region_{number:02d}",
                instance_id=f"{file.parent.name}/region_{number:02d}/original",
                centers_local_m=centers[selected],
                pitch_m=pitch,
                materials=(
                    Material(
                        "effective_mixture",
                        density[selected],
                        "cached_untrained_effective_density_mean",
                    ),
                ),
                fractions=np.ones((int(selected.sum()), 1)),
                construction="unresolved_effective_density_only",
                evidence=("geometric_region_not_semantic_part", str(file)),
            )
        )
    return PhysicalObject(file.parent.name, tuple(parts), centers.min(axis=0))
