"""固定构造下的部件区域、表面显示与物理字段共同编辑。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tactile3d.physics.part_editing import EditSpec, apply_edit, verify_full_recompute
from tactile3d.physics.part_io import file_sha256, load_object, save_object, write_json
from tactile3d.physics.part_state import (
    FloatArray,
    Material,
    PhysicalObject,
    frozen_array,
)

DEFAULT_RGBA = (160, 160, 160, 255)
MASS_RTOL = 1e-10
MASS_ATOL_KG = 1e-12


@dataclass(frozen=True)
class MaterialRegion:
    part_id: str
    region_id: str
    material_slot: int

    def __post_init__(self) -> None:
        if not all(
            isinstance(v, str) and v.strip() for v in (self.part_id, self.region_id)
        ):
            raise ValueError("区域需要明确的part_id和region_id")
        if type(self.material_slot) is not int or self.material_slot < 0:
            raise ValueError("材料槽位必须为非负整数")


def geometry_signature(state: PhysicalObject) -> str:
    """只绑定不可被本次换材改变的域，材料名称与密度不进入几何签名。"""
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            [state.object_id, list(state.grid_origin_m)], ensure_ascii=False
        ).encode()
    )
    for part in state.parts:
        identity = [
            part.part_id,
            part.instance_id,
            part.construction,
            part.pitch_m,
            len(part.materials),
        ]
        digest.update(json.dumps(identity, ensure_ascii=False).encode())
        for value in (part.centers_local_m, part.fractions, part.local_to_world):
            array = np.asarray(value, dtype="<f8")
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


def integer_array(value: Any, ndim: int) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != ndim or array.dtype.kind not in "iu":
        raise ValueError("索引必须为整数数组，不允许截断小数或布尔值")
    result = np.array(array, dtype=np.int64, copy=True)
    if np.any(result < 0):
        raise ValueError("绑定不接受负索引或未标注的面")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class BoundSurface:
    vertices_m: FloatArray
    faces: NDArray[np.int64]
    face_regions: NDArray[np.int64]
    regions: tuple[MaterialRegion, ...]
    geometry_sha256: str
    label_source: str

    def __post_init__(self) -> None:
        vertices = frozen_array(self.vertices_m, 2)
        faces = integer_array(self.faces, 2)
        labels = integer_array(self.face_regions, 1)
        regions = tuple(self.regions)
        if (
            vertices.shape[1:] != (3,)
            or faces.shape[1:] != (3,)
            or not len(vertices)
            or not len(faces)
            or not regions
        ):
            raise ValueError("表面需要非空三角面、顶点和区域")
        if labels.shape != (len(faces),) or faces.max() >= len(vertices):
            raise ValueError("面索引或逐面区域数量错误")
        if labels.max() >= len(regions) or not all(
            isinstance(r, MaterialRegion) for r in regions
        ):
            raise ValueError("区域标签超界或区域描述错误")
        if len({(r.part_id, r.region_id) for r in regions}) != len(regions):
            raise ValueError("同部件内区域ID不能重复")
        if len({(r.part_id, r.material_slot) for r in regions}) != len(regions):
            raise ValueError("当前每个材料槽位只允许绑定一个区域，避免隐式别名编辑")
        if not isinstance(self.label_source, str) or not self.label_source.strip():
            raise ValueError("必须说明表面/体积对应的标注来源")
        if not isinstance(self.geometry_sha256, str) or len(self.geometry_sha256) != 64:
            raise ValueError("需要几何状态SHA256绑定")
        triangle = vertices[faces]
        if np.any(
            np.linalg.norm(
                np.cross(
                    triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
                ),
                axis=1,
            )
            <= 0
        ):
            raise ValueError("表面包含退化三角形")
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "face_regions", labels)
        object.__setattr__(self, "regions", regions)

    def validate_state(self, state: PhysicalObject) -> None:
        """防止几何或占用改变后，旧面标签仍被当作有效的体积对应。"""
        if geometry_signature(state) != self.geometry_sha256:
            raise ValueError("表面绑定已过期：构造、占用、几何或部件实例已改变")
        for region in self.regions:
            part = state.part(region.part_id)
            if region.material_slot >= len(part.materials):
                raise ValueError("区域材料槽位超界")
            material = part.materials[region.material_slot]
            if not np.any(part.fractions[:, region.material_slot] > 0) or np.any(
                material.values(len(part.fractions)) <= 0
            ):
                raise ValueError("区域必须对应实际存在的正密度材料，不能把空腔当材料")

    def selected_faces(self, part_id: str, region_id: str) -> NDArray[np.bool_]:
        for index, region in enumerate(self.regions):
            if (region.part_id, region.region_id) == (part_id, region_id):
                return self.face_regions == index
        raise ValueError(f"未绑定的部件/区域：{part_id}/{region_id}")

    def face_rgba(self, state: PhysicalObject) -> NDArray[np.uint8]:
        """颜色从当前材料状态读取，避免显示仍使用编辑前的独立缓存。"""
        self.validate_state(state)
        colors = []
        for region in self.regions:
            material = state.part(region.part_id).materials[region.material_slot]
            color = material.display_rgba or DEFAULT_RGBA
            if color[3] != 255:
                raise ValueError("当前深度渲染仅支持不透明示意色")
            colors.append(color)
        return np.asarray(colors, dtype=np.uint8)[self.face_regions]


def edit_bound_material(
    state: PhysicalObject,
    surface: BoundSurface,
    part_id: str,
    region_id: str,
    new_material: Material,
    expected_part_version: int,
) -> tuple[PhysicalObject, dict[str, Any]]:
    """保持区域体积与表面不变，同时提交材料物理参数和示意色。

    Args:
        state: 已有明确区域占用的物体；不从外表面猜测内部。
        surface: 按部件和材料槽位标注的米制Mesh。
        part_id: 稳定部件槽位。
        region_id: 明确材料区域，不隐含扩展到整个部件。
        new_material: 新密度、可选弹性参数与必要示意色。
        expected_part_version: 拒绝陈旧选择产生的编辑。

    Returns:
        新状态及区域、守恒条件、质量变化和完整重算核对报告。
    """
    surface.validate_state(state)
    if type(expected_part_version) is not int or expected_part_version < 0:
        raise ValueError("绑定换材必须指定非负整数部件版本")
    selected = surface.selected_faces(part_id, region_id)
    if not isinstance(new_material, Material):
        raise TypeError("新材料必须为Material")
    if new_material.display_rgba is None or new_material.display_rgba[3] != 255:
        raise ValueError("同步编辑需要明确的不透明RGBA示意色")
    if np.any(new_material.density_kg_m3 <= 0):
        raise ValueError("固定占用的换材必须保持正密度；删除材料属于构造编辑")
    part = state.part(part_id)
    region = next(
        r for r in surface.regions if (r.part_id, r.region_id) == (part_id, region_id)
    )
    old = part.materials[region.material_slot]
    before_colors = surface.face_rgba(state)
    result = apply_edit(
        state,
        EditSpec(
            part_id,
            "material_change",
            expected_part_version=expected_part_version,
            material_name=old.name,
            new_material=new_material,
        ),
    )
    surface.validate_state(result.state)
    after_colors = surface.face_rgba(result.state)
    if not np.array_equal(before_colors[~selected], after_colors[~selected]):
        raise ValueError("非目标区域外观发生变化，拒绝提交")
    occupancy = part.fractions[:, region.material_slot]
    # 占用表示宏观材料体积；不能把整个部件包围体积再乘一次填充率。
    change = new_material.values(len(occupancy)) - old.values(len(occupancy))
    expected_delta = float(np.dot(occupancy, change) * part.pitch_m**3)
    actual_delta = result.state.total.mass_kg - state.total.mass_kg
    if not np.isclose(actual_delta, expected_delta, rtol=MASS_RTOL, atol=MASS_ATOL_KG):
        raise ValueError("换材后的质量变化与固定区域积分不一致")
    return result.state, {
        **result.report,
        "region_id": region_id,
        "material_slot": region.material_slot,
        "old_material": old.to_dict(),
        "new_material": new_material.to_dict(),
        "selected_faces": int(selected.sum()),
        "changed_faces": int(np.any(before_colors != after_colors, axis=1).sum()),
        "surface_label_source": surface.label_source,
        "occupied_region_volume_m3": float(occupancy.sum() * part.pitch_m**3),
        "geometry_and_occupancy_unchanged": True,
        "expected_delta_mass_kg": expected_delta,
        "delta_mass_kg": actual_delta,
        "full_recompute_error": verify_full_recompute(result.state),
        "appearance_rendered": False,
        "appearance_bound_to_state": True,
        "elasticity_scope": (
            "declared isotropic material values; no stiffness/strength validation"
        ),
    }


def save_bound_asset(
    state: PhysicalObject, surface: BoundSurface, directory: Path
) -> None:
    """保存能够独立重放的物理与表面绑定，不覆盖原资产。

    Args:
        state: 已校验的共同体积状态。
        surface: 与该状态构造对应的逐面区域标签。
        directory: 尚不存在的目标目录；表面NPZ另有SHA256校验。
    """
    surface.validate_state(state)
    if directory.exists():
        raise FileExistsError(f"拒绝覆盖已有绑定资产：{directory}")
    directory.mkdir(parents=True)
    save_object(state, directory / "physical")
    np.savez_compressed(
        directory / "surface.npz",
        vertices_m=surface.vertices_m,
        faces=surface.faces,
        face_regions=surface.face_regions,
    )
    write_json(
        directory / "binding.json",
        {
            "schema_version": 1,
            "geometry_sha256": surface.geometry_sha256,
            "label_source": surface.label_source,
            "regions": [vars(region) for region in surface.regions],
            "surface_sha256": file_sha256(directory / "surface.npz"),
        },
    )


def load_bound_asset(directory: Path) -> tuple[PhysicalObject, BoundSurface]:
    """检查文件完整性与几何对应后加载，不信任保存时的质量缓存。

    Returns:
        重新积分的物理状态及不可变表面绑定。标注真实性仍需独立验证。
    """
    payload = json.loads((directory / "binding.json").read_text(encoding="utf-8-sig"))
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
        raise ValueError("不支持的绑定版本")
    if payload["surface_sha256"] != file_sha256(directory / "surface.npz"):
        raise ValueError("绑定表面文件SHA256错误")
    state = load_object(directory / "physical")
    with np.load(directory / "surface.npz", allow_pickle=False) as arrays:
        surface = BoundSurface(
            arrays["vertices_m"],
            arrays["faces"],
            arrays["face_regions"],
            tuple(MaterialRegion(**region) for region in payload["regions"]),
            payload["geometry_sha256"],
            payload["label_source"],
        )
    surface.validate_state(state)
    return state, surface
