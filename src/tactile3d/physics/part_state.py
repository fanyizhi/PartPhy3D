"""统一体素域中的不可变部件状态、接口和原始质量矩。"""

from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

FloatArray = NDArray[np.float64]
ARRAY_TOLERANCE = 1e-9
GRID_TOLERANCE = 1e-6
POSITION_TOLERANCE_M = 1e-5
ANGLE_TOLERANCE_RAD = np.deg2rad(0.1)
SIZE_TOLERANCE_M = 1e-5
MAX_GRID_INDEX = 10**9
MAX_INTERFACE_SAMPLES = 50000
CONTACT_SAMPLE_OFFSET_CELLS = 0.25


def frozen_array(value: Any, ndim: int | None = None) -> FloatArray:
    """复制有限数值并冻结，避免外部数组修改使物理缓存失效。"""
    result = np.array(value, dtype=np.float64, copy=True)
    if not np.isfinite(result).all() or (ndim is not None and result.ndim != ndim):
        raise ValueError("数组必须有限且维数正确")
    result.flags.writeable = False
    return result


def rigid_frame(value: Any) -> FloatArray:
    """校验米制刚体变换，禁止把缩放伪装成姿态变换。"""
    frame = frozen_array(value, 2)
    if frame.shape != (4, 4) or not np.allclose(
        frame[3], [0, 0, 0, 1], atol=ARRAY_TOLERANCE, rtol=0
    ):
        raise ValueError("坐标变换必须为4x4齐次矩阵")
    rotation = frame[:3, :3]
    if not np.allclose(
        rotation.T @ rotation, np.eye(3), atol=ARRAY_TOLERANCE, rtol=0
    ) or not np.isclose(np.linalg.det(rotation), 1, atol=ARRAY_TOLERANCE):
        raise ValueError("坐标变换仅允许旋转和平移，不允许缩放或镜像")
    return frame


@dataclass(frozen=True)
class Material:
    """密度可为标量或与部件体素一一对应的字段。"""

    name: str
    density_kg_m3: float | FloatArray
    source: str = "user_specified"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not isinstance(self.source, str):
            raise ValueError("材料名称和来源必须为字符串")
        if not self.name.strip() or not self.source.strip():
            raise ValueError("材料名称和来源不能为空")
        density = frozen_array(self.density_kg_m3)
        if density.ndim > 1 or density.size == 0 or np.any(density < 0):
            raise ValueError("密度必须是非负标量或一维字段")
        object.__setattr__(self, "density_kg_m3", density)

    def values(self, count: int) -> FloatArray:
        if count <= 0:
            raise ValueError("体素数量必须为正")
        density = np.asarray(self.density_kg_m3)
        if density.ndim == 1 and density.shape != (count,):
            raise ValueError("密度字段长度必须与部件体素数量一致")
        return np.broadcast_to(density, (count,))


@dataclass(frozen=True)
class Interface:
    """安装坐标系，不是两个接触面各自朝外的法向坐标系。"""

    name: str
    frame_local: FloatArray
    size_m: FloatArray

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("接口名称不能为空")
        size = frozen_array(self.size_m, 1)
        if size.shape != (2,) or np.any(size <= 0):
            raise ValueError("接口尺寸必须为两个正数，单位为米")
        object.__setattr__(self, "frame_local", rigid_frame(self.frame_local))
        object.__setattr__(self, "size_m", size)


@dataclass(frozen=True)
class MassMoments:
    mass_kg: float
    first_kg_m: FloatArray
    second_kg_m2: FloatArray

    def __post_init__(self) -> None:
        first = frozen_array(self.first_kg_m, 1)
        second = frozen_array(self.second_kg_m2, 2)
        if (
            not np.isfinite(self.mass_kg)
            or self.mass_kg < 0
            or first.shape != (3,)
            or second.shape != (3, 3)
        ):
            raise ValueError("质量矩的数值或形状无效")
        object.__setattr__(self, "first_kg_m", first)
        object.__setattr__(self, "second_kg_m2", second)

    def properties(self) -> dict[str, Any]:
        """从共同坐标系原始矩恢复质心与完整惯量张量。"""
        if self.mass_kg <= 0:
            return {
                "mass_kg": 0.0,
                "center_of_mass_m": None,
                "inertia_kg_m2": np.zeros((3, 3)).tolist(),
            }
        center = self.first_kg_m / self.mass_kg
        central = self.second_kg_m2 - self.mass_kg * np.outer(center, center)
        inertia = np.trace(central) * np.eye(3) - central
        return {
            "mass_kg": float(self.mass_kg),
            "center_of_mass_m": center.tolist(),
            "inertia_kg_m2": ((inertia + inertia.T) / 2).tolist(),
        }


@dataclass(frozen=True)
class PartPhysicalState:
    part_id: str
    instance_id: str
    centers_local_m: FloatArray
    pitch_m: float
    materials: tuple[Material, ...]
    fractions: FloatArray
    local_to_world: FloatArray = field(default_factory=lambda: np.eye(4))
    interfaces: tuple[Interface, ...] = ()
    version: int = 0
    construction: str = "specified_voxel_field"
    evidence: tuple[str, ...] = ()
    mesh_ref: str | None = None
    feature_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        centers = frozen_array(self.centers_local_m, 2)
        fractions = frozen_array(self.fractions, 2)
        materials, interfaces = tuple(self.materials), tuple(self.interfaces)
        if not isinstance(self.part_id, str) or not isinstance(self.instance_id, str):
            raise ValueError("部件槽位ID和实例ID必须为字符串")
        if not self.part_id.strip() or not self.instance_id.strip():
            raise ValueError("部件槽位ID和实例ID不能为空")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("部件版本必须为非负整数")
        if not np.isfinite(self.pitch_m) or self.pitch_m <= 0:
            raise ValueError("体素边长必须为正数，单位为米")
        if centers.shape[1:] != (3,) or not len(centers) or not materials:
            raise ValueError("需要非空(N,3)体素中心和材料列表")
        if fractions.shape != (len(centers), len(materials)):
            raise ValueError("材料占用必须为(N,K)数组")
        if np.any(fractions < 0) or not np.allclose(
            fractions.sum(axis=1), 1, atol=ARRAY_TOLERANCE, rtol=0
        ):
            raise ValueError("含空腔的材料占用必须非负且每行和为1")
        if not all(isinstance(item, Material) for item in materials):
            raise ValueError("部件材料必须使用Material状态")
        if not all(isinstance(item, Interface) for item in interfaces):
            raise ValueError("部件接口必须使用Interface状态")
        if len({item.name for item in materials}) != len(materials):
            raise ValueError("同一部件的材料名称必须唯一")
        if len({item.name for item in interfaces}) != len(interfaces):
            raise ValueError("同一部件的接口名称必须唯一")
        for material in materials:
            material.values(len(centers))
        object.__setattr__(self, "centers_local_m", centers)
        object.__setattr__(self, "fractions", fractions)
        object.__setattr__(self, "materials", materials)
        object.__setattr__(self, "interfaces", interfaces)
        object.__setattr__(self, "local_to_world", rigid_frame(self.local_to_world))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "feature_refs", tuple(self.feature_refs))

    @property
    def centers_world_m(self) -> FloatArray:
        return (
            self.centers_local_m @ self.local_to_world[:3, :3].T
            + self.local_to_world[:3, 3]
        )

    @property
    def density_kg_m3(self) -> FloatArray:
        density = np.column_stack(
            [item.values(len(self.centers_local_m)) for item in self.materials]
        )
        return np.einsum("ij,ij->i", self.fractions, density)

    def interface(self, name: str) -> Interface:
        for item in self.interfaces:
            if item.name == name:
                return item
        raise ValueError(f"部件{self.part_id}不存在接口{name}")

    def moments(self) -> MassMoments:
        """先在局部坐标积分，再解析变换到共同世界坐标。

        Returns:
            含有限立方单元尺寸贡献的原始质量矩，采用SI单位。
        """
        x = self.centers_local_m
        mass = self.density_kg_m3 * self.pitch_m**3
        total = float(mass.sum())
        # 单元是有限立方体，二阶矩必须保留自身尺寸贡献。
        second = np.einsum("i,ij,ik->jk", mass, x, x)
        second += np.eye(3) * total * self.pitch_m**2 / 12
        rotation = self.local_to_world[:3, :3]
        translation = self.local_to_world[:3, 3]
        first = rotation @ (mass @ x)
        # 避免逐体素累加带大平移的二阶矩，降低恢复中心惯量时的消减误差。
        second = (
            rotation @ second @ rotation.T
            + np.outer(first, translation)
            + np.outer(translation, first)
            + total * np.outer(translation, translation)
        )
        return MassMoments(total, first + total * translation, second)


@dataclass(frozen=True)
class Connection:
    part_a: str
    interface_a: str
    part_b: str
    interface_b: str
    relation: str = "fixed"

    def __post_init__(self) -> None:
        if self.part_a == self.part_b or not all(
            [self.part_a, self.interface_a, self.part_b, self.interface_b]
        ):
            raise ValueError("连接需要两个不同部件及明确接口")
        if self.relation != "fixed":
            raise ValueError("首版仅支持已声明的固定安装接口")


def interface_error(
    first: PartPhysicalState,
    first_name: str,
    second: PartPhysicalState,
    second_name: str,
) -> dict[str, float]:
    a, b = first.interface(first_name), second.interface(second_name)
    frame_a = first.local_to_world @ a.frame_local
    frame_b = second.local_to_world @ b.frame_local
    rotation = frame_a[:3, :3].T @ frame_b[:3, :3]
    return {
        "position_m": float(np.linalg.norm(frame_a[:3, 3] - frame_b[:3, 3])),
        "angle_rad": float(np.arccos(np.clip((np.trace(rotation) - 1) / 2, -1, 1))),
        "size_m": float(np.max(np.abs(a.size_m - b.size_m))),
    }


def require_interface_match(errors: dict[str, float]) -> None:
    limits = {
        "position_m": POSITION_TOLERANCE_M,
        "angle_rad": ANGLE_TOLERANCE_RAD,
        "size_m": SIZE_TOLERANCE_M,
    }
    if any(
        not np.isfinite(errors[key]) or errors[key] > limit
        for key, limit in limits.items()
    ):
        raise ValueError(f"安装接口不兼容：{errors}")


@dataclass(frozen=True)
class PhysicalObject:
    object_id: str
    parts: tuple[PartPhysicalState, ...]
    grid_origin_m: FloatArray = field(default_factory=lambda: np.zeros(3))
    connections: tuple[Connection, ...] = ()
    version: int = 0
    _reuse: InitVar[PhysicalObject | None] = None
    _cache: tuple[MassMoments, ...] = field(init=False, repr=False)
    total: MassMoments = field(init=False)
    recomputed_part_ids: tuple[str, ...] = field(init=False)

    def __post_init__(self, _reuse: PhysicalObject | None) -> None:
        parts, connections = tuple(self.parts), tuple(self.connections)
        origin = frozen_array(self.grid_origin_m, 1)
        if not self.object_id.strip() or not parts or origin.shape != (3,):
            raise ValueError("物体需要ID、非空部件和三维网格原点")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("物体版本必须为非负整数")
        if len({part.part_id for part in parts}) != len(parts):
            raise ValueError("部件槽位ID不能重复")
        if len({part.instance_id for part in parts}) != len(parts):
            raise ValueError("部件实例ID不能重复")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "connections", connections)
        object.__setattr__(self, "grid_origin_m", origin)
        indices = [self.grid_indices(part) for part in parts]
        all_indices = np.concatenate(indices)
        if len(np.unique(all_indices, axis=0)) != len(all_indices):
            raise ValueError("体积单元重叠或重复归属，不能重复计算质量")
        for edge in connections:
            require_interface_match(
                interface_error(
                    self.part(edge.part_a),
                    edge.interface_a,
                    self.part(edge.part_b),
                    edge.interface_b,
                )
            )
            a = self.interface_material_sides(edge.part_a, edge.interface_a)
            b = self.interface_material_sides(edge.part_b, edge.interface_b)
            opposite = (
                np.allclose(a, [1, 0], atol=ARRAY_TOLERANCE, rtol=0)
                and np.allclose(b, [0, 1], atol=ARRAY_TOLERANCE, rtol=0)
            ) or (
                np.allclose(a, [0, 1], atol=ARRAY_TOLERANCE, rtol=0)
                and np.allclose(b, [1, 0], atol=ARRAY_TOLERANCE, rtol=0)
            )
            if not opposite:
                raise ValueError("接口接触采样未找到两侧完整材料支持")
        prior = (
            {}
            if _reuse is None
            else {
                part.part_id: (part, moment)
                for part, moment in zip(_reuse.parts, _reuse._cache, strict=True)
            }
        )
        cache, changed = [], []
        for part in parts:
            old = prior.get(part.part_id)
            if old is not None and old[0] is part:
                cache.append(old[1])
            else:
                cache.append(part.moments())
                changed.append(part.part_id)
        total = MassMoments(
            sum(item.mass_kg for item in cache),
            np.sum([item.first_kg_m for item in cache], axis=0),
            np.sum([item.second_kg_m2 for item in cache], axis=0),
        )
        if total.mass_kg <= 0:
            raise ValueError("物体总质量必须为正")
        object.__setattr__(self, "_cache", tuple(cache))
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "recomputed_part_ids", tuple(changed))

    def part(self, part_id: str) -> PartPhysicalState:
        for part in self.parts:
            if part.part_id == part_id:
                return part
        raise ValueError(f"不存在部件：{part_id}")

    def grid_indices(self, part: PartPhysicalState) -> NDArray[np.int64]:
        if not np.isclose(
            part.pitch_m, self.parts[0].pitch_m, atol=ARRAY_TOLERANCE, rtol=0
        ):
            raise ValueError("首版要求所有部件使用同一体素边长")
        rotation = np.abs(part.local_to_world[:3, :3])
        if not np.allclose(rotation, np.round(rotation), atol=ARRAY_TOLERANCE, rtol=0):
            raise ValueError("任意角度体素替换需要重体素化，首版不静默近似")
        grid = (part.centers_world_m - self.grid_origin_m) / part.pitch_m
        if np.any(np.abs(grid) > MAX_GRID_INDEX) or not np.allclose(
            grid, np.round(grid), atol=GRID_TOLERANCE, rtol=0
        ):
            raise ValueError("部件体素没有落在物体的共同网格上")
        return np.round(grid).astype(np.int64)

    def interface_material_sides(self, part_id: str, name: str) -> FloatArray:
        """在声明的矩形安装面两侧采样材料支持，不作强度判定。"""
        part = self.part(part_id)
        interface = part.interface(name)
        counts = np.ceil(interface.size_m / part.pitch_m).astype(int)
        if np.prod(counts) > MAX_INTERFACE_SAMPLES:
            raise ValueError("接口采样超过预算，需要显式调整离散方案")
        axes = [
            (np.arange(n) + 0.5) * size / n - size / 2
            for n, size in zip(counts, interface.size_m, strict=True)
        ]
        xy = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
        frame = part.local_to_world @ interface.frame_local
        occupied = self.grid_indices(part)[part.density_kg_m3 > 0]
        if not len(occupied):
            return np.zeros(2)
        tree = cKDTree(occupied)
        support = []
        for sign in (-1, 1):
            z = np.full(len(xy), sign * CONTACT_SAMPLE_OFFSET_CELLS * part.pitch_m)
            local = np.column_stack([xy, z])
            world = local @ frame[:3, :3].T + frame[:3, 3]
            grid = np.floor((world - self.grid_origin_m) / part.pitch_m + 0.5)
            distance, _ = tree.query(grid)
            support.append(float(np.mean(distance <= GRID_TOLERANCE)))
        return np.array(support)


def bind_surface_points(
    part: PartPhysicalState, points_world_m: FloatArray, max_distance_m: float
) -> tuple[NDArray[np.int64], FloatArray]:
    """在指定部件内绑定近邻，拒绝越过距离阈值的表面映射。

    Args:
        part: 已确定身份的目标部件。
        points_world_m: 待绑定的(N,3)表面锚点。
        max_distance_m: 明确的最大绑定距离。

    Returns:
        部件内体素索引和距离；这不是已学习的跨模态对应。
    """
    points = frozen_array(points_world_m, 2)
    if points.shape[1:] != (3,) or not len(points):
        raise ValueError("表面锚点必须是非空(N,3)数组")
    if not np.isfinite(max_distance_m) or max_distance_m <= 0:
        raise ValueError("最大绑定距离必须为有限正数")
    distance, index = cKDTree(part.centers_world_m).query(points)
    if np.any(distance > max_distance_m):
        raise ValueError("表面锚点超出部件绑定距离，需要重新建立对应")
    return index.astype(np.int64), distance
