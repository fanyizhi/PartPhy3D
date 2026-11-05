"""封闭材料区域Mesh上的直接积分与固定构造编辑，不建立体素网格。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import InitVar, dataclass, field, replace
from typing import Any, Literal

import manifold3d as manifold
import numpy as np
import trimesh
from numpy.typing import NDArray

from tactile3d.physics.part_state import Material, frozen_array

FloatArray = NDArray[np.float64]
OVERLAP_REL_TOL = 1e-9
VOLUME_ABS_TOL_M3 = 1e-14
MOMENT_RTOL = 1e-9
MOMENT_ATOL = 1e-12


def positive_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f"{name}需要有限正标量")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name}需要有限正标量") from error
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name}需要有限正标量")
    return result


@dataclass(frozen=True)
class VolumeMoments:
    volume_m3: float
    center_m: FloatArray
    central_second_m5: FloatArray

    def __post_init__(self) -> None:
        volume = positive_scalar(self.volume_m3, "封闭体积")
        center = frozen_array(self.center_m, 1)
        second = frozen_array(self.central_second_m5, 2)
        if center.shape != (3,) or second.shape != (3, 3):
            raise ValueError("体积矩需要三维质心和3×3中心二阶矩")
        if not np.allclose(second, second.T, atol=MOMENT_ATOL, rtol=MOMENT_RTOL):
            raise ValueError("体积中心二阶矩必须对称")
        if np.linalg.eigvalsh(second).min() < -MOMENT_ATOL:
            raise ValueError("体积中心二阶矩不是半正定，检查方向或几何")
        object.__setattr__(self, "volume_m3", volume)
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "central_second_m5", second)


def integrate_boundary(
    vertices_m: FloatArray, faces: NDArray[np.int64]
) -> VolumeMoments:
    """用有向四面体边界积分恢复体积、一阶矩和中心二阶矩。

    Args:
        vertices_m: 已验证闭合、定向一致的边界坐标，单位m。
        faces: 三角面索引；空腔边界必须朝材料域外侧定向。

    Returns:
        区域几何矩，不含密度；可缓存供多次换材复用。
    """
    vertices = frozen_array(vertices_m, 2)
    indices = np.asarray(faces)
    if (
        vertices.shape[1:] != (3,)
        or not len(vertices)
        or indices.ndim != 2
        or indices.shape[1:] != (3,)
        or not len(indices)
        or indices.dtype.kind not in "iu"
        or indices.min() < 0
        or indices.max() >= len(vertices)
    ):
        raise ValueError("边界积分需要非空(N,3)坐标和合法三角面索引")
    # 在几何附近积分，避免平移到大世界坐标后原始矩相减损失精度。
    reference = vertices.min(0) + np.ptp(vertices, axis=0) / 2
    triangle = vertices[indices] - reference
    a, b, c = triangle[:, 0], triangle[:, 1], triangle[:, 2]
    signed_volume = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6
    volume = float(signed_volume.sum())
    if volume <= VOLUME_ABS_TOL_M3:
        raise ValueError("有向体积非正或小于数值阈值；不取绝对值掩盖反向边界")
    total_vertices = triangle.sum(1)
    center = signed_volume @ total_vertices / (4 * volume)
    # 四面体重心坐标的二阶矩含交叉项，不能只累加表面顶点惯量。
    second = np.einsum("n,ni,nj->ij", signed_volume, total_vertices, total_vertices)
    second += np.einsum("n,nki,nkj->ij", signed_volume, triangle, triangle)
    second = second / 20 - volume * np.outer(center, center)
    return VolumeMoments(volume, center + reference, (second + second.T) / 2)


def _manifold(vertices: FloatArray, faces: NDArray[np.int64]) -> manifold.Manifold:
    value = manifold.Manifold(
        manifold.Mesh64(
            np.ascontiguousarray(vertices, dtype=np.float64),
            np.ascontiguousarray(faces, dtype=np.uint64),
        )
    )
    if value.status() != manifold.Error.NoError or value.is_empty():
        raise ValueError(f"Manifold拒绝该封闭边界：{value.status()}")
    return value


@dataclass(frozen=True)
class ClosedMesh:
    vertices_m: FloatArray
    faces: NDArray[np.int64]
    provenance: str
    moments: VolumeMoments = field(init=False, repr=False)
    _solid: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        vertices = frozen_array(self.vertices_m, 2)
        raw_faces = np.asarray(self.faces)
        if raw_faces.dtype.kind not in "iu" or raw_faces.ndim != 2:
            raise ValueError("Mesh三角面需要二维整数索引")
        faces = np.array(raw_faces, dtype=np.int64, copy=True)
        if (
            vertices.shape[1:] != (3,)
            or not len(vertices)
            or faces.shape[1:] != (3,)
            or not len(faces)
            or faces.min() < 0
            or faces.max() >= len(vertices)
        ):
            raise ValueError("Mesh坐标或三角面索引无效")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("必须记录封闭区域的几何来源")
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError("材料区域必须已闭合且定向一致，不自动封口或填凸包")
        if np.any(mesh.area_faces <= 0):
            raise ValueError("材料区域含退化三角面")
        # Mesh64保留双精度，不通过float32转换来验证米制小结构。
        reference = vertices.min(0) + np.ptp(vertices, axis=0) / 2
        solid = _manifold(vertices - reference, faces).translate(reference)
        moments = integrate_boundary(vertices, faces)
        if not np.isclose(
            solid.volume(), moments.volume_m3, rtol=MOMENT_RTOL, atol=VOLUME_ABS_TOL_M3
        ):
            raise ValueError("边界积分与Manifold体积不一致")
        faces.flags.writeable = False
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "moments", moments)
        object.__setattr__(self, "_solid", solid)

    def mesh(self) -> trimesh.Trimesh:
        return trimesh.Trimesh(self.vertices_m.copy(), self.faces.copy(), process=False)

    def xy_section(self, height_m: float) -> tuple[FloatArray, ...]:
        """返回有向截面轮廓；空腔轮廓保持相反方向，不采样体素。"""
        if np.ndim(height_m) != 0 or not np.isfinite(height_m):
            raise ValueError("截面高度必须是有限标量，单位m")
        return tuple(
            frozen_array(contour, 2)
            for contour in self._solid.slice(float(height_m)).to_polygons()
        )


def boolean_volume(
    first: ClosedMesh,
    second: ClosedMesh,
    operation: Literal["difference", "union", "intersection"],
) -> ClosedMesh:
    """通过双精度Mesh布尔运算建立区域；空结果明确拒绝。"""
    if operation == "difference":
        result = first._solid - second._solid
    elif operation == "union":
        result = first._solid + second._solid
    elif operation == "intersection":
        result = first._solid ^ second._solid
    else:
        raise ValueError(f"不支持的布尔操作：{operation}")
    if result.status() != manifold.Error.NoError or result.is_empty():
        raise ValueError("布尔结果为空或无效，不能作为正体积材料区域")
    payload = result.to_mesh64()
    return ClosedMesh(
        payload.vert_properties[:, :3],
        payload.tri_verts,
        f"{operation}({first.provenance},{second.provenance})",
    )


def closed_shell(outer: ClosedMesh, cavity: ClosedMesh) -> ClosedMesh:
    """只接受严格位于外层内部的空腔，避免把相交切口冒充闭壳。"""
    if np.any(cavity.vertices_m.min(0) <= outer.vertices_m.min(0)) or np.any(
        cavity.vertices_m.max(0) >= outer.vertices_m.max(0)
    ):
        raise ValueError("闭壳空腔必须严格位于外边界内，开放容器需单独建模")
    outside = cavity._solid - outer._solid
    if not outside.is_empty() and outside.volume() > VOLUME_ABS_TOL_M3:
        raise ValueError("空腔穿出外层，不能只按包围盒判断包含")
    return boolean_volume(outer, cavity, "difference")


@dataclass(frozen=True)
class MeshRegion:
    part_id: str
    region_id: str
    domain: ClosedMesh
    material: Material
    fill_fraction: float = 1.0
    version: int = 0

    def __post_init__(self) -> None:
        if not all(
            isinstance(v, str) and v.strip() for v in (self.part_id, self.region_id)
        ):
            raise ValueError("需要明确的部件与材料区域ID")
        if not isinstance(self.domain, ClosedMesh) or not isinstance(
            self.material, Material
        ):
            raise TypeError("区域需要ClosedMesh域和Material材料")
        positive_scalar(self.material.density_kg_m3, "区域密度kg/m³")
        fraction = positive_scalar(self.fill_fraction, "宏观填充比例")
        if fraction > 1 or type(self.version) is not int or self.version < 0:
            raise ValueError("占用不能超过1，版本必须为非负整数")
        object.__setattr__(self, "fill_fraction", fraction)

    @property
    def effective_density(self) -> float:
        return float(self.material.density_kg_m3) * self.fill_fraction


def combine_regions(
    regions: tuple[MeshRegion, ...], reintegrate: bool = False
) -> dict[str, Any]:
    """用区域中心矩及移轴定理合并，换材只更新区域密度系数。

    Returns:
        SI单位质量、世界质心与绕总质心的完整惯量张量。
    """
    if not regions:
        raise ValueError("需要至少一个材料区域")
    moments = [
        (
            integrate_boundary(r.domain.vertices_m, r.domain.faces)
            if reintegrate
            else r.domain.moments
        )
        for r in regions
    ]
    density = np.array([r.effective_density for r in regions])
    masses = density * np.array([m.volume_m3 for m in moments])
    centers = np.array([m.center_m for m in moments])
    anchor = centers[0]
    local_center = masses @ (centers - anchor) / masses.sum()
    delta = centers - anchor - local_center
    geometric_second = np.array([m.central_second_m5 for m in moments])
    own = np.einsum("r,rij->ij", density, geometric_second)
    shifted = np.einsum("r,ri,rj->ij", masses, delta, delta)
    central = own + shifted
    inertia = np.trace(central) * np.eye(3) - central
    return {
        "mass_kg": float(masses.sum()),
        "center_of_mass_m": (local_center + anchor).tolist(),
        "inertia_kg_m2": ((inertia + inertia.T) / 2).tolist(),
    }


@dataclass(frozen=True)
class MeshPhysicalObject:
    object_id: str
    regions: tuple[MeshRegion, ...]
    version: int = 0
    _reuse: InitVar[MeshPhysicalObject | None] = None
    _properties: dict[str, Any] = field(init=False, repr=False, compare=False)

    def __post_init__(self, _reuse: MeshPhysicalObject | None) -> None:
        regions = tuple(self.regions)
        if (
            not isinstance(self.object_id, str)
            or not self.object_id.strip()
            or not regions
        ):
            raise ValueError("物体需要ID和非空区域")
        if type(self.version) is not int or self.version < 0:
            raise ValueError("物体版本必须为非负整数")
        if not all(isinstance(r, MeshRegion) for r in regions):
            raise TypeError("物体必须包含MeshRegion")
        if len({(r.part_id, r.region_id) for r in regions}) != len(regions):
            raise ValueError("部件内区域ID重复")
        reusable = _reuse is not None and (
            isinstance(_reuse, MeshPhysicalObject)
            and [(r.part_id, r.region_id, id(r.domain)) for r in regions]
            == [(r.part_id, r.region_id, id(r.domain)) for r in _reuse.regions]
        )
        if not reusable:
            for index, first in enumerate(regions):
                for second in regions[index + 1 :]:
                    low = np.maximum(
                        first.domain.vertices_m.min(0), second.domain.vertices_m.min(0)
                    )
                    high = np.minimum(
                        first.domain.vertices_m.max(0), second.domain.vertices_m.max(0)
                    )
                    if np.any(high <= low):
                        continue
                    intersection = first.domain._solid ^ second.domain._solid
                    if intersection.status() != manifold.Error.NoError:
                        raise ValueError("材料区域交集检查失败")
                    tolerance = max(
                        VOLUME_ABS_TOL_M3,
                        OVERLAP_REL_TOL
                        * min(
                            first.domain.moments.volume_m3,
                            second.domain.moments.volume_m3,
                        ),
                    )
                    if intersection.volume() > tolerance:
                        raise ValueError(
                            f"材料区域重叠：{first.part_id}/{first.region_id}与"
                            f"{second.part_id}/{second.region_id}"
                        )
        object.__setattr__(self, "regions", regions)
        object.__setattr__(self, "_properties", combine_regions(regions))

    @property
    def properties(self) -> dict[str, Any]:
        """返回独立副本，防止调用方改写已缓存的物理结果。"""
        return deepcopy(self._properties)

    def region(self, part_id: str, region_id: str) -> MeshRegion:
        for region in self.regions:
            if (region.part_id, region.region_id) == (part_id, region_id):
                return region
        raise ValueError(f"不存在的部件/区域：{part_id}/{region_id}")

    def edit_material(
        self,
        part_id: str,
        region_id: str,
        material: Material,
        expected_version: int,
    ) -> tuple[MeshPhysicalObject, dict[str, Any]]:
        """固定Mesh和占用换材，复用已验证域及其几何矩。

        Returns:
            新对象和域复用记录；异常时源对象不变。
        """
        target = self.region(part_id, region_id)
        if type(expected_version) is not int or expected_version != target.version:
            raise ValueError("区域编辑版本过期或类型无效")
        updated = replace(target, material=material, version=target.version + 1)
        result = MeshPhysicalObject(
            self.object_id,
            tuple(updated if r is target else r for r in self.regions),
            self.version + 1,
            self,
        )
        delta = (
            updated.effective_density - target.effective_density
        ) * target.domain.moments.volume_m3
        return result, {
            "part_id": part_id,
            "region_id": region_id,
            "before": self.properties,
            "after": result.properties,
            "expected_delta_mass_kg": delta,
            "delta_mass_kg": result.properties["mass_kg"] - self.properties["mass_kg"],
            "reintegrated_meshes": 0,
            "changed_material_regions": 1,
            "geometry_and_fill_unchanged": True,
            "voxelization_used": False,
            "old_material": target.material.to_dict(),
            "new_material": material.to_dict(),
        }


def verify_mesh_recompute(state: MeshPhysicalObject) -> dict[str, float]:
    """重新遍历全部边界积分，核对缓存更新；这是软件一致性检查。"""
    reference = combine_regions(state.regions, reintegrate=True)
    result = {}
    for key, expected in reference.items():
        actual = state.properties[key]
        if not np.allclose(actual, expected, rtol=MOMENT_RTOL, atol=MOMENT_ATOL):
            raise ValueError(f"Mesh积分缓存核对失败：{key}")
        result[key] = float(np.max(np.abs(np.asarray(actual) - expected)))
    return result
