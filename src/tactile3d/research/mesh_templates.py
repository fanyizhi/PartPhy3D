"""已知有限构造的Mesh候选；仅为受控学习诊断，不推断真实内部。"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import trimesh

from tactile3d.physics.mesh_native import (
    ClosedMesh,
    MeshPhysicalObject,
    MeshRegion,
    boolean_volume,
    closed_shell,
    positive_scalar,
)
from tactile3d.physics.part_state import Material
from tactile3d.research.construction import decode_state

ASPECT_RATIO = (10.0, 12.0, 8.0)
WALL_UNITS = 1.0
MATERIAL_COLORS = ((155, 105, 58, 255), (90, 145, 175, 255))


def box_domain(
    size_m: tuple[float, float, float], center_m: tuple[float, float, float] = (0, 0, 0)
) -> ClosedMesh:
    if len(size_m) != 3 or len(center_m) != 3:
        raise ValueError("长方体尺寸与中心需要三维参数")
    size = np.array([positive_scalar(value, "长方体边长") for value in size_m])
    center = np.asarray(center_m, dtype=float)
    if not np.isfinite(center).all():
        raise ValueError("长方体中心必须有限")
    mesh = trimesh.creation.box(extents=size)
    mesh.apply_translation(center)
    return ClosedMesh(mesh.vertices, mesh.faces, "specified_box_geometry")


@lru_cache(maxsize=1)
def unit_templates() -> tuple[tuple[ClosedMesh, ...], ...]:
    """用Mesh差集生成实心/闭壳/夹层/框架四类候选，无规则网格。"""
    dimensions = np.array(ASPECT_RATIO)
    outer = box_domain(tuple(dimensions))
    core = box_domain(tuple(dimensions - 2 * WALL_UNITS))
    shell = closed_shell(outer, core)
    frame = outer
    for axis in range(3):
        # 三个贯通孔移除最多只有一层边界的面心域，留下十二条实体棱梁。
        size = dimensions - 2 * WALL_UNITS
        size[axis] = dimensions[axis] + 2 * WALL_UNITS
        frame = boolean_volume(frame, box_domain(tuple(size)), "difference")
    filling = boolean_volume(outer, frame, "difference")
    return ((outer,), (shell,), (shell, core), (frame, filling))


def mesh_candidate_volumes(length_unit_m: np.ndarray) -> np.ndarray:
    units = np.asarray(length_unit_m, dtype=float)
    if units.ndim != 1 or not np.isfinite(units).all() or np.any(units <= 0):
        raise ValueError("对象长度单位必须为一维有限正数组")
    basis = np.zeros((4, 2))
    for index, regions in enumerate(unit_templates()):
        for phase, domain in enumerate(regions):
            basis[index, phase] = domain.moments.volume_m3
    return basis[None] * units[:, None, None] ** 3


def mesh_state_from_logits(
    object_id: str,
    length_unit_m: float,
    logits: np.ndarray,
    priors: np.ndarray,
) -> MeshPhysicalObject:
    """MAP候选导出为封闭区域Mesh；混合期望质量另行报告。"""
    scale = positive_scalar(length_unit_m, "对象长度单位")
    if logits.shape != (1, 8):
        raise ValueError("导出仅支持一个对象的八维构造参数")
    decoded = decode_state(logits, priors)
    candidate = int(decoded["probability"][0].argmax())
    regions = []
    for phase, domain in enumerate(unit_templates()[candidate]):
        metric = ClosedMesh(
            domain.vertices_m * scale,
            domain.faces,
            "scaled_known_construction_candidate",
        )
        material = Material(
            f"phase_{phase}",
            float(decoded["density"][0, phase]),
            "learned_synthetic_not_VLM_or_measured",
            display_rgba=MATERIAL_COLORS[phase],
        )
        regions.append(
            MeshRegion(
                "body",
                f"phase_{phase}",
                metric,
                material,
                float(decoded["fill"][0, phase]),
            )
        )
    return MeshPhysicalObject(object_id, tuple(regions))
