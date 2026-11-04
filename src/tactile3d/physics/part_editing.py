"""共享体素物理状态上的可撤销编辑，失败不修改源对象。"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np

from tactile3d.physics.part_state import (
    GRID_TOLERANCE,
    Material,
    PartPhysicalState,
    PhysicalObject,
    interface_error,
    require_interface_match,
)

Operation = Literal["material_change", "construction_change", "replace_part"]
OPERATIONS = {"material_change", "construction_change", "replace_part"}
VOID = Material("void", 0.0, "explicit_empty_space")
MOMENT_RTOL = 1e-10
MOMENT_ATOL = 1e-12


@dataclass(frozen=True)
class EditSpec:
    target_part_id: str
    operation: Operation
    expected_part_version: int | None = None
    material_name: str | None = None
    new_material: Material | None = None
    construction: str | None = None
    wall_thickness_m: float | None = None
    mount_interface: str | None = None
    donor_state_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_part_id, str) or not isinstance(
            self.operation, str
        ):
            raise ValueError("编辑目标和操作必须为字符串")
        if not self.target_part_id.strip() or self.operation not in OPERATIONS:
            raise ValueError("编辑需要有效目标部件及已支持的操作")
        for name in (
            "material_name",
            "construction",
            "mount_interface",
            "donor_state_ref",
        ):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name}必须为非空字符串")
        if self.new_material is not None and not isinstance(
            self.new_material, Material
        ):
            raise ValueError("new_material必须为Material状态")
        if self.expected_part_version is not None and (
            type(self.expected_part_version) is not int
            or self.expected_part_version < 0
        ):
            raise ValueError("预期部件版本必须是非负整数")
        if self.operation == "material_change":
            if not self.material_name or self.new_material is None:
                raise ValueError("材料编辑必须声明旧材料名称和新材料")
            if any(
                value is not None
                for value in (
                    self.construction,
                    self.wall_thickness_m,
                    self.mount_interface,
                    self.donor_state_ref,
                )
            ):
                raise ValueError("材料编辑不能同时夹带构造或替换操作")
        elif self.operation == "construction_change":
            if self.construction not in {"solid", "hollow_box"}:
                raise ValueError("首版构造编辑仅支持solid和hollow_box")
            if not self.material_name or any(
                value is not None
                for value in (
                    self.new_material,
                    self.mount_interface,
                    self.donor_state_ref,
                )
            ):
                raise ValueError("构造编辑必须指定已有材料，不夹带其他操作")
            if self.construction == "hollow_box" and (
                self.wall_thickness_m is None
                or not np.isfinite(self.wall_thickness_m)
                or self.wall_thickness_m <= 0
            ):
                raise ValueError("空心盒需要有限正壁厚，单位为米")
            if self.construction == "solid" and self.wall_thickness_m is not None:
                raise ValueError("实心构造不接受壁厚参数")
        elif any(
            value is not None
            for value in (
                self.material_name,
                self.new_material,
                self.construction,
                self.wall_thickness_m,
            )
        ):
            raise ValueError("替换件的材料和构造应记录在donor状态中")

    @classmethod
    def from_json(cls, path: str | Path) -> EditSpec:
        """读取严格编辑契约，拒绝拼错字段导致的静默失效。"""
        file = Path(path).expanduser().resolve()
        return cls.from_dict(
            json.loads(file.read_text(encoding="utf-8-sig")), file.parent
        )

    @classmethod
    def from_dict(cls, value: object, base_dir: str | Path = ".") -> EditSpec:
        """统一校验单步和批量指令，相对路径以指令文件所在目录为准。"""
        allowed = {item.name for item in fields(cls)}
        if not isinstance(value, dict) or set(value) - allowed:
            raise ValueError("EditSpec不是对象或含有未知字段")
        payload = dict(value)
        if "new_material" in payload:
            payload["new_material"] = Material.from_dict(payload["new_material"])
        if payload.get("donor_state_ref"):
            if not isinstance(payload["donor_state_ref"], str):
                raise ValueError("donor_state_ref必须为目录字符串")
            payload["donor_state_ref"] = str(
                (Path(base_dir).expanduser() / payload["donor_state_ref"]).resolve()
            )
        if not {"target_part_id", "operation"} <= set(payload):
            raise ValueError("EditSpec缺少target_part_id或operation")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        """导出单步指令，供事务和独立JSON共用同一契约。"""
        payload = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if getattr(self, item.name) is not None
        }
        if self.new_material is not None:
            payload["new_material"] = self.new_material.to_dict()
        return payload

    def to_json(self, path: str | Path) -> Path:
        """保存可重放的编辑要求，不把数组对象写成不透明字符串。"""
        file = Path(path).expanduser().resolve()
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        return file


@dataclass(frozen=True)
class EditResult:
    state: PhysicalObject
    report: dict[str, Any]


def construction_fractions(part: PartPhysicalState, spec: EditSpec) -> np.ndarray:
    """在固定几何域中建立实心或网格对齐的封闭盒壳。

    Args:
        part: 编辑前部件，几何域包含实心及空腔单元。
        spec: 声明构造类型和壁厚的编辑要求。

    Returns:
        两列材料/空腔体积占用，未将类别置信度当占用率。
    """
    if spec.operation != "construction_change":
        raise ValueError("构造占用生成需要construction_change操作")
    fractions = np.zeros((len(part.centers_local_m), 2))
    fractions[:, 0] = 1
    if spec.construction == "solid":
        return fractions
    pitch = part.pitch_m
    thickness = float(spec.wall_thickness_m)
    layers = thickness / pitch
    if not np.isclose(layers, round(layers), atol=GRID_TOLERANCE, rtol=0):
        raise ValueError("首版壁厚必须为整数体素层；请细化网格")
    centers = part.centers_local_m
    lower, upper = centers.min(axis=0) - pitch / 2, centers.max(axis=0) + pitch / 2
    counts = np.round((upper - lower) / pitch).astype(np.int64)
    if int(np.prod(counts)) != len(centers):
        raise ValueError("hollow_box仅接受完整矩形盒域，不用包围盒填补缺失几何")
    if np.any(2 * thickness >= upper - lower - pitch * GRID_TOLERANCE):
        raise ValueError("壁厚过大，没有可分辨的内部空腔")
    distance = np.minimum(centers - lower, upper - centers).min(axis=1)
    cavity = distance >= thickness
    fractions[cavity] = [0, 1]
    return fractions


def apply_edit(
    source: PhysicalObject,
    spec: EditSpec,
    donor: PartPhysicalState | None = None,
) -> EditResult:
    """执行一次原子编辑，复用身份未变且输入未变的部件质量矩。

    Args:
        source: 不可变的源物体状态。
        spec: 单一编辑操作及可选版本检查。
        donor: 替换操作需要的已体素化部件。

    Returns:
        新状态和可审计的编辑报告；异常时源状态保持原样。

    Raises:
        ValueError: 版本过期、接口不兼容、重叠或不支持的离散条件。
    """
    old = source.part(spec.target_part_id)
    if (
        spec.expected_part_version is not None
        and old.version != spec.expected_part_version
    ):
        raise ValueError("编辑版本过期，需要重新读取部件状态")
    if spec.operation != "replace_part" and donor is not None:
        raise ValueError("非替换操作不接受donor")
    evidence = (*old.evidence, f"explicit_edit:{spec.operation}")
    interface_checks = []
    if spec.operation == "material_change":
        names = [item.name for item in old.materials]
        if spec.material_name not in names:
            raise ValueError("指定的旧材料不在目标部件中")
        materials = list(old.materials)
        materials[names.index(spec.material_name)] = spec.new_material
        updated = replace(
            old,
            materials=tuple(materials),
            version=old.version + 1,
            evidence=evidence,
            feature_refs=(),
        )
    elif spec.operation == "construction_change":
        material = next(
            (item for item in old.materials if item.name == spec.material_name), None
        )
        if material is None or np.any(material.values(len(old.centers_local_m)) <= 0):
            raise ValueError("构造编辑需要目标部件中已声明的正密度材料")
        updated = replace(
            old,
            materials=(material, VOID),
            fractions=construction_fractions(old, spec),
            version=old.version + 1,
            construction=str(spec.construction),
            evidence=evidence,
            feature_refs=(),
        )
    else:
        if donor is None or not old.interfaces:
            raise ValueError("替换需要donor和明确安装接口")
        if donor.instance_id == old.instance_id:
            raise ValueError("替换件需要不同的实例ID")
        name = spec.mount_interface or old.interfaces[0].name
        target_frame = old.local_to_world @ old.interface(name).frame_local
        pose = target_frame @ np.linalg.inv(donor.interface(name).frame_local)
        updated = replace(
            donor,
            part_id=old.part_id,
            version=old.version + 1,
            local_to_world=pose,
            evidence=(*donor.evidence, *evidence),
            feature_refs=(),
        )
        for interface in old.interfaces:
            errors = interface_error(old, interface.name, updated, interface.name)
            require_interface_match(errors)
            interface_checks.append({"name": interface.name, **errors})
    parts = tuple(updated if part is old else part for part in source.parts)
    state = PhysicalObject(
        source.object_id,
        parts,
        source.grid_origin_m,
        source.connections,
        version=source.version + 1,
        _reuse=source,
    )
    affected_edges = [
        f"{edge.part_a}:{edge.interface_a}|{edge.part_b}:{edge.interface_b}"
        for edge in source.connections
        if old.part_id in {edge.part_a, edge.part_b}
    ]
    return EditResult(
        state,
        {
            "operation": spec.operation,
            "part_id": old.part_id,
            "source_version": source.version,
            "result_version": state.version,
            "old_instance_id": old.instance_id,
            "new_instance_id": updated.instance_id,
            "recomputed_parts": list(state.recomputed_part_ids),
            "reused_parts": [part.part_id for part in source.parts if part is not old],
            "invalidated_fields": ["target_physics", "target_fused_features"],
            "revalidated_interfaces": affected_edges,
            "interface_checks": interface_checks,
            "interface_validation": "frame, size and voxel contact samples",
            "before": source.total.properties(),
            "after": state.total.properties(),
            "appearance_rendered": False,
            "dependency_scope": "explicit fields; no global learned/stress solver",
            "uncertainty_updated": False,
            "strength_validated": False,
        },
    )


def full_recompute(state: PhysicalObject) -> PhysicalObject:
    """独立重算所有部件，作为缓存更新的数值参考。"""
    return PhysicalObject(
        state.object_id,
        state.parts,
        state.grid_origin_m,
        state.connections,
        version=state.version,
    )


def verify_full_recompute(state: PhysicalObject) -> dict[str, float]:
    reference = full_recompute(state)
    result = {}
    for name in ("mass_kg", "first_kg_m", "second_kg_m2"):
        actual, expected = getattr(state.total, name), getattr(reference.total, name)
        result[name] = float(np.max(np.abs(np.asarray(actual) - expected)))
        if not np.allclose(actual, expected, rtol=MOMENT_RTOL, atol=MOMENT_ATOL):
            raise ValueError(f"增量结果与完整重算不一致：{name}")
    return result
