"""明确句式或严格JSON驱动的部件编辑事务，不调用语言模型。"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tactile3d.physics.part_editing import EditSpec, apply_edit, verify_full_recompute
from tactile3d.physics.part_io import load_object
from tactile3d.physics.part_state import Material, PartPhysicalState, PhysicalObject

PROGRAM_SCHEMA_VERSION = 1
MAX_EDIT_STEPS = 64
MAX_COMMAND_LENGTH = 1024
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TOKEN = r"[^\s,。]+"
PREFIX = rf"(?:把|将)\s*(?P<part>{TOKEN})\s*"
PATTERNS = {
    "material_change": re.compile(
        PREFIX
        + rf"的材料\s*(?P<old>{TOKEN})\s*改为\s*(?P<new>{TOKEN})"
        + rf"\s*,\s*密度\s*(?P<density>{NUMBER})\s*kg/(?:m3|m\^3)"
    ),
    "solid": re.compile(PREFIX + rf"设为实心\s*,\s*材料\s*(?P<material>{TOKEN})"),
    "hollow_box": re.compile(
        PREFIX
        + rf"设为空心盒\s*,\s*材料\s*(?P<material>{TOKEN})"
        + rf"\s*,\s*壁厚\s*(?P<thickness>{NUMBER})\s*(?P<unit>mm|cm|m|毫米|厘米|米)"
    ),
    "replace_part": re.compile(
        PREFIX + rf"替换为\s*(?P<donor>{TOKEN})\s*,\s*接口\s*(?P<mount>{TOKEN})"
    ),
}
UNIT_TO_METERS = {
    "mm": 1e-3,
    "cm": 1e-2,
    "m": 1.0,
    "毫米": 1e-3,
    "厘米": 1e-2,
    "米": 1.0,
}
GRAMMAR_HELP = (
    "不支持或不完整的指令。支持：将 leg 的材料 wood 改为 steel，密度 7850 kg/m3；"
    "将 leg 设为空心盒，材料 steel，壁厚 10 mm；"
    "将 leg 设为实心，材料 steel；将 leg 替换为 donor，接口 mount。"
)


@dataclass(frozen=True)
class EditProgram:
    """绑定物体版本的有序操作，步骤中的部件版本可进一步限制。"""

    object_id: str
    expected_object_version: int
    steps: tuple[EditSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.object_id, str) or not self.object_id.strip():
            raise ValueError("编辑程序需要非空object_id")
        if (
            type(self.expected_object_version) is not int
            or self.expected_object_version < 0
        ):
            raise ValueError("expected_object_version必须为非负整数")
        steps = tuple(self.steps)
        if not 1 <= len(steps) <= MAX_EDIT_STEPS:
            raise ValueError(f"编辑步数必须介于1和{MAX_EDIT_STEPS}之间")
        if not all(isinstance(step, EditSpec) for step in steps):
            raise ValueError("所有编辑步骤必须是EditSpec")
        object.__setattr__(self, "steps", steps)

    @classmethod
    def from_json(cls, path: str | Path) -> EditProgram:
        """读取严格批量指令，不允许未知字段或隐式丢弃步骤。"""
        file = Path(path).expanduser().resolve()
        payload = json.loads(file.read_text(encoding="utf-8-sig"))
        required = {"schema_version", "object_id", "expected_object_version", "steps"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError(f"编辑程序字段必须恰为{sorted(required)}")
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"] != PROGRAM_SCHEMA_VERSION
        ):
            raise ValueError("不支持的编辑程序schema_version")
        if not isinstance(payload["steps"], list):
            raise ValueError("steps必须是非空列表")
        if not 1 <= len(payload["steps"]) <= MAX_EDIT_STEPS:
            raise ValueError("编辑步数超出允许范围")
        return cls(
            payload["object_id"],
            payload["expected_object_version"],
            tuple(EditSpec.from_dict(row, file.parent) for row in payload["steps"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PROGRAM_SCHEMA_VERSION,
            "object_id": self.object_id,
            "expected_object_version": self.expected_object_version,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class ProgramResult:
    program: EditProgram
    states: tuple[PhysicalObject, ...]
    reports: tuple[dict[str, Any], ...]
    used_donors: tuple[tuple[str, PartPhysicalState], ...] = ()

    @property
    def final_state(self) -> PhysicalObject:
        return self.states[-1]


def parse_instruction(
    text: str,
    aliases: Mapping[str, str] | None = None,
    donors: Mapping[str, str | Path] | None = None,
) -> EditSpec:
    """将受限中文句式编译为单步指令，拒绝部分匹配。

    Args:
        text: 必须提供操作需要的全部参数，不支持任意自由对话。
        aliases: 显式部件别名，例如左前腿对应leg；不推断视角左右。
        donors: 替换件名称到已持久化物理状态目录的映射。

    Returns:
        严格的单步编辑；实际几何与版本约束在事务中继续校验。
    """
    if not isinstance(text, str) or not 0 < len(text.strip()) <= MAX_COMMAND_LENGTH:
        raise ValueError("指令必须为长度受限的非空字符串")
    normalized = unicodedata.normalize("NFKC", text).strip().removesuffix("。")
    names = {} if aliases is None else dict(aliases)
    candidates = {} if donors is None else dict(donors)
    if any(
        not isinstance(key, str)
        or not isinstance(value, str)
        or not key.strip()
        or not value.strip()
        for key, value in names.items()
    ):
        raise ValueError("部件别名必须一一指向明确的字符串ID")
    for operation, pattern in PATTERNS.items():
        match = pattern.fullmatch(normalized)
        if match is None:
            continue
        row = match.groupdict()
        target = names.get(row["part"], row["part"])
        if operation == "material_change":
            density = float(row["density"])
            if not 0 < density < float("inf"):
                raise ValueError("编辑材料密度必须为有限正数")
            return EditSpec(
                target,
                operation,
                material_name=row["old"],
                new_material=Material(row["new"], density, "explicit_instruction"),
            )
        if operation in {"solid", "hollow_box"}:
            thickness = (
                float(row["thickness"]) * UNIT_TO_METERS[row["unit"]]
                if operation == "hollow_box"
                else None
            )
            return EditSpec(
                target,
                "construction_change",
                material_name=row["material"],
                construction=operation,
                wall_thickness_m=thickness,
            )
        key = row["donor"]
        if key not in candidates:
            raise ValueError(f"未注册替换件：{key}，请提供其物理状态目录")
        return EditSpec(
            target,
            "replace_part",
            mount_interface=row["mount"],
            donor_state_ref=str(Path(candidates[key]).expanduser().resolve()),
        )
    raise ValueError(GRAMMAR_HELP)


def program_from_commands(
    source: PhysicalObject,
    commands: Sequence[str],
    aliases: Mapping[str, str] | None = None,
    donors: Mapping[str, str | Path] | None = None,
) -> EditProgram:
    """显式绑定初始物体，不让别名覆盖另一个真实部件ID。"""
    if isinstance(commands, (str, bytes)):
        raise ValueError("commands必须是指令列表，不能是单个字符串")
    names = {} if aliases is None else dict(aliases)
    ids = {part.part_id for part in source.parts}
    for alias, target in names.items():
        if not isinstance(alias, str) or not isinstance(target, str):
            raise ValueError("部件别名必须指向字符串ID")
        if target not in ids or (alias in ids and alias != target):
            raise ValueError("别名目标不存在，或别名与真实部件ID冲突")
    if not 1 <= len(commands) <= MAX_EDIT_STEPS:
        raise ValueError("编辑步数超出允许范围")
    steps = tuple(parse_instruction(text, names, donors) for text in commands)
    for spec in steps:
        source.part(spec.target_part_id)
    return EditProgram(source.object_id, source.version, steps)


def load_donor(path: str) -> PartPhysicalState:
    candidate = load_object(path)
    if len(candidate.parts) != 1:
        raise ValueError("替换件状态必须恰含一个部件")
    return candidate.parts[0]


def execute_program(
    source: PhysicalObject,
    program: EditProgram,
    donors: Mapping[str, PartPhysicalState] | None = None,
) -> ProgramResult:
    """整组在内存中执行，任何一步失败都不提交状态或写文件。

    Args:
        source: 不可变源物体。
        program: 已绑定物体身份及版本的编辑步骤。
        donors: 可选内存替换件，键与donor_state_ref完全一致。

    Returns:
        全部通过约束及全量重算检查的新状态序列和逐步报告。

    Raises:
        ValueError: 身份、版本、材料、几何、接口或全量核对失败。
    """
    if source.object_id != program.object_id:
        raise ValueError("编辑程序object_id与源物体不匹配")
    if source.version != program.expected_object_version:
        raise ValueError("编辑程序物体版本过期，请重新读取源状态")
    candidates = {} if donors is None else dict(donors)
    if not all(isinstance(part, PartPhysicalState) for part in candidates.values()):
        raise ValueError("内存替换件必须为PartPhysicalState")
    states, reports, bound_steps, used_donors = [source], [], [], {}
    for index, spec in enumerate(program.steps, start=1):
        try:
            current = states[-1]
            if spec.expected_part_version is None:
                spec = replace(
                    spec,
                    expected_part_version=current.part(spec.target_part_id).version,
                )
            donor = None
            if spec.donor_state_ref:
                key = spec.donor_state_ref
                if key not in used_donors:
                    used_donors[key] = (
                        candidates[key] if key in candidates else load_donor(key)
                    )
                donor = used_donors[key]
            start = time.perf_counter()
            result = apply_edit(current, spec, donor)
            edit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            errors = verify_full_recompute(result.state)
            verification_seconds = time.perf_counter() - start
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"第{index}步失败，整组未提交：{error}") from error
        reports.append(
            {
                **result.report,
                "step": index,
                "full_recompute_errors": errors,
                "edit_seconds_including_validation": edit_seconds,
                "full_verification_seconds": verification_seconds,
            }
        )
        states.append(result.state)
        bound_steps.append(spec)
    bound = EditProgram(source.object_id, source.version, tuple(bound_steps))
    return ProgramResult(
        bound, tuple(states), tuple(reports), tuple(used_donors.items())
    )
