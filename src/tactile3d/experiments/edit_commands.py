"""可直接运行的编辑指令入口，默认验证已知合成装配体。"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tactile3d.experiments.part_edit_demo import make_example
from tactile3d.figures.edit_history import plot_edit_history, preview_state_indices
from tactile3d.mesh.state_surface import part_surface
from tactile3d.physics.edit_program import (
    EditProgram,
    ProgramResult,
    execute_program,
    program_from_commands,
)
from tactile3d.physics.part_io import file_sha256, load_object, save_object, write_json
from tactile3d.physics.part_state import PartPhysicalState, PhysicalObject

DEFAULT_OUTPUT = Path("outputs/edit_commands")
DEFAULT_COMMANDS = (
    "将 leg 的材料 wood 改为 steel，密度 7850 kg/m3",
    "将 leg 设为空心盒，材料 steel，壁厚 10 mm",
    "将 leg 替换为 aluminum_leg，接口 mount",
)
REFERENCE_ATOL = 1e-10
REFERENCE_RTOL = 1e-10


def analytic_boxes(
    boxes: list[tuple[tuple[float, float, float], float, float]],
) -> dict[str, Any]:
    """用长方体解析惯量和移轴定理建立独立合成参考。

    Args:
        boxes: 每项为三轴尺寸、世界z质心、密度；负密度项仅用于减去空腔。

    Returns:
        总质量、质心和绕整体质心的完整惯量，使用SI单位。
    """
    sizes = np.array([box[0] for box in boxes], dtype=float)
    if sizes.ndim != 2 or sizes.shape[1:] != (3,) or not len(sizes):
        raise ValueError("解析参考需要非空(N,3)长方体尺寸")
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("解析长方体尺寸必须为有限正数")
    centers = np.zeros_like(sizes)
    centers[:, 2] = [box[1] for box in boxes]
    masses = np.prod(sizes, axis=1) * np.array([box[2] for box in boxes])
    if not np.isfinite(centers).all() or not np.isfinite(masses).all():
        raise ValueError("解析参考坐标和质量必须有限")
    mass = float(masses.sum())
    if mass <= 0:
        raise ValueError("解析装配参考总质量必须为正")
    center = masses @ centers / mass
    inertia = np.zeros((3, 3))
    for dimensions, position, item_mass in zip(sizes, centers, masses, strict=True):
        own = item_mass / 12 * np.diag(np.sum(dimensions**2) - dimensions**2)
        delta = position - center
        inertia += own + item_mass * (
            np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta)
        )
    return {
        "mass_kg": mass,
        "center_of_mass_m": center.tolist(),
        "inertia_kg_m2": inertia.tolist(),
    }


def synthetic_reference() -> list[dict[str, Any]]:
    body = ((0.20, 0.12, 0.04), 0.22, 600.0)
    return [
        analytic_boxes([body, ((0.04, 0.04, 0.20), 0.10, 650.0)]),
        analytic_boxes([body, ((0.04, 0.04, 0.20), 0.10, 7850.0)]),
        analytic_boxes(
            [
                body,
                ((0.04, 0.04, 0.20), 0.10, 7850.0),
                ((0.02, 0.02, 0.18), 0.10, -7850.0),
            ]
        ),
        analytic_boxes(
            [
                body,
                ((0.02, 0.02, 0.18), 0.09, 2700.0),
                ((0.04, 0.04, 0.02), 0.19, 2700.0),
            ]
        ),
    ]


def check_reference(
    result: ProgramResult,
    reference: list[dict[str, Any]],
) -> list[dict[str, float]]:
    if len(reference) != len(result.states):
        raise ValueError("解析参考与状态数量不一致")
    errors = []
    for state, expected in zip(result.states, reference, strict=True):
        actual = state.total.properties()
        row = {}
        for key in ("mass_kg", "center_of_mass_m", "inertia_kg_m2"):
            difference = np.asarray(actual[key]) - expected[key]
            if not np.allclose(
                actual[key], expected[key], rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
            ):
                raise ValueError(f"合成解析参考核对失败：{key}")
            row[key] = float(np.max(np.abs(difference)))
        errors.append(row)
    return errors


def export_result(
    result: ProgramResult,
    output: Path,
    reference: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """成功后导出可移植重放包，拒绝覆盖已有实验。

    Args:
        result: 全部步骤已成功的内存事务。
        output: 尚不存在的新目录。
        reference: 仅默认合成案例具有解析参考，实际物体不自动生成真值。

    Returns:
        带逐步检查、解析误差和导出来源的报告。
    """
    reference_errors = check_reference(result, reference) if reference else None
    output.mkdir(parents=True, exist_ok=False)
    portable_steps, copied = [], {}
    donors = dict(result.used_donors)
    source = result.states[0]
    for spec in result.program.steps:
        if spec.donor_state_ref:
            key = spec.donor_state_ref
            if key not in copied:
                part = donors[key]
                relative = f"donors/donor_{len(copied):03d}"
                save_object(
                    PhysicalObject("donor", (part,), source.grid_origin_m),
                    output / relative,
                )
                copied[key] = relative
            spec = replace(spec, donor_state_ref=copied[key])
        portable_steps.append(spec)
    portable = EditProgram(source.object_id, source.version, tuple(portable_steps))
    write_json(output / "program.json", portable.to_dict())
    surfaces, rows = [], []
    for index, state in enumerate(result.states):
        folder = output / f"state_{index:03d}"
        save_object(state, folder)
        for part_index, part in enumerate(state.parts):
            relative = f"state_{index:03d}/part_{part_index:03d}.ply"
            nonempty = bool(np.any(part.density_kg_m3 > 0))
            if nonempty:
                part_surface(state, part.part_id).export(output / relative)
            surfaces.append(
                {
                    "state": index,
                    "part_id": part.part_id,
                    "mesh": relative if nonempty else None,
                    "status": "exported" if nonempty else "empty_material",
                }
            )
        physics = state.total.properties()
        row = {"state": index, "mass_kg": physics["mass_kg"]}
        row.update(
            dict(
                zip(
                    ("com_x_m", "com_y_m", "com_z_m"),
                    physics["center_of_mass_m"],
                    strict=True,
                )
            )
        )
        for axis, value in enumerate(np.linalg.eigvalsh(physics["inertia_kg_m2"])):
            row[f"principal_inertia_{axis + 1}_kg_m2"] = float(value)
        if reference:
            row["reference_mass_kg"] = reference[index]["mass_kg"]
            row["absolute_mass_error_kg"] = reference_errors[index]["mass_kg"]
        rows.append(row)
    with (output / "physical_updates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_edit_history(result, output, reference)
    report = {
        "status": "complete",
        "reference_type": "analytic_synthetic" if reference else "none",
        "real_object_accuracy_evaluated": False,
        "language_backend": "explicit_grammar_or_validated_json",
        "steps": list(result.reports),
        "state_physics": [state.total.properties() for state in result.states],
        "analytic_reference": reference,
        "analytic_errors": reference_errors,
        "surface_exports": surfaces,
        "geometry_figure_state_indices": list(
            preview_state_indices(len(result.states))
        ),
        "limits": [
            "axis-aligned common grid",
            "solid / closed hollow box only",
            "no learned language model or appearance update",
            "no strength, uncertainty or robot validation",
            "recorded timings are diagnostics, not speedup evidence",
        ],
    }
    write_json(output / "report.json", report)
    files = sorted(path for path in output.rglob("*") if path.is_file())
    source_root = Path(__file__).resolve().parents[1]
    source_files = (
        "physics/edit_program.py",
        "physics/part_editing.py",
        "physics/part_state.py",
        "physics/part_io.py",
        "mesh/state_surface.py",
        "figures/edit_history.py",
        "experiments/edit_commands.py",
        "experiments/part_edit_demo.py",
    )
    write_json(
        output / "manifest.json",
        {
            "status": "complete",
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "source_sha256": {
                name: file_sha256(source_root / name) for name in source_files
            },
            "files": {
                path.relative_to(output).as_posix(): file_sha256(path) for path in files
            },
        },
    )
    return report


def key_value_arguments(values: list[str], label: str) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, target = value.partition("=")
        if not separator or not key.strip() or not target.strip() or key in result:
            raise ValueError(f"{label}需要不重复的名称=值")
        result[key] = target
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, help="包含state.json/cells.npz的源目录")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--plan", type=Path, help="严格的program.json")
    group.add_argument("--command", action="append", help="受限中文指令，可重复")
    parser.add_argument("--alias", action="append", default=[], help="显示名称=part_id")
    parser.add_argument(
        "--donor", action="append", default=[], help="名称=替换件状态目录"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.state and not (args.plan or args.command):
        parser.error("指定真实源状态时必须同时指定--plan或--command")
    if args.plan and (args.alias or args.donor):
        parser.error("JSON程序已经包含明确ID和替换件路径，不接受别名或donor覆盖")
    output = (args.output or DEFAULT_OUTPUT / f"run_{time.time_ns()}").resolve()
    try:
        if output.exists():
            raise FileExistsError(f"拒绝覆盖已有目录：{output}")
        default_demo = not (args.state or args.plan or args.command)
        memory_donors: dict[str, PartPhysicalState] = {}
        if args.state:
            source = load_object(args.state)
        else:
            source, donor = make_example()
            memory_donors[str(Path("synthetic_aluminum_leg").resolve())] = donor
        if args.plan:
            program = EditProgram.from_json(args.plan)
        else:
            aliases = key_value_arguments(args.alias, "--alias")
            candidates = key_value_arguments(args.donor, "--donor")
            if default_demo:
                if aliases or candidates:
                    raise ValueError("自定义别名或替换件时必须提供--command")
                candidates["aluminum_leg"] = next(iter(memory_donors))
            program = program_from_commands(
                source, args.command or DEFAULT_COMMANDS, aliases, candidates
            )
        result = execute_program(source, program, memory_donors)
        report = export_result(
            result,
            output,
            synthetic_reference() if default_demo else None,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"编辑失败：{error}\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "mass_kg": [row["mass_kg"] for row in report["state_physics"]],
                "reference_type": report["reference_type"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
