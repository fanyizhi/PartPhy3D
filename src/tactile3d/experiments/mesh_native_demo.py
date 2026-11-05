"""无需GPU的Mesh原生主线：封闭区域、直接积分、材料编辑与重放。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from tactile3d.figures.mesh_native import (
    plot_mesh_edit,
    plot_mesh_physics,
    plot_mesh_sections,
)
from tactile3d.physics.mesh_asset_io import (
    import_mesh_regions,
    load_mesh_asset,
    save_mesh_asset,
)
from tactile3d.physics.mesh_native import (
    MeshPhysicalObject,
    MeshRegion,
    closed_shell,
    verify_mesh_recompute,
)
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.physics.part_state import Material
from tactile3d.research.mesh_templates import box_domain
from tactile3d.research.workflow import write_csv

OUTER_SIZE_M = (0.20, 0.30, 0.40)
INNER_SIZE_M = (0.16, 0.26, 0.36)
CENTER_M = (0.10, -0.04, 0.20)
SHELL_MATERIAL = Material(
    "shell_polymer", 1000, "known_design_preset", 2e9, 0.35, (175, 125, 75, 255)
)
FILL_MATERIAL = Material(
    "foam", 50, "known_design_preset", display_rgba=(75, 145, 180, 255)
)
EDIT_MATERIAL = Material(
    "aluminum", 2700, "explicit_design_intervention", 69e9, 0.33, (170, 185, 200, 255)
)
REFERENCE_RTOL = 1e-9
REFERENCE_ATOL = 1e-12


def make_mesh_example() -> (
    tuple[MeshPhysicalObject, MeshPhysicalObject, MeshPhysicalObject]
):
    """外壳和内部填充均为显式闭合Mesh，不使用表面标签估算内部体积。"""
    outer = box_domain(OUTER_SIZE_M, CENTER_M)
    inner = box_domain(INNER_SIZE_M, CENTER_M)
    shell = closed_shell(outer, inner)
    solid = MeshPhysicalObject(
        "specified_solid", (MeshRegion("body", "bulk", outer, SHELL_MATERIAL),)
    )
    hollow = MeshPhysicalObject(
        "specified_hollow", (MeshRegion("body", "shell", shell, SHELL_MATERIAL),)
    )
    layered = MeshPhysicalObject(
        "specified_layered",
        (
            MeshRegion("body", "shell", shell, SHELL_MATERIAL),
            MeshRegion("body", "filling", inner, FILL_MATERIAL),
        ),
    )
    return solid, hollow, layered


def analytic_box(
    size: tuple[float, float, float], density: float
) -> tuple[float, np.ndarray]:
    dimensions = np.asarray(size)
    mass = float(np.prod(dimensions) * density)
    inertia = mass / 12 * np.diag(np.sum(dimensions**2) - dimensions**2)
    return mass, inertia


def check_example(states: list[MeshPhysicalObject]) -> dict[str, float]:
    """以盒解析惯量及空腔减法建立独立参考，不重复调用Mesh积分器。"""
    outer_mass, outer_i = analytic_box(OUTER_SIZE_M, 1000)
    inner_mass, inner_i = analytic_box(INNER_SIZE_M, 1000)
    foam_mass, foam_i = analytic_box(INNER_SIZE_M, 50)
    expected = [
        (outer_mass, outer_i),
        (outer_mass - inner_mass, outer_i - inner_i),
        (outer_mass - inner_mass + foam_mass, outer_i - inner_i + foam_i),
        (
            (outer_mass - inner_mass) * 2.7 + foam_mass,
            (outer_i - inner_i) * 2.7 + foam_i,
        ),
    ]
    errors = {}
    for index, (state, (mass, inertia)) in enumerate(
        zip(states, expected, strict=True)
    ):
        for key, reference in (
            ("mass_kg", mass),
            ("center_of_mass_m", CENTER_M),
            ("inertia_kg_m2", inertia),
        ):
            actual = state.properties[key]
            if not np.allclose(
                actual, reference, rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
            ):
                raise ValueError(f"独立解析参考不一致：{index}/{key}")
            errors[f"{index}_{key}"] = float(
                np.max(np.abs(np.asarray(actual) - reference))
            )
    return errors


def run_mesh_demo(
    output: Path,
    input_json: Path | None = None,
    asset: Path | None = None,
    plan: Path | None = None,
    figures: bool = True,
) -> dict[str, Any]:
    """运行纯Mesh验证或用户区域资产，不进行体素化修复。

    Args:
        output: 新结果目录。
        input_json: 用户已声明的区域Mesh输入，需明确比例、来源与材料。
        asset: 先前导出的Mesh原生资产。
        plan: 真实输入必须提供明确的材料编辑JSON。
        figures: 是否生成Matplotlib和逐视角图片。

    Returns:
        原始与编辑物理量、重算核对及使用范围；无独立参考时不报准确性。
    """
    if input_json is not None and asset is not None:
        raise ValueError("input_json与asset只能提供一个")
    if (input_json is not None or asset is not None) and plan is None:
        raise ValueError("用户资产必须提供编辑计划，不套用默认构造或目标")
    if plan is None:
        payload = {
            "part_id": "body",
            "region_id": "shell",
            "expected_version": 0,
            "material": EDIT_MATERIAL.to_dict(),
        }
    else:
        payload = json.loads(plan.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, dict) or set(payload) != {
            "part_id",
            "region_id",
            "expected_version",
            "material",
        }:
            raise ValueError(
                "编辑JSON需要part_id、region_id、expected_version和material"
            )
    material = Material.from_dict(payload["material"])
    states = []
    if input_json is not None:
        original = import_mesh_regions(input_json)
    elif asset is not None:
        original = load_mesh_asset(asset)
    else:
        states = list(make_mesh_example())
        original = states[-1]
    edited, report = original.edit_material(
        payload["part_id"], payload["region_id"], material, payload["expected_version"]
    )
    errors = check_example([*states, edited]) if plan is None and states else None
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running"})
    try:
        write_json(output / "edit_spec.json", payload)
        save_mesh_asset(original, output / "original")
        save_mesh_asset(edited, output / "edited")
        physics_rows = []
        for label, state in (("original", original), ("edited", edited)):
            props = state.properties
            row = {"state": label, "mass_kg": props["mass_kg"]}
            row.update(
                {
                    f"COM_{axis}_m": value
                    for axis, value in zip(
                        "xyz", props["center_of_mass_m"], strict=True
                    )
                }
            )
            row.update(
                {
                    f"I_{i}{j}_kg_m2": value
                    for (i, j), value in np.ndenumerate(props["inertia_kg_m2"])
                }
            )
            physics_rows.append(row)
        write_csv(output / "physics.csv", physics_rows)
        rows = [
            {
                "state": label,
                "part_id": r.part_id,
                "region_id": r.region_id,
                "volume_m3": r.domain.moments.volume_m3,
                "fill_fraction": r.fill_fraction,
                "density_kg_m3": float(r.material.density_kg_m3),
                "effective_density_kg_m3": r.effective_density,
                "material": r.material.name,
                "youngs_modulus_pa": r.material.youngs_modulus_pa,
                "poisson_ratio": r.material.poisson_ratio,
                "triangles": len(r.domain.faces),
                "source": r.material.source,
            }
            for label, state in (("original", original), ("edited", edited))
            for r in state.regions
        ]
        write_csv(output / "regions.csv", rows)
        report.update(
            verification=verify_mesh_recompute(edited),
            analytic_errors=errors,
            real_physical_accuracy_established=False,
            geometry_method="signed_boundary_tetrahedra_and_mesh_CSG",
            scope="known_construction_or_explicit_user_regions_not_image_inference",
        )
        if figures:
            report["views"] = plot_mesh_edit(
                original, edited, payload["part_id"], payload["region_id"], output
            )
            plot_mesh_physics([original, edited], ["Original", "Edited"], output)
            plot_mesh_sections([original, edited], ["Original", "Edited"], output)
        write_json(output / "report.json", report)
        write_json(
            output / "status.json", {"status": "complete", "voxelization_used": False}
        )
        write_json(
            output / "SHA256.json",
            {
                p.relative_to(output).as_posix(): file_sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        )
        return report
    except Exception as error:
        write_json(
            output / "status.json",
            {
                "status": "failed",
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/mesh_native"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    try:
        report = run_mesh_demo(args.output, args.input, args.asset, args.plan)
    except (ValueError, TypeError, OSError, RuntimeError, KeyError) as error:
        parser.exit(1, f"Mesh原生实验失败：{error}\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("before", "after", "delta_mass_kg", "voxelization_used")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
