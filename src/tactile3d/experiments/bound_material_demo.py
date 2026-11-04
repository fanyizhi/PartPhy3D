"""部件换材第一版：明确区域、固定构造、同步显示和物理属性。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from tactile3d.figures.bound_material import plot_physics, render_comparison
from tactile3d.physics.material_binding import (
    BoundSurface,
    MaterialRegion,
    edit_bound_material,
    geometry_signature,
    load_bound_asset,
    save_bound_asset,
)
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.physics.part_state import Material, PartPhysicalState, PhysicalObject
from tactile3d.research.workflow import write_csv

PITCH_M = 0.01
WOOD_DENSITY_KG_M3 = 650.0
SEAT_DENSITY_KG_M3 = 600.0
EDIT_DENSITY_KG_M3 = 2700.0
REFERENCE_RTOL = 1e-10
REFERENCE_ATOL = 1e-12
WOOD = Material(
    "wood",
    WOOD_DENSITY_KG_M3,
    "specified_synthetic_fixture_not_measured",
    10e9,
    0.30,
    (155, 105, 58, 255),
)
SEAT = Material(
    "seat_wood",
    SEAT_DENSITY_KG_M3,
    "specified_synthetic_fixture_not_measured",
    10e9,
    0.30,
    (185, 135, 80, 255),
)
ALUMINUM = Material(
    "aluminum",
    EDIT_DENSITY_KG_M3,
    "specified_synthetic_intervention_not_measured",
    69e9,
    0.33,
    (170, 185, 200, 255),
)
BOXES = (
    ("seat", (-0.20, -0.18, 0.40), (40, 36, 4)),
    ("front_left_leg", (-0.16, -0.14, 0.0), (4, 4, 40)),
    ("front_right_leg", (0.12, -0.14, 0.0), (4, 4, 40)),
    ("rear_left_leg", (-0.16, 0.10, 0.0), (4, 4, 40)),
    ("rear_right_leg", (0.12, 0.10, 0.0), (4, 4, 40)),
)
DEFAULT_PART = "front_left_leg"
DEFAULT_REGION = "bulk"


def make_chair_fixture() -> tuple[PhysicalObject, BoundSurface]:
    """构造已知部件的简化四腿凳，表面和体域都由设计尺寸建立。"""
    parts, meshes, regions, labels = [], [], [], []
    for index, (part_id, lower, counts) in enumerate(BOXES):
        dimensions = np.asarray(counts) * PITCH_M
        center = np.asarray(lower) + dimensions / 2
        centers = (np.indices(counts).reshape(3, -1).T + 0.5) * PITCH_M + lower
        parts.append(
            PartPhysicalState(
                part_id,
                f"fixture:{part_id}",
                centers,
                PITCH_M,
                (SEAT if index == 0 else WOOD,),
                np.ones((len(centers), 1)),
                construction="known_synthetic_box",
                evidence=("explicit_design_geometry",),
            )
        )
        mesh = trimesh.creation.box(extents=dimensions)
        mesh.apply_translation(center)
        meshes.append(mesh)
        regions.append(MaterialRegion(part_id, "bulk", 0))
        labels.extend([index] * len(mesh.faces))
    state = PhysicalObject(
        "known_four_leg_stool", tuple(parts), np.full(3, PITCH_M / 2)
    )
    mesh = trimesh.util.concatenate(meshes)
    surface = BoundSurface(
        mesh.vertices,
        mesh.faces,
        np.asarray(labels, dtype=int),
        tuple(regions),
        geometry_signature(state),
        "known_design_not_predicted_segmentation",
    )
    return state, surface


def analytic_reference(edited: bool) -> dict[str, Any]:
    """独立解析盒惯量和移轴参考，不以预测体素积分作自己的真值。"""
    sizes, positions, masses = [], [], []
    for index, (part_id, lower, counts) in enumerate(BOXES):
        size = np.asarray(counts) * PITCH_M
        density = (
            SEAT_DENSITY_KG_M3
            if index == 0
            else (
                EDIT_DENSITY_KG_M3
                if edited and part_id == DEFAULT_PART
                else WOOD_DENSITY_KG_M3
            )
        )
        sizes.append(size)
        positions.append(np.asarray(lower) + size / 2)
        masses.append(density * float(np.prod(size)))
    dimensions, centers, mass = np.array(sizes), np.array(positions), np.array(masses)
    total = float(mass.sum())
    center = mass @ centers / total
    inertia = np.zeros((3, 3))
    for size, position, value in zip(dimensions, centers, mass, strict=True):
        delta = position - center
        inertia += value / 12 * np.diag(np.sum(size**2) - size**2)
        inertia += value * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    return {
        "mass_kg": total,
        "center_of_mass_m": center.tolist(),
        "inertia_kg_m2": inertia.tolist(),
    }


def run_material_edit(
    output: Path,
    asset: Path | None = None,
    plan: Path | None = None,
    figures: bool = True,
) -> dict[str, Any]:
    """运行绑定资产或默认解析样例，导出可重放结果与真实失败记录。

    Args:
        output: 新实验目录，拒绝覆盖已有证据。
        asset: 已提供区域占用和表面对应的资产；省略时使用设计样例。
        plan: 明确部件、区域、预期版本和新材料的JSON。
        figures: 是否执行固定相机的示意色渲染及物理变化绘图。

    Returns:
        编辑区域、物理变化、完整重算和可见区域检查报告。
    """
    if asset is not None and plan is None:
        raise ValueError("实际资产必须提供明确编辑JSON，不能套用默认椅子选择")
    if plan is not None:
        payload = json.loads(plan.read_text(encoding="utf-8-sig"))
        required = {
            "target_part_id",
            "target_region_id",
            "expected_part_version",
            "new_material",
        }
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError("编辑JSON必须恰好包含部件、区域、版本和新材料")
        material = Material.from_dict(payload["new_material"])
    else:
        payload = {
            "target_part_id": DEFAULT_PART,
            "target_region_id": DEFAULT_REGION,
            "expected_part_version": 0,
            "new_material": ALUMINUM.to_dict(),
        }
        material = ALUMINUM
    before, surface = load_bound_asset(asset) if asset else make_chair_fixture()
    after, report = edit_bound_material(
        before,
        surface,
        payload["target_part_id"],
        payload["target_region_id"],
        material,
        payload["expected_part_version"],
    )
    default_reference = asset is None and plan is None
    if default_reference:
        for state, edited in ((before, False), (after, True)):
            reference = analytic_reference(edited)
            for key, value in state.total.properties().items():
                if not np.allclose(
                    value, reference[key], rtol=REFERENCE_RTOL, atol=REFERENCE_ATOL
                ):
                    raise ValueError(f"独立解析参考不匹配：{key}")
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "edit_spec.json", payload)
    write_json(output / "status.json", {"status": "running"})
    try:
        rows = []
        for label, state in (("before", before), ("after", after)):
            save_bound_asset(state, surface, output / label)
            mesh = trimesh.Trimesh(
                surface.vertices_m,
                surface.faces,
                face_colors=surface.face_rgba(state),
                process=False,
            )
            mesh.export(output / f"{label}.ply")
            row = {"state": label, "mass_kg": state.total.mass_kg}
            row.update(
                {
                    f"COM_{axis}_m": value
                    for axis, value in zip(
                        "xyz", state.total.properties()["center_of_mass_m"], strict=True
                    )
                }
            )
            row.update(
                {
                    f"I_{i}{j}_kg_m2": value
                    for (i, j), value in np.ndenumerate(
                        state.total.properties()["inertia_kg_m2"]
                    )
                }
            )
            rows.append(row)
        write_csv(output / "physics.csv", rows)
        if figures:
            report["views"] = render_comparison(
                before,
                after,
                surface,
                payload["target_part_id"],
                payload["target_region_id"],
                output,
            )
            plot_physics(before, after, report, output)
            report["appearance_rendered"] = True
        report.update(
            status="complete",
            independent_reference="analytic_boxes" if default_reference else None,
            real_object_accuracy_established=False,
            interpretation=(
                "declared material edit; geometry/occupancy fixed; schematic colors"
            ),
        )
        write_json(output / "report.json", report)
        write_json(output / "status.json", {"status": "complete"})
        write_json(
            output / "SHA256.json",
            {
                p.relative_to(output).as_posix(): file_sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        )
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
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/bound_material") / f"run_{time.time_ns()}",
    )
    args = parser.parse_args()
    try:
        report = run_material_edit(args.output, args.asset, args.plan)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.exit(1, f"绑定换材失败：{error}\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "part_id",
                    "region_id",
                    "delta_mass_kg",
                    "selected_faces",
                    "before",
                    "after",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
