"""本机可复现的部件状态/编辑验证，不调用网络或GPU模型。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tactile3d.mesh.state_surface import part_surface
from tactile3d.physics.fields import integrate_density_cells
from tactile3d.physics.part_editing import EditSpec, apply_edit, verify_full_recompute
from tactile3d.physics.part_io import (
    import_development_field,
    load_object,
    save_object,
    write_json,
)
from tactile3d.physics.part_state import (
    Connection,
    Interface,
    Material,
    PartPhysicalState,
    PhysicalObject,
)

DEFAULT_OUTPUT = Path("outputs/part_state_edit_demo")
DEFAULT_PITCH_M = 0.01
BODY_DENSITY_KG_M3 = 600.0
WOOD_DENSITY_KG_M3 = 650.0
STEEL_DENSITY_KG_M3 = 7850.0
ALUMINUM_DENSITY_KG_M3 = 2700.0
AUDIT_DENSITY_FACTOR = 1.1
VERIFY_ATOL = 1e-10
COLORS = ("#617C98", "#D49B46")
FIGURE_DPI = 180
MOUNT_SIZE_M = np.array([0.04, 0.04])
GEOMETRY_FIGSIZE = (14, 12)


def translated(z_m: float) -> np.ndarray:
    frame = np.eye(4)
    frame[2, 3] = z_m
    return frame


def box_part(
    part_id: str,
    counts: tuple[int, int, int],
    material: Material,
    z_m: float,
    interface_z_m: float,
) -> PartPhysicalState:
    """生成已知构造盒体，作为独立于ABO预测的解析验证样例。"""
    if len(counts) != 3 or any(not isinstance(n, int) or n <= 0 for n in counts):
        raise ValueError("盒体需要三个正整数体素计数")
    axes = [(np.arange(n) + 0.5 - n / 2) * DEFAULT_PITCH_M for n in counts]
    centers = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    return PartPhysicalState(
        part_id,
        f"{part_id}/original",
        centers,
        DEFAULT_PITCH_M,
        (material,),
        np.ones((len(centers), 1)),
        translated(z_m),
        (Interface("mount", translated(interface_z_m), MOUNT_SIZE_M),),
        construction="known_synthetic_solid_box",
        evidence=("known_synthetic",),
    )


def make_example() -> tuple[PhysicalObject, PartPhysicalState]:
    body = box_part(
        "body", (20, 12, 4), Material("wood_body", BODY_DENSITY_KG_M3), 0.22, -0.02
    )
    leg = box_part("leg", (4, 4, 20), Material("wood", WOOD_DENSITY_KG_M3), 0.10, 0.10)
    source = PhysicalObject(
        "synthetic_mount_assembly",
        (body, leg),
        np.full(3, DEFAULT_PITCH_M / 2),
        (Connection("body", "mount", "leg", "mount"),),
    )
    x = leg.centers_local_m
    stem = (np.abs(x[:, 0]) < 0.01) & (np.abs(x[:, 1]) < 0.01)
    collar = x[:, 2] > 0.08
    solid = stem | collar
    fractions = np.column_stack([solid, ~solid]).astype(float)
    donor = replace(
        leg,
        part_id="donor",
        instance_id="donor/aluminum_stepped_leg",
        materials=(Material("aluminum", ALUMINUM_DENSITY_KG_M3), Material("void", 0.0)),
        fractions=fractions,
        local_to_world=np.eye(4),
        construction="known_synthetic_stepped_leg",
    )
    return source, donor


def plot_results(states: list[PhysicalObject], output: Path) -> None:
    """独立绘图：已知构造的几何、内部占用及质量矩，不冒充实物结果。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    labels = ["Original wood", "Solid steel", "Hollow steel", "Donor replacement"]
    fig = plt.figure(figsize=GEOMETRY_FIGSIZE)
    fig.subplots_adjust(
        left=0.02, right=0.96, bottom=0.07, top=0.92, wspace=0.10, hspace=0.28
    )
    for column, (state, title) in enumerate(zip(states, labels, strict=True), start=1):
        axis = fig.add_subplot(2, 2, column, projection="3d", computed_zorder=False)
        for index, part in enumerate(state.parts):
            mesh = part_surface(state, part.part_id)
            collection = Poly3DCollection(
                mesh.triangles,
                facecolors=COLORS[index],
                edgecolors="none",
                alpha=0.85,
                zorder=1,
                label=part.part_id,
            )
            axis.add_collection3d(collection)
        center = state.total.properties()["center_of_mass_m"]
        axis.scatter(
            *center, color="#A72C38", marker="x", s=70, label="COM (overlay)", zorder=10
        )
        axis.set(
            xlim=(-0.12, 0.12),
            ylim=(-0.08, 0.08),
            zlim=(0, 0.26),
            xlabel="x (m)",
            ylabel="y (m)",
            zlabel="z (m)",
            title=f"{title}: {state.total.mass_kg:.4f} kg",
        )
        axis.set_box_aspect((0.24, 0.16, 0.26))
        axis.set_xticks([-0.10, 0, 0.10])
        axis.set_yticks([-0.06, 0, 0.06])
        axis.set_zticks([0, 0.10, 0.20])
        axis.legend(loc="upper left", fontsize=8)
    fig.suptitle("Known synthetic construction: interface-compatible edits")
    save_figure(fig, output / "01_geometry_edits")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    positions = np.arange(len(states))
    mass = [state.total.mass_kg for state in states]
    axes[0, 0].bar(positions, mass, color=COLORS[1], label="Integrated mass")
    axes[0, 0].set(ylabel="Mass (kg)", xlabel="Edit state", title="Mass update")
    centers = np.array(
        [state.total.properties()["center_of_mass_m"] for state in states]
    )
    for index, name in enumerate(("x", "y", "z")):
        axes[0, 1].plot(positions, centers[:, index], "o-", label=f"COM {name}")
    axes[0, 1].set(ylabel="COM coordinate (m)", xlabel="Edit state")
    inertia = np.array(
        [
            np.linalg.eigvalsh(state.total.properties()["inertia_kg_m2"])
            for state in states
        ]
    )
    for index in range(3):
        axes[1, 0].plot(
            positions, inertia[:, index], "o-", label=f"Principal {index + 1}"
        )
    axes[1, 0].set(ylabel="Principal inertia (kg m²)", xlabel="Edit state")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.set_xticks(positions, labels, rotation=15)
        axis.legend(fontsize=8)
    shell = states[2].part("leg")
    points = shell.centers_world_m
    plane = np.min(points[:, 1]) + DEFAULT_PITCH_M
    selected = np.isclose(points[:, 1], plane)
    occupied = shell.density_kg_m3[selected] > 0
    for mask, color, name in (
        (occupied, COLORS[1], "Steel"),
        (~occupied, "#DCE5ED", "Void"),
    ):
        axes[1, 1].scatter(
            1000 * points[selected][mask, 0],
            1000 * points[selected][mask, 2],
            c=color,
            marker="s",
            s=65,
            label=name,
        )
    axes[1, 1].set(
        xlabel="x (mm)",
        ylabel="z (mm)",
        title="Hollow steel: interior slice",
        aspect="equal",
    )
    axes[1, 1].set_xticks([-10, 10])
    axes[1, 1].legend(loc="upper left", bbox_to_anchor=(1.05, 1))
    fig.suptitle("Deterministic numerical validation; no learned prediction or real GT")
    save_figure(fig, output / "02_physical_state")
    plt.close(fig)


def save_figure(figure: Any, stem: Path) -> None:
    for suffix in ("png", "svg", "pdf"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=FIGURE_DPI)


def run_demo(output: Path) -> dict[str, Any]:
    source, donor = make_example()
    specs = [
        EditSpec(
            "leg",
            "material_change",
            expected_part_version=0,
            material_name="wood",
            new_material=Material("steel", STEEL_DENSITY_KG_M3),
        ),
        EditSpec(
            "leg",
            "construction_change",
            expected_part_version=1,
            material_name="steel",
            construction="hollow_box",
            wall_thickness_m=DEFAULT_PITCH_M,
        ),
        EditSpec(
            "leg", "replace_part", expected_part_version=2, mount_interface="mount"
        ),
    ]
    states, reports = [source], []
    for spec in specs:
        result = apply_edit(
            states[-1], spec, donor if spec.operation == "replace_part" else None
        )
        result.report["full_recompute_errors"] = verify_full_recompute(result.state)
        states.append(result.state)
        reports.append(result.report)
    for index, state in enumerate(states):
        folder = output / f"state_{index}"
        save_object(state, folder)
        loaded = load_object(folder)
        np.testing.assert_allclose(
            loaded.total.second_kg_m2,
            state.total.second_kg_m2,
            rtol=0,
            atol=VERIFY_ATOL,
        )
        for part in state.parts:
            part_surface(state, part.part_id).export(folder / f"{part.part_id}.ply")
    save_object(
        PhysicalObject("donor", (donor,), source.grid_origin_m), output / "donor"
    )
    for index, spec in enumerate(specs, start=1):
        replay = replace(spec, donor_state_ref="../donor") if index == 3 else spec
        replay.to_json(output / "edit_specs" / f"{index}_{spec.operation}.json")
    plot_results(states, output)
    report = {
        "reference_type": "known_synthetic",
        "trained_model": False,
        "mass_kg": [state.total.mass_kg for state in states],
        "edits": reports,
        "limits": [
            "axis-aligned common grid",
            "no appearance or learned feature update",
            "no strength, deformable simulation or uncertainty validation",
        ],
    }
    write_json(output / "report.json", report)
    return report


def audit_pilot(root: Path, output: Path) -> dict[str, Any]:
    files = sorted((root / "objects").glob("*/physical_field.npz"))
    if len(files) != 10:
        raise ValueError("需要现有固定十物体目录，不能静默改变对象数量")
    rows = []
    for file in files:
        state = import_development_field(file)
        with np.load(file, allow_pickle=False) as arrays:
            reference = integrate_density_cells(
                arrays["centers_m"],
                arrays["density_mean_kg_m3"],
                float(arrays["pitch_m"]),
            )
        actual = state.total.properties()
        np.testing.assert_allclose(
            actual["mass_kg"], reference["mass_kg"], rtol=0, atol=VERIFY_ATOL
        )
        np.testing.assert_allclose(
            actual["inertia_kg_m2"],
            reference["inertia_kg_m2"],
            rtol=0,
            atol=VERIFY_ATOL,
        )
        target = max(state.parts, key=lambda part: part.moments().mass_kg)
        material = Material(
            "explicit_effective_density_scenario",
            target.density_kg_m3 * AUDIT_DENSITY_FACTOR,
            "numerical_scenario_not_material_identification",
        )
        result = apply_edit(
            state,
            EditSpec(
                target.part_id,
                "material_change",
                material_name="effective_mixture",
                new_material=material,
            ),
        )
        errors = verify_full_recompute(result.state)
        expected = state.total.mass_kg + target.moments().mass_kg * (
            AUDIT_DENSITY_FACTOR - 1
        )
        np.testing.assert_allclose(
            result.state.total.mass_kg, expected, rtol=0, atol=VERIFY_ATOL
        )
        folder = output / file.parent.name
        save_object(state, folder / "imported_state")
        round_trip = load_object(folder / "imported_state")
        np.testing.assert_allclose(
            round_trip.total.mass_kg, state.total.mass_kg, rtol=0, atol=VERIFY_ATOL
        )
        write_json(folder / "numerical_edit_report.json", result.report)
        rows.append(
            {
                "scene_id": state.object_id,
                "parts": len(state.parts),
                "mass_before_kg": state.total.mass_kg,
                "mass_after_scenario_kg": result.state.total.mass_kg,
                "mass_import_error_kg": abs(actual["mass_kg"] - reference["mass_kg"]),
                "full_second_moment_error": errors["second_kg_m2"],
                "recomputed_parts": len(result.state.recomputed_part_ids),
            }
        )
    with (output / "audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "objects": len(rows),
        "status": "numerical_state_audit_passed",
        "physical_accuracy_evaluated": False,
        "part_labels": "previous geometric clusters, not semantic truth",
        "uncertainty_migrated": False,
        "source": str(root.resolve()),
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pilot", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--edit-spec", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = DEFAULT_OUTPUT / f"run_{time.time_ns()}"
    if bool(args.state) != bool(args.edit_spec) or (args.pilot and args.state):
        parser.error("--state与--edit-spec必须同时提供，且不能同时使用--pilot")
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        if args.pilot:
            report = audit_pilot(args.pilot, args.output)
        elif args.state:
            spec = EditSpec.from_json(args.edit_spec)
            donor = None
            if spec.donor_state_ref:
                candidate = load_object(spec.donor_state_ref)
                if len(candidate.parts) != 1:
                    raise ValueError("donor状态必须恰含一个部件")
                donor = candidate.parts[0]
            result = apply_edit(load_object(args.state), spec, donor)
            report = result.report
            report["full_recompute_errors"] = verify_full_recompute(result.state)
            save_object(result.state, args.output / "edited_state")
            write_json(args.output / "report.json", report)
        else:
            report = run_demo(args.output)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"运行失败：{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"输出目录：{args.output.resolve()}")


if __name__ == "__main__":
    main()
