"""对条件局部材料域执行同冲量自由刚体响应，检查质量与惯量更新链路。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.physics.mesh_asset_io import load_mesh_asset
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.physics.rigid_response import free_flight, impulse_response

IMPULSE_NS = np.array([0.0, 0.001, 0.0])
DURATION_S = 0.5
SAMPLES = 101
MAX_DRIFT = 1e-6


def plot_response(records: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(
        2,
        len(records),
        figsize=(3.2 * len(records), 5.2),
        layout="constrained",
        squeeze=False,
    )
    for column, row in enumerate(records):
        for phase, color in (("before", "#5A76A8"), ("after", "#C3916C")):
            with np.load(output / row["scene_id"] / f"{phase}.npz") as data:
                time = data["times_s"]
                displacement = np.linalg.norm(
                    data["position_m"] - data["position_m"][0], axis=1
                )
                angular_speed = np.linalg.norm(data["omega_body_rad_s"], axis=1)
            axes[0, column].plot(time, displacement, color=color, label=phase)
            axes[1, column].plot(time, angular_speed, color=color, label=phase)
        axes[0, column].set(
            title=row["scene_id"].split("_")[0],
            xlabel="Time (s)",
            ylabel="COM displacement (m)",
        )
        axes[1, column].set(xlabel="Time (s)", ylabel="Angular speed (rad/s)")
        for axis in axes[:, column]:
            axis.legend(frameon=False, fontsize=8)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "Same imposed impulse | conditional local-layer zero-gravity motion\n"
        "Declared geometry scale; no contact or measured response reference",
        fontsize=10,
    )
    for suffix in ("svg", "png"):
        figure.savefig(output / f"conditional_rigid_response.{suffix}", dpi=300)
    plt.close(figure)


def run(source: Path, output: Path) -> dict[str, Any]:
    """保留全部对象的成功/缺失状态，不将局部片响应称为整物体仿真。"""
    report = json.loads((source / "report.json").read_text(encoding="utf-8"))
    if report["scope"] != "conditional_local_layer_physics_not_real_mass_estimation":
        raise ValueError("需要带条件构造来源的局部域实验报告")
    output.mkdir(parents=True, exist_ok=False)
    scenes = list(dict.fromkeys(r["scene_id"] for r in report["records"]))
    results, missing = [], []
    for scene in scenes:
        entries = [
            r
            for r in report["records"]
            if r["scene_id"] == scene
            and r["status"] == "complete"
            and r["thickness_ratio"] == 0.002
        ]
        if not entries:
            missing.append(
                {"scene_id": scene, "reason": "no_valid_domain_at_fixed_ratio"}
            )
            continue
        record: dict[str, Any] = {"scene_id": scene}
        directory = source / scene / "ratio_0.002"
        states = {
            phase: load_mesh_asset(directory / phase) for phase in ("before", "after")
        }
        reference = states["before"]
        center = np.array(reference.properties["center_of_mass_m"])
        vertices = np.vstack([r.domain.vertices_m for r in reference.regions])
        point = vertices[np.linalg.norm(vertices - center, axis=1).argmax()]
        child = output / scene
        child.mkdir()
        for phase, state in states.items():
            properties = state.properties
            inertia = np.asarray(properties["inertia_kg_m2"])
            center = np.asarray(properties["center_of_mass_m"])
            velocity, omega = impulse_response(
                properties["mass_kg"], inertia, point - center, IMPULSE_NS
            )
            trajectory = free_flight(
                properties["mass_kg"],
                inertia,
                center,
                velocity,
                omega,
                np.linspace(0, DURATION_S, SAMPLES),
                np.zeros(3),
            )
            arrays = {k: v for k, v in trajectory.items() if isinstance(v, np.ndarray)}
            np.savez_compressed(child / f"{phase}.npz", **arrays)
            diagnostics = {k: v for k, v in trajectory.items() if k not in arrays}
            drift = max(diagnostics.values())
            record[phase] = {
                **diagnostics,
                "mass_kg": properties["mass_kg"],
                "impulse_ns": IMPULSE_NS.tolist(),
                "impulse_point_m": point.tolist(),
                "initial_velocity_m_s": velocity.tolist(),
                "initial_omega_rad_s": omega.tolist(),
                "numerical_checks_passed": drift < MAX_DRIFT,
            }
        write_json(child / "report.json", record)
        results.append(record)
    result = {
        "status": (
            "complete"
            if results and not missing
            else (
                "partial_geometry_acceptance" if results else "failed_no_valid_domains"
            )
        ),
        "scope": "conditional_fragment_free_flight_numerical_consistency",
        "source_sha256": file_sha256(source / "report.json"),
        "contact_simulated": False,
        "deformation_simulated": False,
        "real_response_accuracy_established": False,
        "objects": results,
        "missing": missing,
    }
    write_json(output / "report.json", result)
    if results:
        plot_response(results, output)
    else:
        raise RuntimeError("固定厚度下无合法局部域，不能报告刚体验证通过")
    if any(
        not row[phase]["numerical_checks_passed"]
        for row in results
        for phase in ("before", "after")
    ):
        raise RuntimeError("存在刚体积分守恒漂移超限；已保留报告，不标记通过")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
