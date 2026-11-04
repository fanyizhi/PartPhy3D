"""用独立解析参考审计材料编辑能增加哪些观测信息。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
REFERENCE_DENSITY_KG_M3 = 1000.0
ENVELOPE_VOLUME_M3 = 0.001
TRUE_DENSITY_KG_M3 = 500.0
TRUE_FILL_FRACTION = 0.5
TRUE_REST_MASS_KG = 0.4
EDIT_DENSITIES_KG_M3 = (1200.0, 2700.0)
ILLUSTRATIVE_MASS_SIGMA_KG = 0.001
FINITE_DIFFERENCE_STEP = 1e-6
RANK_RELATIVE_TOLERANCE = 1e-10
JACOBIAN_ABSOLUTE_TOLERANCE = 1e-8
BOUNDARY_EPS_MULTIPLIER = 64.0
DEFAULT_OUTPUT = Path("outputs/observability_audit")


def normalized_response(
    parameters: FloatArray, edit_density_ratios: FloatArray
) -> FloatArray:
    """计算无量纲原质量及材料编辑质量。

    Args:
        parameters: 密度比、宏观填充率、归一化其余部件质量。
        edit_density_ratios: 已知新材料密度与固定参考密度之比。

    Returns:
        保持几何、宏观占用及其余部件不变的原／编辑后质量。
    """
    values = np.asarray(parameters, dtype=np.float64)
    edits = np.asarray(edit_density_ratios, dtype=np.float64)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("参数必须为三个有限值：密度比、填充率、其余质量")
    density_ratio, fraction, rest = values
    if density_ratio <= 0 or not 0 < fraction <= 1 or rest < 0:
        raise ValueError("密度必须为正，填充率须在(0,1]，其余质量不能为负")
    if edits.ndim != 1 or not np.isfinite(edits).all() or np.any(edits <= 0):
        raise ValueError("编辑密度比必须为有限正数的一维数组")
    return rest + fraction * np.concatenate(([density_ratio], edits))


def response_jacobian(
    parameters: FloatArray, edit_density_ratios: FloatArray
) -> FloatArray:
    """对无量纲密度、占用和其余质量求导，避免单位影响秩审计。"""
    normalized_response(parameters, edit_density_ratios)
    density_ratio, fraction, _ = np.asarray(parameters, dtype=np.float64)
    edits = np.asarray(edit_density_ratios, dtype=np.float64)
    result = np.zeros((len(edits) + 1, 3), dtype=np.float64)
    result[:, 1] = np.concatenate(([density_ratio], edits))
    result[:, 2] = 1.0
    result[0, 0] = fraction
    return result


def recover_parameters(
    masses_kg: FloatArray,
    edit_densities_kg_m3: FloatArray,
    envelope_volume_m3: float,
) -> dict[str, float]:
    """在声明的简化模型下从三次独立质量观测恢复参数。

    Args:
        masses_kg: 原质量和两个已知密度替换后的装配质量。
        edit_densities_kg_m3: 两个互异、独立已知的新材料密度。
        envelope_volume_m3: 已知外包络体积。

    Returns:
        密度、占用率、材料体积和不变的其余部件质量。
    """
    masses = np.asarray(masses_kg, dtype=np.float64)
    densities = np.asarray(edit_densities_kg_m3, dtype=np.float64)
    if (
        masses.shape != (3,)
        or densities.shape != (2,)
        or not np.isfinite(masses).all()
        or not np.isfinite(densities).all()
        or np.any(masses <= 0)
        or np.any(densities <= 0)
        or not np.isfinite(envelope_volume_m3)
        or envelope_volume_m3 <= 0
    ):
        raise ValueError("需要三个正质量、两个正密度及正外包络体积")
    gap = float(densities[1] - densities[0])
    if abs(gap) <= np.finfo(float).eps * np.max(densities):
        raise ValueError("两次编辑密度相同，不能独立识别材料体积和其余质量")
    volume = float((masses[2] - masses[1]) / gap)
    rest = float(masses[1] - densities[0] * volume)
    if volume <= 0:
        raise ValueError("观测与正材料体积假设不一致")
    density = float((masses[0] - rest) / volume)
    fraction = volume / envelope_volume_m3
    tolerance = BOUNDARY_EPS_MULTIPLIER * np.finfo(float).eps
    if (
        not np.isfinite([density, fraction, rest, volume]).all()
        or not 0 < fraction <= 1 + tolerance
        or rest < -tolerance * np.max(masses)
        or density <= 0
    ):
        raise ValueError("观测不满足合法密度、占用及不变其余质量的简化模型")
    # 仅修正零质量及满占用端点的舍入，不吸收真实测量不一致。
    rest = max(rest, 0.0)
    fraction = min(fraction, 1.0)
    density = float((masses[0] - rest) / volume)
    return {
        "density_kg_m3": density,
        "fill_fraction": fraction,
        "occupied_volume_m3": volume,
        "rest_mass_kg": rest,
    }


def rank_record(
    name: str, parameters: FloatArray, edit_density_ratios: FloatArray
) -> dict[str, Any]:
    """缺少全列秩时不报告有限条件数，避免误导为参数已可辨识。"""
    jacobian = response_jacobian(parameters, edit_density_ratios)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(
        np.count_nonzero(singular_values > singular_values[0] * RANK_RELATIVE_TOLERANCE)
    )
    return {
        "case": name,
        "observations": jacobian.shape[0],
        "unknowns": jacobian.shape[1],
        "rank": rank,
        "condition_number": (
            float(singular_values[0] / singular_values[-1])
            if rank == jacobian.shape[1]
            else None
        ),
    }


def audit_observability() -> dict[str, Any]:
    """检查解析Jacobian、负控制与测量噪声的局部放大。

    Returns:
        解析审计记录；不包含模型训练或真实物体准确性结论。
    """
    mass_scale = REFERENCE_DENSITY_KG_M3 * ENVELOPE_VOLUME_M3
    parameters = np.array(
        [
            TRUE_DENSITY_KG_M3 / REFERENCE_DENSITY_KG_M3,
            TRUE_FILL_FRACTION,
            TRUE_REST_MASS_KG / mass_scale,
        ]
    )
    densities = np.array(EDIT_DENSITIES_KG_M3)
    edits = densities / REFERENCE_DENSITY_KG_M3
    basis = np.eye(3) * FINITE_DIFFERENCE_STEP
    numerical = np.column_stack(
        [
            (
                normalized_response(parameters + offset, edits)
                - normalized_response(parameters - offset, edits)
            )
            / (2 * FINITE_DIFFERENCE_STEP)
            for offset in basis
        ]
    )
    analytic = response_jacobian(parameters, edits)
    if not np.allclose(numerical, analytic, rtol=0, atol=JACOBIAN_ABSOLUTE_TOLERANCE):
        raise ValueError("解析Jacobian与独立有限差分不一致")
    masses = normalized_response(parameters, edits) * mass_scale
    recovered = recover_parameters(masses, densities, ENVELOPE_VOLUME_M3)
    alternative = np.array([0.325, 0.4, 0.52])
    alternative_masses = normalized_response(alternative, edits) * mass_scale
    if not np.allclose(masses[:2], alternative_masses[:2]):
        raise ValueError("单次编辑的不可辨识负控制构造错误")
    if np.isclose(masses[2], alternative_masses[2]):
        raise ValueError("第二次独立编辑未区分预设负控制")
    cases = (
        ("original_only", np.array([])),
        ("one_edit", edits[:1]),
        ("two_distinct_edits", edits),
        ("two_duplicate_edits", np.repeat(edits[0], 2)),
        ("two_near_duplicate_edits", np.array([edits[0], edits[0] + 0.001])),
    )
    # 仅计独立称量噪声；密度、尺寸、附件变化还需额外误差传播。
    sigma = np.sqrt(2) * ILLUSTRATIVE_MASS_SIGMA_KG / ENVELOPE_VOLUME_M3
    return {
        "scope": "analytic_scalar_model_not_learned_or_real_object_validation",
        "assumptions": [
            "known_envelope_volume_and_two_exact_edit_densities",
            "single_uniform_material_phase_plus_void",
            "unchanged_geometry_macro_occupancy_and_rest_mass",
            "independent_mass_observations_available_for_the_audit",
        ],
        "model_training_executed": False,
        "real_object_accuracy_validated": False,
        "rgb_only_test_identifiability_established": False,
        "jacobian_absolute_error": float(np.max(np.abs(numerical - analytic))),
        "masses_kg": masses.tolist(),
        "alternative_masses_kg": alternative_masses.tolist(),
        "recovered": recovered,
        "cases": [rank_record(name, parameters, values) for name, values in cases],
        "illustrative_noise": {
            "assumed_independent_mass_sigma_kg": ILLUSTRATIVE_MASS_SIGMA_KG,
            "distinct_edits_fill_sigma": float(sigma / abs(np.diff(densities)[0])),
            "one_kg_m3_density_gap_fill_sigma": float(sigma),
            "excludes_density_geometry_and_manufacturing_uncertainty": True,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUT / f"run_{time.time_ns()}"
    try:
        if output.exists():
            raise FileExistsError(f"拒绝覆盖已有审计：{output}")
        report = audit_observability()
        output.mkdir(parents=True, exist_ok=False)
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        with (output / "observability.csv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(report["cases"][0]))
            writer.writeheader()
            writer.writerows(report["cases"])
    except (OSError, ValueError) as error:
        parser.exit(1, f"可辨识性审计失败：{error}\n")
    print(json.dumps({"output": str(output.resolve()), "cases": report["cases"]}))


if __name__ == "__main__":
    main()
