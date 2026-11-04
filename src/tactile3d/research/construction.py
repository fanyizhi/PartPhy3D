"""有限构造候选与共享体积状态，所有数值使用SI单位。"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from tactile3d.physics.part_state import Material, PartPhysicalState, PhysicalObject

Array = NDArray[np.float64]
CONSTRUCTIONS = ("solid", "closed_shell", "shell_filled", "frame_filled")
GRID_SHAPE = (10, 12, 8)
MIN_FILL = 0.02
DENSITY_LOG_RANGE = 1.5
TRAIN_EDITS = ((0, 1200.0), (0, 2700.0), (1, 50.0), (1, 150.0))
TEST_EDIT = (0, 7850.0)


def candidate_masks(shape: tuple[int, int, int] = GRID_SHAPE) -> Array:
    """构造互斥材料区域，空腔保留为空而不是重复扣除密度。

    Args:
        shape: 各轴单元数，至少4，边界厚度为一个单元。

    Returns:
        (候选数, 单元数, 2)区域指示数组；微观材料孔隙不在此处扣除。
    """
    if len(shape) != 3 or any(type(n) is not int or n < 4 for n in shape):
        raise ValueError("构造网格需要三个不小于4的整数")
    index = np.indices(shape).reshape(3, -1).T
    boundary_count = ((index == 0) | (index == np.array(shape) - 1)).sum(1)
    shell, frame = boundary_count > 0, boundary_count >= 2
    result = np.zeros((len(CONSTRUCTIONS), len(index), 2))
    result[0, :, 0] = 1
    result[1, shell, 0] = 1
    result[2, shell, 0], result[2, ~shell, 1] = 1, 1
    result[3, frame, 0], result[3, ~frame, 1] = 1, 1
    return result


def candidate_volumes(pitch_m: Array) -> Array:
    pitch = np.asarray(pitch_m, dtype=float)
    if pitch.ndim != 1 or not np.isfinite(pitch).all() or np.any(pitch <= 0):
        raise ValueError("pitch必须为有限正数的一维数组")
    return candidate_masks().sum(1)[None] * pitch[:, None, None] ** 3


def decode_state(logits: Array, priors: Array) -> dict[str, Array]:
    if logits.ndim != 2 or logits.shape[1] != 8:
        raise ValueError("状态输出需要(N,8)：4个构造logit和两组密度/占用参数")
    if priors.shape != (len(logits), 2) or np.any(priors <= 0):
        raise ValueError("密度先验需要(N,2)正数")
    if not np.isfinite(logits).all() or not np.isfinite(priors).all():
        raise ValueError("状态输入不能含NaN或无穷")
    probability = np.exp(logits[:, :4] - logits[:, :4].max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    density_latent = np.tanh(logits[:, 4:6])
    sigmoid = 1 / (1 + np.exp(-np.clip(logits[:, 6:8], -60, 60)))
    return {
        "probability": probability,
        "density": priors * np.exp(DENSITY_LOG_RANGE * density_latent),
        "fill": MIN_FILL + (1 - MIN_FILL) * sigmoid,
        "density_derivative": DENSITY_LOG_RANGE * (1 - density_latent**2),
        "fill_derivative": (1 - MIN_FILL) * sigmoid * (1 - sigmoid),
    }


def mass_forward(
    logits: Array,
    priors: Array,
    volumes: Array,
    edits: tuple[tuple[int, float], ...] = TRAIN_EDITS,
    hard: bool = False,
) -> tuple[Array, dict[str, Any]]:
    """原始和编辑质量均由同一状态积分，训练时边缘化有限候选。

    Returns:
        (N,1+编辑数)质量和用于解析反向传播的中间量。
    """
    state = decode_state(logits, priors)
    if volumes.shape != (len(logits), 4, 2) or np.any(volumes < 0):
        raise ValueError("候选体积需要(N,4,2)非负数组")
    if not np.isfinite(volumes).all() or np.any(volumes.sum((1, 2)) <= 0):
        raise ValueError("每个对象需要正的有限体积")
    for phase, density in edits:
        if phase not in (0, 1) or not np.isfinite(density) or density <= 0:
            raise ValueError("编辑需要合法区域索引和正密度")
    densities = np.repeat(state["density"][:, None], len(edits) + 1, axis=1)
    for column, (phase, density) in enumerate(edits, 1):
        densities[:, column, phase] = density
    occupied = volumes * state["fill"][:, None]
    branch_mass = np.einsum("nkr,ner->nek", occupied, densities)
    probability = state["probability"].copy()
    if hard:
        probability[:] = np.eye(4)[state["probability"].argmax(1)]
    mass = np.einsum("nk,nek->ne", probability, branch_mass)
    return mass, {
        **state,
        "used_probability": probability,
        "densities": densities,
        "branch_mass": branch_mass,
        "volumes": volumes,
        "edits": edits,
        "mass": mass,
    }


def mass_backward(gradient: Array, cache: dict[str, Any]) -> Array:
    """对平滑候选混合求导；硬导出不伪装成可微操作。"""
    if gradient.shape != cache["mass"].shape or not np.isfinite(gradient).all():
        raise ValueError("质量梯度形状错误或含非法数值")
    p, volumes = cache["probability"], cache["volumes"]
    result = np.zeros((len(gradient), 8))
    dp = np.einsum("ne,nek->nk", gradient, cache["branch_mass"])
    result[:, :4] = p * (dp - (dp * p).sum(1, keepdims=True))
    mean_volume = np.einsum("nk,nkr->nr", p, volumes)
    d_density = np.zeros_like(mean_volume)
    d_fill = np.zeros_like(mean_volume)
    for column in range(gradient.shape[1]):
        live = np.ones(2)
        if column:
            live[cache["edits"][column - 1][0]] = 0
        d_density += gradient[:, column, None] * mean_volume * cache["fill"] * live
        d_fill += (
            gradient[:, column, None] * mean_volume * cache["densities"][:, column]
        )
    result[:, 4:6] = d_density * cache["density"] * cache["density_derivative"]
    result[:, 6:8] = d_fill * cache["fill_derivative"]
    return result


def export_state(
    object_id: str, pitch_m: float, logits: Array, priors: Array
) -> PhysicalObject:
    """把单对象MAP候选导出为可编辑状态，不把混合候选伪装成一个实物。"""
    if not np.isfinite(pitch_m) or pitch_m <= 0:
        raise ValueError("体素边长必须为有限正数")
    if logits.shape != (1, 8):
        raise ValueError("导出仅接收一个对象")
    decoded = decode_state(logits, priors)
    candidate = int(decoded["probability"][0].argmax())
    region = candidate_masks()[candidate]
    fractions = region * decoded["fill"][0]
    fractions = np.column_stack((fractions, 1 - fractions.sum(1)))
    centers = (np.indices(GRID_SHAPE).reshape(3, -1).T + 0.5) * pitch_m
    part = PartPhysicalState(
        "body",
        object_id + ":body",
        centers,
        pitch_m,
        (
            Material("outer", float(decoded["density"][0, 0]), "learned_synthetic"),
            Material("inner", float(decoded["density"][0, 1]), "learned_synthetic"),
            Material("void", 0.0, "explicit_empty_space"),
        ),
        fractions,
        construction=CONSTRUCTIONS[candidate],
        evidence=("synthetic_box_diagnostic_not_ABO_or_VLM",),
    )
    return PhysicalObject(object_id, (part,), np.full(3, pitch_m / 2))
