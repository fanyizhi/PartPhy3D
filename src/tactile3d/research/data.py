"""独立解析真值的分组配对数据；不是ABO，也不是VLM伪标签。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tactile3d.research.construction import GRID_SHAPE, TRAIN_EDITS, Array

VARIANTS_PER_FAMILY = 6
FEATURE_NAMES = (
    "log_pitch_m",
    "log_outer_density_prior",
    "log_inner_density_prior",
    "appearance_r",
    "appearance_g",
    "appearance_b",
    "context_cue",
    "semantic_solid",
    "semantic_shell",
    "semantic_layered",
    "semantic_frame",
)
OUTER_PRIORS = np.array([550.0, 1100.0, 2700.0, 7800.0])
INNER_PRIORS = np.array([35.0, 90.0, 240.0])
APPEARANCE = np.array(
    [[0.53, 0.32, 0.16], [0.28, 0.44, 0.64], [0.66, 0.68, 0.70], [0.36, 0.38, 0.40]]
)


def analytic_volumes(pitch: Array) -> Array:
    """长方体、闭壳与十二棱梁的解析体积，独立于体素计数实现。"""
    size = pitch[:, None] * np.asarray(GRID_SHAPE)
    outer = np.prod(size, axis=1)
    inner = np.prod(size - 2 * pitch[:, None], axis=1)
    shell = outer - inner
    # 十二根棱梁在八角各重叠三次，容斥后扣除两次角体积。
    frame = 4 * pitch**2 * size.sum(1) - 16 * pitch**3
    result = np.zeros((len(pitch), 4, 2))
    result[:, 0, 0] = outer
    result[:, 1, 0] = shell
    result[:, 2, 0], result[:, 2, 1] = shell, inner
    result[:, 3, 0], result[:, 3, 1] = frame, outer - frame
    return result


def make_dataset(count: int, seed: int) -> tuple[dict[str, Array], dict[str, Array]]:
    """生成同族变体，并按族而不是按图片分割，避免相似变体泄漏。

    Args:
        count: 至少60，且为每族6个实例的整数倍。
        seed: 固定的伪随机种子。

    Returns:
        不含物理真值的输入，以及单独保存的解析监督标签。
    """
    if count < 60 or count % VARIANTS_PER_FAMILY:
        raise ValueError("对象数至少60，且必须是6的整数倍")
    rng = np.random.default_rng(seed)
    families = count // VARIANTS_PER_FAMILY
    groups = np.repeat(np.arange(families), VARIANTS_PER_FAMILY)
    order = rng.permutation(families)
    cut1, cut2 = int(0.65 * families), int(0.82 * families)
    family_split = np.empty(families, dtype=int)
    family_split[order[:cut1]] = 0
    family_split[order[cut1:cut2]] = 1
    family_split[order[cut2:]] = 2
    semantic = rng.integers(0, 4, families)[groups]
    outer_material = rng.integers(0, len(OUTER_PRIORS), families)[groups]
    inner_material = rng.integers(0, len(INNER_PRIORS), families)[groups]
    pitch = np.exp(rng.uniform(np.log(0.006), np.log(0.018), count))
    priors = np.column_stack(
        (OUTER_PRIORS[outer_material], INNER_PRIORS[inner_material])
    )
    construction = np.where(
        rng.random(count) < 0.75, semantic, rng.integers(0, 4, count)
    )
    context = rng.uniform(0.15, 0.95, count)
    family_effect = rng.normal(0, 0.08, (families, 2))[groups]
    density = priors * np.exp(family_effect + rng.normal(0, 0.10, (count, 2)))
    fill = np.clip(
        np.column_stack((0.45 + 0.50 * context, 0.35 + 0.60 * context))
        + rng.normal(0, 0.05, (count, 2)),
        0.05,
        1.0,
    )
    rgb = np.clip(APPEARANCE[outer_material] + rng.normal(0, 0.06, (count, 3)), 0, 1)
    features = np.column_stack(
        (np.log(pitch), np.log(priors), rgb, context, np.eye(4)[semantic])
    )
    volumes = analytic_volumes(pitch)[np.arange(count), construction]
    edited_density = np.repeat(density[:, None], len(TRAIN_EDITS) + 1, axis=1)
    for column, (phase, value) in enumerate(TRAIN_EDITS, 1):
        edited_density[:, column, phase] = value
    masses = np.einsum("nr,ner->ne", volumes * fill, edited_density)
    inputs = {
        "features": features,
        "priors": priors,
        "pitch_m": pitch,
        "group": groups,
        "split": family_split[groups],
    }
    labels = {
        "construction": construction,
        "density": density,
        "fill": fill,
        "mass_kg": masses,
    }
    return inputs, labels


def save_dataset(path: Path, count: int, seed: int) -> None:
    inputs, labels = make_dataset(count, seed)
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path / "inputs.npz", **inputs)
    np.savez_compressed(path / "labels.npz", **labels)


def load_inputs(path: Path) -> dict[str, Array]:
    with np.load(path, allow_pickle=False) as stored:
        required = {"features", "priors", "pitch_m", "group", "split"}
        if set(stored.files) != required:
            raise ValueError("输入数据字段不匹配，禁止混入评测标签")
        result = {name: stored[name].copy() for name in stored.files}
    n = len(result["pitch_m"])
    shapes = {
        "features": (n, len(FEATURE_NAMES)),
        "priors": (n, 2),
        "pitch_m": (n,),
        "group": (n,),
        "split": (n,),
    }
    for name, shape in shapes.items():
        if result[name].shape != shape or not np.isfinite(result[name]).all():
            raise ValueError(f"输入字段形状或数值非法：{name}")
    if np.any(result["priors"] <= 0) or np.any(result["pitch_m"] <= 0):
        raise ValueError("尺度和密度先验必须为正")
    if not set(np.unique(result["split"])) == {0, 1, 2}:
        raise ValueError("必须具有train/calibration/test三个分割")
    for group in np.unique(result["group"]):
        if len(np.unique(result["split"][result["group"] == group])) != 1:
            raise ValueError("同族对象跨分割，存在数据泄漏")
    return result
