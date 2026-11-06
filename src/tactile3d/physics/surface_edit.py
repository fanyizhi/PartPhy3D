"""可见表面的原子换材；未闭合的材料域不产生质量或惯量。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from tactile3d.physics.materials import MaterialPrior


@dataclass(frozen=True)
class SurfaceEditResult:
    changed: NDArray[np.bool_]
    density_kg_m3: NDArray[np.float64]
    youngs_pa: NDArray[np.float64]
    colors_rgba: NDArray[np.uint8]
    tokens: dict[str, Any]


def edit_surface_fields(
    labels: NDArray[np.int32],
    density_kg_m3: NDArray[np.float64],
    youngs_pa: NDArray[np.float64],
    colors_rgba: NDArray[np.uint8],
    tokens: dict[str, Any],
    part_id: str,
    material: MaterialPrior,
    color_rgba: tuple[int, int, int, int],
) -> SurfaceEditResult:
    """同步替换逐面物性和部件状态，输入不变，非法编辑不产生部分写入。

    Args:
        labels: 与同一个 Mesh 绑定的逐面部件索引，0 表示未知。
        density_kg_m3: 密度条件先验；未知值允许为 NaN。
        youngs_pa: 杨氏模量条件先验；未知值允许为 NaN。
        colors_rgba: 当前逐面材料示意颜色。
        tokens: 含唯一 part_id、part_index 的部件数据契约。
        part_id: 需要替换的可见部件。
        material: 明确给定的新材料先验，不解释为实测物性。
        color_rgba: 新材料的示意颜色。

    Returns:
        独立的更新数组和 Token。此函数不构造内部域、不估计质量。
    """
    labels = np.asarray(labels)
    density = np.asarray(density_kg_m3, dtype=float)
    youngs = np.asarray(youngs_pa, dtype=float)
    colors = np.asarray(colors_rgba)
    if labels.ndim != 1 or not labels.size or labels.dtype.kind not in "iu":
        raise ValueError("表面编辑需要非空一维整数部件标签")
    if density.shape != labels.shape or youngs.shape != labels.shape:
        raise ValueError("逐面物性长度必须与部件标签一致")
    if colors.shape != (len(labels), 4) or colors.dtype != np.uint8:
        raise ValueError("逐面颜色需要(N,4)的uint8数组")
    for values in (density, youngs):
        if np.isinf(values).any() or np.any(values[np.isfinite(values)] <= 0):
            raise ValueError("已知物性需要有限正数，未知物性只能用NaN")
    proposed_color = np.asarray(color_rgba)
    if (
        proposed_color.shape != (4,)
        or proposed_color.dtype.kind not in "iu"
        or np.any((proposed_color < 0) | (proposed_color > 255))
    ):
        raise ValueError("新颜色需要4个0至255的整数")
    parts = tokens.get("parts", [])
    ids = [p.get("part_id") for p in parts]
    indices = [p.get("part_index") for p in parts]
    if (
        not parts
        or any(not isinstance(i, str) or not i.strip() for i in ids)
        or len(set(ids)) != len(ids)
        or any(not isinstance(i, int) or isinstance(i, bool) or i <= 0 for i in indices)
        or len(set(indices)) != len(indices)
        or not set(np.unique(labels)).issubset({0, *indices})
    ):
        raise ValueError("部件ID/索引必须唯一、合法且覆盖标签")
    if part_id not in ids:
        raise ValueError(f"未知编辑目标：{part_id}")
    material_values = [material.density_kg_m3.mean, material.youngs_modulus_pa.mean]
    if not np.isfinite(material_values).all() or min(material_values) <= 0:
        raise ValueError("目标材料的密度与杨氏模量先验必须为有限正数")
    index = ids.index(part_id)
    changed = labels == indices[index]
    if not changed.any():
        raise ValueError("编辑目标没有已观测的Mesh面")
    updated_density, updated_youngs, updated_colors = (
        density.copy(),
        youngs.copy(),
        colors.copy(),
    )
    updated_density[changed] = material.density_kg_m3.mean
    updated_youngs[changed] = material.youngs_modulus_pa.mean
    updated_colors[changed] = proposed_color.astype(np.uint8)
    updated_tokens = deepcopy(tokens)
    updated_tokens["parts"][index].update(
        surface_material=material.name,
        material_assignment_source="explicit_edit_configuration",
        density_prior_kg_m3=material.density_kg_m3.mean,
        youngs_prior_pa=material.youngs_modulus_pa.mean,
    )
    return SurfaceEditResult(
        changed, updated_density, updated_youngs, updated_colors, updated_tokens
    )
