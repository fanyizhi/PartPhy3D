"""冻结InstantMesh的表面单元权重稀疏解码；不改变网格分辨率或面数预算。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import MethodType
from typing import Any


def decode_surface_weights(
    decoder: Any,
    sampled_features: Any,
    cube_indices: Any,
    stats: list[dict[str, Any]],
) -> tuple[Any, Any, Any]:
    """保留全部SDF/形变查询，仅跳过非零交叉单元的权重MLP。

    Args:
        decoder: 从固定检查点加载且处于eval的原始解码器。
        sampled_features: 原始三平面采样特征，批量必须为1。
        cube_indices: 固定规则网格的八顶点索引。
        stats: 接收本次执行计数的列表，不把计数缩减当作整体加速。

    Returns:
        与原始接口相同的SDF、形变和单元权重。训练和空场景明确拒绝。
    """
    import torch

    if decoder.training or sampled_features.ndim != 4 or sampled_features.shape[0] != 1:
        raise ValueError("稀疏权重诊断仅支持eval模式、批量1的冻结解码器")
    if cube_indices.ndim != 2 or cube_indices.shape[1] != 8:
        raise ValueError("需要八顶点网格单元索引")
    batch, planes, count, channels = sampled_features.shape
    features = sampled_features.permute(0, 2, 1, 3).reshape(
        batch, count, planes * channels
    )
    sdf = decoder.net_sdf(features)
    deformation = decoder.net_deformation(features)
    resolution = round(len(cube_indices) ** (1 / 3))
    full_regular_grid = (
        resolution**3 == len(cube_indices) and (resolution + 1) ** 3 == count
    )
    if full_regular_grid:
        inner = sdf[0, :, 0].reshape([resolution + 1] * 3)[1:-1, 1:-1, 1:-1]
        if not (inner > 0).any() or not (inner < 0).any():
            raise ValueError("原始空场修正会改变SDF，必须使用全量解码")
    signs = sdf[0, :, 0] < 0
    occupied = signs[cube_indices].sum(1)
    active = (occupied > 0) & (occupied < 8)
    if not active.any():
        raise ValueError("无零交叉单元；原始空场修正可能改变活跃集，拒绝稀疏替代")
    selected = cube_indices[active]
    gathered = torch.index_select(features, 1, selected.reshape(-1)).reshape(
        batch, len(selected), 8 * features.shape[-1]
    )
    active_weights = decoder.net_weight(gathered) * 0.1
    weights = active_weights.new_zeros(
        (batch, len(cube_indices), active_weights.shape[-1])
    )
    weights[:, active] = active_weights
    stats.append(
        {
            "sdf_vertices_queried": count,
            "full_weight_cells": len(cube_indices),
            "evaluated_weight_cells": len(selected),
            "weight_cell_fraction": len(selected) / len(cube_indices),
            "geometry_resolution_changed": False,
            "trained": False,
            "original_empty_field_guard_checked": full_regular_grid,
        }
    )
    return sdf, deformation, weights


@contextmanager
def sparse_weight_decoder(model: Any) -> Iterator[list[dict[str, Any]]]:
    """在作用域内替换冻结权重查询，异常时也恢复原方法。"""
    decoder = model.synthesizer.decoder
    original = decoder.get_geometry_prediction
    stats: list[dict[str, Any]] = []

    def sparse(self: Any, features: Any, indices: Any) -> tuple[Any, Any, Any]:
        return decode_surface_weights(self, features, indices, stats)

    decoder.get_geometry_prediction = MethodType(sparse, decoder)
    try:
        yield stats
    finally:
        decoder.get_geometry_prediction = original
