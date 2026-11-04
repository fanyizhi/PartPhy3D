"""保留GLB/Scene节点实例与变换，避免把家具部件堆到原点。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def load_triangle_mesh(path: str | Path, process: bool = True) -> trimesh.Trimesh:
    """加载共同坐标系三角Mesh，包含同一几何的全部场景实例。

    Args:
        path: Mesh或场景文件。
        process: 是否允许上游加载器合并重复顶点。

    Returns:
        施加场景节点变换后的组合三角Mesh，仍使用原文件长度单位。
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Mesh文件不存在：{path}")
    loaded = trimesh.load(path, force=None, process=process)
    if isinstance(loaded, trimesh.Scene):
        instances = []
        for node in loaded.graph.nodes_geometry:
            transform, name = loaded.graph[node]
            geometry = loaded.geometry[name]
            if not isinstance(geometry, trimesh.Trimesh):
                raise TypeError("场景含非三角几何，请显式选择需要的表面")
            instance = geometry.copy()
            instance.apply_transform(transform)
            instances.append(instance)
        if not instances:
            raise ValueError("场景没有三角几何实例")
        loaded = trimesh.util.concatenate(instances)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise ValueError("输入不是含面的三角Mesh，点云不能直接计算封闭体积")
    if not np.isfinite(loaded.vertices).all():
        raise ValueError("Mesh坐标包含NaN或无穷")
    return loaded
