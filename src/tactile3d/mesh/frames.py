"""显式区分glTF资产坐标与ABO渲染相机坐标，旋转不改变长度尺度。"""

from __future__ import annotations

import numpy as np
import trimesh

GLTF_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
GLTF_TO_Z_UP.setflags(write=False)


def mesh_in_camera_frame(
    mesh: trimesh.Trimesh, source_frame: str = "camera"
) -> tuple[trimesh.Trimesh, dict[str, object]]:
    """把声明的资产坐标转换到ABO的Z向上相机世界坐标。

    Args:
        mesh: 有限顶点的三角Mesh，不要求闭合。
        source_frame: camera表示已对齐；gltf_y_up表示原生ABO glTF轴约定。

    Returns:
        不修改原对象的变换后Mesh，以及可审计的坐标变换记录。
    """
    if source_frame not in {"camera", "gltf_y_up"}:
        raise ValueError("坐标约定只能是camera或gltf_y_up，不能自动猜测")
    if not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise ValueError("坐标转换需要有限坐标的非空三角Mesh")
    transform = GLTF_TO_Z_UP if source_frame == "gltf_y_up" else np.eye(4)
    result = mesh.copy()
    result.apply_transform(transform)
    return result, {
        "source_coordinate_frame": source_frame,
        "target_coordinate_frame": "ABO_camera_world_z_up",
        "mesh_to_camera_world": transform.tolist(),
        "length_scale_changed": False,
        "transform_source": "declared_coordinate_convention_not_fitted_to_mass",
    }
