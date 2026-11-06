"""六张真实标定视图直接进入冻结InstantMesh重建器的诊断入口。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.research.observed_cameras import calibrated_observed_cameras

CHECKPOINT_SHA256 = "329f7aae9b583fd7c1b27d0221463db7b808a03c48a1f7aa26649af6a03b91a1"
IMAGE_SIZE = 320
SEED = 17


def load_frozen_reconstructor(
    repository: Path, checkpoint: Path, encoder_config: Path
) -> Any:
    """全部参数必须由校验过的官方权重严格加载，不使用随机编码器。"""
    import torch
    from omegaconf import OmegaConf
    from transformers import ViTConfig, ViTImageProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("InstantMesh诊断需要CUDA GPU")
    if file_sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("InstantMesh权重与冻结的官方文件SHA256不一致")
    sys.path.insert(0, str(repository))
    from src.models.encoder.dino import ViTModel
    from src.models.encoder.dino_wrapper import DinoWrapper
    from src.models.geometry.rep_3d.flexicubes_geometry import FlexiCubesGeometry
    from src.models.lrm_mesh import InstantMesh

    config = OmegaConf.load(repository / "configs/instant-mesh-large.yaml")
    old_builder = DinoWrapper._build_dino

    def local_encoder(model_name: str) -> tuple[Any, Any]:
        if model_name != "facebook/dino-vitb16":
            raise ValueError("未验证的编码器配置")
        vit_config = ViTConfig.from_pretrained(
            str(encoder_config), local_files_only=True
        )
        processor = ViTImageProcessor.from_pretrained(
            str(encoder_config), local_files_only=True
        )
        return ViTModel(vit_config, add_pooling_layer=False), processor

    # 完整检查点含DINO；避免先下载一份马上会被覆盖的预训练权重。
    DinoWrapper._build_dino = staticmethod(local_encoder)
    try:
        model = InstantMesh(**dict(config.model_config.params))
    finally:
        DinoWrapper._build_dino = staticmethod(old_builder)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    prefix = "lrm_generator."
    weights = {
        key[len(prefix) :]: value
        for key, value in state["state_dict"].items()
        if key.startswith(prefix)
    }
    model.load_state_dict(weights, strict=True)
    del state, weights
    model = model.to("cuda").eval()
    # 顶点色导出只查询几何和材质场，不需要创建可微光栅器或编译CUDA插件。
    model.geometry = FlexiCubesGeometry(
        grid_res=model.grid_res,
        scale=model.grid_scale,
        renderer=None,
        render_type="neural_render",
        device="cuda",
    )
    return model


def run_reconstruction(args: argparse.Namespace) -> dict[str, Any]:
    """从真实RGB与相机重建，输出模型坐标和原相机坐标Mesh。

    Args:
        args: 显式指定场景、上游代码、官方检查点、编码器配置与新输出目录。

    Returns:
        本次真实执行的拓扑、运行时间、显存和尺度限制记录。
    """
    import torch
    import trimesh

    scene, output = args.scene.resolve(), args.output.resolve()
    camera_path = scene / "transforms.json"
    for path in (
        camera_path,
        args.repository / "src/models/lrm_mesh.py",
        args.checkpoint,
        args.encoder_config / "config.json",
        args.encoder_config / "preprocessor_config.json",
    ):
        if not path.is_file():
            raise FileNotFoundError(f"缺少重建依赖：{path}")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有实验：{output}")
    camera = json.loads(camera_path.read_text())
    indices, cameras, transform = calibrated_observed_cameras(camera)
    image_paths, images = [], []
    for index in indices:
        path = (scene / camera["frames"][index]["file_path"]).resolve()
        if not path.is_relative_to(scene):
            raise ValueError("观测图像路径必须位于场景目录")
        with Image.open(path) as source:
            if source.size != (int(camera["w"]), int(camera["h"])):
                raise ValueError("原图尺寸与标定不符，不能静默缩放内参")
            rgba = source.convert("RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            rgb = Image.alpha_composite(white, rgba).convert("RGB")
            image = rgb.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BICUBIC)
            images.append(np.asarray(image, dtype=np.float32) / 255)
        image_paths.append(path)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "backend": "frozen_InstantMesh_observed_calibrated_views_diagnostic",
        "official_single_image_pipeline_executed": False,
        "zero123_executed": False,
        "our_adaptive_generation_executed": False,
        "observed_views": 6,
        "camera_file": str(camera_path),
        "camera_sha256": file_sha256(camera_path),
        "input_images": [
            {"path": str(path), "sha256": file_sha256(path)} for path in image_paths
        ],
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "encoder_config_sha256": file_sha256(args.encoder_config / "config.json"),
        "encoder_processor_sha256": file_sha256(
            args.encoder_config / "preprocessor_config.json"
        ),
        "upstream_model_sha256": file_sha256(
            args.repository / "src/models/lrm_mesh.py"
        ),
        "adapter_sha256": file_sha256(Path(__file__)),
        "camera_adapter_sha256": file_sha256(
            PROJECT / "src/tactile3d/research/observed_cameras.py"
        ),
        "coordinate_transform": transform,
        "image_size": IMAGE_SIZE,
        "seed": SEED,
        "source_reference_mesh_read": False,
        "mass_labels_read": False,
        "metric_volume_established": False,
        "differentiable_rasterizer_executed": False,
    }
    write_json(output / "protocol.json", protocol)
    np.save(output / "input_cameras.npy", cameras)
    shutil.copyfile(Path(__file__), output / "adapter_source.py")
    torch.manual_seed(SEED)
    start = time.perf_counter()
    try:
        model = load_frozen_reconstructor(
            args.repository, args.checkpoint, args.encoder_config
        )
        tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
        tensor = tensor.contiguous().unsqueeze(0).to("cuda")
        camera_tensor = torch.from_numpy(cameras).unsqueeze(0).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        inference_start = time.perf_counter()
        with torch.no_grad():
            planes = model.forward_planes(tensor, camera_tensor)
            vertices, faces, colors = model.extract_mesh(planes, use_texture_map=False)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - inference_start
        if len(faces) == 0 or len(vertices) == 0 or not np.isfinite(vertices).all():
            raise ValueError("模型未输出有限的非空三角Mesh")
        normalized = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=colors, process=False
        )
        normalized.export(output / "model_space.ply")
        restored = normalized.copy()
        restored.vertices = vertices / transform[
            "model_units_per_camera_unit"
        ] + np.asarray(transform["center_camera_units"])
        restored.export(output / "camera_space.ply")
        result = {
            "status": "complete",
            "backend": protocol["backend"],
            "vertices": len(vertices),
            "faces": len(faces),
            "is_watertight": bool(restored.is_watertight),
            "is_winding_consistent": bool(restored.is_winding_consistent),
            "is_volume": bool(restored.is_volume),
            "bounds_camera_units": restored.bounds.tolist(),
            "volume_camera_units_cubed": (
                float(restored.volume) if restored.is_volume else None
            ),
            "camera_unit_to_meter": None,
            "inference_seconds": inference_seconds,
            "total_seconds_including_load": time.perf_counter() - start,
            "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()),
            "mesh_sha256": file_sha256(output / "camera_space.ply"),
        }
        write_json(output / "report.json", result)
        return result
    except (OSError, ValueError, RuntimeError, ImportError, KeyError) as error:
        write_json(
            output / "failure.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        (output / "traceback.txt").write_text(traceback.format_exc())
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    result = run_reconstruction(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
