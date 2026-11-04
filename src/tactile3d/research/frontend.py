"""官方InstantMesh命令适配：默认只检查，执行必须显式选择。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from tactile3d.mesh.io import load_triangle_mesh
from tactile3d.physics.part_io import file_sha256, write_json

DEFAULT_CONFIG = "configs/instant-mesh-large.yaml"
DEFAULT_TIMEOUT_SECONDS = 1800


def instantmesh_plan(
    repository: Path,
    python: Path,
    image: Path,
    output: Path,
    unit_to_meter: float,
    scale_source: str,
    config: str = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """验证显式依赖，生成无shell插值的可审查命令。

    Args:
        repository: 用户已安装并审查的官方代码目录。
        python: 该模型环境的Python解释器绝对路径。
        image: 单张真实观测，官方入口生成的其他视角只能算辅助图。
        output: 全新的输出目录。
        unit_to_meter: 外部尺度标定结果，不能读取测试质量反推。
        scale_source: 尺度来源的可追溯说明。
        config: 上游配置相对路径。

    Returns:
        输入哈希、预期产物和完整参数列表；不导入torch或分配GPU。
    """
    repository, python = repository.resolve(), python.resolve()
    image, output = image.resolve(), output.resolve()
    configuration = (repository / config).resolve()
    if not configuration.is_relative_to(repository):
        raise ValueError("模型配置必须位于指定上游仓库内")
    for path in (repository / "run.py", configuration, python, image):
        if not path.is_file():
            raise FileNotFoundError(f"缺少依赖文件：{path}")
    if image.suffix.lower() not in {".png", ".jpg", ".webp"}:
        raise ValueError("官方入口支持png/jpg/webp，请先显式转换图像")
    with Image.open(image) as observed:
        observed.verify()
    if not np.isfinite(unit_to_meter) or unit_to_meter <= 0 or not scale_source.strip():
        raise ValueError("必须声明有限正的米制比例及独立尺度来源")
    if output.exists():
        raise FileExistsError("拒绝覆盖已有重建目录")
    if not configuration.stem.startswith("instant-mesh"):
        raise ValueError("本适配器仅检查InstantMesh配置")
    copied_image = output / ("observed" + image.suffix.lower())
    upstream_output = output / "upstream"
    command = [
        str(python),
        str(repository / "run.py"),
        str(configuration),
        str(copied_image),
        "--output_path",
        str(upstream_output),
        "--seed",
        "17",
        "--export_texmap",
    ]
    expected = upstream_output / configuration.stem / "meshes/observed.obj"
    return {
        "backend": "official_instantmesh_external_process",
        "repository": str(repository),
        "configuration": str(configuration),
        "configuration_sha256": file_sha256(configuration),
        "run_py_sha256": file_sha256(repository / "run.py"),
        "image": str(image),
        "image_sha256": file_sha256(image),
        "copied_image": str(copied_image),
        "output": str(output),
        "expected_mesh": str(expected),
        "command": command,
        "unit_to_meter": unit_to_meter,
        "scale_source": scale_source,
        "observed_views": 1,
        "generated_views_are_observations": False,
        "adaptive_decoder": False,
    }


def execute_instantmesh(
    plan: dict[str, Any],
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """仅启动本次明确请求的前端子进程，保存失败日志，不触碰队列。"""
    if timeout_seconds <= 0:
        raise ValueError("超时必须为正整数")
    output = Path(plan["output"])
    if file_sha256(Path(plan["image"])) != plan["image_sha256"]:
        raise ValueError("图像已在计划生成后变化")
    for field, path_field in (
        ("configuration_sha256", "configuration"),
        ("run_py_sha256", None),
    ):
        path = (
            Path(plan[path_field])
            if path_field
            else Path(plan["repository"]) / "run.py"
        )
        if file_sha256(path) != plan[field]:
            raise ValueError("上游代码或配置在检查后发生变化")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan["image"], plan["copied_image"])
    status = {**plan, "status": "running", "scope": "frontend_only"}
    write_json(output / "frontend.json", status)
    start = time.perf_counter()
    try:
        with (output / "upstream.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                plan["command"],
                cwd=plan["repository"],
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"InstantMesh退出码{completed.returncode}，查看upstream.log"
            )
        expected = Path(plan["expected_mesh"])
        if not expected.is_file():
            raise FileNotFoundError("上游没有产生约定的OBJ，不自动寻找其他Mesh替代")
        mesh = load_triangle_mesh(expected, process=False)
        mesh.apply_scale(plan["unit_to_meter"])
        mesh.export(output / "metric_mesh.glb")
        status.update(
            status="complete",
            mesh_sha256=file_sha256(expected),
            metric_mesh="metric_mesh.glb",
            watertight=bool(mesh.is_watertight),
            note="Non-watertight output requires explicit repair before volume use",
        )
    except Exception as error:
        status.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        status["seconds"] = time.perf_counter() - start
        write_json(output / "frontend.json", status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--unit-to-meter", type=float, required=True)
    parser.add_argument("--scale-source", required=True)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        plan = instantmesh_plan(
            args.repository,
            args.python,
            args.image,
            args.output,
            args.unit_to_meter,
            args.scale_source,
            args.config,
        )
        report = (
            execute_instantmesh(plan)
            if args.execute
            else {**plan, "status": "checked_not_executed"}
        )
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"前端失败：{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
