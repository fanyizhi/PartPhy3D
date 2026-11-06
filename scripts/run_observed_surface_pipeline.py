"""串行执行真实部件定位、分割、冲突拒绝和编辑；不声称完成内部物理估计。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.physics.part_io import file_sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """逐步留存命令、日志与失败，不覆盖结果，不读取质量标签。

    Args:
        args: 模型、已生成Mesh、材料假设和观察图像的位置。

    Returns:
        每步状态与证据范围；完整真实物理方法状态始终独立记录。
    """
    for name in ("root", "predictions", "model"):
        path = getattr(args, name).resolve()
        if not path.is_dir():
            raise ValueError(f"{name}目录不存在：{path}")
        setattr(args, name, path)
    for name in ("sam_checkpoint", "clip_checkpoint", "mesh_manifest"):
        path = getattr(args, name).resolve()
        if not path.is_file():
            raise ValueError(f"{name}文件不存在：{path}")
        setattr(args, name, path)
    ids = json.loads(
        (ROOT / "configs/five_object_pilot.json").read_text(encoding="utf-8")
    )["object_ids"][: args.limit]
    meshes = json.loads(args.mesh_manifest.read_text(encoding="utf-8"))
    prediction_hashes = {}
    for scene in ids:
        mesh = Path(meshes[scene]["path"])
        if not mesh.is_file() or file_sha256(mesh) != meshes[scene]["sha256"]:
            raise ValueError(f"生成Mesh清单校验失败：{scene}")
        path = args.predictions / "objects" / scene / "prediction.json"
        if not path.is_file():
            raise ValueError(f"缺少先前材料假设：{scene}")
        prediction_hashes[scene] = file_sha256(path)
    if args.edit_spec is not None:
        args.edit_spec = args.edit_spec.resolve()
        if not args.edit_spec.is_file():
            raise ValueError("编辑规格文件不存在")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "scope": "observed_surface_pipeline_not_complete_physical_inference",
        "objects": ids,
        "mesh_manifest_sha256": file_sha256(args.mesh_manifest),
        "prediction_sha256": prediction_hashes,
        "steps": [],
        "mass_integration_executed": False,
        "physics_accuracy_established": False,
        "status": "running",
    }
    common = [
        "--root",
        args.root,
        "--predictions",
        args.predictions,
        "--output",
        output / "parts",
        "--limit",
        str(args.limit),
    ]
    steps = [
        (
            "locate",
            "run_part_surface_maps.py",
            [
                "--phase",
                "locate",
                "--model",
                args.model,
                *common,
            ],
        ),
        (
            "segment",
            "run_part_surface_maps.py",
            [
                "--phase",
                "segment",
                "--sam-checkpoint",
                args.sam_checkpoint,
                "--mesh-manifest",
                args.mesh_manifest,
                *common,
            ],
        ),
        (
            "observe",
            "observe_part_points.py",
            [
                "--locations",
                output / "parts/locations",
                "--model",
                args.model,
                "--root",
                args.root,
                "--output",
                output / "point_materials",
            ],
        ),
        (
            "refine",
            "refine_surface_conflicts.py",
            [
                "--source",
                output / "parts/fields",
                "--root",
                args.root,
                "--checkpoint",
                args.clip_checkpoint,
                "--point-evidence",
                output / "point_materials",
                "--output",
                output / "refined_fields",
            ],
        ),
        (
            "render_edit",
            "render_surface_maps.py",
            [
                "--fields",
                output / "refined_fields",
                "--root",
                args.root,
                "--output",
                output / "figures",
                *(["--edit-spec", args.edit_spec] if args.edit_spec else []),
            ],
        ),
    ]
    try:
        for name, script, options in steps:
            command = [sys.executable, str(ROOT / "scripts" / script)]
            command.extend(str(value) for value in options)
            step = {"name": name, "command": command, "status": "running"}
            report["steps"].append(step)
            write_json(output / "pipeline.json", report)
            start = time.perf_counter()
            with (output / f"{name}.log").open("wb") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    shell=False,
                    check=False,
                )
            step.update(
                seconds=time.perf_counter() - start, returncode=completed.returncode
            )
            if completed.returncode:
                step["status"] = "failed"
                raise RuntimeError(f"{name}失败；请检查{output / (name + '.log')}")
            step["status"] = "complete"
            print(f"COMPLETED {name}", flush=True)
        report["status"] = "complete"
    except (OSError, RuntimeError) as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write_json(output / "pipeline.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/abo500_five")
    parser.add_argument("--limit", type=int, default=1, choices=range(1, 6))
    for name in (
        "predictions",
        "model",
        "sam-checkpoint",
        "clip-checkpoint",
        "mesh-manifest",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--edit-spec", type=Path)
    args = parser.parse_args()
    try:
        result = run_pipeline(args)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"流水线失败：{error}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
