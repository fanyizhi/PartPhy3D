"""将固定五物体的已完成重建结果绑定为哈希清单，拒绝混入参考Mesh。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.physics.part_io import file_sha256, write_json


def build(runs: Path, output: Path, limit: int) -> None:
    """只接受带重建协议和完成报告的camera_space.ply，不从任意文件猜来源。"""
    if not 1 <= limit <= 5:
        raise ValueError("固定先导队列需要1到5个物体")
    ids = json.loads((ROOT / "configs/five_object_pilot.json").read_text())[
        "object_ids"
    ]
    manifest = {}
    for scene in ids[:limit]:
        directory = runs / scene
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        if (
            report.get("status") != "complete"
            or protocol.get("source_reference_mesh_read") is not False
            or protocol.get("mass_labels_read") is not False
            or protocol.get("backend")
            != "frozen_InstantMesh_observed_calibrated_views_diagnostic"
        ):
            raise ValueError(f"{scene}不是通过指定图像重建协议产生的Mesh")
        mesh = (directory / "camera_space.ply").resolve()
        digest = file_sha256(mesh)
        if digest != report["mesh_sha256"]:
            raise ValueError(f"{scene}生成结果哈希错误")
        manifest[scene] = {"path": str(mesh), "sha256": digest}
    if output.exists():
        raise FileExistsError("拒绝覆盖既有Mesh绑定")
    write_json(output, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    build(args.runs, args.output, args.limit)


if __name__ == "__main__":
    main()
