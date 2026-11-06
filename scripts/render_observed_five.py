"""展示固定五对象真实生成Mesh，不把参考GLB用作生成结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.experiments.five_object_pilot import read_cohort
from tactile3d.figures.bound_material import raster_face_ids, shade_face_colors
from tactile3d.physics.part_io import file_sha256, write_json

NAMES = ("Chaise", "USB hub", "Ladder", "Mirror", "Sleeve")
RESOLUTION = 300
AZIMUTH = -55.0


def render_gallery(results: Path, data_root: Path, output: Path) -> dict[str, Any]:
    """并列原图和固定方向生成Mesh；仅展示诊断，不推断材料真值。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ids = read_cohort(Path("configs/five_object_pilot.json"))["object_ids"]
    if len(ids) != len(NAMES):
        raise ValueError("当前图只支持冻结的五物体先导队列")
    if output.exists():
        raise FileExistsError("拒绝覆盖已有画廊")
    output.mkdir(parents=True)
    figure, axes = plt.subplots(
        2, 5, figsize=(13, 6.3), layout="constrained", squeeze=False
    )
    reports = []
    for index, (scene_id, name) in enumerate(zip(ids, NAMES, strict=True)):
        run = (
            results / "observed_instantmesh_gate_20261106_v3"
            if index == 0
            else results / "observed_instantmesh_five_20261106" / scene_id
        )
        evaluation = (
            results / "observed_instantmesh_gate_evaluation_20261106_v1"
            if index == 0
            else run.parent / (scene_id + "_evaluation")
        )
        report = json.loads((run / "report.json").read_text())
        scores = json.loads((evaluation / "report.json").read_text())
        mesh_path = run / "camera_space.ply"
        if file_sha256(mesh_path) != report["mesh_sha256"]:
            raise ValueError("Mesh哈希与真实执行报告不一致")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        surface = SimpleNamespace(vertices_m=mesh.vertices, faces=mesh.faces)
        visible = raster_face_ids(surface, AZIMUTH, RESOLUTION)
        rgba = np.rint(mesh.visual.vertex_colors[mesh.faces].mean(axis=1))
        rendered = shade_face_colors(surface, rgba.astype(np.uint8), visible)
        path = data_root / "scenes" / scene_id / "images" / f"{scene_id}_00.png"
        with Image.open(path) as source:
            rgba_image = source.convert("RGBA").resize((RESOLUTION, RESOLUTION))
            image = Image.alpha_composite(
                Image.new("RGBA", rgba_image.size, "white"), rgba_image
            ).convert("RGB")
            axes[0, index].imshow(image)
        axes[1, index].imshow(rendered)
        Image.fromarray(rendered).save(output / f"{scene_id}_generated.png")
        axes[0, index].set_title(f"{index+1:02d}  {name}")
        axes[1, index].set_title(
            f"{report['faces']:,} faces\nHeld-out IoU {scores['mean_iou_heldout']:.3f}",
            fontsize=9,
        )
        reports.append(
            {
                "scene_id": scene_id,
                "mesh_sha256": report["mesh_sha256"],
                "faces": report["faces"],
                "is_watertight": report["is_watertight"],
                "mean_iou_heldout": scores["mean_iou_heldout"],
                "metric_scale_established": False,
            }
        )
    for row, label in zip(axes, ("Observed RGB", "Generated Mesh"), strict=True):
        for index, axis in enumerate(row):
            axis.set(xlabel="u (pixel)", xticks=(0, 150, 299), yticks=(0, 150, 299))
            axis.set_ylabel(f"{label}\nv (pixel)" if index == 0 else "v (pixel)")
            axis.tick_params(labelsize=7)
    figure.suptitle(
        "Frozen InstantMesh adapted to six calibrated real views\n"
        "Generated Mesh: common viewing direction and normalized display scale",
        fontsize=11,
    )
    for suffix in ("svg", "png"):
        figure.savefig(
            output / f"observed_five_gallery.{suffix}", dpi=300, bbox_inches="tight"
        )
    plt.close(figure)
    summary = {
        "scope": "five_object_observed_multiview_frontend_diagnostic",
        "our_complete_method_evaluated": False,
        "official_single_image_InstantMesh_pipeline": False,
        "reference_glb_rendered_as_generated_result": False,
        "rows": reports,
        "mean_heldout_silhouette_iou": float(
            np.mean([r["mean_iou_heldout"] for r in reports])
        ),
    }
    write_json(output / "report.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/abo500_five"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(render_gallery(args.results, args.data_root, args.output), indent=2)
    )


if __name__ == "__main__":
    main()
