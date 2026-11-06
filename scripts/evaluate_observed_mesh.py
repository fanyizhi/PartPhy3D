"""以未输入重建器的视角检验轮廓，不读取参考Mesh或质量标签。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.silhouette import iou, silhouette

RESOLUTION = 256


def plot_projection(examples: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        2,
        len(examples),
        figsize=(3.5 * len(examples), 6),
        squeeze=False,
        layout="constrained",
    )
    for index, example in enumerate(examples):
        axes[0, index].imshow(example["image"])
        axes[1, index].imshow(example["image"])
        axes[1, index].contour(
            example["target"], levels=[0.5], colors=["#4FA38F"], linewidths=1
        )
        axes[1, index].contour(
            example["predicted"], levels=[0.5], colors=["#A56F4B"], linewidths=1
        )
        axes[0, index].set_title(f"Held-out view {example['frame_index']:02d}")
        axes[1, index].set_title(f"Silhouette IoU = {example['iou']:.3f}")
    for axis in axes.flat:
        axis.set(xlabel="u (pixel)", ylabel="v (pixel)")
    figure.legend(
        handles=[
            Line2D([0], [0], color="#4FA38F", label="Observed alpha silhouette"),
            Line2D(
                [0],
                [0],
                color="#A56F4B",
                label="Frozen InstantMesh adapted to real views",
            ),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    for suffix in ("svg", "png"):
        figure.savefig(
            output / f"heldout_projection.{suffix}", dpi=300, bbox_inches="tight"
        )
    plt.close(figure)


def evaluate(run: Path, output: Path) -> dict[str, Any]:
    """冻结变换下评估全部视角；不进行参考几何或轮廓驱动的对齐。"""
    protocol = json.loads((run / "protocol.json").read_text())
    reconstruction = json.loads((run / "report.json").read_text())
    camera_path = Path(protocol["camera_file"])
    if file_sha256(camera_path) != protocol["camera_sha256"]:
        raise ValueError("相机文件在重建后发生变化")
    mesh_path = run / "camera_space.ply"
    if file_sha256(mesh_path) != reconstruction["mesh_sha256"]:
        raise ValueError("生成Mesh与重建报告哈希不一致")
    camera = json.loads(camera_path.read_text())
    mesh = trimesh.load_mesh(mesh_path, process=False)
    selected = set(protocol["coordinate_transform"]["selected_frame_indices"])
    heldout = [i for i in range(len(camera["frames"])) if i not in selected]
    if len(heldout) < 3:
        raise ValueError("至少保留三个未输入的视角用于诊断")
    # 展示索引在看分数之前固定，避免挑选最好的三个结果。
    preview_ids = {heldout[int(i)] for i in np.linspace(0, len(heldout) - 1, 3)}
    rows, examples = [], []
    for index, frame in enumerate(camera["frames"]):
        path = (camera_path.parent / frame["file_path"]).resolve()
        if not path.is_relative_to(camera_path.parent.resolve()):
            raise ValueError("相机图像路径越出场景目录")
        with Image.open(path) as source:
            if "A" not in source.getbands():
                raise ValueError("此诊断要求真实alpha，不用模型预测替代")
            image = source.resize((RESOLUTION, RESOLUTION))
            target = np.asarray(image.getchannel("A")) >= 128
            rgba = np.asarray(image).copy()
        predicted = silhouette(
            mesh.vertices,
            mesh.faces,
            np.asarray(frame["transform_matrix"]),
            camera,
            RESOLUTION,
        )
        score = iou(predicted, target)
        rows.append(
            {
                "frame_index": index,
                "used_for_reconstruction": index in selected,
                "silhouette_iou": score,
                "image_sha256": file_sha256(path),
            }
        )
        if index in preview_ids:
            examples.append(
                {
                    "frame_index": index,
                    "image": rgba,
                    "predicted": predicted,
                    "target": target,
                    "iou": score,
                }
            )
    result = {
        "scope": "observed_view_adaptation_geometry_diagnostic",
        "reference_mesh_used": False,
        "mass_labels_used": False,
        "posthoc_alignment": False,
        "reconstruction_protocol_sha256": file_sha256(run / "protocol.json"),
        "resolution": RESOLUTION,
        "input_views": len(selected),
        "heldout_views": len(heldout),
        "mean_iou_input": float(
            np.mean([r["silhouette_iou"] for r in rows if r["used_for_reconstruction"]])
        ),
        "mean_iou_heldout": float(
            np.mean(
                [r["silhouette_iou"] for r in rows if not r["used_for_reconstruction"]]
            )
        ),
        "rows": rows,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "report.json", result)
    with (output / "per_view.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_projection(examples, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.run, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
