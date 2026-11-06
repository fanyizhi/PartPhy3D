"""用固定三视角alpha轮廓核验ABO参考Mesh坐标，不读取称重标签。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.experiments.five_object_pilot import read_cohort
from tactile3d.mesh.frames import mesh_in_camera_frame
from tactile3d.mesh.io import load_triangle_mesh
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.silhouette import iou, silhouette

RESOLUTION = 256
VIEWS = (0, 10, 20)
COLORS = ("#C96A6A", "#5A76A8")


def plot_audit(examples: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(
        1, len(examples), figsize=(15, 3.3), squeeze=False, layout="constrained"
    )
    for index, (axis, example) in enumerate(zip(axes[0], examples, strict=True), 1):
        axis.imshow(example["image"])
        for key, color in zip(("before", "after"), COLORS, strict=True):
            axis.contour(
                example[key].astype(float), levels=[0.5], colors=[color], linewidths=1.0
            )
        axis.set(xlabel="u (pixel)", ylabel="v (pixel)", title=f"Object {index:02d}")
    figure.legend(
        handles=[
            Line2D([0], [0], color=c, label=n)
            for c, n in zip(
                COLORS, ("Unconverted axes", "Declared glTF → camera axes"), strict=True
            )
        ],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.10),
        frameon=False,
    )
    for suffix in ("svg", "png"):
        figure.savefig(
            output / f"reference_camera_alignment.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def audit(root: Path, output: Path) -> dict[str, Any]:
    ids = read_cohort(Path("configs/five_object_pilot.json"))["object_ids"]
    output.mkdir(parents=True, exist_ok=False)
    rows, examples = [], []
    for scene_id in ids:
        camera_path = root / "scenes" / scene_id / "transforms.json"
        camera = json.loads(camera_path.read_text())
        mesh_path = root / "meshes_reference" / scene_id / "model.glb"
        mesh = load_triangle_mesh(mesh_path)
        aligned, report = mesh_in_camera_frame(mesh, "gltf_y_up")
        for view in VIEWS:
            frame = next(
                f
                for f in camera["frames"]
                if Path(f["file_path"]).stem.endswith(f"_{view:02d}")
            )
            pose = np.asarray(frame["transform_matrix"])
            path = root / "scenes" / scene_id / frame["file_path"]
            with Image.open(path) as original:
                if "A" not in original.getbands():
                    raise ValueError(f"缺少原始alpha通道：{path}")
                image = original.resize((RESOLUTION, RESOLUTION))
                target = np.asarray(image.getchannel("A")) >= 128
                image = np.asarray(image).copy()
            before = silhouette(mesh.vertices, mesh.faces, pose, camera, RESOLUTION)
            after = silhouette(
                aligned.vertices, aligned.faces, pose, camera, RESOLUTION
            )
            row = {
                "scene_id": scene_id,
                "view": view,
                "iou_unconverted": iou(before, target),
                "iou_declared_frame": iou(after, target),
                "image_sha256": file_sha256(path),
                "mesh_sha256": file_sha256(mesh_path),
                "camera_sha256": file_sha256(camera_path),
            }
            rows.append(row)
            if view == 0:
                examples.append({"image": image, "before": before, "after": after})
        write_json(output / f"{scene_id}_transform.json", report)
    result = {
        "scope": "reference_geometry_camera_coordinate_audit",
        "ground_truth_mass_read": False,
        "reconstruction_executed": False,
        "views_per_object": len(VIEWS),
        "resolution": RESOLUTION,
        "rows": rows,
        "mean_iou_unconverted": float(np.mean([r["iou_unconverted"] for r in rows])),
        "mean_iou_declared_frame": float(
            np.mean([r["iou_declared_frame"] for r in rows])
        ),
    }
    write_json(output / "report.json", result)
    with (output / "alignment.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_audit(examples, output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/abo500_five"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
