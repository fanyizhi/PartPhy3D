from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from tactile3d.utils.io import ensure_dir, write_json


def make_demo_dataset(root: str | Path) -> tuple[Path, Path]:
    root = ensure_dir(root)
    mesh_root = ensure_dir(root.parent / f"{root.name}_meshes")
    colors = [(180, 185, 190), (160, 90, 65), (80, 120, 175), (205, 200, 185), (70, 75, 80)]
    extents = [
        (0.10, 0.08, 0.06),
        (0.12, 0.07, 0.05),
        (0.08, 0.08, 0.10),
        (0.15, 0.05, 0.04),
        (0.09, 0.09, 0.09),
    ]
    weights: dict[str, float] = {}
    split = {"train": [], "val": [], "test": []}
    for index, (color, size) in enumerate(zip(colors, extents)):
        scene_id = f"scene{index:04d}"
        scene_dir = ensure_dir(root / "scenes" / scene_id)
        image_dir = ensure_dir(scene_dir / "images")
        for view_id in range(3):
            image = Image.new("RGB", (96, 96), (245, 245, 245))
            draw = ImageDraw.Draw(image)
            offset = view_id * 2
            draw.rounded_rectangle(
                (20 + offset, 25, 76 - offset, 72),
                radius=8,
                fill=color,
                outline=(35, 35, 35),
                width=2,
            )
            image.save(image_dir / f"image{view_id:04d}.png")
        frames = [
            {
                "file_path": f"images/image{view_id:04d}.png",
                "transform_matrix": np.asarray(
                    [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0 + 0.05 * view_id],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ).tolist(),
            }
            for view_id in range(3)
        ]
        write_json(
            scene_dir / "transforms.json",
            {"w": 96, "h": 96, "fl_x": 80.0, "fl_y": 80.0, "frames": frames},
        )
        mesh = trimesh.creation.box(extents=size)
        mesh.export(mesh_root / f"{scene_id}.ply")
        weights[scene_id] = float(np.prod(size) * (900.0 + 100.0 * index))
        target_split = "train" if index < 3 else "val" if index == 3 else "test"
        split[target_split].append(scene_id)
    write_json(root / "filtered_product_weights.json", weights)
    write_json(root / "splits.json", split)
    return root, mesh_root
