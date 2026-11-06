"""真实VLM定位、SAM2掩码和生成Mesh的可见表面属性对应。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.part_lifting import add_view_votes, finalize_votes, mask_labels
from tactile3d.vision.vlm_interior import _parse_response_content

VIEWS = (0, 10, 20)
RESOLUTION = 384
SAM_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"


def scene_parts(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    ids = json.loads(
        Path("configs/five_object_pilot.json").read_text(encoding="utf-8")
    )["object_ids"]
    return [
        (
            scene,
            json.loads(
                (args.predictions / "objects" / scene / "prediction.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        for scene in ids[: args.limit]
    ]


def locate(args: argparse.Namespace) -> None:
    """固定语义ID后逐视角定位，材料假设不在跨视角过程中偷偷更改。"""
    output = args.output / "locations"
    output.mkdir(parents=True, exist_ok=False)
    model = LocalVLM(
        args.model, max_tokens=1500, max_pixels=RESOLUTION**2, quantized=True
    )
    write_json(output / "model.json", model.provenance)
    for scene, prediction in scene_parts(args):
        directory = output / scene
        directory.mkdir()
        parts = prediction["parts"]
        table = [
            {"part_id": p["part_id"], "name": p.get("name", p["part_id"])}
            for p in parts
        ]
        prompt = (
            "Locate the specified functional parts in this product image. "
            "Do not invent new IDs. Do not use a whole-object box for every part. "
            "For hidden or unresolvable parts, use null for both box and point. "
            "Return JSON {parts: [{part_id: string, box: [x0,y0,x1,y1] or null, "
            "point: [x,y] or null}]}. Coordinates are normalized to [0,1000]. "
            "The point must lie on the visible target material inside its box, "
            "not on an occluder or background. Specified parts: " + json.dumps(table)
        )
        records = []
        for view in VIEWS:
            path = args.root / "scenes" / scene / "images" / f"{scene}_{view:02d}.png"
            with Image.open(path) as image:
                response = model.generate(
                    prompt,
                    [image.copy()],
                    "Localize visible object parts. Return strict JSON only.",
                )
            (directory / f"{view:02d}_raw.txt").write_text(
                response.text, encoding="utf-8"
            )
            rows = _parse_response_content(response.text)["parts"]
            known = {p["part_id"] for p in parts}
            if not isinstance(rows, list) or len({r["part_id"] for r in rows}) != len(
                rows
            ):
                raise ValueError("部件定位列表重复或类型错误")
            for row in rows:
                if row["part_id"] not in known:
                    raise ValueError("VLM擅自新增部件ID")
                box, point = row.get("box"), row.get("point")
                if box is not None:
                    values = np.asarray(box, dtype=float)
                    if (
                        values.shape != (4,)
                        or not np.isfinite(values).all()
                        or np.any((values < 0) | (values > 1000))
                        or values[0] >= values[2]
                        or values[1] >= values[3]
                    ):
                        raise ValueError("无效定位框")
                if point is not None:
                    values = np.asarray(point, dtype=float)
                    if (
                        values.shape != (2,)
                        or not np.isfinite(values).all()
                        or np.any((values < 0) | (values > 1000))
                    ):
                        raise ValueError("无效定位点")
            records.append(
                {"view": view, "image_sha256": file_sha256(path), "parts": rows}
            )
            write_json(
                directory / "localization.json",
                {
                    "parts": parts,
                    "views": records,
                    "materials_source": "cached_VLM_hypotheses_not_ground_truth",
                },
            )
            print(f"LOCATED {scene} {view}", flush=True)


def segment(args: argparse.Namespace) -> None:
    """只给实际观测到的表面分配条件先验；不创建未经证实的闭合体积域。"""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if file_sha256(args.sam_checkpoint) != SAM_SHA256:
        raise ValueError("SAM2权重SHA256错误")
    output = args.output / "fields"
    output.mkdir(parents=True, exist_ok=False)
    sam = SAM2ImagePredictor(
        build_sam2(
            "configs/sam2.1/sam2.1_hiera_s.yaml",
            str(args.sam_checkpoint),
            device="cuda",
            apply_postprocessing=False,
        )
    )
    meshes = json.loads(args.mesh_manifest.read_text(encoding="utf-8"))
    results = []
    for scene, prediction in scene_parts(args):
        directory = output / scene
        directory.mkdir()
        location = json.loads(
            (args.output / "locations" / scene / "localization.json").read_text(
                encoding="utf-8"
            )
        )
        parts = location["parts"]
        mesh_path = Path(meshes[scene]["path"])
        if file_sha256(mesh_path) != meshes[scene]["sha256"]:
            raise ValueError("生成Mesh哈希不符")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        camera = json.loads(
            (args.root / "scenes" / scene / "transforms.json").read_text(
                encoding="utf-8"
            )
        )
        votes = np.zeros((len(mesh.faces), len(parts) + 1))
        summaries = []
        for record in location["views"]:
            view = record["view"]
            path = args.root / "scenes" / scene / "images" / f"{scene}_{view:02d}.png"
            if file_sha256(path) != record["image_sha256"]:
                raise ValueError("观测图像发生变化")
            with Image.open(path) as source:
                rgba = source.convert("RGBA")
                image = np.asarray(
                    Image.alpha_composite(
                        Image.new("RGBA", rgba.size, "white"), rgba
                    ).convert("RGB")
                ).copy()
                foreground = (
                    np.asarray(rgba.getchannel("A").resize((RESOLUTION, RESOLUTION)))
                    >= 128
                )
            masks = np.zeros((len(parts), RESOLUTION, RESOLUTION), dtype=bool)
            quality = np.zeros(len(parts))
            boxes = {p["part_id"]: p for p in record["parts"]}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                sam.set_image(image)
                for index, part in enumerate(parts):
                    row = boxes.get(part["part_id"], {})
                    if row.get("box") is None:
                        continue
                    scale = np.array([image.shape[1], image.shape[0]]) / 1000
                    box = np.asarray(row["box"]) * np.tile(scale, 2)
                    point = (
                        np.asarray(row["point"]) * scale
                        if row.get("point") is not None
                        else None
                    )
                    found, scores, _ = sam.predict(
                        box=box,
                        point_coords=point[None] if point is not None else None,
                        point_labels=np.ones(1) if point is not None else None,
                        multimask_output=True,
                    )
                    best = int(np.argmax(scores))
                    mask = Image.fromarray(found[best].astype(np.uint8) * 255)
                    masks[index] = (
                        np.asarray(
                            mask.resize(
                                (RESOLUTION, RESOLUTION), Image.Resampling.NEAREST
                            )
                        )
                        > 0
                    )
                    quality[index] = float(scores[best])
            labels = mask_labels(masks, quality, foreground)
            frame = next(
                f
                for f in camera["frames"]
                if Path(f["file_path"]).stem.endswith(f"_{view:02d}")
            )
            ids = add_view_votes(
                votes, mesh, np.asarray(frame["transform_matrix"]), camera, labels
            )
            np.savez_compressed(
                directory / f"view_{view:02d}.npz",
                masks=masks,
                quality=quality,
                labels=labels,
                face_ids=ids,
            )
            summaries.append(
                {
                    "view": view,
                    "sam_quality": quality.tolist(),
                    "labeled_foreground_fraction": float(
                        np.count_nonzero(labels) / max(foreground.sum(), 1)
                    ),
                }
            )
        labels, agreement, support = finalize_votes(mesh, votes)
        np.savez_compressed(
            directory / "surface_fields.npz",
            labels=labels,
            agreement=agreement,
            observed_views=support,
            votes=votes,
        )
        tokens = [{"part_index": i + 1, **p} for i, p in enumerate(parts)]
        write_json(
            directory / "part_tokens.json",
            {
                "parts": tokens,
                "material_priors": prediction["material_priors"],
                "physics_source": (
                    "configured_material_priors_conditioned_on_VLM_hypotheses"
                ),
                "closed_material_volumes_established": False,
                "camera_unit_to_meter": None,
                "mesh": str(mesh_path),
                "mesh_sha256": file_sha256(mesh_path),
            },
        )
        area = mesh.area_faces
        result = {
            "scene_id": scene,
            "observed_area_fraction": float(area[support > 0].sum() / area.sum()),
            "labeled_area_fraction": float(area[labels > 0].sum() / area.sum()),
            "views": summaries,
            "segmentation_accuracy_established": False,
        }
        write_json(directory / "report.json", result)
        results.append(result)
        print("SEGMENTED", scene, flush=True)
    write_json(
        output / "report.json",
        {
            "objects": results,
            "ground_truth_physics_used": False,
            "mass_integration_executed": False,
            "sam2_sha256": SAM_SHA256,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("locate", "segment"), required=True)
    parser.add_argument("--root", type=Path, default=Path("data/abo500_five"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--sam-checkpoint", type=Path)
    parser.add_argument("--mesh-manifest", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.limit <= 5:
        parser.error("limit需要1到5")
    if args.phase == "locate":
        if args.model is None or not args.model.is_dir():
            parser.error("locate需要--model指定本地模型目录")
        locate(args)
    else:
        if args.sam_checkpoint is None or not args.sam_checkpoint.is_file():
            parser.error("segment需要--sam-checkpoint指定已下载权重")
        if args.mesh_manifest is None or not args.mesh_manifest.is_file():
            parser.error("segment需要--mesh-manifest指定生成Mesh清单")
        segment(args)


if __name__ == "__main__":
    main()
