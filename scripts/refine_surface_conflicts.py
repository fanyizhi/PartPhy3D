"""在固定SAM缓存上检验材料语义冲突，保留未知，不改原始预测。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.part_lifting import add_view_votes, finalize_votes, mask_labels

SEMANTIC_GAP = 0.025
DUPLICATE_IOU = 0.97
DUPLICATE_GAP = 0.015
SPECIFICITY_POWER = 0.25


def refine(
    source: Path,
    root: Path,
    checkpoint: Path,
    output: Path,
    cached_scores: Path | None = None,
    point_evidence: Path | None = None,
) -> None:
    """用点位观察或CLIP拒绝明显矛盾；两者均不是真实材料标注。"""
    import open_clip
    import torch

    if cached_scores is None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=str(checkpoint)
        )
        model = model.to("cuda").eval()
        tokenizer = open_clip.get_tokenizer("ViT-B-16")
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source / "report.json").read_text(encoding="utf-8"))
    reports = []
    for obj in summary["objects"]:
        scene = obj["scene_id"]
        previous = source / scene
        dest = output / scene
        dest.mkdir()
        token = json.loads((previous / "part_tokens.json").read_text(encoding="utf-8"))
        parts = token["parts"]
        names = sorted(token["material_priors"])
        if cached_scores is None:
            with torch.inference_mode(), torch.autocast("cuda"):
                text = model.encode_text(
                    tokenizer(
                        [
                            "a close-up image of a surface made of "
                            + n.replace("_", " ")
                            for n in names
                        ]
                    ).to("cuda")
                )
                text = text / text.norm(dim=-1, keepdim=True)
        if file_sha256(Path(token["mesh"])) != token["mesh_sha256"]:
            raise ValueError("缓存Mesh已改变")
        mesh = trimesh.load_mesh(token["mesh"], process=False)
        candidates = sorted(
            {
                names.index(p["surface_material"])
                for p in parts
                if p["surface_material"] in names
            }
        )
        votes = np.zeros((len(mesh.faces), len(parts) + 1))
        audits = []
        camera = json.loads(
            (root / "scenes" / scene / "transforms.json").read_text(encoding="utf-8")
        )
        point_views = {}
        if point_evidence is not None:
            observed = json.loads(
                (point_evidence / scene / "point_evidence.json").read_text(
                    encoding="utf-8"
                )
            )
            point_views = {
                v["view"]: {p["part_id"]: p["observed_material"] for p in v["parts"]}
                for v in observed["views"]
            }
            if set(point_views) != {0, 10, 20}:
                raise ValueError("可见点材料观测尚未完成")
        for view in (0, 10, 20):
            with np.load(previous / f"view_{view:02d}.npz") as cached:
                masks = cached["masks"].copy()
                quality = cached["quality"].copy()
            with Image.open(
                root / "scenes" / scene / "images" / f"{scene}_{view:02d}.png"
            ) as image:
                rgba = image.convert("RGBA").resize((masks.shape[2], masks.shape[1]))
                rgb = Image.alpha_composite(
                    Image.new("RGBA", rgba.size, "white"), rgba
                ).convert("RGB")
                foreground = np.asarray(rgba.getchannel("A")) >= 128
            cached_ids = None
            if cached_scores is not None:
                with np.load(cached_scores / scene / f"view_{view:02d}.npz") as old:
                    if not np.array_equal(old["masks"], masks):
                        raise ValueError("缓存语义特征与当前掩码不一致")
                    scores = old["clip_scores"].copy()
                    cached_ids = old["face_ids"].copy()
            else:
                crops = []
                for mask in masks:
                    y, x = np.where(mask)
                    crops.append(
                        preprocess(
                            rgb.crop((x.min(), y.min(), x.max() + 1, y.max() + 1))
                        )
                        if len(x)
                        else preprocess(Image.new("RGB", (32, 32), "white"))
                    )
                with torch.inference_mode(), torch.autocast("cuda"):
                    encoded = model.encode_image(torch.stack(crops).to("cuda"))
                    encoded = encoded / encoded.norm(dim=-1, keepdim=True)
                    scores = (encoded @ text.T).float().cpu().numpy()
            original = quality.copy()
            target_scores = np.full(len(parts), -1.0)
            for i, p in enumerate(parts):
                if point_evidence is not None:
                    if (
                        point_views[view].get(p["part_id"], "unknown")
                        != p["surface_material"]
                    ):
                        quality[i] = 0
                    continue
                if p["surface_material"] in names:
                    target_scores[i] = scores[i, names.index(p["surface_material"])]
                    if (
                        candidates
                        and scores[i, candidates].max() - target_scores[i]
                        > SEMANTIC_GAP
                    ):
                        quality[i] = 0
            duplicates = []
            for i in range(len(parts)):
                for j in range(i + 1, len(parts)):
                    overlap = (masks[i] & masks[j]).sum() / max(
                        (masks[i] | masks[j]).sum(), 1
                    )
                    if (
                        overlap > DUPLICATE_IOU
                        and quality[i] >= 0.65
                        and quality[j] >= 0.65
                    ):
                        duplicates.append([parts[i]["part_id"], parts[j]["part_id"]])
                        if abs(target_scores[i] - target_scores[j]) > DUPLICATE_GAP:
                            quality[i if target_scores[i] < target_scores[j] else j] = 0
                        else:
                            quality[[i, j]] = 0
            labels = mask_labels(
                masks,
                quality,
                foreground,
                specificity_power=SPECIFICITY_POWER,
                prefer_nested=point_evidence is not None,
            )
            frame = next(
                f
                for f in camera["frames"]
                if Path(f["file_path"]).stem.endswith(f"_{view:02d}")
            )
            ids = add_view_votes(
                votes,
                mesh,
                np.asarray(frame["transform_matrix"]),
                camera,
                labels,
                cached_face_ids=cached_ids,
            )
            np.savez_compressed(
                dest / f"view_{view:02d}.npz",
                masks=masks,
                quality=original,
                effective_quality=quality,
                labels=labels,
                face_ids=ids,
                clip_scores=scores,
            )
            audits.append(
                {
                    "view": view,
                    "duplicate_part_pairs": duplicates,
                    "rejected_parts": [
                        parts[i]["part_id"]
                        for i in range(len(parts))
                        if original[i] >= 0.65 and quality[i] == 0
                    ],
                    "clip_material_names": names,
                    "clip_cosines": scores.tolist(),
                }
            )
        labels, agreement, support = finalize_votes(mesh, votes)
        np.savez_compressed(
            dest / "surface_fields.npz",
            labels=labels,
            agreement=agreement,
            observed_views=support,
            votes=votes,
        )
        write_json(dest / "part_tokens.json", token)
        result = {
            "scene_id": scene,
            "labeled_area_fraction": float(
                mesh.area_faces[labels > 0].sum() / mesh.area
            ),
            "source_labeled_area_fraction": obj["labeled_area_fraction"],
            "semantic_checks": audits,
            "segmentation_accuracy_established": False,
        }
        write_json(dest / "report.json", result)
        reports.append(result)
        print("REFINED", scene, flush=True)
    write_json(
        output / "report.json",
        {
            "objects": reports,
            "source_sha256": file_sha256(source / "report.json"),
            "clip_checkpoint_sha256": file_sha256(checkpoint),
            "heuristic_thresholds": {
                "semantic_gap": SEMANTIC_GAP,
                "duplicate_iou": DUPLICATE_IOU,
                "duplicate_gap": DUPLICATE_GAP,
                "specificity_power": SPECIFICITY_POWER,
            },
            "physics_accuracy_established": False,
            "semantic_basis": (
                "visible_point_VLM_observations"
                if point_evidence is not None
                else "declared_part_candidate_materials"
            ),
            "score_cache": str(cached_scores) if cached_scores else None,
            "point_evidence": str(point_evidence) if point_evidence else None,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("data/abo500_five"))
    p.add_argument("--cached-scores", type=Path)
    p.add_argument("--point-evidence", type=Path)
    a = p.parse_args()
    refine(a.source, a.root, a.checkpoint, a.output, a.cached_scores, a.point_evidence)


if __name__ == "__main__":
    main()
