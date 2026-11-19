"""Verify the material vocabulary on local mask crops from the same input views."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json
from replay_abo50_material_coverage import assign
from run_abo50_density_maps import CASES
from run_nerf_cached_baseline import CLIP_SHA256

MATERIAL_TEXT = {
    "wood": "wood",
    "steel": "steel",
    "aluminum": "aluminum",
    "plastic": "plastic",
    "glass": "glass",
    "ceramic": "ceramic",
    "rubber": "rubber",
    "foam": "foam",
    "leather": "leather",
    "fabric": "woven fabric",
    "stone": "stone",
}
VIEWS = (0, 10, 20)
MIN_MARGIN = 0.015


def crop_part(rgb: np.ndarray, mask: np.ndarray) -> Image.Image | None:
    y, x = np.nonzero(mask)
    if len(x) < 64:
        return None
    padding = max(2, round(max(np.ptp(x), np.ptp(y)) * 0.08))
    crop = np.s_[
        max(0, y.min() - padding) : y.max() + padding + 1,
        max(0, x.min() - padding) : x.max() + padding + 1,
    ]
    isolated = np.where(mask[crop][..., None], rgb[crop], 127)
    image = Image.fromarray(isolated)
    canvas = Image.new("RGB", (max(image.size),) * 2, (127,) * 3)
    canvas.paste(image, tuple((a - b) // 2 for a, b in zip(canvas.size, image.size)))
    return canvas


def main() -> None:
    import open_clip
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("observations", "source", "clip", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.clip) != CLIP_SHA256:
        raise ValueError("Expected the pinned OpenCLIP model")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.clip)
    )
    model = model.cuda().eval()
    tokenize = open_clip.get_tokenizer("ViT-B-16")
    names = list(MATERIAL_TEXT)
    prompts = [
        f"a close-up photograph of a surface made of {name}"
        for name in MATERIAL_TEXT.values()
    ]
    with torch.inference_mode():
        text = torch.nn.functional.normalize(
            model.encode_text(tokenize(prompts).cuda()), dim=-1
        )

    args.output.mkdir()
    meshes = read_json(args.source / "meshes.json")
    write_json(args.output / "meshes.json", meshes)
    original = read_json(args.source / "density_manifest.json")
    results = []
    for scene, record in zip(CASES, original["scenes"], strict=True):
        field = args.source / "surface" / "fields" / scene
        tokens = read_json(field / "part_tokens.json")
        mesh_path = Path(meshes[scene]["path"])
        if digest(mesh_path) != tokens["mesh_sha256"]:
            raise ValueError("Mesh used for part masks changed")
        mesh = trimesh.load_mesh(mesh_path, process=False)
        crops, ids = [], []
        for view in VIEWS:
            with np.load(field / f"view_{view:02d}.npz") as view_data:
                masks = view_data["masks"]
            path = (
                args.observations
                / "scenes"
                / scene
                / "images"
                / f"{scene}_{view:02d}.png"
            )
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB").resize(masks.shape[1:][::-1]))
            for part, mask in zip(tokens["parts"], masks, strict=True):
                crop = crop_part(rgb, mask)
                if crop is not None:
                    crops.append(preprocess(crop))
                    ids.append((int(part["part_index"]), view))
        audits = {p["part_index"]: [] for p in tokens["parts"]}
        if crops:
            with torch.inference_mode():
                vectors = torch.nn.functional.normalize(
                    model.encode_image(torch.stack(crops).cuda()), dim=-1
                )
                values, positions = (vectors @ text.T).topk(2, dim=1)
            for (index, view), value, position in zip(
                ids, values.cpu().tolist(), positions.cpu().tolist(), strict=True
            ):
                audits[index].append(
                    {
                        "view": view,
                        "choice": names[position[0]],
                        "margin": value[0] - value[1],
                    }
                )
        changes = []
        for part in tokens["parts"]:
            old = part["surface_material"]
            votes = Counter(
                row["choice"]
                for row in audits[part["part_index"]]
                if row["margin"] >= MIN_MARGIN
            )
            best = votes.most_common(1)
            if best and best[0][1] >= 2 and best[0][0] != old:
                part["surface_material"] = best[0][0]
                changes.append(
                    {"part": part["name"], "before": old, "after": best[0][0]}
                )
        with np.load(field / "surface_fields.npz") as stored:
            materials, _, labels = assign(mesh, tokens, stored["votes"])
        with np.load(args.source / f"{scene}.npz") as stored:
            ids_image = stored["face_ids"]
            foreground = stored["foreground"]
        visible = foreground & (ids_image >= 0)
        observed = labels[ids_image[visible]]
        results.append(
            {
                **record,
                "coverage_after": float(np.count_nonzero(observed) / foreground.sum()),
                "changes": changes,
                "crop_predictions": audits,
            }
        )
        child = args.output / "surface" / "fields" / scene
        child.mkdir(parents=True)
        write_json(
            child / "part_tokens.json",
            {
                **tokens,
                "parts": [
                    {"part_index": i + 1, "surface_material": name}
                    for i, name in enumerate(materials)
                ],
            },
        )
        np.savez_compressed(child / "surface_fields.npz", labels=labels)
        print(scene, changes, flush=True)
    write_json(
        args.output / "report.json",
        {
            "views": VIEWS,
            "margin": MIN_MARGIN,
            "minimum_agreeing_views": 2,
            "clip_sha256": CLIP_SHA256,
            "scenes": results,
        },
    )


if __name__ == "__main__":
    main()
