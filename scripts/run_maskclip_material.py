"""Run a 2D MaskCLIP value-feature baseline on frozen ABO figure views."""

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


MATERIAL_NAMES = (
    "wood",
    "metal",
    "fabric",
    "plastic",
    "rubber",
    "foam",
    "leather",
    "electronics",
    "paint",
    "glass",
    "ceramic",
    "stone",
)
TEMPLATES = ("a photo of {} material", "a close-up of a {} surface")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_features(visual, images):
    """Use only the value path of the last ViT attention layer."""
    patches = visual.conv1(images)
    batch, width, height, cols = patches.shape
    tokens = patches.flatten(2).transpose(1, 2)
    cls = visual.class_embedding.expand(batch, 1, width)
    position = visual.positional_embedding
    native = visual.grid_size
    patch_position = position[1:].reshape(*native, width).permute(2, 0, 1)
    patch_position = (
        F.interpolate(
            patch_position[None], (height, cols), mode="bicubic", align_corners=False
        )
        .flatten(2)
        .transpose(1, 2)
    )
    tokens = visual.ln_pre(
        torch.cat((cls, tokens), dim=1)
        + torch.cat((position[None, :1], patch_position), dim=1)
    )
    blocks = visual.transformer.resblocks
    for block in list(blocks)[:-1]:
        tokens = block(tokens)
    last = blocks[-1]
    attn = last.attn
    normalized = last.ln_1(tokens[:, 1:])
    values = F.linear(
        normalized, attn.in_proj_weight[-width:], attn.in_proj_bias[-width:]
    )
    features = visual.ln_post(attn.out_proj(values)) @ visual.proj
    return F.normalize(features, dim=-1).reshape(batch, height, cols, -1)


def observation(data):
    if "rgba" in data:
        rgba = data["rgba"]
        alpha = rgba[..., 3:] / 255.0
        rgb = np.rint(rgba[..., :3] * alpha + 255 * (1 - alpha)).astype(np.uint8)
        return rgb, rgba[..., 3] >= 128
    rgb = data["rgb"]
    if rgb.dtype.kind == "f":
        rgb = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb, data["foreground"]


def predict(model, embeddings, mean, std, rgb, foreground):
    resized = np.asarray(
        Image.fromarray(rgb).resize((448, 448), Image.Resampling.BICUBIC)
    ).copy()
    image = torch.from_numpy(resized).cuda().permute(2, 0, 1)[None] / 255
    features = value_features(model.visual, (image - mean) / std)
    scores = (features @ embeddings.T).permute(0, 3, 1, 2)
    score_map = F.interpolate(
        scores, size=foreground.shape, mode="bilinear", align_corners=False
    )[0]
    labels = (1 + score_map.argmax(0)).cpu().numpy().astype(np.uint8)
    labels[~foreground] = 0
    return labels


def evaluate(root, output, estimator):
    from benchmark_physseg_material_pairs import (
        OBJECTS,
        SIZE,
        aggregate,
        score,
        target_map,
    )

    archive_path = root / "PhysSeg-15.zip"
    cohort = json.loads((root / "observations/cohort.json").read_bytes())
    rows = []
    with ZipFile(archive_path) as archive:
        for scene in OBJECTS:
            paths = {}
            for path in archive.namelist():
                fields = path.split("/")
                if (
                    len(fields) == 5
                    and fields[2] == scene
                    and fields[4] in {"fabric.png", "wood.png"}
                ):
                    paths.setdefault(int(fields[3].rsplit("_", 1)[-1]), {})[
                        Path(fields[4]).stem
                    ] = path
            if len(paths) != 5:
                raise ValueError(f"Expected five PhysSeg views for {scene}")
            camera_file = f"scenes/{scene}/transforms.json"
            root_obs = root / "observations"
            if digest(root_obs / camera_file) != cohort["files_sha256"][camera_file]:
                raise ValueError(f"Camera changed for {scene}")
            camera = json.loads((root_obs / camera_file).read_bytes())
            for view, masks in sorted(paths.items()):
                path = f"scenes/{scene}/{camera['frames'][view]['file_path']}"
                rgb_path = root_obs / path
                image_hash = digest(rgb_path)
                if image_hash != cohort["files_sha256"][path]:
                    raise ValueError(f"Frozen PhysSeg observation changed: {path}")
                with Image.open(rgb_path) as image:
                    rgba = np.array(image.convert("RGBA").resize((SIZE, SIZE)))
                labels = estimator(*observation({"rgba": rgba}))
                cached = output / f"{scene}_{view:02d}.npz"
                if cached.exists():
                    with np.load(cached) as prior:
                        if not np.array_equal(labels, prior["maskclip"]):
                            raise ValueError(
                                "Displayed mask differs from the scored prediction"
                            )
                rows.append(
                    {
                        "scene_id": scene,
                        "view": view,
                        "rgb_sha256": image_hash,
                        "counts": score(
                            target_map(archive, masks, camera), labels, MATERIAL_NAMES
                        ),
                    }
                )
    return {
        "annotation_sha256": digest(archive_path),
        "rows": rows,
        "metrics": aggregate([row["counts"] for row in rows]),
    }


@torch.inference_mode()
def run(args):
    import open_clip

    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(args.checkpoint)
    )
    model = model.cuda().eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    text = tokenizer(
        [
            template.format(material)
            for material in MATERIAL_NAMES
            for template in TEMPLATES
        ]
    ).cuda()
    embeddings = F.normalize(model.encode_text(text), dim=-1)
    embeddings = F.normalize(
        embeddings.reshape(len(MATERIAL_NAMES), len(TEMPLATES), -1).mean(1), dim=-1
    )
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073), device="cuda")[
        None, :, None, None
    ]
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711), device="cuda")[
        None, :, None, None
    ]

    def estimator(rgb, foreground):
        return predict(model, embeddings, mean, std, rgb, foreground)

    args.output.mkdir(exist_ok=False)
    records = []
    for source in args.source:
        for path in sorted(source.glob("*.npz")):
            with np.load(path) as data:
                rgb, foreground = observation(data)
            labels = estimator(rgb, foreground)
            target = args.output / path.name
            if target.exists():
                raise ValueError(f"Duplicate figure file {path.name}")
            np.savez_compressed(target, maskclip=labels, maskclip_names=MATERIAL_NAMES)
            records.append(
                {
                    "file": path.name,
                    "input_sha256": digest(path),
                    "labels_sha256": digest(target),
                }
            )
    (args.output / "manifest.json").write_text(
        json.dumps(
            {
                "method": "MaskCLIP (ViT-B/16, frozen DataComp OpenCLIP weights)",
                "backbone_sha256": digest(args.checkpoint),
                "input_pixels": 448,
                "materials": MATERIAL_NAMES,
                "text_templates": TEMPLATES,
                "output": "foreground argmax, direct 2D without mesh or optional refinements",
                "views": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if args.physseg_eval is not None:
        report = evaluate(args.physseg_eval, args.output, estimator)
        (args.output / "physseg_score.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physseg-eval", type=Path)
    run(parser.parse_args())
