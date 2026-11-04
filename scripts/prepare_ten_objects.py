"""Fetch only the ten fixed test assets; never read their target masses."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import shutil
import time
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

from tactile3d.experiments.single_object import digest
from tactile3d.utils.io import ensure_dir, read_json, write_json


def prepare(root, assets, output):
    selection = read_json(root/"splits.json")["test"][:10]
    manifest = {"selection": "first 10 published test IDs, no replacements",
                "mass_labels_read": False, "known_development_object": selection[0],
                "split_sha256": digest(root/"splits.json"), "object_ids": selection}
    write_json(output/"selection_before_evaluation.json", manifest)

    def fetch(scene_id):
        asin = scene_id.split("_")[0]
        folder = ensure_dir(assets/scene_id)
        target = folder/"model.glb"
        url = f"https://amazon-berkeley-objects.s3.amazonaws.com/3dmodels/original/{asin[-1]}/{asin}.glb"
        old = Path("data/abo500_single")/scene_id/"model.glb"
        started = time.perf_counter()
        try:
            if not target.exists():
                if old.exists():
                    shutil.copyfile(old, target)
                else:
                    request = urllib.request.Request(url, headers={"User-Agent": "PartPhy3D research"})
                    with (urllib.request.urlopen(request, timeout=25) as response,
                          target.with_suffix(".partial").open("wb") as stream):
                        shutil.copyfileobj(response, stream, length=1024*1024)
                    target.with_suffix(".partial").replace(target)
            with target.open("rb") as stream:
                if stream.read(4) != b"glTF":
                    raise ValueError("Downloaded asset is not GLB")
            return {"scene_id": scene_id, "status": "available", "url": url,
                    "mesh_path": str(target.resolve()), "bytes": target.stat().st_size,
                    "sha256": digest(target), "seconds": time.perf_counter()-started}
        except Exception as exc:  # noqa: BLE001 - retain per-asset download failures
            return {"scene_id": scene_id, "status": "download_failed", "url": url,
                    "error": str(exc), "seconds": time.perf_counter()-started}

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        rows = []
        for row in executor.map(fetch, selection):
            rows.append(row)
            print(json.dumps(row), flush=True)
    write_json(output/"asset_downloads.json", rows)
    canvas = Image.new("RGB", (1500, 1000), "white")
    draw = ImageDraw.Draw(canvas)
    for index, scene_id in enumerate(selection):
        scene = root/"scenes"/scene_id
        info = read_json(scene/"info.json")
        frames = read_json(scene/"transforms.json")["frames"]
        view = int(info["idx_to_caption"])
        image = Image.open(scene/frames[view]["file_path"]).convert("RGBA")
        white = Image.new("RGBA", image.size, "white")
        white.alpha_composite(image)
        alpha_box = image.getchannel("A").getbbox()
        cropped = white.crop(alpha_box).convert("RGB")
        cropped.thumbnail((280, 370))
        x, y = (index % 5)*300, (index // 5)*500
        canvas.paste(cropped, (x+(300-cropped.width)//2, y+45+(370-cropped.height)//2))
        draw.text((x+8, y+8), f"{index+1:02d} {scene_id[:10]}", fill="black")
        caption = info["caption"]
        draw.text((x+8, y+430), caption[:42], fill="black")
        draw.text((x+8, y+449), caption[42:84], fill="black")
    canvas.save(output/"selected_objects.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("data/abo500_ten"))
    parser.add_argument("--output", type=Path, default=Path("outputs/abo500_ten"))
    args = parser.parse_args()
    prepare(args.root, args.assets, ensure_dir(args.output))
