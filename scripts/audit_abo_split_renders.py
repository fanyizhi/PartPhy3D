"""Check cross-split ABO render similarity using foreground RGB on a common canvas."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist


def descriptor(dataset, scene):
    views, hashes = [], []
    for index in (0, 8, 20):
        path = dataset / "scenes" / scene / "images" / f"{scene}_{index:02d}.png"
        with Image.open(path) as source:
            rgba = source.convert("RGBA")
        hashes.append(hashlib.sha256(rgba.tobytes()).hexdigest())
        alpha = rgba.getchannel("A")
        bounds = alpha.getbbox()
        if bounds is None:
            raise ValueError(f"Empty alpha mask for {scene}")
        rgba = rgba.crop(bounds)
        rgba.thumbnail((32, 32), Image.Resampling.LANCZOS)
        canvas = Image.new("RGBA", (32, 32), "white")
        canvas.alpha_composite(rgba, ((32-rgba.width)//2, (32-rgba.height)//2))
        views.append(np.asarray(canvas.convert("RGB"), dtype=np.float32).ravel()/255)
    return np.concatenate(views), tuple(hashes)


def main(dataset, output):
    splits = json.loads((dataset/"splits.json").read_bytes())
    ordered = sum((splits[k] for k in ("train", "val", "test")), [])
    vectors, digests = zip(*(descriptor(dataset, i) for i in ordered))
    train = splits["train"]
    holdout = splits["val"] + splits["test"]
    features = np.stack(vectors)
    distance = cdist(features[len(train):], features[:len(train)], "euclidean") / np.sqrt(features.shape[1])
    nearest = np.argmin(distance, axis=1)
    rows = []
    for j, scene in enumerate(holdout):
        rows.append(dict(scene_id=scene, split="val" if scene in splits["val"] else "test",
                         train_nearest=train[nearest[j]], rgb_rmse=float(distance[j, nearest[j]]),
                         identical_three_views=digests[len(train)+j] in digests[:len(train)]))
    result = dict(description="Three white-composited 32px foreground crops. Low RGB distance flags candidates for manual product-family review.",
                  n_train=len(train), rows=sorted(rows,key=lambda r:r["rgb_rmse"]))
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    main(args.dataset, args.output)
