"""Matched visual/structural ridge ablation. Hyperparameters see only train folds."""

import hashlib
import json
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from evaluate_structure_cues import ROOT, WORK, errors, read


def visual(ids):
    locations = {}
    original = Path("/root/autodl-tmp/PartPhy3D/outputs/abo50_20261119/clip462")
    manifests = [
        original / "manifest.json",
        *ROOT.parent.glob("abo100_20261120/batch_*/clip_inputs/manifest.json"),
        *ROOT.glob("*_???/clip_inputs/manifest.json"),
    ]
    for manifest in manifests:
        for row in read(manifest):
            locations[row["scene_id"]] = manifest.parent, row["files"]
    checkpoint = Path("/root/autodl-tmp/models/open_clip_pytorch_model.bin")
    with checkpoint.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    assert (
        digest.hexdigest()
        == "80905176486d952914c1deb3b753b8e653ab54aa4ffb14208bc5007cf7643a16"
    )
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(checkpoint)
    )
    model = model.cuda().eval()
    vectors = []
    for start in range(0, len(ids), 12):
        images = []
        for key in ids[start : start + 12]:
            folder, files = locations[key]
            assert len(files) == 3
            for name, expected in sorted(files.items()):
                path = folder / name
                assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
                with Image.open(path) as im:
                    images.append(preprocess(im))
        with torch.inference_mode():
            z = torch.nn.functional.normalize(
                model.encode_image(torch.stack(images).cuda())
            )
            z = torch.nn.functional.normalize(z.reshape(-1, 3, z.shape[-1]).mean(1))
            vectors.extend(z.cpu().numpy())
    return np.asarray(vectors, dtype=float)


def fit_predict(rgb, structure, truth, i, j, weight, alpha):
    # Approximate unit total variance per structural feature block, fit on train only.
    scaler = StandardScaler().fit(structure[i])
    x = np.column_stack(
        [rgb, scaler.transform(structure) * weight / np.sqrt(structure.shape[1])]
    )
    model = Ridge(alpha=alpha, solver="cholesky").fit(x[i], np.log(truth[i]))
    return np.exp(model.predict(x[j]))


def main():
    records = read(WORK / "structural_ablation.json")["records"]
    ids = list(records)
    rgb = visual(ids)
    s = np.array([records[k]["x"] for k in ids])
    gt = np.array([records[k]["gt"] for k in ids])
    parts = {
        split: np.array([i for i, k in enumerate(ids) if records[k]["split"] == split])
        for split in ("train", "val", "test")
    }
    train = parts["train"]
    folds = list(KFold(5, shuffle=True, random_state=25).split(train))
    results = {}
    for name, weights in (("image", [0.0]), ("joint", [0.0, 0.03, 0.1, 0.3, 1.0])):
        trials = []
        for w in weights:
            for a in (0.001, 0.01, 0.1, 1.0, 10.0):
                oof = np.zeros(len(train))
                for left, right in folds:
                    oof[right] = fit_predict(
                        rgb, s, gt, train[left], train[right], w, a
                    )
                trials.append((errors(gt[train], oof).mean(0)[1], w, a, oof))
        score, w, a, oof = min(trials, key=lambda t: t[0])
        pred = fit_predict(rgb, s, gt, train, np.arange(len(ids)), w, a)
        pred[train] = oof
        results[name] = dict(
            weight=w,
            alpha=a,
            cv_alde=score,
            metrics={
                split: errors(gt[ix], pred[ix]).mean(0).tolist()
                for split, ix in parts.items()
            },
            predictions=dict(zip(ids, pred.tolist())),
        )
    intervals = {}
    for split in ("val", "test"):
        ix = parts[split]
        a, b = (
            np.array([results[n]["predictions"][ids[j]] for j in ix])
            for n in ("joint", "image")
        )
        diff = errors(gt[ix], a) - errors(gt[ix], b)
        rng = np.random.default_rng(25)
        intervals[split] = np.quantile(
            diff[rng.integers(0, len(ix), size=(10000, len(ix)))].mean(1),
            [0.025, 0.975],
            axis=0,
        ).tolist()
    (WORK / "matched_models.json").write_text(
        json.dumps(dict(results=results, paired_intervals=intervals), indent=2)
    )
    print(
        {
            n: {k: v for k, v in data.items() if k != "predictions"}
            for n, data in results.items()
        },
        intervals,
    )


if __name__ == "__main__":
    main()
