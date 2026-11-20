"""Fit train-only image mass and length controls with held-out validation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_abo50_mass_hypotheses import paired_comparison
from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo50_mass_hypotheses import ROOT, VIEWS
from run_component_mass_stability import sealed_rows
from tactile3d.models.calibration import mass_metrics
from tactile3d.models.scale_ridge import LogScaleRidge, fit_log_scale, scale_metrics

ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0)


def convert_observation(original: Path, derived: Path) -> None:
    with Image.open(original) as image:
        rgba = image.convert("RGBA")
        white = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
        white.thumbnail((256, 256), Image.Resampling.LANCZOS)
        white.convert("RGB").save(derived, quality=93, subsampling=0)


def prepare(source: Path, evaluation: list[Path], output: Path) -> None:
    splits = read_json(source / "splits.json")
    heldout = [scene for path in evaluation for scene in read_json(path)["object_ids"]]
    seen = set()
    rows = []
    output.mkdir(parents=True, exist_ok=False)
    for split, scenes in (
        ("train", splits["train"]),
        ("val", splits["val"]),
        ("test", heldout),
    ):
        if split == "test" and not set(scenes) <= set(splits["test"]):
            raise ValueError("Test scenes must be drawn from the official test split")
        products = {scene.split("_")[0] for scene in scenes}
        if len(products) != len(scenes) or products & seen:
            raise ValueError("Duplicate or shared product across splits")
        seen.update(products)
        for scene in scenes:
            files = {}
            for view in VIEWS:
                name = f"{scene}_{view:02d}"
                original = source / "scenes" / scene / "images" / f"{name}.png"
                derived = output / f"{name}.jpg"
                convert_observation(original, derived)
                files[derived.name] = digest(derived)
            rows.append(dict(scene_id=scene, split=split, files=files))
    write_json(output / "manifest.json", rows)
    weights = read_json(source / "filtered_product_weights.json")
    catalog = read_json(ROOT / "configs/evidence/abo500_catalog_dimensions.json")[
        "objects"
    ]
    training = {}
    for row in rows:
        if row["split"] == "test":
            continue
        scene = row["scene_id"]
        training[scene] = dict(
            mass_kg=weights.get(scene, weights.get(scene.split("_")[0])),
            length_m=catalog[scene].get("max_dimension_m"),
        )
    write_json(output / "train_val_labels.json", training)


def predict(
    dataset: Path,
    checkpoint: Path,
    output: Path,
    frozen_head: Path | None = None,
    device: str = "cuda",
) -> None:
    import open_clip
    import torch

    checkpoint_hash = digest(checkpoint)
    if (
        checkpoint_hash
        != read_json(ROOT / "configs/nerf2physics_lock.json")["clip_sha256"]
    ):
        raise ValueError("CLIP checkpoint mismatch")
    manifest = dataset / "manifest.json"
    rows = read_json(manifest)
    labels = read_json(dataset / "train_val_labels.json") if frozen_head is None else {}
    if frozen_head is None and labels.keys() != {
        r["scene_id"] for r in rows if r["split"] in {"train", "val"}
    }:
        raise ValueError("Only train and validation products may supply labels")
    output.mkdir(parents=True, exist_ok=False)
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained=str(checkpoint)
    )
    model = model.to(device).eval()
    features = []
    for start in range(0, len(rows), 12):
        chunk = rows[start : start + 12]
        imgs = []
        for r in chunk:
            for filename, checksum in r["files"].items():
                path = dataset / filename
                if digest(path) != checksum:
                    raise ValueError(f"Image hash mismatch: {filename}")
                with Image.open(path) as image:
                    imgs.append(preprocess(image))
        with torch.inference_mode():
            z = torch.nn.functional.normalize(
                model.encode_image(torch.stack(imgs).to(device))
            )
            z = torch.nn.functional.normalize(
                z.reshape(len(chunk), len(VIEWS), -1).mean(1)
            )
        features.append(z.float().cpu().numpy())
    x = np.concatenate(features)
    outputs = {
        r["scene_id"]: {}
        for r in rows
        if frozen_head is not None or r["split"] == "test"
    }
    protocol = dict(
        manifest_sha256=digest(manifest),
        labels_sha256=(
            digest(dataset / "train_val_labels.json") if frozen_head is None else None
        ),
        checkpoint_sha256=checkpoint_hash,
        n_views=len(VIEWS),
        heads={},
    )
    test = np.array([r["split"] == "test" for r in rows])
    if frozen_head is not None:
        with np.load(frozen_head, allow_pickle=False) as head:
            fit = LogScaleRidge(
                head["feature_mean"],
                head["coefficients"],
                float(head["log_length_mean"]),
                float(head["alpha"]),
            )
        protocol["frozen_mass_head_sha256"] = digest(frozen_head)
        for scene, value in zip(outputs, fit.predict(x), strict=True):
            outputs[scene]["mass_kg"] = float(value)
    for field in ("length_m", "mass_kg") if frozen_head is None else ():
        targets = [labels.get(r["scene_id"], {}).get(field) for r in rows]
        y = np.array([float(t) if t is not None else np.nan for t in targets])
        train = np.array([r["split"] == "train" for r in rows]) & np.isfinite(y)
        val = np.array([r["split"] == "val" for r in rows]) & np.isfinite(y)
        candidates = [fit_log_scale(x[train], y[train], alpha) for alpha in ALPHAS]
        best = min(
            candidates,
            key=lambda f: scale_metrics(y[val], f.predict(x[val]))[
                "mean_absolute_log_error"
            ],
        )
        protocol["heads"][field] = dict(
            alpha=best.alpha,
            train_objects=int(train.sum()),
            val_objects=int(val.sum()),
            validation=scale_metrics(y[val], best.predict(x[val])),
        )
        np.savez_compressed(
            output / f"{field}_head.npz",
            feature_mean=best.feature_mean,
            coefficients=best.coefficients,
            log_length_mean=best.log_length_mean,
            alpha=best.alpha,
        )
        for scene, value in zip(outputs, best.predict(x[test]), strict=True):
            outputs[scene][field] = float(value)
    write_json(output / "protocol.json", protocol)
    result = output / "predictions.json"
    write_json(result, outputs)
    write_json(output / "prediction_seal.json", {"sha256": digest(result)})


def evaluate(args):
    clip_path = args.run / "predictions.json"
    clip = read_sealed_clip(args.run)
    old = sealed_rows(args.prior)
    appliance = sealed_rows(args.appliance) if args.appliance else {}
    winding = sealed_rows(args.winding)
    catalog = sealed_rows(args.catalog_scale)
    nerf_path = args.nerf / "official_predictions.json"
    if (
        digest(nerf_path)
        != read_json(args.nerf / "prediction_seal.json")[nerf_path.name]
    ):
        raise ValueError("NeRF2Physics official cache prediction seal mismatch")
    nerf = read_json(nerf_path)
    weights = read_json(args.weights)
    dimensions = read_json(ROOT / "configs/evidence/abo500_catalog_dimensions.json")[
        "objects"
    ]

    def summarize(items, reference):
        methods = {}
        for name in sorted(items[0].keys() - {"scene_id", "target"}):
            pairs = [(r["target"], r[name], r[reference]) for r in items if name in r]
            if pairs:
                y, x, _ = zip(*pairs, strict=True)
                methods[name] = dict(
                    metrics=mass_metrics(y, x),
                    delta_vs_reference=paired_comparison(pairs),
                )
        return methods

    cohorts = []
    for path in args.evaluation:
        ids = read_json(path)["object_ids"]
        items = []
        for scene in ids:
            target = weights.get(scene, weights.get(scene.split("_")[0]))
            if not target:
                raise ValueError(f"Missing catalogue mass for {scene}")
            item = dict(
                scene_id=scene,
                target=target,
                clip=clip[scene]["mass_kg"],
                nerf=float(np.mean(nerf[scene])),
            )
            if scene in old:
                physical = old[scene]["construction_mass_kg"]
                direct = old[scene]["direct_mass_kg"]
                item.update(
                    physical=physical,
                    direct=direct,
                    physical_clip=(physical * item["clip"]) ** 0.5,
                    direct_clip=(direct * item["clip"]) ** 0.5,
                    physical_direct=(physical * direct) ** 0.5,
                )
                if args.appliance:
                    corrected = appliance.get(scene, {}).get("role_mass_kg", physical)
                    item["physical_role_replay"] = corrected
                    item["physical_role_clip"] = combine_role_mass(
                        corrected, item["clip"]
                    )
            items.append(item)
        ref = "direct" if all("direct" in r for r in items) else "nerf"
        correction = None
        if args.appliance and ref == "direct":
            correction = paired_comparison(
                [(r["target"], r["physical_role_clip"], r["physical"]) for r in items]
            )
        cohorts.append(
            dict(
                n=len(ids),
                reference=ref,
                stats=summarize(items, ref),
                per_scene=items,
                physical_role_clip_vs_previous_physical=correction,
            )
        )

    same = [
        scene
        for scene in winding
        if "construction_mass_kg" in winding[scene]
        and "construction_mass_kg" in catalog[scene]
    ]
    scale_rows = [
        dict(
            scene_id=scene,
            target=weights.get(scene, weights.get(scene.split("_")[0])),
            visual_scale=winding[scene]["construction_mass_kg"],
            catalog_scale=catalog[scene]["construction_mass_kg"],
        )
        for scene in same
    ]
    measured = [
        scene
        for scene in old
        if "predicted_max_length_m" in old[scene]
        and dimensions[scene].get("max_dimension_m") is not None
    ]
    true_length = np.array([dimensions[s]["max_dimension_m"] for s in measured])
    scales = dict(
        n=len(measured),
        clip=scale_metrics(
            true_length, np.array([clip[s]["length_m"] for s in measured])
        ),
        old_vlm=scale_metrics(
            true_length, np.array([old[s]["predicted_max_length_m"] for s in measured])
        ),
    )
    write_json(
        args.output,
        dict(
            cohorts=cohorts,
            prediction_sha256=digest(clip_path),
            same_object_scale_only=dict(
                n=len(same),
                stats=summarize(scale_rows, "visual_scale"),
                per_scene=scale_rows,
            ),
            catalogue_length=scales,
        ),
    )


def read_sealed_clip(run: Path) -> dict:
    source = run / "predictions.json"
    if digest(source) != read_json(run / "prediction_seal.json")["sha256"]:
        raise ValueError("Image head prediction seal mismatch")
    return read_json(source)


def combine_role_mass(physical_kg: float, prior_kg: float) -> float:
    """Equal log-domain weighting keeps the prediction between the two inputs."""
    if not all(math.isfinite(mass) and mass > 0 for mass in (physical_kg, prior_kg)):
        raise ValueError("Mass inputs must be finite and positive")
    return math.exp((math.log(physical_kg) + math.log(prior_kg)) / 2)


def select_abo50(args):
    """Write the selected 50-object replay before opening any evaluation labels."""
    scenes = read_json(args.cohort)["object_ids"]
    if len(scenes) != 50 or len(set(scenes)) != 50:
        raise ValueError("The selected replay requires 50 distinct frozen objects")
    prior = sealed_rows(args.prior)
    clip = read_sealed_clip(args.run)
    appliance = sealed_rows(args.appliance)
    if not set(scenes) <= (prior.keys() & clip.keys()) or not appliance.keys() <= set(
        scenes
    ):
        raise ValueError("Prediction sources do not cover the requested cohort")
    rows = []
    for scene in scenes:
        corrected = appliance.get(scene, {}).get("role_mass_kg")
        volume_mass = (
            prior[scene]["construction_mass_kg"] if corrected is None else corrected
        )
        prior_mass = clip[scene]["mass_kg"]
        rows.append(
            dict(
                scene_id=scene,
                mass_kg=combine_role_mass(volume_mass, prior_mass),
                construction_mass_kg=volume_mass,
                image_prior_kg=prior_mass,
                appliance_replay_used=corrected is not None,
            )
        )
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        dict(
            cohort_sha256=digest(args.cohort),
            sources={
                name: digest(getattr(args, name) / "predictions.json")
                for name in ("run", "prior", "appliance")
            },
            scope="Exploratory selected fusion replay, no evaluation labels loaded",
        ),
    )
    output = args.output / "predictions.json"
    write_json(output, rows)
    write_json(args.output / "prediction_seal.json", {"sha256": digest(output)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    pack = sub.add_parser("prepare")
    pack.add_argument("--source", type=Path, required=True)
    pack.add_argument("--evaluation", type=Path, nargs="+", required=True)
    run = sub.add_parser("predict")
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--frozen-head", type=Path)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    score = sub.add_parser("score")
    for name in ("run", "prior", "weights", "nerf", "winding", "catalog-scale"):
        score.add_argument(f"--{name}", type=Path, required=True)
    score.add_argument("--evaluation", type=Path, nargs="+", required=True)
    score.add_argument("--appliance", type=Path)
    select = sub.add_parser("select")
    for name in ("run", "prior", "appliance", "cohort"):
        select.add_argument(f"--{name}", type=Path, required=True)
    for child in (pack, run, score, select):
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.source, args.evaluation, args.output)
    elif args.action == "predict":
        predict(
            args.dataset, args.checkpoint, args.output, args.frozen_head, args.device
        )
    elif args.action == "score":
        evaluate(args)
    else:
        select_abo50(args)


if __name__ == "__main__":
    main()
