"""Apply the frozen image prior, assemble sealed ABO-100 predictions, and score them."""

import argparse
import math
import re
from pathlib import Path

from evaluate_abo50_mass_hypotheses import paired_comparison
from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo_image_mass_control import (
    combine_role_mass,
    convert_observation,
    read_sealed_clip,
)
from run_abo50_mass_hypotheses import VIEWS
from run_component_mass_stability import sealed_rows
from tactile3d.models.calibration import mass_metrics


APPLIANCE_CUES = r"\b(fan|grilles?|grills?|vents?|ventilation|filters?|compressor|condenser|radiator)\b"


def write_score_table(output, stats, records):
    labels = {
        "official_100": "官方测试集",
        "prior_50": "原 50 件",
        "additional_50": "新增 50 件",
        "additional_excluding_declared_development": "新增批排除已声明开发对象",
    }
    lines = [
        "# ABO-100 质量评分",
        "",
        "| 样本 | 方法 | 件数 | ADE (kg) ↓ | ALDE ↓ | APE (%) ↓ | MnRE ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group, result in stats.items():
        for key, name in (("ours", "Ours"), ("nerf", "NeRF2Physics 官方缓存")):
            m = result[key]
            lines.append(
                f"| {labels[group]} | {name} | {result['n']} | {m['ade_kg']:.3f} | {m['alde']:.4f} | {100*m['ape']:.2f} | {m['mnre']:.4f} |"
            )
    lines += [
        "",
        f"全组仅用图像先验回退 {stats['official_100']['image_only_fallback']} 件。原 50 件是既有封存结果，其中包含探索性修正。",
        "",
        "| 对象 | 批次 | 目录质量 (kg) | Ours (kg) | NeRF2Physics (kg) | 仅图像先验 |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in records:
        lines.append(
            f"| `{row['scene_id']}` | {labels[row['batch']]} | {row['target_mass_kg']:.3f} | {row['mass_kg']:.3f} | {row['nerf_mass_kg']:.3f} | {'是' if row.get('image_only_fallback') else '否'} |"
        )
    output.with_name(f"{output.stem}_table.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def prepare_clip(observations, output):
    cohort = read_json(observations / "cohort.json")
    output.mkdir(exist_ok=False)
    rows = []
    for scene in cohort["object_ids"]:
        files = {}
        for view in VIEWS:
            name = f"{scene}_{view:02d}"
            relative = f"scenes/{scene}/images/{name}.png"
            if digest(observations / relative) != cohort["files_sha256"][relative]:
                raise ValueError(f"Input hash mismatch: {relative}")
            derived = output / f"{name}.jpg"
            convert_observation(observations / relative, derived)
            files[derived.name] = digest(derived)
        rows.append(dict(scene_id=scene, split="test", files=files))
    write_json(output / "manifest.json", rows)


def candidates(construction):
    return [
        scene
        for scene, row in sealed_rows(construction).items()
        if any(
            re.search(APPLIANCE_CUES, part["name"], flags=re.I)
            for part in row.get("parts", [])
        )
    ]


def select(args):
    cohort = read_json(args.cohort)
    previous = sealed_rows(args.previous)
    clip = read_sealed_clip(args.clip)
    construction = sealed_rows(args.construction)
    appliance = sealed_rows(args.appliance) if args.appliance else {}
    if (
        list(previous) != cohort["prior_object_ids"]
        or list(construction) != cohort["additional_object_ids"]
        or set(clip) != set(construction)
    ):
        raise ValueError("Inconsistent membership or order of the two frozen batches")
    if not appliance.keys() <= construction.keys():
        raise ValueError("Unexpected appliance outputs outside extension")
    records = []
    for scene in cohort["object_ids"]:
        if scene in previous:
            records.append(previous[scene] | {"batch": "prior_50"})
            continue
        child = construction[scene]
        corrected = appliance.get(scene, {}).get("role_mass_kg")
        physical = (
            corrected if corrected is not None else child.get("construction_mass_kg")
        )
        image = clip[scene]["mass_kg"]
        records.append(
            dict(
                scene_id=scene,
                batch="additional_50",
                mass_kg=(
                    combine_role_mass(physical, image)
                    if physical is not None
                    else image
                ),
                construction_mass_kg=physical,
                image_prior_kg=image,
                appliance_replay_used=corrected is not None,
                image_only_fallback=physical is None,
                direct_vlm_kg=child.get("direct_mass_kg"),
            )
        )
    args.output.mkdir(exist_ok=False)
    write_json(
        args.output / "protocol.json",
        dict(
            cohort_sha256=digest(args.cohort),
            mass_labels_used=False,
            source_sha256={
                key: digest(getattr(args, key) / "predictions.json")
                for key in ("previous", "clip", "construction", "appliance")
                if getattr(args, key)
            },
            appliance_part_name_cues=APPLIANCE_CUES,
            scope="Official 100 test objects, archived first 50 and prospective additional 50 with fixed automatic appliance routing",
            fallback="Image prior only if the complete geometry/construction pipeline cannot return a valid mass",
        ),
    )
    result = args.output / "predictions.json"
    write_json(result, records)
    write_json(args.output / "prediction_seal.json", dict(sha256=digest(result)))


def score(args):
    cohort = read_json(args.cohort)
    current = sealed_rows(args.run)
    if list(current) != cohort["object_ids"] or read_json(args.run / "protocol.json")[
        "cohort_sha256"
    ] != digest(args.cohort):
        raise ValueError("Predictions differ from the official cohort")
    baseline = args.nerf / "official_predictions.json"
    if digest(baseline) != read_json(args.nerf / "prediction_seal.json")[baseline.name]:
        raise ValueError("Invalid official NeRF2Physics seal")
    nerf = read_json(baseline)
    weights = read_json(args.weights)
    records = []
    for scene, row in current.items():
        target = weights.get(scene, weights.get(scene.split("_")[0]))
        estimates = (*nerf[scene], row["mass_kg"], target)
        if not all(
            isinstance(m, (int, float)) and math.isfinite(m) and m > 0
            for m in estimates
        ):
            raise ValueError(f"Missing or invalid mass on {scene}")
        records.append(
            row | {"target_mass_kg": target, "nerf_mass_kg": sum(nerf[scene]) / 2}
        )
    groups = {
        "official_100": records,
        **{
            name: [r for r in records if r["batch"] == name]
            for name in ("prior_50", "additional_50")
        },
    }
    previously_exposed = set(cohort.get("historical_development_ids", []))
    if previously_exposed:
        groups["additional_excluding_declared_development"] = [
            r
            for r in groups["additional_50"]
            if r["scene_id"] not in previously_exposed
        ]
    stats = {}
    for name, batch in groups.items():
        y, ours, baseline = zip(
            *[(x["target_mass_kg"], x["mass_kg"], x["nerf_mass_kg"]) for x in batch]
        )
        stats[name] = dict(
            n=len(batch),
            image_only_fallback=sum(x.get("image_only_fallback", False) for x in batch),
            ours=mass_metrics(y, ours),
            nerf=mass_metrics(y, baseline),
            delta_vs_nerf=paired_comparison(list(zip(y, ours, baseline))),
        )
    write_json(
        args.output,
        dict(
            prediction_sha256=digest(args.run / "predictions.json"),
            cohorts=stats,
            per_scene=records,
        ),
    )
    write_score_table(args.output, stats, records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name, fields in (
        ("prepare-clip", ("observations", "output")),
        ("appliance-candidates", ("construction",)),
        ("select", ("cohort", "previous", "clip", "construction", "output")),
        ("score", ("cohort", "run", "weights", "nerf", "output")),
    ):
        child = sub.add_parser(name)
        for field in fields:
            child.add_argument(f"--{field}", type=Path, required=True)
        if name == "select":
            child.add_argument("--appliance", type=Path)
    args = parser.parse_args()
    if args.action == "prepare-clip":
        prepare_clip(args.observations, args.output)
    elif args.action == "appliance-candidates":
        print("\n".join(candidates(args.construction)))
    elif args.action == "select":
        select(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
