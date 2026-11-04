from __future__ import annotations

from pathlib import Path

import numpy as np

from tactile3d.config import resolve_config_path
from tactile3d.mesh.reconstruction import build_reconstructor
from tactile3d.mesh.repair import repair_mesh
from tactile3d.mesh.segmentation import segment_mesh
from tactile3d.physics.estimation import estimate_parts, parts_to_json
from tactile3d.physics.materials import load_material_priors
from tactile3d.utils.io import ensure_dir, write_json


def run_inference(image_path: str | Path, cfg: dict, output_dir: str | Path) -> dict:
    output_dir = ensure_dir(output_dir)
    rng = np.random.default_rng(int(cfg.get("seed", 7)))
    reconstructor = build_reconstructor(cfg)

    material_path = resolve_config_path(
        cfg.get("materials", {}).get("priors", "configs/material_priors.yaml"),
        cfg.get("_config_dir"),
    )
    priors = load_material_priors(material_path)

    samples: list[dict] = []
    for sample_id in range(int(cfg.get("num_samples", 8))):
        sample_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        hypothesis = reconstructor.sample(image_path, sample_rng)
        mesh, repair_report = repair_mesh(
            hypothesis.mesh,
            strategy=str(cfg.get("repair_strategy", "fill_then_hull")),
        )
        if cfg.get("require_watertight", True) and not mesh.is_watertight:
            continue

        segmentation = segment_mesh(
            mesh,
            num_parts=int(cfg.get("num_parts", 4)),
            cfg=cfg,
            rng=sample_rng,
        )
        parts = estimate_parts(
            mesh,
            segmentation,
            priors,
            material_temperature=float(cfg.get("materials", {}).get("temperature", 0.8)),
            rng=sample_rng,
            image_features=hypothesis.metadata,
        )
        part_payload = parts_to_json(parts)
        sample_payload = {
            "sample_id": sample_id,
            "log_prob": hypothesis.log_prob,
            "mesh_metadata": hypothesis.metadata,
            "repair": repair_report,
            "is_watertight": bool(mesh.is_watertight),
            "volume_m3": float(abs(mesh.volume)) if mesh.is_volume else 0.0,
            "mass_kg": float(sum(p.mass_kg for p in parts)),
            "parts": part_payload,
        }
        samples.append(sample_payload)

        if cfg.get("output", {}).get("save_mesh_samples", True):
            mesh.export(output_dir / f"mesh_sample_{sample_id:03d}.ply")
        if cfg.get("output", {}).get("save_part_json", True):
            write_json(output_dir / f"parts_sample_{sample_id:03d}.json", part_payload)

    if not samples:
        raise RuntimeError("No valid mesh samples survived watertightness filtering.")

    summary = summarize_samples(samples)
    summary["image"] = str(image_path)
    summary["num_valid_samples"] = len(samples)
    summary["num_requested_samples"] = int(cfg.get("num_samples", 8))
    write_json(output_dir / "summary.json", summary)
    return summary


def summarize_samples(samples: list[dict]) -> dict:
    masses = np.asarray([s["mass_kg"] for s in samples], dtype=np.float64)
    volumes = np.asarray([s["volume_m3"] for s in samples], dtype=np.float64)
    watertight = np.asarray([s["is_watertight"] for s in samples], dtype=np.float64)

    max_parts = max(len(s["parts"]) for s in samples)
    part_summaries = []
    for part_id in range(max_parts):
        part_values = [s["parts"][part_id] for s in samples if part_id < len(s["parts"])]
        if not part_values:
            continue
        part_summaries.append(
            {
                "part_id": part_id,
                "volume_m3": _stats([p["volume_m3"] for p in part_values]),
                "density_kg_m3": _stats([p["density_kg_m3"] for p in part_values]),
                "youngs_modulus_pa": _stats([p["youngs_modulus_pa"] for p in part_values]),
                "mass_kg": _stats([p["mass_kg"] for p in part_values]),
                "material_distribution": _categorical([p["sampled_material"] for p in part_values]),
            }
        )

    return {
        "object": {
            "mass_kg": _stats(masses),
            "volume_m3": _stats(volumes),
            "watertight_rate": float(watertight.mean()),
        },
        "parts": part_summaries,
    }


def _stats(values) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "q05": float(np.quantile(arr, 0.05)),
        "q50": float(np.quantile(arr, 0.50)),
        "q95": float(np.quantile(arr, 0.95)),
    }


def _categorical(values: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    total = float(sum(counts.values()))
    return {k: v / total for k, v in sorted(counts.items())}
