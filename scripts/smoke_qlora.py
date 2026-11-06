"""两步QLoRA工程验收；合成已知物理样本不用于宣称真实材料识别精度。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.figures.bound_material import raster_face_ids, shade_face_colors
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.training.train_vlm import add_training_arguments, run_training

CASES = (
    ((0.10, 0.20, 0.30), 650.0, (175, 125, 75, 255)),
    ((0.20, 0.10, 0.15), 2700.0, (140, 160, 180, 255)),
    ((0.15, 0.16, 0.11), 1050.0, (130, 100, 170, 255)),
    ((0.12, 0.18, 0.25), 500.0, (100, 160, 130, 255)),
)
STEPS = 2


def prepare(output: Path) -> dict[str, Any]:
    """生成对象组、图像和答案均独立的四个确定性工程样本。"""
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    for index, (dimensions, density, rgba) in enumerate(CASES):
        mesh = trimesh.creation.box(extents=dimensions)
        surface = SimpleNamespace(vertices_m=mesh.vertices, faces=mesh.faces)
        face_ids = raster_face_ids(surface, -55, 160)
        colors = np.tile(np.asarray(rgba, np.uint8), (len(mesh.faces), 1))
        image = images / f"known_box_{index}.png"
        Image.fromarray(shade_face_colors(surface, colors, face_ids)).save(image)
        prompt = (
            "This rendered object is a specified solid rectangular box. "
            f"Dimensions in metres: {dimensions}; density: {density} kg/m^3. "
            "Compute mass = density * length * width * height. "
            'Return JSON with the single numeric field "mass_kg".'
        )
        answer = json.dumps({"mass_kg": round(density * math.prod(dimensions), 8)})
        row = {
            "group_id": f"synthetic_known_box_{index}",
            "images": [str(image.resolve())],
            "provenance": {"source": "prescribed_geometry_and_density_not_ABO"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ],
        }
        rows["train" if index < 2 else "val"].append(row)
    for split, records in rows.items():
        (output / f"{split}.jsonl").write_text(
            "\n".join(json.dumps(row) for row in records) + "\n", encoding="utf-8"
        )
    report = {
        "scope": "synthetic_training_integration_only",
        "samples": 4,
        "train_groups": 2,
        "validation_groups": 2,
        "ABO_images_or_labels_used": False,
        "supervision": "analytic_mass_from_specified_box_and_density",
    }
    write_json(output / "dataset_scope.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--output", type=Path, default=Path("outputs/qlora_smoke"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    dataset = args.output / "data"
    prepare(dataset)
    if not args.execute:
        print("Prepared synthetic engineering data; training not started.")
        return
    import torch

    train_parser = argparse.ArgumentParser()
    add_training_arguments(train_parser)
    train_args = train_parser.parse_args(
        [
            "--model",
            args.model,
            "--train-data",
            str(dataset / "train.jsonl"),
            "--val-data",
            str(dataset / "val.jsonl"),
            "--output",
            str(args.output / "adapter"),
            "--qlora",
            "--max-steps",
            str(STEPS),
            "--logging-steps",
            "1",
            "--eval-steps",
            "1",
            "--gradient-accumulation",
            "1",
            "--lora-rank",
            "8",
            "--lora-alpha",
            "16",
            "--min-pixels",
            "4096",
            "--max-pixels",
            "196608",
            "--run-scope",
            "integration_smoke",
        ]
    )
    original = Path(args.model) / "model.safetensors"
    original_hash = file_sha256(original) if original.is_file() else None
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    run_training(train_args)
    torch.cuda.synchronize()
    if original_hash and original_hash != file_sha256(original):
        raise RuntimeError("训练错误改写了基座权重")
    report = json.loads((args.output / "adapter/training_report.json").read_text())
    adapter = args.output / "adapter/adapter_model.safetensors"
    if (
        report["optimizer_steps"] != STEPS
        or report["nonzero_lora_B_tensors"] == 0
        or not adapter.is_file()
    ):
        raise RuntimeError("QLoRA没有完成约定的更新与adapter保存")
    report.update(
        seconds=time.perf_counter() - start,
        peak_gpu_bytes=int(torch.cuda.max_memory_allocated()),
        base_weights_unchanged=True,
        adapter_bytes=adapter.stat().st_size,
        valid_finetuning_accuracy_experiment=False,
    )
    write_json(args.output / "smoke_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
