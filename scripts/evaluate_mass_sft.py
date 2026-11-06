"""同一NF4推理条件下评价基座与质量SFT适配器，保留失败覆盖率。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.vlm_interior import _parse_response_content

PIXELS = 384 * 384


def evaluate(
    model: Path, dataset: Path, output: Path, adapter: Path | None = None
) -> dict[str, Any]:
    """封存测试预测后读取标签；不将测试答案拼接入提示。"""
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    rows = [
        json.loads(line)
        for line in (dataset / "test_inputs.jsonl").read_text().splitlines()
    ]
    output.mkdir(parents=True, exist_ok=False)
    vlm = LocalVLM(
        model, max_tokens=96, max_pixels=PIXELS, quantized=True, adapter_path=adapter
    )
    write_json(
        output / "protocol.json",
        {
            "model": vlm.provenance,
            "test_inputs_sha256": file_sha256(dataset / "test_inputs.jsonl"),
            "dataset_manifest_sha256": file_sha256(dataset / "dataset_manifest.json"),
            "prompt": manifest["prompt"],
            "system": manifest["system"],
            "target_mass_passed_to_inference": False,
            "adapter_sha256": (
                file_sha256(adapter / "adapter_model.safetensors") if adapter else None
            ),
        },
    )
    predictions, failures = [], []
    for row in rows:
        scene = row["scene_id"]
        try:
            with Image.open(row["images"][0]) as image:
                response = vlm.generate(
                    manifest["prompt"], [image.copy()], manifest["system"]
                )
            (output / f"{scene}.txt").write_text(response.text, encoding="utf-8")
            value = _parse_response_content(response.text)["mass_kg"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("mass_kg不是数值")
            if not math.isfinite(value) or value <= 0:
                raise ValueError("mass_kg须为有限正数")
            predictions.append({"scene_id": scene, "predicted_mass_kg": value})
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            failures.append({"scene_id": scene, "error": str(error)})
        print(f"EVALUATED {scene} {len(predictions)}/{len(rows)}", flush=True)
    write_json(output / "predictions.json", predictions)
    seal = file_sha256(output / "predictions.json")
    write_json(output / "prediction_seal.json", {"sha256": seal})
    labels = json.loads((dataset / "evaluation_labels.json").read_text())
    target = [labels[p["scene_id"]] for p in predictions]
    estimated = [p["predicted_mass_kg"] for p in predictions]
    result = {
        "requested": len(rows),
        "successful": len(predictions),
        "failures": failures,
        "coverage": len(predictions) / len(rows),
        "metrics_successful_objects_only": (
            mass_metrics(target, estimated) if predictions else None
        ),
        "physical_material_accuracy_established": False,
    }
    write_json(output / "report.json", result)
    if file_sha256(output / "predictions.json") != seal:
        raise ValueError("封存预测发生变化")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.model, args.dataset, args.output, args.adapter), indent=2
        )
    )


if __name__ == "__main__":
    main()
