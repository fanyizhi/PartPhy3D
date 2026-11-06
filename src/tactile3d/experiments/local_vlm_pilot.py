"""真实多视角VLM开发基线；不把语言模型猜测冒充Mesh物理积分。"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import shutil
import traceback
from pathlib import Path
from typing import Any

from PIL import Image

from tactile3d.experiments.five_object_pilot import read_cohort
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.vlm_interior import _parse_response_content

VIEWS = (0, 10, 20)
DEFAULT_COHORT = Path("configs/five_object_pilot.json")
DEFAULT_PRIORS = Path("configs/material_priors.yaml")
SYSTEM = (
    "You analyze public product images for a scientific development experiment. "
    "Return strict JSON only. Images show appearance, not hidden material truth. "
    "Distinguish observations from hypotheses. Never claim measured properties."
)
PROMPT = """{view_count} image(s) show the same object from camera view(s).
Estimate its total mass from visual evidence and propose visible functional parts.
No measured dimensions, mass, or interior labels are supplied. Do not invent them.
Use ordinary product-size assumptions and explain the main uncertainty briefly.
For upholstered products, distinguish cover, foam and support frame. For electronics,
distinguish casing, circuit boards and metal components. Hidden structure is uncertain.
Use at most 6 parts. Material names must come from: {materials}, unknown, void.
Bounding boxes refer ONLY to the FIRST image, normalized to [0,1000]. Return null
for an occluded part rather than invent its position. Boxes are proposals, not masks.
Return this JSON schema, without Markdown:
{{"object_description":"short description",
  "mass_kg":{{"estimate":1.0,"plausible_low":0.5,"plausible_high":2.0}},
  "main_uncertainty":"short statement",
  "parts":[{{"part_id":"p0","name":"functional part",
    "surface_material":"allowed material",
    "interior_material":"allowed material or unknown",
    "construction":"solid / hollow / layered / frame / flexible / unknown",
    "alternative":"one plausible alternative, not a fact",
    "bbox_first_view_1000":[100,100,900,900]}}]}}
The mass bounds are subjective plausibility bounds, NOT calibrated quantiles.
"""
TYPE_ONLY_PROMPT = """{view_count} image(s) show the SAME product.
Identify the product category, visible functional parts and plausible construction.
Estimate the total mass of the identified product using ordinary product-size
assumptions. No measured dimensions or mass labels are supplied. State uncertainty.
Distinguish visible cover from hidden foam, frame, circuitry, fill or void.
Hidden construction is a hypothesis. Return at most six meaningful parts.
Allowed material names: {materials}, unknown, void.
For each part, locate ONLY that part in the FIRST image. Use a normalized 0-1000
box [x_min,y_min,x_max,y_max]. Return null if you cannot localize that part.
Do not give every part the same whole-object box. Do not invent occluded positions.
Return a JSON object with these fields (the following are TYPES, not values):
object_description: string describing the observed product
mass_kg: object with estimate, plausible_low, plausible_high, all positive numbers
main_uncertainty: brief string about size or hidden construction
parts: list of objects, each with:
  part_id: unique string
  name: visible functional part name
  surface_material: allowed material name
  interior_material: allowed material name or unknown or void
  construction: solid / hollow / layered / frame / flexible / unknown
  alternative: brief plausible alternative, not a measured fact
  bbox_first_view_1000: four numbers as above, or null
Plausibility bounds are subjective, not calibrated quantiles. Return JSON only.
"""


def quality_flags(
    payload: dict[str, Any], materials: set[str] | None = None
) -> dict[str, Any]:
    """语义退化只标记不删样本，避免通过排除失败提高质量指标。"""
    mass = payload["mass_kg"]
    copied = [mass[k] for k in ("plausible_low", "estimate", "plausible_high")]
    boxes = [
        tuple(p["bbox_first_view_1000"])
        for p in payload["parts"]
        if p["bbox_first_view_1000"] is not None
    ]
    return {
        "matches_numeric_example": copied == [0.5, 1.0, 2.0],
        "identical_part_boxes": len(boxes) > 1 and len(set(boxes)) == 1,
        "localized_part_count": len(boxes),
        "unique_box_count": len(set(boxes)),
        "fine_segmentation_established": False,
        "physical_properties_measured": False,
        "solid_with_void_interior": [
            p["part_id"]
            for p in payload["parts"]
            if p.get("construction") == "solid" and p["interior_material"] == "void"
        ],
        "unmapped_materials": sorted(
            {
                p[key]
                for p in payload["parts"]
                for key in ("surface_material", "interior_material")
                if materials is not None
                and p[key] not in materials | {"unknown", "void"}
            }
        ),
    }


def validate_response(raw: str, materials: set[str]) -> dict[str, Any]:
    """拒绝非法数值或越界标框，模型格式失败保留为失败案例。"""
    payload = _parse_response_content(raw)
    mass = payload.get("mass_kg", {})
    if not isinstance(mass, dict):
        raise TypeError("质量预测必须是对象")
    values = []
    for key in ("plausible_low", "estimate", "plausible_high"):
        value = mass.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise TypeError(f"质量字段不是数值：{key}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"质量字段须为有限正数：{key}")
        values.append(float(value))
    if values != sorted(values):
        raise ValueError("质量下界、估计与上界次序错误")
    parts = payload.get("parts")
    if not isinstance(parts, list) or not 1 <= len(parts) <= 64:
        raise ValueError("部件列表须含1到64项")
    ids = []
    for part in parts:
        if not isinstance(part, dict):
            raise TypeError("部件必须是对象")
        part_id = part.get("part_id")
        if not isinstance(part_id, str) or not part_id.strip():
            raise ValueError("部件缺少稳定标识")
        ids.append(part_id)
        for key in ("surface_material", "interior_material"):
            value = part.get(key)
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"材料名称必须是非空字符串：{key}")
        box = part.get("bbox_first_view_1000")
        if box is not None:
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("标框需要4个坐标或null")
            if any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or not 0 <= v <= 1000
                for v in box
            ):
                raise ValueError("标框坐标须在0到1000之间")
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("标框必须有正面积")
    if len(ids) != len(set(ids)):
        raise ValueError("部件标识重复")
    return payload


def infer_pilot(
    model_path: Path,
    data_root: Path,
    output: Path,
    limit: int = 5,
    cohort: Path = DEFAULT_COHORT,
    view_count: int = 3,
    prompt_mode: str = "typed_schema",
) -> dict[str, Any]:
    """先封存全部图像预测，再读取独立称重标签。

    Returns:
        模型信息、逐对象成功/失败记录，以及仅针对成功对象的开发评价。
    """
    selection = read_cohort(cohort, limit)
    if view_count not in (1, 3):
        raise ValueError("当前先导只比较1视角与3视角")
    views = VIEWS[:view_count]
    templates = {"typed_schema": TYPE_ONLY_PROMPT, "numeric_example": PROMPT}
    if prompt_mode not in templates:
        raise ValueError("未知提示方案")
    priors = load_material_priors(DEFAULT_PRIORS)
    names = {p.name for p in priors}
    prompt = templates[prompt_mode].format(
        materials=", ".join(sorted(names)), view_count=view_count
    )
    inputs = []
    for scene_id in selection["object_ids"]:
        paths = [
            data_root / "scenes" / scene_id / "images" / f"{scene_id}_{view:02d}.png"
            for view in views
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"缺少原始图像：{path}")
            with Image.open(path) as image:
                image.verify()
        inputs.append((scene_id, paths))
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "selection": selection,
        "views": views,
        "source_sha256": {
            "pilot": file_sha256(Path(__file__)),
            "local_vlm": file_sha256(Path(__file__).parents[1] / "vision/local_vlm.py"),
        },
        "system": SYSTEM,
        "prompt": prompt,
        "prompt_mode": prompt_mode,
        "scope": "image_only_direct_VLM_development_baseline",
        "reference_mesh_used": False,
        "instantmesh_executed": False,
        "fine_part_segmentation": False,
        "mesh_physics_prediction": False,
        "intervals_are_calibrated": False,
        "target_mass_passed_to_inference": False,
        "material_parameter_source": "config_prior_not_measured_or_VLM_regression",
        "inputs": [
            {
                "scene_id": scene,
                "images": [
                    {"path": str(path), "sha256": file_sha256(path)} for path in paths
                ],
            }
            for scene, paths in inputs
        ],
        "priors_sha256": file_sha256(DEFAULT_PRIORS),
    }
    write_json(output / "protocol.json", protocol)
    snapshots = output / "source"
    snapshots.mkdir()
    shutil.copyfile(Path(__file__), snapshots / "local_vlm_pilot.py")
    shutil.copyfile(
        Path(__file__).parents[1] / "vision/local_vlm.py",
        snapshots / "local_vlm.py",
    )
    model = LocalVLM(model_path)
    write_json(output / "model.json", model.provenance)
    predictions, failures = [], []
    for scene_id, paths in inputs:
        scene_output = output / "objects" / scene_id
        scene_output.mkdir(parents=True)
        try:
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(image.copy())
            response = model.generate(prompt, images, SYSTEM)
            (scene_output / "raw_response.txt").write_text(
                response.text, encoding="utf-8"
            )
            write_json(
                scene_output / "runtime.json",
                {
                    key: value
                    for key, value in dataclasses.asdict(response).items()
                    if key != "text"
                },
            )
            payload = validate_response(response.text, names)
            payload["scene_id"] = scene_id
            payload["source"] = "local_VLM_uncalibrated_hypothesis"
            payload["quality_flags"] = quality_flags(payload, names)
            payload["object_index"] = selection["object_ids"].index(scene_id) + 1
            payload["material_priors"] = {
                p.name: {
                    "density_mean_kg_m3": p.density_kg_m3.mean,
                    "youngs_mean_pa": p.youngs_modulus_pa.mean,
                }
                for p in priors
            }
            write_json(scene_output / "prediction.json", payload)
            predictions.append(payload)
            print(f"PREDICTED {scene_id}", flush=True)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            (scene_output / "traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            failure = {
                "scene_id": scene_id,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            write_json(scene_output / "failure.json", failure)
            failures.append(failure)
            print(f"FAILED {scene_id}: {error}", flush=True)
    prediction_path = output / "sealed_predictions.json"
    write_json(prediction_path, predictions)
    seal = file_sha256(prediction_path)
    write_json(output / "prediction_seal.json", {"sha256": seal})
    weights_path = data_root / "filtered_product_weights.json"
    weights = json.loads(weights_path.read_text())
    evaluated = [
        {
            "scene_id": p["scene_id"],
            "target_mass_kg": weights[p["scene_id"]],
            "predicted_mass_kg": p["mass_kg"]["estimate"],
        }
        for p in predictions
    ]
    metrics = (
        mass_metrics(
            [p["target_mass_kg"] for p in evaluated],
            [p["predicted_mass_kg"] for p in evaluated],
        )
        if evaluated
        else None
    )
    if file_sha256(prediction_path) != seal:
        raise ValueError("评价过程中封存预测发生变化")
    result = {
        "status": "complete" if not failures else "incomplete",
        "scope": protocol["scope"],
        "requested": limit,
        "successful": len(predictions),
        "quality_flagged": sum(
            p["quality_flags"]["matches_numeric_example"]
            or p["quality_flags"]["identical_part_boxes"]
            or bool(p["quality_flags"]["unmapped_materials"])
            or bool(p["quality_flags"]["solid_with_void_interior"])
            for p in predictions
        ),
        "failures": failures,
        "metrics_successful_objects_only": metrics,
        "evaluation": evaluated,
        "labels_sha256": file_sha256(weights_path),
    }
    write_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/abo500_five"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--view-count", type=int, choices=(1, 3), default=3)
    parser.add_argument(
        "--prompt-mode",
        choices=("typed_schema", "numeric_example"),
        default="typed_schema",
    )
    args = parser.parse_args()
    result = infer_pilot(
        args.model_path,
        args.root,
        args.output,
        args.limit,
        view_count=args.view_count,
        prompt_mode=args.prompt_mode,
    )
    if result["successful"]:
        from tactile3d.figures.vlm_pilot import make_vlm_figures

        make_vlm_figures(args.output, args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
