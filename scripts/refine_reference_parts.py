"""Resolve component identity with one isolated target per VLM query."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image
from run_reference_mesh_study import load_prepared

from tactile3d.figures.mesh_fields import render_mesh
from tactile3d.physics.part_io import write_json
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.vlm_interior import _parse_response_content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/root/autodl-tmp/models/Qwen3-VL-8B-Instruct"),
    )
    args = parser.parse_args()
    out = args.source / "views_contacts"
    comparison = json.loads((out / "view_count_results.json").read_text())
    count = comparison["selected_views"]
    selections = json.loads((out / "view_selection.json").read_text())[str(count)]
    roles = json.loads((out / "role_review.json").read_text())["roles"]
    _mesh, labels, pieces, texture = load_prepared(args.source)
    manifest = json.loads((args.source / "input_manifest.json").read_text())
    model = LocalVLM(args.model, max_tokens=200, max_pixels=384**2, quantized=True)
    records = []
    for index, part in enumerate(pieces):
        isolated = render_mesh(
            part.vertices, part.faces, texture[labels == index], resolution=450
        )
        Image.fromarray(isolated).save(out / f"focused_component_{index:02d}.png")
        images = [
            Image.open(out / f"view_{s['frame_index']:02d}.png").convert("RGB")
            for s in selections
        ]
        images += [
            Image.open(out / "global_component_positions.png").convert("RGB"),
            Image.fromarray(isolated),
        ]
        prompt = (
            f"Identify ONLY component {index} of the chaise. The LAST image is the isolated "
            "target. Other images show original views and numbered whole-object context. "
            "Do not assign a different component's name. Geometry in meters: "
            + json.dumps(manifest["components"][index])
            + ". Return JSON {component_id:integer,functional_role:string,surface_material:string,evidence:string}. "
            "Choose functional_role exactly from cushion, seat, backrest, armrest, leg, frame_rail, underside, unknown. "
            "Choose surface_material from wood, fabric, metal, plastic, unknown. Keep evidence under 20 words."
        )
        response = model.generate(
            prompt,
            images,
            "Identify the isolated target using global spatial context; JSON only.",
            seed=1601 + index,
        )
        record = {
            "component_id": index,
            "prompt": prompt,
            "response": asdict(response),
            "valid": False,
        }
        try:
            answer = _parse_response_content(response.text)
            if answer["component_id"] != index:
                raise ValueError("Wrong component identity")
            record["answer"] = answer
            record["valid"] = True
            record["role_correct"] = answer["functional_role"] == roles[str(index)]
        except (KeyError, ValueError, TypeError) as error:
            record["error"] = str(error)
        records.append(record)
        write_json(out / f"focused_answer_{index:02d}.json", record)
        print(
            f"Focused {index}: valid={record['valid']} role_correct={record.get('role_correct')}",
            flush=True,
        )
    write_json(
        out / "focused_results.json",
        {
            "selected_original_view_count": count,
            "role_accuracy": sum(r.get("role_correct", False) for r in records)
            / len(roles),
            "generation_seconds": sum(r["response"]["seconds"] for r in records),
            "scope": "development diagnostic with target-isolated images; not equal-cost global comparison",
            "records": records,
        },
    )


if __name__ == "__main__":
    main()
