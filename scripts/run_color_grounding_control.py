"""Isolate the effect of explicit per-component visual evidence on identity binding."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image

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
    directory = out / "color_control"
    directory.mkdir(exist_ok=False)
    selected = json.loads((out / "view_count_results.json").read_text())[
        "selected_views"
    ]
    views = json.loads((out / "view_selection.json").read_text())[str(selected)]
    manifest = json.loads((args.source / "input_manifest.json").read_text())
    # Only sampled RGB is added here; no geometry dimensions or evaluation labels.
    table = [
        {"component_id": r["component_id"], "mean_texture_rgb": r["mean_texture_rgb"]}
        for r in manifest["components"]
    ]
    prompt = (
        "Assign all 15 numbered regions. Per-component sampled RGB below is visual "
        "evidence from the reference texture and helps bind the correct isolated component "
        "to its number. It is not a material label. Use images and color jointly. "
        "Return JSON {components:[{component_id:integer,functional_role:string,surface_material:string}]}. "
        "Role vocabulary: cushion, seat, backrest, armrest, leg, frame_rail, underside, unknown. "
        "Material vocabulary: wood, fabric, metal, plastic, unknown. Use each ID exactly once. "
        "Visual color observations: " + json.dumps(table)
    )
    paths = [out / f"view_{r['frame_index']:02d}.png" for r in views]
    paths += [
        args.source / "component_sheet.png",
        out / "global_component_positions.png",
    ]
    images = [Image.open(p).convert("RGB") for p in paths]
    model = LocalVLM(args.model, max_tokens=1400, max_pixels=768**2, quantized=True)
    write_json(directory / "model.json", model.provenance)
    records = []
    for trial in range(3):
        response = model.generate(
            prompt,
            images,
            "Bind component identity to its own visual evidence; JSON only.",
            seed=1201 + trial,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        record = {
            "trial": trial,
            "view_count": selected,
            "prompt": prompt,
            "images": [p.name for p in paths],
            "response": asdict(response),
            "valid": False,
        }
        try:
            answer = _parse_response_content(response.text)
            if sorted(r["component_id"] for r in answer["components"]) != list(
                range(15)
            ):
                raise ValueError("Incomplete component identities")
            record.update(answer=answer, valid=True)
        except (KeyError, ValueError, TypeError) as error:
            record["error"] = str(error)
        records.append(record)
        write_json(directory / f"trial_{trial}.json", record)
        print(
            f"Color evidence trial {trial}: valid={record['valid']}, {response.seconds:.1f}s",
            flush=True,
        )
    write_json(
        directory / "answers.json",
        {
            "selected_views": selected,
            "records": records,
            "scope": "additional visual-evidence control; geometry-only descriptors excluded from VLM prompt",
        },
    )


if __name__ == "__main__":
    main()
