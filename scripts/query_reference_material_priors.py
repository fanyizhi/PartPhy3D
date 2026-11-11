"""Obtain conditional material priors after structural inference is sealed.

This shared stage is excluded from the equal-call structural-arm comparison.
It estimates nominal properties, not experimental ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
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
    out = args.source
    states = json.loads((out / "consensus_states.json").read_text())
    names = sorted(
        {c["surface_material"] for c in states["progressive"]["components"]}
        - {"unknown"}
    )
    directory = out / "property_queries"
    directory.mkdir(exist_ok=False)
    model = LocalVLM(args.model, max_tokens=1100, max_pixels=384**2, quantized=True)
    write_json(directory / "model.json", model.provenance)
    image = Image.open(out / "observation_00.png").convert("RGB")
    prompt = (
        "Estimate typical material-property PRIORS for these already identified surface "
        f"materials: {json.dumps(names)}. These are conditional nominal estimates, not "
        "measurements and not whole-upholstery effective properties. Give reasonable "
        "numeric central values and broad ranges for density and Young's modulus when "
        "a material-level value is meaningful. For wood use longitudinal modulus, unknown "
        "species. For textile use an explicitly assumed effective in-plane tensile modulus. "
        "Define friction as static friction for a clean DRY stainless-steel slider against "
        "this material under light contact at room temperature. State that exact surface "
        "finish is unresolved. Define hardness using Shore D only when meaningful; "
        "textile has no useful bulk Shore D hardness and must be null. Do not copy null "
        "schema examples for all other properties. Return JSON {materials:[{name:string, "
        "density_kg_m3:number, density_range:[low,high], youngs_modulus_pa:number or null, "
        "modulus_direction:string, friction_coefficient:number or null, "
        "friction_contact_pair:'dry stainless steel slider / material', "
        "hardness_value:number or null, hardness_scale:'Shore D' or null, "
        "assumptions:string}]}. Do not fabricate a literature citation."
    )
    records = []
    for index in range(3):
        response = model.generate(
            prompt,
            [image],
            "Return conditional material priors as JSON.",
            seed=891 + index,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        record = {"prompt": prompt, "response": asdict(response), "valid": False}
        try:
            answer = _parse_response_content(response.text)
            rows = answer["materials"]
            if not isinstance(rows, list) or {r["name"] for r in rows} != set(names):
                raise ValueError("Property material names do not match fixed state")
            record["answer"] = answer
            record["valid"] = True
        except (KeyError, ValueError, TypeError) as error:
            record["error"] = str(error)
        records.append(record)
        write_json(directory / f"query_{index}.json", record)
        print(
            f"Property query {index}: valid={record['valid']}, {response.seconds:.1f}s",
            flush=True,
        )
    for arm in states:
        for name in names:
            rows = [
                r
                for record in records
                if record["valid"]
                for r in record["answer"]["materials"]
                if r["name"] == name
            ]
            bound = states[arm]["materials"].setdefault(name, {})
            for key in ("density_kg_m3", "youngs_modulus_pa", "friction_coefficient"):
                values = [
                    r[key]
                    for r in rows
                    if type(r.get(key)) in (int, float)
                    and np.isfinite(r[key])
                    and r[key] > 0
                    and (
                        key != "friction_coefficient"
                        or isinstance(r.get("friction_contact_pair"), str)
                    )
                ]
                if values:
                    bound[key] = float(np.median(values))
                    bound[f"{key}_source"] = (
                        "separate shared VLM conditional material-prior query"
                    )
            bound["hardness_records"] = [
                {"value": r["hardness_value"], "scale": "Shore D"}
                for r in rows
                if name != "fabric"
                and type(r.get("hardness_value")) in (int, float)
                and 0 <= r["hardness_value"] <= 100
                and str(r.get("hardness_scale", "")).lower() == "shore d"
            ]
            bound["friction_contact_pair"] = (
                "clean dry stainless-steel slider / material, light contact, room temperature; finish unresolved"
            )
            bound["assumptions"] = [r.get("assumptions") for r in rows]
    write_json(directory / "records.json", records)
    write_json(
        directory / "seal.json",
        {
            "sha256": hashlib.sha256(
                (directory / "records.json").read_bytes()
            ).hexdigest(),
            "shared_between_structural_arms": True,
            "separate_from_18_call_structure_comparison": True,
        },
    )
    state_path = out / "consensus_states_with_properties.json"
    write_json(state_path, states)
    write_json(
        out / "active_state.json",
        {
            "path": state_path.name,
            "sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        },
    )


if __name__ == "__main__":
    main()
