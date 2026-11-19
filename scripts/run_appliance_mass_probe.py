"""Test RGB-only functional disambiguation and hidden assembly mass."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo50_mass_hypotheses import ROOT, SYSTEM, VIEWS
from run_component_mass_stability import integrate
from tactile3d.physics.materials import load_material_priors
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.vlm_interior import _parse_response_content

PROMPT = """Identify the device function from the three photographs. Distinguish
window air conditioner, portable air conditioner, compressor dehumidifier,
filter air purifier, bare computer case, and ordinary electric fan. Compare
the front controls and outlet with the rear intake, coils, exhaust and ports.
If the evidence cannot distinguish functions, return undecided.
Return only a compact JSON object with class, cue, longest_cm, groups. class
must be one of cooling, dehumidifier, purifier, case, fan, undecided. Only for
cooling or dehumidifier fill groups, covering shell, chassis, heat exchangers,
compressor and fan assemblies as applicable. Give dimensions in cm, not mass.
Each group is [name,material,shape,count,[length,width,height],fill,evidence].
Use only block and sheet; sheet height is wall thickness. For bulky assemblies
estimate local bounding volume and the solid material fraction in (0,1],
excluding air. Use sheet for panels and block with fill for coils and motors.
Materials: steel, aluminum, plastic, rubber, copper. evidence is visible or
inferred. Put no dimensions outside this compact format. If class is other
than cooling or dehumidifier use an empty groups list."""
CLASSES = {"cooling", "dehumidifier", "purifier", "case", "fan", "undecided"}


def run(args):
    cohort_path = args.observations / "cohort.json"
    cohort = read_json(cohort_path)
    if not set(args.scenes) <= set(cohort["object_ids"]):
        raise ValueError("Probe objects must come from the specified cohort")
    model = LocalVLM(
        args.model, max_tokens=620, max_pixels=448**2, quantized=not args.bf16
    )
    priors = ROOT / "configs/material_priors.yaml"
    densities = {p.name: p.density_kg_m3.mean for p in load_material_priors(priors)}
    densities["copper"] = 8960.0
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        dict(
            object_ids=args.scenes,
            cohort_sha256=digest(cohort_path),
            code_sha256=digest(Path(__file__)),
            model=model.provenance,
            prompt=PROMPT,
            views=list(VIEWS),
            priors_sha256=digest(priors),
            copper_kg_m3=densities["copper"],
            quantized=not args.bf16,
            catalogue_and_mass_labels_used=False,
        ),
    )
    rows = []
    for scene in args.scenes:
        images = []
        for view in VIEWS:
            path = f"scenes/{scene}/images/{scene}_{view:02d}.png"
            original = args.observations / path
            if digest(original) != cohort["files_sha256"][path]:
                raise ValueError(f"Input hash differs: {path}")
            with Image.open(original) as image:
                images.append(image.copy())
        response = model.generate(PROMPT, images, SYSTEM)
        (args.output / f"{scene}.txt").write_text(response.text, encoding="utf-8")
        row = {"scene_id": scene, "peak_gpu_bytes": response.peak_gpu_bytes}
        try:
            parsed = _parse_response_content(response.text)
            cls = parsed["class"]
            if cls not in CLASSES:
                raise ValueError("Unexpected appliance function")
            row.update(device_class=cls, cue=parsed["cue"])
            if cls in {"cooling", "dehumidifier"}:
                groups = integrate(parsed, densities)
                row.update(
                    parts=groups,
                    mass_kg=sum(p["mass_kg"] for p in groups),
                    visible_mass_kg=sum(
                        p["mass_kg"] for p in groups if p["evidence"] == "visible"
                    ),
                )
        except (AttributeError, KeyError, TypeError, ValueError) as error:
            row["error"] = str(error)
        rows.append(row)
        print(
            scene,
            row.get("device_class"),
            row.get("mass_kg"),
            row.get("error"),
            flush=True,
        )
    result = args.output / "predictions.json"
    write_json(result, rows)
    write_json(args.output / "prediction_seal.json", {"sha256": digest(result)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenes", nargs="+", required=True)
    parser.add_argument("--bf16", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
