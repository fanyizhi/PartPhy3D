"""Compare direct visual mass and construction-conditioned mesh integration."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import trimesh
from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.physics.materials import load_material_priors
from tactile3d.vision.local_vlm import LocalVLM
from tactile3d.vision.vlm_interior import _parse_response_content

VIEWS = (0, 8, 20)
SYSTEM = (
    "You infer physical hypotheses about one empty, unpackaged household object. "
    "Return a single JSON object and distinguish visible evidence from assumptions."
)
DIRECT = "Estimate its mass in kilograms from these views. Return JSON with mass_kg."
CONSTRUCTION = """Estimate its maximum physical dimension in meters.
List one to six physical components, grouping interchangeable repeated parts.
For each, select a material from {materials}. Allocate fractions of the total outer
mesh-enclosed volume and of the exterior surface area to the component. Include the
potential void within each shell in its outer volume share. Use solid for filled
components, shell for thin walls around empty volumes, and partial for structures
with a separate, unknown material occupancy fraction. A filled foam component is
solid because the foam prior already describes its bulk density. For shell give
typical wall thickness in mm. For partial give occupied fraction between 0 and 1.
Return JSON with max_length_m and parts. Each part has name, evidence, material,
volume_share, area_share, construction, and thickness_mm or occupied_fraction
where applicable. Estimate dimensions and construction without estimating mass.
"""


def positive(value, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def shares(parts: list[dict], field: str) -> list[float]:
    raw = [float(part[field]) for part in parts]
    if any(not math.isfinite(item) or not 0 <= item <= 1 for item in raw):
        raise ValueError(f"Invalid {field}")
    total = positive(sum(raw), field)
    return [item / total for item in raw]


def integrate(payload: dict, mesh: trimesh.Trimesh, densities: dict) -> dict:
    """Use a fixed explicit envelope. Shell thickness gives a first-order proxy."""
    if not mesh.is_volume:
        raise ValueError(
            "Envelope must have consistent winding and positive closed volume"
        )
    length = positive(payload["max_length_m"], "max_length_m")
    scale = length / positive(mesh.extents.max(), "mesh extent")
    area, volume = float(mesh.area * scale**2), float(mesh.volume * scale**3)
    parts = payload["parts"]
    if not isinstance(parts, list) or not 1 <= len(parts) <= 6:
        raise ValueError("Expected one to six component hypotheses")
    rows = []
    for part, v, a in zip(
        parts, shares(parts, "volume_share"), shares(parts, "area_share")
    ):
        density = densities[part["material"]]
        outer = v * volume
        kind = part["construction"]
        if kind == "solid":
            occupied = outer
        elif kind == "shell":
            thickness = positive(part["thickness_mm"], "shell thickness") / 1000
            occupied = min(outer, a * area * thickness)
        elif kind == "partial":
            fraction = positive(part["occupied_fraction"], "occupied_fraction")
            if fraction > 1:
                raise ValueError("occupied_fraction must not exceed one")
            occupied = outer * fraction
        else:
            raise ValueError(f"Unsupported construction: {kind}")
        rows.append(
            {
                **part,
                "density_kg_m3": density,
                "outer_volume_m3": outer,
                "occupied_volume_m3": occupied,
                "mass_kg": density * occupied,
                "all_solid_mass_kg": density * outer,
            }
        )
    return {
        "area_m2": area,
        "envelope_volume_m3": volume,
        "predicted_max_length_m": length,
        "parts": rows,
        "construction_mass_kg": positive(
            sum(r["mass_kg"] for r in rows), "construction mass"
        ),
        "all_solid_mass_kg": sum(r["all_solid_mass_kg"] for r in rows),
    }


def infer(args: argparse.Namespace) -> None:
    manifest = read_json(args.observations / "cohort.json")
    cases = manifest["object_ids"]
    prior_path = ROOT / "configs/material_priors.yaml"
    densities = {p.name: p.density_kg_m3.mean for p in load_material_priors(prior_path)}
    args.output.mkdir()
    model = LocalVLM(args.model, max_tokens=1200, max_pixels=448**2, quantized=True)
    prompts = {
        "direct": DIRECT,
        "construction": CONSTRUCTION.format(materials=", ".join(densities)),
    }
    write_json(
        args.output / "protocol.json",
        {
            "scope": "ABO-50 mass prototype with conditional component volume allocation",
            "cohort_sha256": digest(args.observations / "cohort.json"),
            "code_sha256": digest(Path(__file__)),
            "material_priors_sha256": digest(prior_path),
            "vlm_view_indices": list(VIEWS),
            "prompts": prompts,
            "model": model.provenance,
            "mass_labels_used_for_inference": False,
            "scale_source": "VLM estimate from observations, no catalogue dimensions",
        },
    )
    rows = []
    for scene in cases:
        row = {"scene_id": scene, "errors": {}}
        child = args.output / scene
        child.mkdir()
        images = []
        for index in VIEWS:
            path = (
                args.observations
                / "scenes"
                / scene
                / "images"
                / f"{scene}_{index:02d}.png"
            )
            with Image.open(path) as source:
                images.append(source.copy())
        for name, prompt in prompts.items():
            try:
                response = model.generate(prompt, images, SYSTEM)
                (child / f"{name}.txt").write_text(response.text, encoding="utf-8")
                row[f"{name}_vlm_seconds"] = response.seconds
                parsed = _parse_response_content(response.text)
                if name == "direct":
                    row["direct_mass_kg"] = positive(parsed["mass_kg"], "direct mass")
                else:
                    run = args.reconstructions / scene
                    mesh_path = run / "camera_space.ply"
                    if (
                        digest(mesh_path)
                        != read_json(run / "report.json")["mesh_sha256"]
                    ):
                        raise ValueError("Mesh differs from reconstruction provenance")
                    row.update(
                        integrate(
                            parsed,
                            trimesh.load_mesh(mesh_path, process=False),
                            densities,
                        )
                    )
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
                row["errors"][name] = str(error)
        rows.append(row)
        write_json(child / "result.json", row)
        print(scene, sorted(row["errors"]), flush=True)
    result = args.output / "predictions.json"
    write_json(result, rows)
    write_json(args.output / "prediction_seal.json", {"sha256": digest(result)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--reconstructions", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    infer(parser.parse_args())


if __name__ == "__main__":
    main()
