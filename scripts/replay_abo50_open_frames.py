"""Route canopy-and-support assemblies to a local sheet and tube model."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from PIL import Image

from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo50_mass_hypotheses import (
    ROOT,
    VIEWS,
    LocalVLM,
    _parse_response_content,
    load_material_priors,
    positive,
)

PROMPT = """Estimate the construction dimensions of the assembled shelter in
these views. Describe only matter, excluding all air below the canopy. Group
identical pieces. The estimated overall longest dimension is {length} m. Return
one JSON with groups: each group contains name, material, kind, count and the
dimensions below. Material choices: {materials}.
Kinds: sheet (area_m2 per piece, thickness_mm); tube (length_m per piece,
diameter_mm, wall_mm); box (dimensions_mm, three positive numbers per piece;
use only for small compact fittings); rod (length_m per piece, diameter_mm;
use only when not hollow). Give positive values and integer counts. Include all
main visible material groups, including roof fabric, tubes and base fittings
where present. Thickness and diameter are in millimetres. Do not estimate mass.
"""
ENCLOSURE_PROMPT = """Describe the material-bearing components of this panel-built
object in the images. Its estimated longest dimension is {length} m. Exclude the
empty space inside or between panels. Return JSON with groups, each containing
name, material, kind, count and dimensions. Materials: {materials}.
Use sheet for panels (area_m2 per piece, thickness_mm; add solid_fraction between
0 and 1 for grilles or perforated panels). Use box (three dimensions_mm) only for
compact solid components. For a support use tube (length_m, diameter_mm,
wall_mm) or rod (length_m, diameter_mm). Include visible components and mark any
plausible hidden component in its name. Counts are positive integers. Do not
estimate mass."""


def is_open_canopy(row: dict) -> bool:
    names = " ".join(part["name"].lower() for part in row["parts"])
    return "canopy" in names and bool(
        re.search(r"\b(poles?|legs?|supports?|posts?)\b", names)
    )


def is_panel_enclosure(row: dict) -> bool:
    names = " ".join(part["name"].lower() for part in row["parts"])
    solid = sum(
        part["occupied_volume_m3"]
        for part in row["parts"]
        if part["construction"] == "solid"
    )
    return (
        bool(re.search(r"\b(grille|grill|vents?|drawers?|doors?|cabinet)\b", names))
        and solid > row["envelope_volume_m3"] / 4
    )


def integrate_groups(
    groups: list[dict], density: dict, length: float, *, panel_enclosure=False
) -> list[dict]:
    if not isinstance(groups, list) or not 2 <= len(groups) <= 12:
        raise ValueError("Expected a list of 2–12 material groups")
    kinds = {g["kind"] for g in groups}
    if "sheet" not in kinds:
        raise ValueError("At least one panel is required")
    if not panel_enclosure and not kinds.intersection({"tube", "rod"}):
        raise ValueError("A shelter needs sheet and support groups")
    assessed = []
    seen = set()
    for group in groups:
        identity = json.dumps(group, sort_keys=True)
        if identity in seen:
            continue
        seen.add(identity)
        group = dict(group)
        kind, count = group["kind"], group["count"]
        if isinstance(group.get("dimensions"), dict):
            for key, value in group["dimensions"].items():
                group.setdefault(key, value)
            if kind == "box" and "dimensions_mm" not in group:
                depth = "length_mm" if "length_mm" in group else "depth_mm"
                group["dimensions_mm"] = [
                    group[key] for key in (depth, "width_mm", "height_mm")
                ]
        elif "dimensions" in group:
            dims = group["dimensions"]
            expected = 2 if kind == "rod" else 3
            if not isinstance(dims, list) or len(dims) != expected:
                raise ValueError("Incorrect compact dimension list length")
            dims = [positive(value, "dimension in millimetres") for value in dims]
            if kind == "sheet":
                group.setdefault("area_m2", dims[0] * dims[1] * 1e-6)
                group.setdefault("thickness_mm", dims[2])
            elif kind in {"tube", "rod"}:
                group.setdefault("length_m", dims[0] * 1e-3)
                group.setdefault("diameter_mm", dims[1])
                if kind == "tube":
                    group.setdefault("wall_mm", dims[2])
            elif kind == "box":
                group.setdefault("dimensions_mm", dims)
        if type(count) is not int or not 1 <= count <= 120:
            raise ValueError("Invalid number of pieces")
        if kind == "sheet":
            area = positive(group["area_m2"], "sheet area")
            thickness = positive(group["thickness_mm"], "sheet thickness") * 1e-3
            if count * area > 4 * length**2 or thickness > (
                0.06 if panel_enclosure else 0.01
            ):
                raise ValueError(
                    "Sheet dimensions not consistent with a thin shelter panel"
                )
            fraction = positive(group.get("solid_fraction", 1), "panel solid fraction")
            if fraction > 1:
                raise ValueError("Panel solid fraction must not exceed 1")
            volume = area * thickness * fraction
        elif kind in {"tube", "rod"}:
            span = positive(group["length_m"], "support length")
            diameter = positive(group["diameter_mm"], "support diameter") * 1e-3
            ratio = 1 if panel_enclosure and kind == "rod" else 0.20
            if span > 2 * length or diameter > min(0.15, ratio * span):
                raise ValueError(
                    "Support dimensions not consistent with a slender part"
                )
            if kind == "tube":
                wall = positive(group["wall_mm"], "tube wall") * 1e-3
                if 2 * wall >= diameter:
                    raise ValueError("Tube wall leaves no interior")
                volume = math.pi * wall * (diameter - wall) * span
            else:
                volume = math.pi * (diameter / 2) ** 2 * span
        elif kind == "box":
            raw = group["dimensions_mm"]
            if len(raw) != 3:
                raise ValueError("A fitting needs three dimensions")
            dimensions = [positive(x, "fitting dimension") * 1e-3 for x in raw]
            max_fraction = 0.35 if panel_enclosure else 0.10
            if max(dimensions) > min(0.25, max_fraction * length):
                raise ValueError("Box is only for small compact fittings")
            volume = math.prod(dimensions)
        else:
            raise ValueError(f"Unknown component kind: {kind}")
        assessed.append(
            {
                **group,
                "volume_m3": volume * count,
                "mass_kg": volume * count * density[group["material"]],
            }
        )
    return assessed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("benchmark", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    source_mode = parser.add_mutually_exclusive_group(required=True)
    source_mode.add_argument("--model", type=Path)
    source_mode.add_argument("--raw-run", type=Path)
    parser.add_argument("--panel-enclosures", action="store_true")
    args = parser.parse_args()
    source = args.source / "predictions.json"
    seal = digest(source)
    if read_json(args.source / "prediction_seal.json")["sha256"] != seal:
        raise ValueError("Source prediction seal mismatch")
    protocol = read_json(args.source / "protocol.json")
    obs = args.benchmark / "observations"
    cohort_path = obs / "cohort.json"
    manifest = read_json(cohort_path)
    prior_path = ROOT / "configs/material_priors.yaml"
    rows = read_json(source)
    if (
        [r["scene_id"] for r in rows] != manifest["object_ids"]
        or digest(cohort_path) != protocol["cohort_sha256"]
        or digest(prior_path) != protocol["material_priors_sha256"]
    ):
        raise ValueError("Cohort or material prior differs from the sealed run")
    densities = {p.name: p.density_kg_m3.mean for p in load_material_priors(prior_path)}
    args.output.mkdir()
    model = (
        LocalVLM(args.model, max_tokens=1000, max_pixels=448**2, quantized=True)
        if args.model
        else None
    )
    template = ENCLOSURE_PROMPT if args.panel_enclosures else PROMPT
    route = is_panel_enclosure if args.panel_enclosures else is_open_canopy
    write_json(
        args.output / "protocol.json",
        {
            **protocol,
            "scope": "exploratory local-dimension assembly replay",
            "source_prediction_sha256": seal,
            "routing": route.__name__,
            "code_sha256": digest(Path(__file__)),
            "local_dimension_prompt": template,
            "compact_dimensions_unit": "mm",
            "archived_responses_replayed": args.raw_run is not None,
            "mass_labels_used_for_inference": False,
        },
    )
    for row in rows:
        if not route(row):
            continue
        scene = row["scene_id"]
        child = args.output / scene
        child.mkdir()
        images = []
        for index in VIEWS:
            relative = f"scenes/{scene}/images/{scene}_{index:02d}.png"
            path = obs / relative
            if digest(path) != manifest["files_sha256"][relative]:
                raise ValueError(f"Image differs for {scene}")
            with Image.open(path) as image:
                images.append(image.copy())
        length = positive(row["predicted_max_length_m"], "object length")
        prompt = template.format(length=length, materials=", ".join(densities))
        row["envelope_construction_mass_kg"] = row.pop("construction_mass_kg")
        for attempt in range(1, 3):
            if args.raw_run:
                raw = args.raw_run / scene / f"attempt_{attempt}.txt"
                if not raw.is_file():
                    break
                content = raw.read_text(encoding="utf-8")
            else:
                content = model.generate(
                    prompt, images, "Return a physical component inventory in JSON."
                ).text
            (child / f"attempt_{attempt}.txt").write_text(content, encoding="utf-8")
            try:
                groups = integrate_groups(
                    _parse_response_content(content)["groups"],
                    densities,
                    length,
                    panel_enclosure=args.panel_enclosures,
                )
            except (TypeError, ValueError, KeyError) as error:
                row["open_frame_error"] = str(error)
                prompt = f"{prompt}\nValidation error: {error}. Return a corrected inventory."
            else:
                row.pop("open_frame_error", None)
                row["open_frame_groups"] = groups
                row["accepted_response_sha256"] = digest(
                    child / f"attempt_{attempt}.txt"
                )
                row["construction_mass_kg"] = sum(g["mass_kg"] for g in groups)
                break
        write_json(child / "result.json", row)
        print(
            scene,
            row.get("construction_mass_kg"),
            row.get("open_frame_error"),
            flush=True,
        )
    result = args.output / "predictions.json"
    write_json(result, rows)
    write_json(args.output / "prediction_seal.json", {"sha256": digest(result)})


if __name__ == "__main__":
    main()
