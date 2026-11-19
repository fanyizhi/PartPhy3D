"""Replay mass hypotheses with winding repair or a catalogue length."""

from __future__ import annotations

import argparse
from pathlib import Path

import trimesh

from prepare_abo50_benchmark import digest, read_json, write_json
from run_abo50_mass_hypotheses import (
    ROOT,
    _parse_response_content,
    integrate,
    load_material_priors,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image-only-run", "reconstructions", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--repair-winding", action="store_true")
    args = parser.parse_args()
    if args.catalog is None and not args.repair_winding:
        parser.error("Choose catalogue scale, winding repair, or both")
    source = args.image_only_run / "predictions.json"
    source_seal = digest(source)
    if read_json(args.image_only_run / "prediction_seal.json")["sha256"] != source_seal:
        raise ValueError("Image-only prediction seal mismatch")
    initial = read_json(args.image_only_run / "protocol.json")
    catalog = read_json(args.catalog) if args.catalog else None
    if catalog is not None and (
        catalog["scope"] != "catalog_dimension_assisted_input_no_mass_labels"
        or catalog["mass_labels_included"] is not False
    ):
        raise ValueError("Expected a dimensions-only catalogue export")
    prior_path = ROOT / "configs/material_priors.yaml"
    if digest(prior_path) != initial["material_priors_sha256"]:
        raise ValueError("Material priors differ from sealed prediction protocol")
    densities = {p.name: p.density_kg_m3.mean for p in load_material_priors(prior_path)}
    rows = []
    for original in read_json(source):
        scene = original["scene_id"]
        row = {"scene_id": scene}
        if catalog is None and "direct_mass_kg" in original:
            row["direct_mass_kg"] = original["direct_mass_kg"]
        try:
            run = args.reconstructions / scene
            mesh_path = run / "camera_space.ply"
            mesh_hash = digest(mesh_path)
            if mesh_hash != read_json(run / "report.json")["mesh_sha256"]:
                raise ValueError("Mesh differs from reconstruction report")
            mesh = trimesh.load_mesh(mesh_path, process=False)
            row["winding_repaired"] = False
            if "parts" in original:
                if mesh_hash != original["mesh_sha256"]:
                    raise ValueError(
                        "Mesh differs from original construction prediction"
                    )
                payload = {
                    "max_length_m": original["predicted_max_length_m"],
                    "parts": original["parts"],
                }
            elif (
                args.repair_winding
                and mesh.is_watertight
                and not mesh.is_winding_consistent
            ):
                raw = args.image_only_run / scene / "construction.txt"
                payload = _parse_response_content(raw.read_text(encoding="utf-8"))
                row["raw_construction_sha256"] = digest(raw)
                trimesh.repair.fix_winding(mesh)
                row["winding_repaired"] = bool(mesh.is_volume)
            else:
                raise ValueError("No reusable construction hypothesis")
            if catalog is not None:
                dimensions = catalog["objects"][scene]
                if dimensions["status"] != "available":
                    raise ValueError("Catalogue length unavailable or ambiguous")
                payload["max_length_m"] = dimensions["max_dimension_m"]
            row.update(integrate(payload, mesh, densities))
            if catalog is not None:
                row["catalog_max_dimension_m"] = row.pop("predicted_max_length_m")
            row["mesh_sha256_before_winding_repair"] = mesh_hash
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            row["error"] = str(error)
        rows.append(row)
    args.output.mkdir()
    write_json(
        args.output / "protocol.json",
        {
            "scope": "replay of image-only construction",
            "cohort_sha256": initial["cohort_sha256"],
            "source_prediction_sha256": source_seal,
            "catalogue_sha256": digest(args.catalog) if args.catalog else None,
            "repair_winding": args.repair_winding,
            "material_priors_sha256": digest(prior_path),
            "mass_labels_used_for_inference": False,
        },
    )
    result = args.output / "predictions.json"
    write_json(result, rows)
    write_json(args.output / "prediction_seal.json", {"sha256": digest(result)})


if __name__ == "__main__":
    main()
