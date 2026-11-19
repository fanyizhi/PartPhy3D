"""Replay a sealed appliance proposal with explicit material-bearing domains."""

from __future__ import annotations

import argparse
import math
from itertools import permutations
from pathlib import Path

from prepare_abo50_benchmark import digest, write_json
from run_abo50_mass_hypotheses import ROOT
from run_component_mass_stability import sealed_rows
from tactile3d.physics.materials import load_material_priors

PANEL_THICKNESS_M = {"steel": 0.0008, "aluminum": 0.001, "plastic": 0.0025}
FIN_THICKNESS_M = 0.00012
FIN_PITCH_M = 0.0016
COIL_DEPTH_LIMIT_M = 0.060
COPPER_TUBE_VOLUME_FRACTION = 0.03
COPPER_DENSITY_KG_M3 = 8960.0
FACTORS = ("geometry", "occupancy", "density")


def attribute_product_change(old, new):
    """Average incremental mass changes over the six factor orders."""
    changes = dict.fromkeys(FACTORS, 0.0)
    orders = tuple(permutations(range(len(FACTORS))))
    for order in orders:
        current = list(old)
        for i in order:
            before = math.prod(current)
            current[i] = new[i]
            changes[FACTORS[i]] += (math.prod(current) - before) / len(orders)
    if not math.isclose(
        math.fsum(changes.values()),
        math.prod(new) - math.prod(old),
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("Material mass attribution does not close")
    return changes


def allocate(parts, density):
    corrected = []
    for part in parts:
        name = part["name"].casefold()
        if not any(term in name for term in ("shell", "chassis", "heat exchanger")):
            corrected.append(part)
            continue
        a, b, c = sorted(v / 100 for v in part["dims_cm"])
        count = part["count"]
        material = part["material"]
        if "shell" in name:
            reference = (
                2 * (a * b + a * c + b * c) * PANEL_THICKNESS_M[material] * count
            )
            domains = {material: reference * 0.75}
        elif "chassis" in name:
            reference = b * c * PANEL_THICKNESS_M[material] * 1.6 * count
            domains = {material: reference}
        else:
            reference = b * c * min(a, COIL_DEPTH_LIMIT_M) * count
            domains = {
                "aluminum": reference * FIN_THICKNESS_M / FIN_PITCH_M,
                "copper": reference * COPPER_TUBE_VOLUME_FRACTION,
            }
        volume = sum(domains.values())
        mass = sum(density[k] * v for k, v in domains.items())
        old = (
            part["volume_m3"] / part["occupied_fraction"],
            part["occupied_fraction"],
            density[material],
        )
        new = (reference, volume / reference, mass / volume)
        if not math.isclose(math.prod(old), part["mass_kg"], rel_tol=1e-12):
            raise ValueError(
                "Source part does not obey its sealed material mass equation"
            )
        corrected.append(
            part
            | {
                "material_volumes_m3": domains,
                "volume_m3": volume,
                "mass_kg": mass,
                "attribution": {
                    "before": dict(zip(FACTORS, old)),
                    "after": dict(zip(FACTORS, new)),
                    "delta_kg": attribute_product_change(old, new),
                },
            }
        )
    return corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-invalid", action="store_true")
    args = parser.parse_args()
    records = sealed_rows(args.run)
    priors = ROOT / "configs/material_priors.yaml"
    density = {p.name: p.density_kg_m3.mean for p in load_material_priors(priors)}
    density["copper"] = COPPER_DENSITY_KG_M3
    corrected = []
    for row in records.values():
        if "parts" in row:
            try:
                parts = allocate(row["parts"], density)
                if args.skip_invalid and not any("attribution" in p for p in parts):
                    raise ValueError("No supported physical role")
            except (KeyError, TypeError, ValueError) as error:
                if not args.skip_invalid:
                    raise
                corrected.append(row | {"role_replay_error": str(error)})
                continue
            total = sum(r["mass_kg"] for r in parts)
            attributed = {
                factor: math.fsum(
                    p["attribution"]["delta_kg"][factor]
                    for p in parts
                    if "attribution" in p
                )
                for factor in FACTORS
            }
            residual = total - row["mass_kg"] - math.fsum(attributed.values())
            if abs(residual) > 1e-10:
                raise ValueError("Object-level mass attribution does not close")
            row = row | {
                "role_parts": parts,
                "role_mass_kg": total,
                "attribution_delta_kg": attributed,
                "attribution_residual_kg": residual,
            }
        corrected.append(row)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output / "protocol.json",
        {
            "source_sha256": digest(args.run / "predictions.json"),
            "code_sha256": digest(Path(__file__)),
            "priors_sha256": digest(priors),
            "panel_thickness_m": PANEL_THICKNESS_M,
            "fin_thickness_m": FIN_THICKNESS_M,
            "fin_pitch_m": FIN_PITCH_M,
            "coil_depth_limit_m": COIL_DEPTH_LIMIT_M,
            "copper_tube_volume_fraction": COPPER_TUBE_VOLUME_FRACTION,
            "copper_density_kg_m3": COPPER_DENSITY_KG_M3,
            "scope": "Exploratory role constraints after inspecting the raw six-case probe; nominal engineering assumptions",
            "mass_labels_read_by_replay": False,
        },
    )
    result = args.output / "predictions.json"
    write_json(result, corrected)
    write_json(args.output / "prediction_seal.json", {"sha256": digest(result)})


if __name__ == "__main__":
    main()
