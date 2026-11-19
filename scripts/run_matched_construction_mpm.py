"""Compare declared solid, hollow and filled bodies with an identical exterior."""

import argparse
from pathlib import Path

import numpy as np
import trimesh

from prepare_abo50_benchmark import write_json
from run_abo50_mass_hypotheses import ROOT
from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.mesh_particles import Construction, sample_mesh_particles
from tactile3d.physics.mpm_validation import coarsen_material_points, run_mpm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    priors = load_material_priors(ROOT / "configs/material_priors.yaml")
    rho = {p.name: p.density_kg_m3.mean for p in priors}
    outside = trimesh.creation.box((0.20,) * 3)
    inside = trimesh.creation.box((0.16,) * 3)
    shell_volume = outside.volume - inside.volume
    cases = (
        (
            "generated_solid",
            Construction.from_evidence(surface="plastic", solid=True),
            None,
            outside.volume * rho["plastic"],
        ),
        (
            "generated_hollow",
            Construction.from_generated_partition("plastic"),
            inside,
            shell_volume * rho["plastic"],
        ),
        (
            "visual_hollow_hypothesis",
            Construction.from_visual_hypothesis("plastic", "hollow", thickness_m=0.02),
            None,
            shell_volume * rho["plastic"],
        ),
        (
            "generated_foam_core",
            Construction.from_generated_partition("plastic", "foam"),
            inside,
            shell_volume * rho["plastic"] + inside.volume * rho["foam"],
        ),
    )
    rows = []
    for name, layout, partition, exact in cases:
        points = sample_mesh_particles(
            outside,
            layout,
            priors,
            finest_m=0.001,
            bulk_m=0.020,
            poisson={"plastic": 0.30, "foam": 0.25},
            inner_mesh=partition,
        )
        if name == "generated_solid":
            factor = outside.volume / points["volume_m3"].sum()
            points["volume_m3"] *= factor
            points["mass_kg"] *= factor
        export = args.output / f"{name}_mpm_input.npz"
        np.savez_compressed(export, **coarsen_material_points(points, 0.012))
        with np.load(export, allow_pickle=False) as loaded:
            loaded_points = dict(loaded)
        result = run_mpm(
            loaded_points, dx=0.026, steps=80, strain_rate=10, timestep_s=5e-7
        )
        mass = float(loaded_points["mass_kg"].sum())
        rows.append(
            dict(
                case=name,
                mpm_particles=len(loaded_points["mass_kg"]),
                cavity_particles=int(
                    np.sum(np.abs(loaded_points["position_m"]).max(1) < 0.08)
                ),
                materials=loaded_points["material_names"].tolist(),
                analytic_mass_kg=float(exact),
                particle_mass_kg=mass,
                relative_error=float(mass / exact - 1),
                elastic_energy_j=result["elastic_energy_j"],
                min_J=float(result["jacobian"].min()),
                mass_transfer_residual_kg=result["max_p2g_mass_residual_kg"],
            )
        )
    write_json(
        args.output / "matched_constructions.json",
        dict(
            exterior_m=0.2,
            inner_m=0.16,
            timestep_s=5e-7,
            steps=80,
            grid_dx_m=0.026,
            strain_rate_s_inv=10,
            cases=rows,
        ),
    )


if __name__ == "__main__":
    main()
