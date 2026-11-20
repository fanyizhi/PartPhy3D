"""Isolate cover geometry using analytic two-material test bodies."""

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh

from prepare_abo50_benchmark import write_json
from run_abo50_mass_hypotheses import ROOT
from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.mesh_particles import Construction, sample_mesh_particles
from tactile3d.physics.mpm_validation import coarsen_material_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    priors = load_material_priors(ROOT / "configs/material_priors.yaml")
    rho = {p.name: p.density_kg_m3.mean for p in priors}
    rows = []
    for thickness in (0.002, 0.008):
        extents = np.array([0.10, 0.08, 0.04])
        radius, height = 0.05, 0.04
        cases = (
            (
                "box",
                trimesh.creation.box(extents),
                trimesh.creation.box(extents - 2 * thickness),
                math.prod(extents),
                math.prod(extents - 2 * thickness),
            ),
            (
                "cylinder",
                trimesh.creation.cylinder(radius, height, sections=128),
                trimesh.creation.cylinder(
                    radius - thickness, height - 2 * thickness, sections=128
                ),
                math.pi * radius**2 * height,
                math.pi * (radius - thickness) ** 2 * (height - 2 * thickness),
            ),
        )
        for name, outer, inner, outer_exact, inner_exact in cases:
            truth = (
                rho["fabric"] * (outer_exact - inner_exact) + rho["foam"] * inner_exact
            )
            layout = Construction.from_generated_partition("fabric", "foam")
            points = sample_mesh_particles(
                outer,
                layout,
                priors,
                finest_m=0.0015,
                bulk_m=0.012,
                poisson={"fabric": 0.3, "foam": 0.25},
                inner_mesh=inner,
            )
            packed = coarsen_material_points(points, 0.008)
            estimated = float(packed["mass_kg"].sum())
            if not np.isclose(estimated, truth, rtol=6e-4):
                raise ValueError(
                    f"Control disagrees with independent analytic volume: {name}"
                )
            area_cover = min(outer.area * thickness, outer.volume)
            rows.append(
                dict(
                    shape=name,
                    thickness_m=thickness,
                    analytic_mass_kg=truth,
                    all_fabric_mass_kg=rho["fabric"] * outer_exact,
                    area_thickness_mass_kg=rho["fabric"] * area_cover
                    + rho["foam"] * (outer.volume - area_cover),
                    explicit_domain_particle_mass_kg=estimated,
                    relative_error=abs(estimated - truth) / truth,
                    particle_count=len(packed["mass_kg"]),
                )
            )
    write_json(
        args.output,
        dict(
            scope="Analytic geometry controls with specified material priors and inner interfaces; no ABO labels or visible thickness estimation",
            cases=rows,
        ),
    )


if __name__ == "__main__":
    main()
