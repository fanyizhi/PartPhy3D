"""Evaluate particle centre of mass and inertia against exact cuboid integrals."""
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parents[1] / "paper_assets"
PARTICLES = BASE / "abo50_mass_stability_20261119/matched_mpm"
OUTPUT = BASE / "abo500_20261120/particle_moments.json"


def cuboid(dimensions, centre, density):
    """Raw zeroth, first, and second moments in the common object frame."""
    size, x = np.asarray(dimensions), np.asarray(centre)
    mass = float(np.prod(size) * density)
    return mass, mass * x, mass * (np.diag(size**2 / 12) + np.outer(x, x))


def inertia_at_com(mass, first, second):
    centred = second - np.outer(first, first) / mass
    return np.trace(centred) * np.eye(3) - centred


def run():
    centre = np.zeros(3)
    shell = [cuboid([.2]*3, centre, 1050), cuboid([.16]*3, centre, -1050)]
    core = cuboid([.16]*3, centre, 65)
    wood = cuboid([.02,.16,.16], [-.07,0,0], 650)
    foam = cuboid([.08,.16,.16], [-.02,0,0], 65)
    shapes = {"generated_solid": shell[:1], "generated_hollow": shell,
              "visual_hollow_hypothesis": shell, "generated_foam_core": shell + [core],
              "generated_frame_partial_infill": shell + [wood,foam]}
    original = json.loads((PARTICLES/"matched_constructions.json").read_bytes())
    expected_mass = {r["case"]: r["analytic_mass_kg"] for r in original["cases"]}
    result = {}
    for name, geometry in shapes.items():
        mass, first, second = (sum(parts) for parts in zip(*geometry))
        if not np.isclose(mass, expected_mass[name], rtol=1e-10):
            raise ValueError(f"Declared construction differs from the MPM benchmark: {name}")
        target = inertia_at_com(mass, first, second)
        with np.load(PARTICLES/f"{name}_mpm_input.npz", allow_pickle=False) as data:
            w, x = data["mass_kg"], data["position_m"]
            mean = np.average(x, weights=w, axis=0)
            raw = (x*w[:, None]).T @ x
            estimate = inertia_at_com(w.sum(), mean*w.sum(), raw)
        # Uniform reference is granted the analytic mass (oracle scalar control).
        uniform = mass * .2**2 / 6 * np.eye(3)
        result[name] = dict(target_mass_kg=mass, particle_mass_kg=float(w.sum()),
            target_com_m=(first/mass).tolist(), particle_com_m=mean.tolist(),
            particle_com_error_mm=float(np.linalg.norm(mean-first/mass)*1000),
            uniform_com_error_mm=float(np.linalg.norm(first/mass)*1000),
            particle_inertia_relative=float(np.linalg.norm(estimate-target)/np.linalg.norm(target)),
            uniform_inertia_relative=float(np.linalg.norm(uniform-target)/np.linalg.norm(target)))
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, row in result.items():
        print(name, *(round(row[k],5) for k in ("particle_com_error_mm","uniform_com_error_mm",
            "particle_inertia_relative","uniform_inertia_relative")))


if __name__ == "__main__":
    run()
