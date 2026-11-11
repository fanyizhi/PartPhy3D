"""Finite-cell physical integration and idealized isotropic mixture bounds."""

from __future__ import annotations

import numpy as np


def material_conditioned_shell(depth, pitch, thicknesses, choices, outer, core):
    """Average conditional shell/core fields without decoupling material and thickness.

    ``outer`` and ``core`` map each physical field to aligned Monte Carlo samples.
    The returned shell-volume samples use the same material choices as density.
    Thickness is the conditional nominal value; its within-material uncertainty
    is not sampled here. Artificial part boundaries must not enter ``depth``.
    """
    depth = np.asarray(depth, dtype=float)
    thicknesses = np.asarray(thicknesses, dtype=float)
    choices = np.asarray(choices)
    if (depth.ndim != 1 or not len(depth) or not np.isfinite(depth).all()
            or np.any(depth < 0) or not np.isfinite(pitch) or pitch <= 0
            or thicknesses.ndim != 1 or not np.isfinite(thicknesses).all()
            or np.any(thicknesses < 0) or choices.ndim != 1 or not len(choices)
            or choices.dtype.kind not in "iu" or np.any(choices < 0)
            or np.any(choices >= len(thicknesses)) or not outer
            or set(outer) != set(core)):
        raise ValueError("Invalid conditional shell geometry or material choices")
    outside = {key: np.asarray(values, dtype=float) for key, values in outer.items()}
    inside = {key: np.asarray(values, dtype=float) for key, values in core.items()}
    if any(values.shape != choices.shape or not np.isfinite(values).all()
           or np.any(values < 0) for values in (*outside.values(), *inside.values())):
        raise ValueError("Conditional fields require aligned nonnegative samples")
    means = {key: np.zeros(len(depth)) for key in outside}
    seconds = {key: np.zeros(len(depth)) for key in outside}
    shell_volumes = np.zeros(len(choices))
    for material in np.unique(choices):
        selected = choices == material
        fraction = selected.mean()
        shell = np.clip((thicknesses[material] - np.maximum(depth - pitch, 0)) / pitch, 0, 1)
        inner = 1 - shell
        shell_volumes[selected] = shell.sum() * pitch**3
        for key in means:
            a, b = outside[key][selected], inside[key][selected]
            means[key] += fraction * (shell * a.mean() + inner * b.mean())
            seconds[key] += fraction * (
                shell**2 * np.mean(a*a) + inner**2 * np.mean(b*b)
                + 2 * shell * inner * np.mean(a*b))
    return {"mean": means,
            "std": {key: np.sqrt(np.maximum(seconds[key] - means[key]**2, 0)) for key in means},
            "shell_volume_samples_m3": shell_volumes}


def integrate_density_cells(centers, density, pitch, part_ids=None):
    """Integrate disjoint, axis-aligned, constant-density cubic cells."""
    x = np.asarray(centers, dtype=float)
    rho = np.asarray(density, dtype=float)
    if (x.ndim != 2 or x.shape[1] != 3 or rho.shape != (len(x),)
            or not len(x) or not np.isfinite(x).all() or not np.isfinite(rho).all()
            or np.any(rho < 0) or not np.isfinite(pitch) or pitch <= 0):
        raise ValueError("Require finite (N,3) centers, nonnegative densities and positive pitch")
    weights = rho * pitch**3
    mass = float(weights.sum())
    if mass <= 0:
        raise ValueError("Positive mass is required")
    center = weights @ x / mass
    offsets = x - center
    q = np.einsum("i,ij,ik->jk", weights, offsets, offsets)
    inertia = np.trace(q) * np.eye(3) - q + np.eye(3) * mass * pitch**2 / 6
    result = {"mass_kg": mass, "center_of_mass_m": center.tolist(),
              "inertia_kg_m2": inertia.tolist()}
    if part_ids is not None:
        ids = np.asarray(part_ids)
        if ids.shape != (len(x),) or ids.dtype.kind not in "iu" or np.any(ids < 0):
            raise ValueError("Invalid part IDs")
        result["part_mass_kg"] = np.bincount(ids, weights=weights).tolist()
    return result


def isotropic_mixture_bounds(fractions, youngs, poisson):
    """Voigt/Reuss bulk/shear bounds; phase E=0 denotes ideal void.

    These are idealized mixture bounds, not predicted contact stiffness or a
    constitutive description of fabric, anisotropic timber or nonlinear foam.
    """
    a = np.asarray(fractions, dtype=float)
    e, nu = np.asarray(youngs, float), np.asarray(poisson, float)
    if (a.ndim != 2 or a.shape[1] != len(e) or nu.shape != e.shape
            or not np.isfinite(a).all() or not np.isfinite(e).all()
            or not np.isfinite(nu).all() or np.any(a < -1e-9)
            or not np.allclose(a.sum(axis=1), 1) or np.any(e < 0)
            or np.any(nu <= -1) or np.any(nu >= 0.5)):
        raise ValueError("Invalid phase fractions or isotropic phase properties")
    bulk = e / (3 * (1 - 2 * nu))
    shear = e / (2 * (1 + nu))
    kv, gv = a @ bulk, a @ shear
    solid = e > 0
    has_void = a[:, ~solid].sum(axis=1) > 1e-12
    kr = np.divide(1, a[:, solid] @ (1 / bulk[solid]),
                   out=np.zeros(len(a)), where=~has_void)
    gr = np.divide(1, a[:, solid] @ (1 / shear[solid]),
                   out=np.zeros(len(a)), where=~has_void)

    def to_young(k, g):
        return np.divide(9*k*g, 3*k+g, out=np.zeros_like(k), where=(3*k+g) > 0)

    return to_young(kr, gr), to_young(kv, gv)
