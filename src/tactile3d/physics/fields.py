"""Finite-cell physical integration and idealized isotropic mixture bounds."""

from __future__ import annotations

import numpy as np


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
