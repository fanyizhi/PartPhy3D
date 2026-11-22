"""Small CPU 3D MLS-MPM check for exported material points (SI units)."""

from itertools import product

import numpy as np


def coarsen_material_points(data: dict, pitch_m: float) -> dict:
    """Preserve per-material mass/volume, choosing representatives from the original points."""
    positions = data["position_m"]
    cells = np.floor((positions - positions.min(0)) / pitch_m).astype(np.int64)
    groups = np.column_stack((cells, data["material_id"]))
    _, inverse = np.unique(groups, axis=0, return_inverse=True)
    volume = np.bincount(inverse, weights=data["volume_m3"])
    centroid = np.column_stack(
        [
            np.bincount(inverse, weights=coordinate * data["volume_m3"]) / volume
            for coordinate in positions.T
        ]
    )
    distance = np.sum((positions - centroid[inverse]) ** 2, axis=1)
    order = np.lexsort((distance, inverse))
    first = np.r_[True, np.diff(inverse[order]) != 0]
    representatives = order[first]
    result = {
        name: values[representatives]
        for name, values in data.items()
        if name != "material_names"
    }
    result["material_names"] = data["material_names"]
    result["volume_m3"] = volume
    result["mass_kg"] = np.bincount(inverse, weights=data["mass_kg"])
    return result


def run_mpm(
    data: dict,
    *,
    dx: float,
    steps: int,
    strain_rate: float = 0.0,
    timestep_s: float | None = None,
    on_step=None,
    floor_z_m: float | None = None,
    initial_vertical_speed: float = 0.,
) -> dict:
    """Advance compressible neo-Hookean points in free space with gravity.

    A compressive initial velocity gradient exercises the constitutive update.
    Quadratic B-spline MLS/APIC transfers use a single perfectly bonded grid.
    """
    if not (np.isfinite(dx) and dx > 0 and steps > 0 and np.isfinite(strain_rate)):
        raise ValueError("Invalid MPM grid or loading settings")
    x = np.array(data["position_m"], dtype=float, copy=True)
    origin = x.min(0) - 4 * dx
    shape = tuple(np.ceil((x.max(0) - origin) / dx).astype(int) + 5)
    mass = data["mass_kg"]
    volume = data["volume_m3"]
    youngs, nu = data["youngs_pa"], data["poisson"]
    mu = youngs / (2 * (1 + nu))
    lam = youngs * nu / ((1 + nu) * (1 - 2 * nu))
    gravity = np.array([0.0, 0.0, -9.81])
    v = np.zeros_like(x)
    v[:, 2] = -strain_rate * (x[:, 2] - x[:, 2].mean())
    v[:, 2] += initial_vertical_speed
    c = np.zeros((len(x), 3, 3))
    c[:, 2, 2] = -strain_rate
    dt = (
        0.18
        * dx
        / (
            np.sqrt((lam + 2 * mu) * volume / mass).max()
            + np.linalg.norm(v, axis=1).max()
        )
    )
    if timestep_s is not None:
        if not np.isfinite(timestep_s) or not 0 < timestep_s <= dt:
            raise ValueError("Requested timestep exceeds the solver stability bound")
        dt = timestep_s
    eye = np.eye(3)
    deformation = np.broadcast_to(eye, c.shape).copy()
    offsets = np.asarray(list(product(range(3), repeat=3)))
    initial = x.copy()
    initial_velocity = v.copy()
    max_mass_residual = 0.0
    max_momentum_residual = 0.0
    floor_nodes = (origin[2] + np.arange(np.prod(shape)) % shape[2] * dx <= floor_z_m) if floor_z_m is not None else None
    for step in range(1, steps + 1):
        cells = (x - origin) / dx
        base = np.floor(cells - 0.5).astype(int)
        nodes = base[:, None, :] + offsets
        if np.any(nodes < 0) or np.any(nodes >= shape):
            raise ValueError("MPM particles left the padded grid")
        indices = np.ravel_multi_index(tuple(nodes.transpose(2, 0, 1)), shape).ravel()
        local = cells - base
        weights = np.stack(
            (
                0.5 * (1.5 - local) ** 2,
                0.75 - (local - 1) ** 2,
                0.5 * (local - 0.5) ** 2,
            ),
            axis=1,
        )
        w = np.prod(weights[:, offsets, np.arange(3)], axis=-1)
        relative = (offsets - local[:, None, :]) * dx
        jacobian = np.linalg.det(deformation)
        if np.any(jacobian <= 0) or not np.isfinite(jacobian).all():
            raise ValueError("MPM produced inverted or non-finite deformation")
        stress = (
            mu[:, None, None] * (deformation @ deformation.transpose(0, 2, 1) - eye)
            + (lam * np.log(jacobian))[:, None, None] * eye
        )
        affine = (
            mass[:, None, None] * c - (4 * dt / dx**2) * volume[:, None, None] * stress
        )
        p2g_momentum = mass[:, None, None] * v[:, None, :] + np.einsum(
            "pij,pnj->pni", affine, relative
        )
        grid_mass = np.bincount(
            indices, weights=(mass[:, None] * w).ravel(), minlength=np.prod(shape)
        )
        grid_momentum = np.column_stack(
            [
                np.bincount(
                    indices, weights=(w * component).ravel(), minlength=len(grid_mass)
                )
                for component in p2g_momentum.transpose(2, 0, 1)
            ]
        )
        max_mass_residual = max(max_mass_residual, abs(grid_mass.sum() - mass.sum()))
        occupied = grid_mass > 0
        grid_momentum[occupied] /= grid_mass[occupied, None]
        grid_momentum[occupied] += dt * gravity
        if floor_nodes is not None:
            # Frictionless contact on background nodes, with no tensile reaction.
            grid_momentum[floor_nodes, 2] = np.maximum(0., grid_momentum[floor_nodes, 2])
        weighted_velocity = w[:, :, None] * grid_momentum[indices].reshape(
            relative.shape
        )
        next_velocity = weighted_velocity.sum(1)
        momentum_error = np.sum(
            mass[:, None] * (next_velocity - v - dt * gravity), axis=0
        )
        max_momentum_residual = max(
            max_momentum_residual, np.linalg.norm(momentum_error)
        )
        v = next_velocity
        c = (4 / dx**2) * np.einsum("pni,pnj->pij", weighted_velocity, relative)
        deformation = (eye + dt * c) @ deformation
        x += dt * v
        if on_step is not None:
            on_step(step, step * dt, x.copy())
    jacobian = np.linalg.det(deformation)
    if not (
        np.isfinite(x).all() and np.isfinite(jacobian).all() and np.all(jacobian > 0)
    ):
        raise ValueError("Final MPM state is not physically admissible")
    free_motion = (
        steps * dt * initial_velocity + (steps * (steps + 1) / 2) * dt**2 * gravity
    )
    log_j = np.log(jacobian)
    energy_density = (
        mu / 2 * (np.sum(deformation**2, axis=(1, 2)) - 3)
        - mu * log_j
        + lam / 2 * log_j**2
    )
    return dict(
        position_m=x,
        jacobian=jacobian,
        displacement_m=x - initial,
        dt_s=dt,
        duration_s=steps * dt,
        elastic_energy_j=float(volume @ energy_density),
        max_p2g_mass_residual_kg=float(max_mass_residual),
        max_step_momentum_residual_kg_m_s=float(max_momentum_residual),
        max_deviation_from_ballistic_m=float(
            np.linalg.norm(x - initial - free_motion, axis=1).max()
        ),
    )
