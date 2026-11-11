"""Closest triangle-surface pairs, including edge and intersection candidates."""

from __future__ import annotations

import numpy as np
import trimesh


def closest_surface_pair(first: trimesh.Trimesh, second: trimesh.Trimesh) -> dict:
    """Compute a geometric surface minimum to floating-point precision.

    Vertex/face queries establish an upper bound. Triangle bounding boxes prune
    pairs, followed by segment/triangle intersections and segment/segment minima.
    This is a surface-distance query, not a contact-force or penetration solver.
    """
    if not len(first.faces) or not len(second.faces):
        raise ValueError("Two nonempty triangle surfaces are required")
    p, distance, _ = trimesh.proximity.closest_point(second, first.vertices)
    i = int(np.argmin(distance))
    best_a, best_b = first.vertices[i].copy(), p[i].copy()
    best = float(distance[i])
    p, distance, _ = trimesh.proximity.closest_point(first, second.vertices)
    i = int(np.argmin(distance))
    if distance[i] < best:
        best_a, best_b, best = (
            p[i].copy(),
            second.vertices[i].copy(),
            float(distance[i]),
        )
    ta, tb = first.triangles, second.triangles
    diagonal = max(
        float(np.linalg.norm(first.extents)),
        float(np.linalg.norm(second.extents)),
        1e-12,
    )
    tolerance = diagonal * 1e-12
    separation = np.maximum(
        0,
        np.maximum(
            ta.min(1)[:, None] - tb.max(1)[None], tb.min(1)[None] - ta.max(1)[:, None]
        ),
    )
    ai, bi = np.nonzero(np.linalg.norm(separation, axis=2) <= best + tolerance)
    a, b = ta[ai], tb[bi]
    # An edge may pierce a face with every vertex at a positive distance.
    for edges, triangles in ((a, b), (b, a)):
        e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        for edge in range(3):
            start = edges[:, edge]
            direction = edges[:, (edge + 1) % 3] - start
            h = np.cross(direction, e2)
            determinant = np.einsum("ij,ij->i", e1, h)
            valid = np.abs(determinant) > diagonal**3 * 1e-14
            inv = np.divide(
                1.0, determinant, out=np.zeros_like(determinant), where=valid
            )
            offset = start - triangles[:, 0]
            u = np.einsum("ij,ij->i", offset, h) * inv
            q = np.cross(offset, e1)
            v = np.einsum("ij,ij->i", direction, q) * inv
            t = np.einsum("ij,ij->i", e2, q) * inv
            hit = (
                valid
                & (u >= -1e-12)
                & (v >= -1e-12)
                & (u + v <= 1 + 1e-12)
                & (t >= -1e-12)
                & (t <= 1 + 1e-12)
            )
            if hit.any():
                k = int(np.flatnonzero(hit)[0])
                point = start[k] + t[k] * direction[k]
                return {
                    "point_first_m": point.tolist(),
                    "point_second_m": point.tolist(),
                    "distance_m": 0.0,
                    "triangle_pair_candidates": len(ai),
                    "method": "triangle intersection / boundary minimum",
                }
    for edge_a in range(3):
        for edge_b in range(3):
            p, q = a[:, edge_a], b[:, edge_b]
            u, v = a[:, (edge_a + 1) % 3] - p, b[:, (edge_b + 1) % 3] - q
            w = p - q
            aa = np.einsum("ij,ij->i", u, u)
            bb = np.einsum("ij,ij->i", u, v)
            cc = np.einsum("ij,ij->i", v, v)
            dd = np.einsum("ij,ij->i", u, w)
            ee = np.einsum("ij,ij->i", v, w)
            denominator = aa * cc - bb**2
            s = np.divide(
                bb * ee - cc * dd,
                denominator,
                out=np.full_like(aa, np.nan),
                where=np.abs(denominator) > diagonal**4 * 1e-16,
            )
            t = np.divide(
                aa * ee - bb * dd,
                denominator,
                out=np.full_like(aa, np.nan),
                where=np.abs(denominator) > diagonal**4 * 1e-16,
            )
            candidates = [
                (s, t),
                (np.zeros_like(s), np.clip(ee / np.maximum(cc, 1e-30), 0, 1)),
                (np.ones_like(s), np.clip((ee + bb) / np.maximum(cc, 1e-30), 0, 1)),
                (np.clip(-dd / np.maximum(aa, 1e-30), 0, 1), np.zeros_like(t)),
                (np.clip((bb - dd) / np.maximum(aa, 1e-30), 0, 1), np.ones_like(t)),
            ]
            for s, t in candidates:
                point_a, point_b = p + s[:, None] * u, q + t[:, None] * v
                distances = np.linalg.norm(point_a - point_b, axis=1)
                distances[
                    (s < 0) | (s > 1) | (t < 0) | (t > 1) | ~np.isfinite(distances)
                ] = np.inf
                if len(distances) and distances.min() < best:
                    k = int(np.argmin(distances))
                    best_a, best_b, best = point_a[k], point_b[k], float(distances[k])
    return {
        "point_first_m": best_a.tolist(),
        "point_second_m": best_b.tolist(),
        "distance_m": best,
        "triangle_pair_candidates": len(ai),
        "method": "vertex-face / edge-edge minimum with bounding-box pruning",
    }
