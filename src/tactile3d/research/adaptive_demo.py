"""已知解析场上的构造预算实验；不冒充InstantMesh提速结果。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from tactile3d.mesh.adaptive_construction import Array, refine_surface
from tactile3d.physics.part_io import write_json
from tactile3d.research.workflow import write_csv

DEFAULT_LEVELS = 3
SPARSE_FRACTION = 0.25
REFERENCE_SUBDIVISIONS = 6


def radius(directions: Array) -> Array:
    theta = np.arctan2(directions[:, 1], directions[:, 0])
    north = np.exp(-np.sum((directions - [0, 0, 1]) ** 2, axis=1) / 0.10)
    return (
        1 + 0.18 * north + 0.05 * np.sin(6 * theta) * (1 - directions[:, 2] ** 2) ** 2
    )


def project(points: Array) -> Array:
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms <= 0):
        raise ValueError("径向场不能投影原点")
    directions = points / norms[:, None]
    return directions * radius(directions)[:, None]


def importance(points: Array) -> Array:
    direction = points / np.linalg.norm(points, axis=1, keepdims=True)
    return np.exp(-np.sum((direction - [0, 0, 1]) ** 2, axis=1) / 0.3)


def plot_adaptive(output: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), layout="constrained")
    colors = {"uniform": "#777777", "task_adaptive": "#0072B2"}
    for method, color in colors.items():
        values = [row for row in rows if row["method"] == method]
        for axis, key, label in zip(
            axes,
            (
                "mean_radial_residual_m",
                "task_radial_residual_m",
                "relative_volume_error",
            ),
            (
                "Mean radial residual (m)",
                "Task region radial residual (m)",
                "Relative volume error",
            ),
            strict=True,
        ):
            axis.plot(
                [row["faces"] for row in values],
                [row[key] for row in values],
                "o-",
                color=color,
                label=method,
            )
            axis.set(
                xlabel="Constructed triangles", ylabel=label, xscale="log", yscale="log"
            )
            axis.spines[["top", "right"]].set_visible(False)
            axis.legend(fontsize=8)
    fig.savefig(output / "adaptive_construction.svg")
    fig.savefig(output / "adaptive_construction.png", dpi=240)
    plt.close(fig)


def run_adaptive(
    output: Path, levels: int = DEFAULT_LEVELS, figures: bool = True
) -> list[dict[str, Any]]:
    if not 1 <= levels <= 4:
        raise ValueError("演示轮数需要在1到4之间")
    output.mkdir(parents=True, exist_ok=False)
    reference = trimesh.creation.icosphere(subdivisions=REFERENCE_SUBDIVISIONS)
    reference.vertices = project(reference.vertices)
    reference_volume = reference.volume
    initial = trimesh.creation.icosphere(subdivisions=1)
    initial.vertices = project(initial.vertices)
    rows = []
    for method, fraction in (("uniform", 1.0), ("task_adaptive", SPARSE_FRACTION)):
        current, queries = initial.copy(), len(initial.vertices)
        start = time.perf_counter()
        for level in range(1, levels + 1):
            current, report = refine_surface(current, project, importance, fraction)
            queries += report["field_query_points"]
            center = current.triangles_center
            residual = np.abs(
                np.linalg.norm(center, axis=1) - np.linalg.norm(project(center), axis=1)
            )
            task = center[:, 2] / np.linalg.norm(center, axis=1) > 0.65
            rows.append(
                {
                    "method": method,
                    "level": level,
                    "faces": len(current.faces),
                    "query_points": queries,
                    "mean_radial_residual_m": float(
                        np.average(residual, weights=current.area_faces)
                    ),
                    "task_radial_residual_m": float(
                        np.average(residual[task], weights=current.area_faces[task])
                    ),
                    "relative_volume_error": abs(current.volume / reference_volume - 1),
                    "construction_seconds": time.perf_counter() - start,
                    "watertight": bool(current.is_watertight),
                }
            )
            current.export(output / f"{method}_{level}.ply")
    write_csv(output / "adaptive_metrics.csv", rows)
    write_json(
        output / "protocol.json",
        {
            "source": "known_analytic_radial_field",
            "upstream_inference_measured": False,
            "reference_subdivisions": REFERENCE_SUBDIVISIONS,
            "metric": "area_weighted_radial_residual_not_Chamfer_or_signed_distance",
            "task_region": "direction_z_gt_0.65",
            "sparse_fraction": SPARSE_FRACTION,
            "standard_operation": "conforming_red_green_edge_refinement",
            "note": "Not an InstantMesh adaptive decoder or a novelty claim",
        },
    )
    if figures:
        plot_adaptive(output, rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/adaptive"))
    parser.add_argument("--levels", type=int, default=DEFAULT_LEVELS)
    args = parser.parse_args()
    print(json.dumps(run_adaptive(args.output, args.levels), indent=2))


if __name__ == "__main__":
    main()
