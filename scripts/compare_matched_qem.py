"""Match global-QEM baselines to each component method's achieved face count."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import trimesh
from run_reference_mesh_study import closed_properties, load_prepared

from tactile3d.mesh.task_lod import simplify_components, simplify_with_lineage
from tactile3d.physics.part_io import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    out = args.source
    mesh, labels, pieces, _texture = load_prepared(out)
    edit = json.loads((out / "edit.json").read_text())
    reference = edit["before"]
    original = json.loads((out / "lod" / "metrics.json").read_text())
    points, _ = trimesh.sample.sample_surface(mesh, 12000, seed=711)
    results = []
    for row in original:
        if row["method"] == "global_qem":
            continue
        bulk_density = np.array(
            [
                edit["before_density_kg_m3"] if i in edit["wood_components"] else np.nan
                for i in range(len(pieces))
            ]
        )

        def global_run(row=row):
            return simplify_with_lineage(mesh, labels, row["actual_faces"])

        def component_run(row=row, bulk_density=bulk_density):
            return simplify_components(
                pieces,
                row["target_faces"],
                bulk_density if row["method"] == "sensitivity_qem" else None,
            )

        global_run()
        component_run()
        times, component_times = [], []
        result = None
        for repetition in range(15):
            order = [(global_run, times), (component_run, component_times)]
            if repetition % 2:
                order.reverse()
            for operation, samples in order:
                start = perf_counter()
                candidate = operation()
                samples.append(perf_counter() - start)
                if operation is global_run:
                    result = candidate
        row = {
            **row,
            "median_seconds": float(np.median(component_times)),
            "seconds_samples": component_times,
            "timing_iqr_seconds": np.quantile(component_times, [0.25, 0.75]).tolist(),
        }
        reduced = result.mesh
        subset = [
            (
                reduced.submesh(
                    [np.flatnonzero(result.face_components == i)],
                    append=True,
                    repair=False,
                )
                if np.any(result.face_components == i)
                else trimesh.Trimesh()
            )
            for i in range(len(pieces))
        ]
        physics = closed_properties(
            subset, edit["wood_components"], edit["before_density_kg_m3"]
        )
        _, distances, _ = trimesh.proximity.closest_point(reduced, points)
        errors = None
        if physics is not None:
            errors = {
                "mass_relative": abs(physics["mass_kg"] / reference["mass_kg"] - 1),
                "com_error_mm": float(
                    np.linalg.norm(
                        np.array(physics["center_of_mass_m"])
                        - reference["center_of_mass_m"]
                    )
                    * 1000
                ),
                "inertia_relative": float(
                    np.linalg.norm(
                        np.array(physics["inertia_kg_m2"]) - reference["inertia_kg_m2"]
                    )
                    / np.linalg.norm(reference["inertia_kg_m2"])
                ),
            }
        baseline = {
            "method": "matched_global_qem",
            "actual_faces": len(reduced.faces),
            "median_seconds": float(np.median(times)),
            "seconds_samples": times,
            "timing_iqr_seconds": np.quantile(times, [0.25, 0.75]).tolist(),
            "surface_p95_mm": float(np.quantile(distances, 0.95) * 1000),
            "surface_mean_mm": float(distances.mean() * 1000),
            "wood_errors": errors,
        }
        pair = {
            "component_method": row,
            "matched_global": baseline,
            "absolute_face_count_difference": abs(
                len(reduced.faces) - row["actual_faces"]
            ),
        }
        results.append(pair)
        print(json.dumps(pair), flush=True)
    write_json(
        out / "matched_qem.json",
        {
            "scope": "matched achieved face counts; same reference and sample points",
            "timing_protocol": "one warmup per method, 15 alternating paired repeats, VLM processes finished",
            "pairs": results,
        },
    )


if __name__ == "__main__":
    main()
