"""Transfer PUGS-Qwen's native dense outputs to the shared ABO comparison mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from prepare_abo50_benchmark import digest, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shared-run", "pugs-run", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    args = parser.parse_args()
    scene = args.scene
    files = args.pugs_run / "scenes" / scene / "pred_result_density"
    report = args.pugs_run / "reports" / scene
    predictions = report / "official_predictions.json"
    if (
        digest(predictions)
        != read_json(report / "prediction_seal.json")[predictions.name]
    ):
        raise ValueError("PUGS-Qwen mass prediction seal differs")
    manifest = read_json(args.shared_run / "density_manifest.json")
    scene_report = next(row for row in manifest["scenes"] if row["scene_id"] == scene)
    mesh_path = Path(read_json(args.shared_run / "meshes.json")[scene]["path"])
    if digest(mesh_path) != scene_report["mesh_sha256"]:
        raise ValueError("Shared mesh changed")
    mesh = trimesh.load_mesh(mesh_path, process=False)
    points_path = files / "dense_points.npy"
    density_path = files / "dense_pred_density.npy"
    points = np.load(points_path)
    intervals = np.load(density_path)
    if points.shape != (len(intervals), 3) or intervals.shape != (len(points), 2):
        raise ValueError("PUGS points and density intervals do not correspond")
    finite = np.isfinite(points).all(axis=1) & np.isfinite(intervals).all(axis=1)
    if not finite.any():
        raise ValueError("PUGS returned no finite density samples")
    with np.load(args.shared_run / f"{scene}.npz") as previous:
        maps = dict(previous)
    visible = maps["foreground"] & (maps["face_ids"] >= 0)
    ids, inv = np.unique(maps["face_ids"][visible], return_inverse=True)
    distance, index = cKDTree(points[finite]).query(mesh.triangles_center[ids])
    density = intervals[finite].mean(axis=1)[index]
    max_distance = 0.02 * float(mesh.extents.max())
    density[distance > max_distance] = np.nan
    pugs = np.full(visible.shape, np.nan)
    pugs[visible] = density[inv]
    maps["pugs"] = pugs
    args.output.mkdir()
    np.savez_compressed(args.output / f"{scene}.npz", **maps)
    write_json(
        args.output / "density_manifest.json",
        {
            **manifest,
            "scenes": [
                {
                    **scene_report,
                    "pugs_foreground_coverage": float(
                        np.isfinite(pugs).sum() / maps["foreground"].sum()
                    ),
                    "pugs_max_transfer_distance_camera_units": max_distance,
                    "pugs_source_points_sha256": digest(points_path),
                    "pugs_density_interval_sha256": digest(density_path),
                    "pugs_mass_interval_kg": read_json(predictions)[scene],
                }
            ],
        },
    )


if __name__ == "__main__":
    main()
