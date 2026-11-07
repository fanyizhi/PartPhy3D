"""检查稀疏解码与全量解码的几何矩差异，不将全量结果当作真实几何。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tactile3d.physics.part_io import file_sha256, write_json


def audit(run: Path, output: Path) -> None:
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    records = []
    for row in report["objects"]:
        path = run / row["scene_id"] / "geometry_equivalence.npz"
        with np.load(path) as data:
            dense = trimesh.Trimesh(
                data["dense_vertices"], data["dense_faces"], process=False
            )
            sparse = trimesh.Trimesh(
                data["sparse_vertices"], data["sparse_faces"], process=False
            )
        if not dense.is_volume or not sparse.is_volume:
            raise ValueError("比较对象不是闭合正体积，不能核对体积与惯量")
        records.append(
            {
                "scene_id": row["scene_id"],
                "arrays_sha256": file_sha256(path),
                "relative_volume_difference": abs(
                    float(sparse.volume / dense.volume - 1)
                ),
                "COM_difference_over_bbox_diagonal": float(
                    np.linalg.norm(sparse.center_mass - dense.center_mass)
                    / np.linalg.norm(dense.extents)
                ),
                "uniform_unit_density_inertia_relative_difference": float(
                    np.linalg.norm(sparse.moment_inertia - dense.moment_inertia)
                    / np.linalg.norm(dense.moment_inertia)
                ),
            }
        )
    if output.exists():
        raise FileExistsError("拒绝覆盖几何矩审计")
    write_json(
        output,
        {
            "scope": "numerical_moment_drift_vs_dense_not_ground_truth_accuracy",
            "units": "dimensionless_relative_errors_in_same_model_coordinate_system",
            "reference_is_ground_truth": False,
            "real_mass_accuracy_established": False,
            "source_report_sha256": file_sha256(run / "report.json"),
            "objects": records,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)


if __name__ == "__main__":
    main()
