"""生成Mesh局部材料域的条件物理实验，不输出整物体真实质量。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tactile3d.physics.materials import load_material_priors
from tactile3d.physics.mesh_asset_io import save_mesh_asset
from tactile3d.physics.mesh_native import (
    ClosedMesh,
    MeshPhysicalObject,
    MeshRegion,
    boolean_volume,
    verify_mesh_recompute,
)
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.physics.part_state import Material
from tactile3d.physics.patch_volume import extrude_patch, piecewise_patch_layer

ROOT = Path(__file__).resolve().parents[1]
THICKNESS_RATIOS = (0.001, 0.002, 0.005)
COLORS = ("#5A76A8", "#C3916C")


def plot_sensitivity(rows: list[dict[str, Any]], output: Path) -> None:
    """横轴为假设厚度，纵轴仅为所选局部材料域质量。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scenes = list(dict.fromkeys(r["scene_id"] for r in rows))
    if not scenes:
        return
    figure, axes = plt.subplots(
        1,
        len(scenes),
        figsize=(3.2 * len(scenes), 3.2),
        layout="constrained",
        squeeze=False,
    )
    for ax, scene in zip(axes[0], scenes, strict=True):
        subset = [r for r in rows if r["scene_id"] == scene]
        for key, label, color in zip(
            ("before_mass_kg", "after_mass_kg"),
            ("Before", "After material edit"),
            COLORS,
            strict=True,
        ):
            ax.plot(
                [r["thickness_ratio"] for r in subset],
                [r[key] for r in subset],
                "o-",
                label=label,
                color=color,
            )
        ax.set(
            xlabel="Assumed axial thickness / max extent",
            ylabel="Selected-layer mass (kg; conditional)",
            title=scene.split("_")[0],
            xticks=THICKNESS_RATIOS,
            xlim=(0.0007, 0.0053),
        )
        if len(subset) < len(THICKNESS_RATIOS):
            ax.text(
                0.03,
                0.97,
                f"{len(THICKNESS_RATIOS) - len(subset)} rejected",
                transform=ax.transAxes,
                va="top",
                fontsize=8,
            )
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
    length = rows[0]["assumed_max_length_m"]
    catalog_scale = any(r["catalog_scale_assisted"] for r in rows)
    learned_scale = any(r["learned_scale"] for r in rows)
    figure.suptitle(
        (
            "Image-predicted scale; assumed layer thickness | "
            if learned_scale
            else (
                "Catalog max-extent scale; assumed layer thickness | "
                if catalog_scale
                else f"Declared max extent {length:g} m; not measured | "
            )
        )
        + "local layer only, not whole-object mass",
        fontsize=10,
    )
    for suffix in ("svg", "png"):
        figure.savefig(output / f"patch_mass_sensitivity.{suffix}", dpi=300)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """从固定标签构建薄层并换材，几何失败逐条保留，不替换对象。

    Args:
        args: 冻结表面标签、显式编辑规格和假设长度。

    Returns:
        条件质量/COM/惯量及失败记录；整物体质量保持为空。
    """
    if not np.isfinite(args.assumed_max_length_m) or args.assumed_max_length_m <= 0:
        raise ValueError("假设最大长度必须是有限正数")
    report = json.loads((args.fields / "report.json").read_text(encoding="utf-8"))
    specs = json.loads(args.edit_spec.read_text(encoding="utf-8"))
    priors = {
        p.name: p for p in load_material_priors(ROOT / "configs/material_priors.yaml")
    }
    catalog = None
    learned = None
    if args.catalog_dimensions is not None and args.scale_predictions is not None:
        raise ValueError("商品尺度与学习尺度只能选择一种输入协议")
    if args.catalog_dimensions is not None:
        catalog = json.loads(args.catalog_dimensions.read_text(encoding="utf-8"))
        if (
            catalog.get("scope") != "catalog_dimension_assisted_input_no_mass_labels"
            or catalog.get("mass_labels_included") is not False
        ):
            raise ValueError("商品尺度输入必须来自排除质量标签的尺寸证据")
    if args.scale_predictions is not None:
        seal_path = args.scale_predictions.parent / "prediction_seal.json"
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        if file_sha256(args.scale_predictions) != seal["sha256"]:
            raise ValueError("学习尺度预测封存校验失败")
        protocol = json.loads(
            (args.scale_predictions.parent / "protocol.json").read_text(
                encoding="utf-8"
            )
        )
        if (
            protocol.get("scope")
            != "image_only_scale_baseline_catalog_supervised_training"
        ):
            raise ValueError("未识别的学习尺度协议")
        learned = {
            r["scene_id"]: r
            for r in json.loads(args.scale_predictions.read_text(encoding="utf-8"))
        }
    args.output.mkdir(parents=True, exist_ok=False)
    rows, failures, records = [], [], []
    for item in report["objects"]:
        scene = item["scene_id"]
        directory = args.fields / scene
        token = json.loads((directory / "part_tokens.json").read_text(encoding="utf-8"))
        spec = specs[scene]
        part = next(p for p in token["parts"] if p["part_id"] == spec["part_id"])
        source_path = Path(token["mesh"])
        if file_sha256(source_path) != token["mesh_sha256"]:
            raise ValueError(f"{scene}生成Mesh哈希错误")
        source = trimesh.load_mesh(source_path, process=False)
        length = args.assumed_max_length_m
        if catalog is not None:
            measurement = catalog["objects"][scene]
            if measurement["status"] != "available":
                raise ValueError(f"{scene}没有无冲突的商品尺寸，禁止静默回退假定尺度")
            length = float(measurement["max_dimension_m"])
            if not np.isfinite(length) or length <= 0:
                raise ValueError("商品尺寸必须为有限正数")
        if learned is not None:
            length = float(learned[scene]["max_length_m"])
            if not np.isfinite(length) or length <= 0:
                raise ValueError("学习尺度预测必须为有限正数")
        scale = length / float(source.extents.max())
        mesh = source.copy()
        mesh.vertices *= scale
        with np.load(directory / "surface_fields.npz") as stored:
            selected = np.flatnonzero(stored["labels"] == part["part_index"])
        child = args.output / scene
        child.mkdir()
        for ratio in THICKNESS_RATIOS:
            try:
                start = time.perf_counter()
                outer = ClosedMesh(
                    mesh.vertices, mesh.faces, "generated_outer_envelope"
                )
                builder = piecewise_patch_layer if args.piecewise else extrude_patch
                built = builder(
                    mesh,
                    selected,
                    length * ratio,
                    "declared_axial_layer_hypothesis_not_observed_internal_structure",
                )
                patch, pieces = built if args.piecewise else (built, 1)
                clipped = boolean_volume(patch, outer, "intersection")
                old = priors[part["surface_material"]]
                new = priors[spec["new_material"]]
                state = MeshPhysicalObject(
                    scene,
                    (
                        MeshRegion(
                            part["part_id"],
                            "conditional_surface_layer",
                            clipped,
                            Material(
                                old.name,
                                old.density_kg_m3.mean,
                                "configured_conditional_prior",
                                youngs_modulus_pa=old.youngs_modulus_pa.mean,
                            ),
                        ),
                    ),
                )
                updated, edit = state.edit_material(
                    part["part_id"],
                    "conditional_surface_layer",
                    Material(
                        new.name,
                        new.density_kg_m3.mean,
                        "explicit_edit_conditional_prior",
                        youngs_modulus_pa=new.youngs_modulus_pa.mean,
                    ),
                    0,
                )
                residual = verify_mesh_recompute(updated)
                prefix = child / f"ratio_{ratio:g}"
                save_mesh_asset(state, prefix / "before")
                save_mesh_asset(updated, prefix / "after")
                evidence = {
                    "scope": "conditional_local_layer_not_complete_real_object",
                    "source_generated_mesh_sha256": token["mesh_sha256"],
                    "source_fields_sha256": file_sha256(
                        directory / "surface_fields.npz"
                    ),
                    "part_id": part["part_id"],
                    "selected_face_count": len(selected),
                    "scale_verified": False,
                    "thickness_verified": False,
                    "assumed_max_length_m": length,
                    "catalog_scale_assisted": catalog is not None,
                    "learned_scale": learned is not None,
                    "scale_source": (
                        "image_only_scale_prediction"
                        if learned is not None
                        else (
                            "catalog_max_extent_normalization"
                            if catalog is not None
                            else "declared_normalization_not_measurement"
                        )
                    ),
                    "axial_thickness_ratio": ratio,
                }
                for phase in ("before", "after"):
                    asset_path = prefix / phase / "asset.json"
                    asset = json.loads(asset_path.read_text(encoding="utf-8"))
                    asset["evidence"] = evidence
                    write_json(asset_path, asset)
                write_json(prefix / "part_domain_binding.json", evidence)
                row = {
                    "scene_id": scene,
                    "part_id": part["part_id"],
                    "construction_pieces": pieces,
                    "thickness_ratio": ratio,
                    "assumed_max_length_m": length,
                    "catalog_scale_assisted": catalog is not None,
                    "learned_scale": learned is not None,
                    "conditional_unit_to_meter": scale,
                    "volume_m3": clipped.moments.volume_m3,
                    "before_mass_kg": state.properties["mass_kg"],
                    "after_mass_kg": updated.properties["mass_kg"],
                    "delta_mass_kg": edit["delta_mass_kg"],
                    "retained_extrusion_volume_fraction": (
                        clipped.moments.volume_m3 / patch.moments.volume_m3
                    ),
                    "seconds": time.perf_counter() - start,
                }
                if learned is not None:
                    bounds = np.array(
                        [
                            learned[scene]["conformal_scale_low_m"],
                            learned[scene]["conformal_scale_high_m"],
                        ]
                    )
                    if (
                        not np.isfinite(bounds).all()
                        or not 0 < bounds[0] <= length <= bounds[1]
                    ):
                        raise ValueError("学习尺度区间不合法")
                    for phase, properties in (
                        ("before", state.properties),
                        ("after", updated.properties),
                    ):
                        row[f"scale_only_{phase}_mass_low_kg"] = (
                            properties["mass_kg"] * (bounds[0] / length) ** 3
                        )
                        row[f"scale_only_{phase}_mass_high_kg"] = (
                            properties["mass_kg"] * (bounds[1] / length) ** 3
                        )
                write_json(
                    prefix / "conditional_physics.json",
                    {
                        **row,
                        "before": state.properties,
                        "after": updated.properties,
                        "edit": edit,
                        "full_reintegration_residual": residual,
                        "is_total_object_mass": False,
                        "scale_verified": False,
                        "thickness_verified": False,
                        "unknown_other_parts": True,
                    },
                )
                rows.append(row)
                records.append({**row, "status": "complete"})
            except ValueError as error:
                failure = {
                    "scene_id": scene,
                    "thickness_ratio": ratio,
                    "status": "rejected_geometry",
                    "reason": str(error),
                }
                failures.append(failure)
                records.append(failure)
        print("PATCH_PHYSICS", scene, flush=True)
    if rows:
        with (args.output / "conditional_masses.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    result = {
        "status": (
            "complete"
            if rows and len(rows) == len(records)
            else ("partial_geometry_acceptance" if rows else "failed_no_valid_domains")
        ),
        "scope": "conditional_local_layer_physics_not_real_mass_estimation",
        "input_fields_sha256": file_sha256(args.fields / "report.json"),
        "edit_spec_sha256": file_sha256(args.edit_spec),
        "declared_assumptions": {
            "max_extent_m": (
                None
                if catalog is not None or learned is not None
                else args.assumed_max_length_m
            ),
            "scale_source": (
                "image_only_learned_scale"
                if learned is not None
                else (
                    "catalog_dimension_assisted_not_RGB_only"
                    if catalog is not None
                    else "normalization_experiment_not_independent_measurement"
                )
            ),
            "thickness_ratios": list(THICKNESS_RATIOS),
            "thickness_source": "declared_sensitivity_grid_not_VLM_or_measurement",
        },
        "whole_object_mass_kg": None,
        "physical_accuracy_established": False,
        "voxelization_used": False,
        "piecewise_directional_construction": args.piecewise,
        "catalog_dimensions_sha256": (
            file_sha256(args.catalog_dimensions) if args.catalog_dimensions else None
        ),
        "scale_predictions_sha256": (
            file_sha256(args.scale_predictions) if args.scale_predictions else None
        ),
        "scale_only_mass_range_is_full_physics_confidence_interval": False,
        "records": records,
        "failures": failures,
    }
    write_json(args.output / "report.json", result)
    plot_sensitivity(rows, args.output)
    if not rows:
        raise RuntimeError("所有闭合域候选均被拒绝；已保存失败记录")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fields", type=Path, default=ROOT / "examples/observed_surface_demo"
    )
    parser.add_argument(
        "--edit-spec",
        type=Path,
        default=ROOT / "examples/observed_surface_demo/edit_spec.json",
    )
    parser.add_argument("--assumed-max-length-m", type=float, default=1.0)
    parser.add_argument("--piecewise", action="store_true")
    parser.add_argument("--catalog-dimensions", type=Path)
    parser.add_argument("--scale-predictions", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/conditional_patch_physics"
    )
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
