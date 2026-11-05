"""Mesh体积基底上的共享构造与编辑响应学习，不执行体素化。"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from tactile3d.experiments.mesh_native_demo import EDIT_MATERIAL
from tactile3d.figures.mesh_native import plot_mesh_edit, plot_mesh_physics
from tactile3d.physics.mesh_asset_io import save_mesh_asset
from tactile3d.physics.mesh_native import verify_mesh_recompute
from tactile3d.physics.part_io import file_sha256, write_json
from tactile3d.research.construction import TEST_EDIT, mass_forward
from tactile3d.research.data import load_inputs, save_dataset
from tactile3d.research.learning import (
    CLASS_WEIGHT,
    DEFAULT_EPOCHS,
    DEFAULT_HIDDEN,
    direct_predict,
    fit_direct,
    fit_shared,
)
from tactile3d.research.mesh_templates import (
    mesh_candidate_volumes,
    mesh_state_from_logits,
)
from tactile3d.research.plots import plot_results
from tactile3d.research.workflow import (
    DATA_SEED,
    DEFAULT_COUNT,
    DEFAULT_SEEDS,
    evaluate,
    write_csv,
)


def run_native_learning(
    output: Path,
    count: int = DEFAULT_COUNT,
    epochs: int = DEFAULT_EPOCHS,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    figures: bool = True,
) -> dict[str, Any]:
    """把封闭Mesh积分得到的候选体积接入相同的可微响应监督。

    Args:
        output: 新目录，预测在评测前封存。
        count: 受控合成对象数，按对象族切分；不等于五物体真实先导。
        epochs: 固定轮数，不用测试集选择。
        seeds: 相同数据划分上的初始化种子。
        figures: 是否输出数值图与一个MAP资产的材料编辑图。

    Returns:
        来源与作用范围、计时和独立评价指标文件。
    """
    if epochs < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("训练轮数和初始化种子无效")
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    manifest: dict[str, Any] = {
        "status": "running",
        "representation": "closed_material_region_mesh",
        "voxelization_used": False,
        "real_VLM_training": False,
        "data_source": "analytic_synthetic_proxy_features_not_ABO",
        "python": platform.python_version(),
        "epochs": epochs,
        "seeds": list(seeds),
        "network": f"11D_proxy_MLP_hidden_{DEFAULT_HIDDEN}",
        "construction_CE_weight": CLASS_WEIGHT,
    }
    write_json(output / "run.json", manifest)
    try:
        save_dataset(output / "dataset", count, DATA_SEED)
        inputs = load_inputs(output / "dataset/inputs.npz")
        with np.load(output / "dataset/labels.npz", allow_pickle=False) as value:
            labels = {key: value[key].copy() for key in value.files}
        train = inputs["split"] == 0
        mean = inputs["features"][train].mean(0)
        scale = np.maximum(inputs["features"][train].std(0), 1e-6)
        features = (inputs["features"] - mean) / scale
        volumes = mesh_candidate_volumes(inputs["pitch_m"])
        np.savez_compressed(output / "mesh_geometry_basis.npz", volumes_m3=volumes)
        reference = inputs["priors"][:, 0] * volumes[:, 0, 0]
        all_metrics, all_predictions, all_history = [], [], []
        for seed in seeds:
            folder = output / f"seed_{seed}"
            folder.mkdir()
            predictions = {}
            for method, weight in (("mass_only", 0.0), ("shared_response", 1.0)):
                model, history = fit_shared(
                    features[train],
                    inputs["priors"][train],
                    volumes[train],
                    labels["mass_kg"][train],
                    labels["construction"][train],
                    seed,
                    epochs,
                    DEFAULT_HIDDEN,
                    weight,
                )
                model.save(folder / f"{method}.npz", mean, scale)
                logits, _ = model.forward(features)
                predictions[method + "_logits"] = logits
                predictions[method] = mass_forward(logits, inputs["priors"], volumes)[0]
                predictions[method + "_hard"] = mass_forward(
                    logits, inputs["priors"], volumes, hard=True
                )[0]
                predictions[method + "_unseen"] = mass_forward(
                    logits, inputs["priors"], volumes, (TEST_EDIT,)
                )[0][:, 1]
                all_history.extend(
                    {"seed": seed, "method": method, **row} for row in history
                )
            model, history = fit_direct(
                features[train],
                reference[train],
                labels["mass_kg"][train],
                labels["construction"][train],
                seed,
                epochs,
                DEFAULT_HIDDEN,
            )
            model.save(
                folder / "direct_response.npz",
                np.r_[mean, np.zeros(3)],
                np.r_[scale, np.ones(3)],
            )
            predictions["direct_response"] = direct_predict(model, features, reference)
            predictions["direct_response_unseen"] = direct_predict(
                model, features, reference, (TEST_EDIT,)
            )[:, 1]
            all_history.extend(
                {"seed": seed, "method": "direct_response", **row} for row in history
            )
            filename = folder / "predictions.npz"
            np.savez_compressed(filename, **predictions)
            seal = file_sha256(filename)
            write_json(
                folder / "prediction_seal.json",
                {"sha256": seal, "geometry_backend": "Mesh64_CSG_boundary_integrals"},
            )
            metrics, rows = evaluate(inputs, labels, predictions, seed)
            if file_sha256(filename) != seal:
                raise ValueError("评测修改了封存预测")
            all_metrics.extend(metrics)
            all_predictions.extend(rows)
            if seed == seeds[0]:
                index = int(np.flatnonzero(inputs["split"] == 2)[0])
                logits = predictions["shared_response_logits"][index : index + 1]
                priors = inputs["priors"][index : index + 1]
                state = mesh_state_from_logits(
                    f"synthetic_{index}",
                    float(inputs["pitch_m"][index]),
                    logits,
                    priors,
                )
                edited, report = state.edit_material(
                    "body", "phase_0", EDIT_MATERIAL, 0
                )
                save_mesh_asset(state, folder / "original")
                save_mesh_asset(edited, folder / "edited")
                hard_mass = mass_forward(
                    logits, priors, volumes[index : index + 1], hard=True
                )[0][0, 0]
                if not np.isclose(
                    state.properties["mass_kg"], hard_mass, rtol=1e-9, atol=1e-12
                ):
                    raise ValueError("MAP Mesh资产质量与共享状态预测不一致")
                report.update(
                    full_recompute=verify_mesh_recompute(edited),
                    soft_original_mass_kg=float(
                        predictions["shared_response"][index, 0]
                    ),
                    hard_original_mass_kg=state.properties["mass_kg"],
                )
                if figures:
                    report["views"] = plot_mesh_edit(
                        state, edited, "body", "phase_0", folder
                    )
                    plot_mesh_physics([state, edited], ["MAP", "Edited MAP"], folder)
                write_json(folder / "edit_report.json", report)
        write_csv(output / "metrics.csv", all_metrics)
        write_csv(output / "test_predictions.csv", all_predictions)
        write_csv(output / "training.csv", all_history)
        if figures:
            plot_results(output, all_metrics, all_predictions, all_history)
        manifest.update(
            status="complete",
            seconds=time.perf_counter() - start,
            candidates="solid, closed_shell, shell_filled, frame_filled",
            limitations=[
                "Known template catalogue and synthetic proxy features",
                "No pixel or Mesh feature encoder trained",
                "No calibrated internal-structure probabilities",
                "Real five-object pilot and VLM results remain separate",
            ],
        )
        write_json(output / "run.json", manifest)
        return manifest
    except Exception as error:
        manifest.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        write_json(output / "run.json", manifest)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/mesh_learning"))
    parser.add_argument("--objects", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    try:
        report = run_native_learning(
            args.output, args.objects, args.epochs, tuple(args.seeds)
        )
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Mesh学习实验失败：{error}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
