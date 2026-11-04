"""固定对象清单的可审计质量对比，不导入论文表格充当复现结果。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from tactile3d.figures.comparison import generate_comparison_figures
from tactile3d.models.calibration import mass_metrics
from tactile3d.physics.part_io import file_sha256, write_json

DEFAULT_PILOT = Path("outputs/abo500_ten_v2")
DEFAULT_OUTPUT = Path("outputs/comparisons")
BOOTSTRAP_SAMPLES = 2000
RANDOM_SEED = 17
FIGURE_FORMATS = ("png", "svg", "pdf")
PAPER_METHODS = ("NeRF2Physics", "PUGS", "PhysVGGT", "Gen2Physics", "VoMP")
SCENARIOS = ("Surface solid", "Shell only", "Construction prior")
MASS_TOLERANCE = 1e-9
REGISTRY_VERSION = 1


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("不能输出空对比表")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_prediction_seal(root: Path) -> None:
    """验证旧实验的已封存预测；兼容清单中的Windows路径分隔符。"""
    root = root.resolve()
    seal = read_json(root / "prediction_seal.json")
    if not isinstance(seal, dict) or "predictions.json" not in seal:
        raise ValueError("缺少有效预测封存清单")
    for name, expected in seal.items():
        file = (root / name.replace("\\", "/")).resolve()
        if not file.is_relative_to(root) or file_sha256(file) != expected:
            raise ValueError(f"预测封存校验失败：{name}")


def prepare_pilot_registry(pilot: Path, output: Path) -> Path:
    """将已运行的三种规则对照接入统一协议，不生成新预测。

    Args:
        pilot: 旧十物体实验目录，必须有预测封存和评价CSV。
        output: 已创建的新输出目录。

    Returns:
        可编辑的对比注册表；论文基线及学习式完整方法保持待运行。
    """
    verify_prediction_seal(pilot)
    cohort = read_json(pilot / "selection_before_evaluation.json")["object_ids"]
    if len(cohort) != 10 or len(set(cohort)) != 10:
        raise ValueError("开发实验必须保持原固定十物体清单")
    targets = {}
    label_source = pilot / "mass_metrics_per_object.csv"
    with label_source.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            scene, value = row["scene_id"], float(row["target_mass_kg"])
            if scene not in cohort:
                raise ValueError("旧评价CSV含固定清单外物体")
            if scene in targets and not np.isclose(
                targets[scene], value, rtol=0, atol=MASS_TOLERANCE
            ):
                raise ValueError("同一物体的参考质量不一致")
            targets[scene] = value
    if set(targets) != set(cohort):
        raise ValueError("评价参考质量未覆盖固定清单")
    write_json(output / "targets.json", targets)
    batch = read_json(pilot / "predictions.json")
    methods = []
    for index, scenario in enumerate(SCENARIOS):
        rows = []
        for prediction in batch["predictions"]:
            if prediction["status"] != "predicted":
                continue
            stats = prediction["scenarios"][scenario]
            rows.append(
                {
                    "scene_id": prediction["scene_id"],
                    "target_mass_kg": targets[prediction["scene_id"]],
                    "object": {
                        "mass_kg": {key: stats[key] for key in ("mean", "q05", "q95")}
                    },
                    "interval_semantics": "uncalibrated_prior_quantiles",
                }
            )
        relative = f"runs/rule_{index}.json"
        write_json(output / relative, {"method": scenario, "predictions": rows})
        methods.append(
            {
                "name": scenario,
                "status": "executed",
                "kind": "rule_baseline",
                "predictions_path": relative,
                "input_condition": "reference ABO mesh + dataset views + cached priors",
                "evidence_path": str((pilot / "prediction_seal.json").resolve()),
            }
        )
    for name in (*PAPER_METHODS, "PartPhy3D learned full pipeline"):
        methods.append(
            {
                "name": name,
                "status": "pending",
                "kind": "proposed_method"
                if name.startswith("PartPhy3D")
                else "reproduced_paper",
                "predictions_path": None,
                "input_condition": "not run; fair input protocol to be declared",
                "evidence_path": None,
            }
        )
    registry = {
        "schema_version": REGISTRY_VERSION,
        "context": "development",
        "cohort_ids": cohort,
        "targets_path": "targets.json",
        "methods": methods,
        "source_pilot": str(pilot.resolve()),
        "prediction_seal_verified": True,
        "targets_source": str(label_source.resolve()),
        "targets_source_sha256": file_sha256(label_source),
        "not_a_heldout_final_test": True,
    }
    path = output / "registry.json"
    write_json(path, registry)
    return path


def normalized_rows(
    path: Path, cohort: list[str], targets: dict[str, float]
) -> tuple[list[dict[str, Any]], list[str]]:
    """检查固定对象、重复ID、单位数值和参考标签一致性。"""
    payload = read_json(path)
    source = payload.get("predictions") if isinstance(payload, dict) else payload
    if not isinstance(source, list) or not source:
        raise ValueError("结果需要非空predictions列表；原生结果请先规范化导入")
    result, seen = {}, set()
    for row in source:
        scene = row["scene_id"]
        if not isinstance(scene, str) or not scene or scene in seen:
            raise ValueError("物体ID无效或重复，不能自动平均多个视角结果")
        seen.add(scene)
        if scene not in targets:
            continue
        if row.get("status", "predicted") not in {"predicted", "success"}:
            continue
        stats = dict(row["object"]["mass_kg"])
        value = float(stats["mean"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{scene}的质量预测必须有限且为正")
        embedded = row.get("target_mass_kg", targets[scene])
        if not np.isfinite(float(embedded)) or not np.isclose(
            float(embedded), targets[scene], rtol=0, atol=MASS_TOLERANCE
        ):
            raise ValueError(f"{scene}的参考质量与固定标签不一致")
        if ("q05" in stats) != ("q95" in stats):
            raise ValueError("分位数上下界必须同时提供")
        if "q05" in stats:
            bounds = np.array([stats["q05"], stats["q95"]], dtype=float)
            if not np.isfinite(bounds).all() or bounds[0] < 0 or bounds[0] > bounds[1]:
                raise ValueError("分位数区间无效")
        result[scene] = {
            **row,
            "target_mass_kg": targets[scene],
            "object": {"mass_kg": stats},
        }
    missing = [scene for scene in cohort if scene not in result]
    return [result[scene] for scene in cohort if scene in result], missing


def compare_registry(
    registry_path: Path, output: Path, bootstrap_samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    """输出完整状态表；不完整方法不计算主指标，也不缩小固定队列。"""
    if type(bootstrap_samples) is not int or bootstrap_samples < 100:
        raise ValueError("bootstrap次数必须为至少100的整数")
    registry_path = registry_path.resolve()
    registry = read_json(registry_path)
    if registry.get("schema_version") != REGISTRY_VERSION:
        raise ValueError("不支持的对比注册表版本")
    if registry.get("context") not in {
        "development",
        "heldout_test",
        "synthetic_validation",
    }:
        raise ValueError("必须声明development、heldout_test或synthetic_validation")
    cohort = registry["cohort_ids"]
    if not cohort or not all(isinstance(scene, str) and scene for scene in cohort):
        raise ValueError("固定对象清单必须为非空字符串列表")
    if len(set(cohort)) != len(cohort):
        raise ValueError("固定对象清单不能重复")
    base = registry_path.parent
    target_file = base / registry["targets_path"]
    raw_targets = read_json(target_file)
    targets = {scene: float(raw_targets[scene]) for scene in cohort}
    mass_metrics(list(targets.values()), list(targets.values()))
    names = [method["name"] for method in registry["methods"]]
    if not names or not all(isinstance(name, str) and name.strip() for name in names):
        raise ValueError("方法名称必须为非空字符串")
    if len(set(names)) != len(names):
        raise ValueError("方法名称必须非空且唯一")
    statuses, runs = [], {}
    for index, method in enumerate(registry["methods"]):
        row = {
            "method": method["name"],
            "status": method["status"],
            "kind": method["kind"],
            "input_condition": method["input_condition"],
            "n_requested": len(cohort),
            "n_valid": 0,
            "n_missing": len(cohort),
            "ade_kg": None,
            "alde": None,
            "ape_percent": None,
            "mnre": None,
            "rmse_kg": None,
            "reason": "not run",
            "predictions_sha256": None,
            "evidence_sha256": None,
        }
        if method["status"] == "pending":
            statuses.append(row)
            continue
        if method["status"] != "executed":
            raise ValueError("注册表状态仅支持executed和pending")
        try:
            source_file = base / method["predictions_path"]
            if method.get("evidence_path"):
                row["evidence_sha256"] = file_sha256(base / method["evidence_path"])
            elif method["kind"] in {"reproduced_paper", "proposed_method"}:
                raise ValueError("论文基线或完整方法必须提供实际运行证据文件")
            row["predictions_sha256"] = file_sha256(source_file)
            rows, missing = normalized_rows(source_file, cohort, targets)
            row.update(n_valid=len(rows), n_missing=len(missing))
            if missing:
                row.update(status="incomplete", reason="missing: " + ", ".join(missing))
            else:
                expected = [targets[item["scene_id"]] for item in rows]
                values = [float(item["object"]["mass_kg"]["mean"]) for item in rows]
                metrics = mass_metrics(expected, values)
                row.update(
                    ade_kg=metrics["ade_kg"],
                    alde=metrics["alde"],
                    ape_percent=100 * metrics["ape"],
                    mnre=metrics["mnre"],
                    rmse_kg=metrics["rmse_kg"],
                    reason="",
                )
                path = output / "normalized_runs" / f"run_{index}.json"
                write_json(path, {"predictions": rows})
                runs[method["name"]] = path
        except (OSError, ValueError, KeyError, TypeError) as error:
            row.update(status="invalid", reason=str(error))
            for metric in ("ade_kg", "alde", "ape_percent", "mnre", "rmse_kg"):
                row[metric] = None
        statuses.append(row)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "mass_comparison.csv", statuses)
    comparison = None
    if len(runs) >= 2:
        comparison = generate_comparison_figures(
            runs,
            output / "figures",
            bootstrap_samples=bootstrap_samples,
            seed=RANDOM_SEED,
            formats=FIGURE_FORMATS,
            cohort_ids=cohort,
            evaluation_context=registry["context"],
        )
    table = [
        "| Method | Status | N | ADE kg ↓ | ALDE ↓ | APE % ↓ | MnRE ↑ |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in statuses:
        scores = [
            "—" if row[key] is None else f"{row[key]:.4f}"
            for key in ("ade_kg", "alde", "ape_percent", "mnre")
        ]
        table.append(
            f"| {row['method']} | {row['status']} | "
            f"{row['n_valid']}/{row['n_requested']} | " + " | ".join(scores) + " |"
        )
    (output / "mass_comparison.md").write_text(
        f"Context: {registry['context']}. Input conditions are listed in CSV.\n\n"
        + "\n".join(table)
        + "\n",
        encoding="utf-8",
    )
    report = {
        "context": registry["context"],
        "cohort_ids": cohort,
        "registry_sha256": file_sha256(registry_path),
        "targets_sha256": file_sha256(target_file),
        "methods": statuses,
        "num_scored_methods": len(runs),
        "num_pending_methods": sum(row["status"] == "pending" for row in statuses),
        "num_failed_methods": sum(
            row["status"] in {"invalid", "incomplete"} for row in statuses
        ),
        "published_numbers_injected": False,
        "comparison": comparison,
    }
    write_json(output / "benchmark_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    args = parser.parse_args()
    output = args.output or DEFAULT_OUTPUT / f"run_{time.time_ns()}"
    try:
        if args.bootstrap_samples < 100:
            raise ValueError("bootstrap次数至少为100")
        output.mkdir(parents=True, exist_ok=False)
        registry = args.registry or prepare_pilot_registry(args.pilot, output)
        report = compare_registry(registry, output, args.bootstrap_samples)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"对比失败：{error}\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "context",
                    "num_scored_methods",
                    "num_pending_methods",
                    "num_failed_methods",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"输出目录：{output.resolve()}")
    if report["num_failed_methods"]:
        parser.exit(2, "部分方法无效或不完整，请查看状态表。\n")


if __name__ == "__main__":
    main()
