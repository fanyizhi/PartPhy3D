from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tactile3d.eval.benchmark import compare_registry, main, verify_prediction_seal
from tactile3d.eval.import_baseline import import_mass_baseline
from tactile3d.figures.comparison import generate_comparison_figures
from tactile3d.models.calibration import mass_metrics


def save(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def prediction(scene: str, mean: float, target: float) -> dict[str, object]:
    return {
        "scene_id": scene,
        "target_mass_kg": target,
        "object": {"mass_kg": {"mean": mean}},
    }


def registry(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    save(tmp_path / "targets.json", {"a": 1.0, "b": 2.0})
    save(tmp_path / "run.json", {"predictions": rows})
    return save(
        tmp_path / "registry.json",
        {
            "schema_version": 1,
            "context": "synthetic_validation",
            "cohort_ids": ["a", "b"],
            "targets_path": "targets.json",
            "methods": [
                {
                    "name": "Synthetic fixture",
                    "kind": "rule_baseline",
                    "status": "executed",
                    "predictions_path": "run.json",
                    "input_condition": "known synthetic numbers",
                },
                {
                    "name": "PUGS",
                    "kind": "reproduced_paper",
                    "status": "pending",
                    "predictions_path": None,
                    "input_condition": "not run",
                },
            ],
        },
    )


def test_known_metrics_and_pending_method_has_no_scores(tmp_path: Path) -> None:
    path = registry(tmp_path, [prediction("a", 2.0, 1.0), prediction("b", 1.0, 2.0)])
    report = compare_registry(path, tmp_path / "output", bootstrap_samples=100)
    scored, pending = report["methods"]
    assert scored["ade_kg"] == pytest.approx(1.0)
    assert scored["alde"] == pytest.approx(np.log(2))
    assert scored["ape_percent"] == pytest.approx(75.0)
    assert scored["mnre"] == pytest.approx(0.5)
    assert scored["rmse_kg"] == pytest.approx(1.0)
    assert pending["status"] == "pending"
    assert pending["ade_kg"] is None
    assert report["published_numbers_injected"] is False


def test_missing_prediction_does_not_shrink_cohort_or_gain_score(
    tmp_path: Path,
) -> None:
    path = registry(tmp_path, [prediction("a", 1.0, 1.0)])
    report = compare_registry(path, tmp_path / "output", bootstrap_samples=100)
    row = report["methods"][0]
    assert report["cohort_ids"] == ["a", "b"]
    assert row["status"] == "incomplete"
    assert row["n_valid"] == 1 and row["n_missing"] == 1
    assert row["ade_kg"] is None
    assert report["num_scored_methods"] == 0


def test_mismatched_labels_are_invalid_not_silently_replaced(tmp_path: Path) -> None:
    path = registry(tmp_path, [prediction("a", 1.0, 5.0), prediction("b", 2.0, 2.0)])
    report = compare_registry(path, tmp_path / "output", bootstrap_samples=100)
    assert report["methods"][0]["status"] == "invalid"
    assert report["methods"][0]["mnre"] is None


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -1.0])
def test_invalid_mass_is_rejected_instead_of_clipped(value: float) -> None:
    with pytest.raises(ValueError):
        mass_metrics([1.0], [value])
    with pytest.raises(ValueError):
        mass_metrics([value], [1.0])


def test_declared_cohort_is_strict_in_plot_entry(tmp_path: Path) -> None:
    first = save(tmp_path / "first.json", [prediction("a", 1.0, 1.0)])
    second = save(tmp_path / "second.json", [prediction("a", 1.0, 1.0)])
    with pytest.raises(ValueError, match="missing declared cohort"):
        generate_comparison_figures(
            {"A": first, "B": second},
            tmp_path / "fig",
            bootstrap_samples=100,
            cohort_ids=["a", "b"],
        )


def test_native_ranges_are_not_probability_quantiles(tmp_path: Path) -> None:
    source = save(tmp_path / "native.json", {"a": [1.0, 3.0]})
    targets = save(tmp_path / "targets.json", {"a": 2.5})
    output = tmp_path / "normalized.json"
    import_mass_baseline(source, targets, output, method="synthetic_adapter_test")
    row = json.loads(output.read_text())["predictions"][0]
    assert row["object"]["mass_kg"]["mean"] == 2.0
    assert "q05" not in row["object"]["mass_kg"]
    assert row["interval_semantics"] == "range_bounds_not_quantiles"


def test_explicit_quantiles_require_independent_point_prediction(
    tmp_path: Path,
) -> None:
    source = save(tmp_path / "native.json", {"a": {"q05": 1.0, "q95": 3.0}})
    targets = save(tmp_path / "targets.json", {"a": 2.0})
    with pytest.raises(ValueError, match="do not define a mean"):
        import_mass_baseline(source, targets, tmp_path / "bad.json", method="test")
    save(source, {"a": {"mean": 2.2, "q05": 1.0, "q95": 3.0}})
    import_mass_baseline(source, targets, tmp_path / "good.json", method="test")
    result = json.loads((tmp_path / "good.json").read_text())["predictions"][0]
    assert result["object"]["mass_kg"]["q05"] == 1.0
    assert result["interval_semantics"] == "reported_quantiles"


def test_multiple_views_are_not_silently_averaged(tmp_path: Path) -> None:
    source = save(tmp_path / "native.json", {"a_view0": 1.0, "a_view1": 2.0})
    targets = save(tmp_path / "targets.json", {"a": 1.5})
    with pytest.raises(ValueError, match="Multiple prediction IDs"):
        import_mass_baseline(source, targets, tmp_path / "bad.json", method="test")


def test_sealed_prediction_paths_cannot_escape_directory(tmp_path: Path) -> None:
    save(
        tmp_path / "prediction_seal.json",
        {"../escape": "invalid", "predictions.json": "x"},
    )
    with pytest.raises(ValueError, match="封存校验失败"):
        verify_prediction_seal(tmp_path)


def test_cli_nonzero_exit_for_incomplete_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = registry(tmp_path, [prediction("a", 1.0, 1.0)])
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare", "--registry", str(path), "--output", str(tmp_path / "output")],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
