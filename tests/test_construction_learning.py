from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tactile3d.research.construction import (
    TRAIN_EDITS,
    candidate_masks,
    candidate_volumes,
    export_state,
    mass_forward,
)
from tactile3d.research.data import analytic_volumes, load_inputs, make_dataset
from tactile3d.research.learning import state_loss, train_models
from tactile3d.research.network import Network
from tactile3d.research.workflow import (
    calibrate_interval,
    mass_metrics,
    predict_models,
    run_experiment,
)


def test_candidate_integrals_match_independent_analytic_geometry() -> None:
    pitch = np.array([0.003, 0.013, 0.027])
    np.testing.assert_allclose(
        candidate_volumes(pitch), analytic_volumes(pitch), rtol=1e-13
    )
    assert np.all(candidate_masks().sum(2) <= 1)


def test_physics_loss_gradient_matches_central_difference() -> None:
    inputs, labels = make_dataset(60, 32)
    rng = np.random.default_rng(47)
    logits = rng.normal(0, 0.4, (3, 8))
    arguments = (
        inputs["priors"][:3],
        candidate_volumes(inputs["pitch_m"][:3]),
        labels["mass_kg"][:3],
        labels["construction"][:3],
        1.0,
    )
    _, gradient = state_loss(logits, *arguments)
    step = 1e-6
    numerical = np.empty_like(logits)
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += step
        minus[index] -= step
        numerical[index] = (
            state_loss(plus, *arguments)[0] - state_loss(minus, *arguments)[0]
        ) / (2 * step)
    np.testing.assert_allclose(gradient, numerical, atol=2e-8, rtol=2e-5)


def test_network_gradient_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    network = Network(3, 2, 4, 12)
    features = np.array([[0.2, -0.3, 0.7], [-0.4, 0.1, 0.2]])
    output, cache = network.forward(features)
    gradient = network.backward(output, cache)
    for name, parameter in network.parameters.items():
        for index in np.ndindex(parameter.shape):
            original = parameter[index]
            parameter[index] = original + 1e-6
            plus = np.sum(network.forward(features)[0] ** 2) / 2
            parameter[index] = original - 1e-6
            minus = np.sum(network.forward(features)[0] ** 2) / 2
            parameter[index] = original
            assert gradient[name][index] == pytest.approx(
                (plus - minus) / 2e-6, abs=1e-8
            )
    network.save(tmp_path / "model.npz", np.zeros(3), np.ones(3))
    restored, _, _ = Network.load(tmp_path / "model.npz")
    np.testing.assert_array_equal(restored.forward(features)[0], output)


def test_training_ignores_held_out_labels_and_inference_never_receives_them() -> None:
    inputs, labels = make_dataset(60, 22)
    other = {key: value.copy() for key, value in labels.items()}
    other["mass_kg"][inputs["split"] != 0] *= 13
    models, mean, scale, _ = train_models(inputs, labels, 9, 3, 8)
    changed, _, _, _ = train_models(inputs, other, 9, 3, 8)
    for name in models:
        for key in models[name].parameters:
            np.testing.assert_array_equal(
                models[name].parameters[key], changed[name].parameters[key]
            )
    prediction = predict_models(inputs, models, mean, scale)
    assert prediction["shared_response"].shape == (60, len(TRAIN_EDITS) + 1)


def test_exported_mass_matches_hard_candidate_not_soft_mixture() -> None:
    logits, prior = np.zeros((1, 8)), np.array([[550.0, 90.0]])
    logits[0, 2] = 2.1
    state = export_state("example", 0.01, logits, prior)
    hard = mass_forward(logits, prior, candidate_volumes(np.array([0.01])), hard=True)[
        0
    ]
    soft = mass_forward(logits, prior, candidate_volumes(np.array([0.01])))[0]
    assert state.total.mass_kg == pytest.approx(hard[0, 0], rel=1e-12)
    assert not np.isclose(hard[0, 0], soft[0, 0])


def test_group_leakage_and_label_fields_in_input_are_rejected(tmp_path: Path) -> None:
    inputs, _ = make_dataset(60, 21)
    inputs["split"][0] = (inputs["split"][1] + 1) % 3
    np.savez(tmp_path / "input.npz", **inputs)
    with pytest.raises(ValueError, match="同族"):
        load_inputs(tmp_path / "input.npz")
    np.savez(tmp_path / "input.npz", **inputs, mass_kg=np.ones(60))
    with pytest.raises(ValueError, match="评测标签"):
        load_inputs(tmp_path / "input.npz")


def test_metrics_and_finite_sample_interval() -> None:
    values = mass_metrics(np.array([2.0, 1.0]), np.array([1.0, 2.0]))
    assert values["ADE_kg"] == 1
    assert values["ALDE_ln"] == pytest.approx(np.log(2))
    assert values["MnRE"] == 0.5
    with pytest.raises(ValueError, match="校准对象不足"):
        calibrate_interval(np.ones(3), np.ones(3))


def test_calibration_counts_independent_families_not_correlated_objects() -> None:
    with pytest.raises(ValueError, match="校准对象不足"):
        calibrate_interval(np.ones(20), np.ones(20), np.repeat([0, 1], 10))
    truth = np.ones(60)
    predicted = np.exp(np.arange(60) / 100)
    result = calibrate_interval(predicted, truth, np.repeat(np.arange(10), 6))
    assert result == pytest.approx(0.59)


def test_full_cpu_workflow_exports_sealed_editable_results(tmp_path: Path) -> None:
    output = tmp_path / "run"
    report = run_experiment(output, 384, 6, (7,), 8, figures=False)
    assert report["status"] == "complete"
    assert report["ABO_experiment"] is False
    for name in (
        "metrics.csv",
        "test_predictions.csv",
        "training.csv",
        "seed_7/prediction_seal.json",
        "seed_7/example/steel.ply",
    ):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        run_experiment(output, 60, 1, (7,), 8, figures=False)
