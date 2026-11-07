from __future__ import annotations

import numpy as np
import pytest

from tactile3d.models.scale_ridge import (
    calibrate_log_scale,
    fit_log_scale,
    scale_metrics,
)


def test_log_scale_regression_recovers_known_positive_mapping() -> None:
    x = np.linspace(-1, 1, 50)[:, None]
    y = np.exp(0.2 + 0.8 * x[:, 0])
    model = fit_log_scale(x[:40], y[:40], 1e-8)
    np.testing.assert_allclose(model.predict(x[40:]), y[40:], rtol=1e-7)
    assert scale_metrics(y[40:], model.predict(x[40:]))["within_factor_1_25"] == 1
    with pytest.raises(ValueError, match="正长度"):
        fit_log_scale(x, -y, 0.1)


def test_dual_and_primal_ridge_solutions_agree() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8, 12))
    y = np.exp(rng.normal(size=8))
    model = fit_log_scale(x, y, 0.3)
    centered = x - x.mean(0)
    expected = np.linalg.solve(
        centered.T @ centered + 0.3 * np.eye(12),
        centered.T @ (np.log(y) - np.log(y).mean()),
    )
    np.testing.assert_allclose(model.coefficients, expected, atol=1e-12)


def test_conformal_uses_finite_sample_rank_and_rejects_tiny_calibration() -> None:
    assert calibrate_log_scale(np.exp(np.arange(9)), np.ones(9), 0.8) == 7
    with pytest.raises(ValueError, match="样本不足"):
        calibrate_log_scale(np.ones(2), np.ones(2), 0.95)
