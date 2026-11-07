"""用独立长度监督拟合对数尺度；不读取商品质量或拟合测试对象。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogScaleRidge:
    feature_mean: np.ndarray
    coefficients: np.ndarray
    log_length_mean: float
    alpha: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != len(self.feature_mean)
            or not np.isfinite(x).all()
        ):
            raise ValueError("尺度预测特征维度错误或含非有限数值")
        values = np.exp(
            (x - self.feature_mean) @ self.coefficients + self.log_length_mean
        )
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("尺度预测溢出或非正，拒绝用于体积计算")
        return values


def fit_log_scale(
    features: np.ndarray, lengths_m: np.ndarray, alpha: float
) -> LogScaleRidge:
    """在训练集中心化后求解岭回归，低样本时采用对偶形式。

    Args:
        features: 训练图像特征，形状(N,D)。
        lengths_m: 独立商品最大尺寸，单位m，不能是质量反推的长度。
        alpha: 正则化强度，只允许根据独立验证集选择。

    Returns:
        可应用于新对象的对数尺度模型。
    """
    x, y = np.asarray(features, dtype=float), np.asarray(lengths_m, dtype=float)
    if (
        x.ndim != 2
        or len(x) < 2
        or x.shape[1] == 0
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.any(y <= 0)
        or not np.isfinite(alpha)
        or alpha <= 0
    ):
        raise ValueError("岭回归需要至少2个有限样本、正长度和正alpha")
    mean = x.mean(0)
    centered = x - mean
    target = np.log(y)
    intercept = float(target.mean())
    target -= intercept
    if len(x) <= x.shape[1]:
        coefficients = centered.T @ np.linalg.solve(
            centered @ centered.T + alpha * np.eye(len(x)),
            target,
        )
    else:
        coefficients = np.linalg.solve(
            centered.T @ centered + alpha * np.eye(x.shape[1]),
            centered.T @ target,
        )
    mean.flags.writeable = False
    coefficients.flags.writeable = False
    return LogScaleRidge(mean, coefficients, intercept, float(alpha))


def scale_metrics(target_m: np.ndarray, prediction_m: np.ndarray) -> dict[str, float]:
    target, pred = np.asarray(target_m), np.asarray(prediction_m)
    if (
        target.ndim != 1
        or not len(target)
        or target.shape != pred.shape
        or not np.isfinite(target).all()
        or not np.isfinite(pred).all()
        or np.any(target <= 0)
        or np.any(pred <= 0)
    ):
        raise ValueError("尺度评价需要等长的有限正数组")
    return {
        "mae_m": float(np.mean(np.abs(pred - target))),
        "ape": float(np.mean(np.abs(pred - target) / target)),
        "mean_absolute_log_error": float(np.mean(np.abs(np.log(pred / target)))),
        "within_factor_1_25": float(
            np.mean(np.maximum(pred / target, target / pred) <= 1.25)
        ),
    }


def calibrate_log_scale(
    target_m: np.ndarray, prediction_m: np.ndarray, coverage: float = 0.9
) -> float:
    """独立校准集上的有限样本保形分位数，不解释为VLM后验概率。"""
    scale_metrics(target_m, prediction_m)
    if not 0 < coverage < 1:
        raise ValueError("目标覆盖率需要位于0到1之间")
    errors = np.abs(np.log(np.asarray(target_m) / np.asarray(prediction_m)))
    rank = int(np.ceil((len(errors) + 1) * coverage))
    if rank > len(errors):
        raise ValueError("独立校准样本不足以给出所要求的有限区间")
    return float(np.sort(errors)[rank - 1])
