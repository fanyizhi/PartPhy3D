"""小型NumPy网络，仅验证学习链路，不代替VLM或几何编码器。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tactile3d.research.construction import Array

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPSILON = 1e-8


class Network:
    def __init__(self, inputs: int, outputs: int, hidden: int, seed: int) -> None:
        if min(inputs, outputs, hidden) <= 0:
            raise ValueError("网络维数必须为正")
        rng = np.random.default_rng(seed)
        self.parameters = {
            "w1": rng.normal(0, 1 / np.sqrt(inputs), (inputs, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0, 0.1 / np.sqrt(hidden), (hidden, outputs)),
            "b2": np.zeros(outputs),
        }
        self.first = {key: np.zeros_like(v) for key, v in self.parameters.items()}
        self.second = {key: np.zeros_like(v) for key, v in self.parameters.items()}
        self.step = 0

    def forward(self, features: Array) -> tuple[Array, tuple[Array, Array]]:
        if features.ndim != 2 or features.shape[1] != len(self.parameters["w1"]):
            raise ValueError("特征维数不匹配")
        if not np.isfinite(features).all():
            raise ValueError("特征必须有限")
        hidden = np.tanh(features @ self.parameters["w1"] + self.parameters["b1"])
        return hidden @ self.parameters["w2"] + self.parameters["b2"], (
            features,
            hidden,
        )

    def backward(self, gradient: Array, cache: tuple[Array, Array]) -> dict[str, Array]:
        features, hidden = cache
        delta = (gradient @ self.parameters["w2"].T) * (1 - hidden**2)
        return {
            "w1": features.T @ delta,
            "b1": delta.sum(0),
            "w2": hidden.T @ gradient,
            "b2": gradient.sum(0),
        }

    def update(self, gradients: dict[str, Array], rate: float) -> None:
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("学习率必须为正")
        if gradients.keys() != self.parameters.keys():
            raise ValueError("梯度参数集合不匹配")
        for name, gradient in gradients.items():
            if gradient.shape != self.parameters[name].shape:
                raise ValueError("梯度形状不匹配")
            if not np.isfinite(gradient).all():
                raise ValueError("非有限梯度，停止训练以保留失败证据")
        self.step += 1
        for name, gradient in gradients.items():
            self.first[name] *= ADAM_BETA1
            self.first[name] += (1 - ADAM_BETA1) * gradient
            self.second[name] *= ADAM_BETA2
            self.second[name] += (1 - ADAM_BETA2) * gradient**2
            mean = self.first[name] / (1 - ADAM_BETA1**self.step)
            variance = self.second[name] / (1 - ADAM_BETA2**self.step)
            self.parameters[name] -= rate * mean / (np.sqrt(variance) + ADAM_EPSILON)

    def save(self, path: Path, mean: Array, scale: Array) -> None:
        np.savez_compressed(
            path, **self.parameters, feature_mean=mean, feature_scale=scale
        )

    @classmethod
    def load(cls, path: Path) -> tuple[Network, Array, Array]:
        with np.load(path, allow_pickle=False) as value:
            required = {"w1", "b1", "w2", "b2", "feature_mean", "feature_scale"}
            if set(value.files) != required:
                raise ValueError("模型文件字段不匹配")
            net = cls(
                value["w1"].shape[0], value["w2"].shape[1], value["w1"].shape[1], 0
            )
            for key in net.parameters:
                item = value[key]
                if (
                    item.shape != net.parameters[key].shape
                    or not np.isfinite(item).all()
                ):
                    raise ValueError("模型参数形状或数值非法")
                net.parameters[key] = item.copy()
            mean, scale = value["feature_mean"].copy(), value["feature_scale"].copy()
        if mean.shape != (len(net.parameters["w1"]),) or scale.shape != mean.shape:
            raise ValueError("特征归一化参数维数不匹配")
        if (
            not np.isfinite(mean).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0)
        ):
            raise ValueError("特征归一化参数非法")
        return net, mean, scale


def huber(value: Array, delta: float = 0.5) -> tuple[float, Array]:
    if value.size == 0 or not np.isfinite(value).all() or delta <= 0:
        raise ValueError("Huber需要非空有限数据和正阈值")
    absolute = np.abs(value)
    loss = np.where(absolute <= delta, value**2 / 2, delta * (absolute - delta / 2))
    return float(loss.mean()), np.clip(value, -delta, delta) / value.size
