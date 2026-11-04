"""共享状态响应监督及直接回归对照，解析梯度均可独立差分检查。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from tactile3d.research.construction import (
    TRAIN_EDITS,
    Array,
    candidate_volumes,
    mass_backward,
    mass_forward,
)
from tactile3d.research.network import Network, huber

DEFAULT_EPOCHS = 350
DEFAULT_HIDDEN = 32
DEFAULT_RATE = 0.01
CLASS_WEIGHT = 0.15
RESPONSE_WEIGHT = 1.0


def state_loss(
    logits: Array,
    priors: Array,
    volumes: Array,
    masses: Array,
    construction: Array,
    response_weight: float,
) -> tuple[float, Array]:
    if (
        masses.shape != (len(logits), len(TRAIN_EDITS) + 1)
        or not np.isfinite(masses).all()
        or np.any(masses <= 0)
        or construction.shape != (len(logits),)
        or not np.isfinite(construction).all()
        or np.any(construction != construction.astype(int))
        or np.any((construction < 0) | (construction >= 4))
        or not np.isfinite(response_weight)
        or response_weight < 0
    ):
        raise ValueError("监督质量、构造标签或响应损失权重非法")
    predicted, cache = mass_forward(logits, priors, volumes)
    loss, dm = huber(np.log(predicted[:, :1] / masses[:, :1]))
    derivative = np.zeros_like(predicted)
    derivative[:, :1] = dm / predicted[:, :1]
    if response_weight:
        # 编辑差值允许为负，不能取对数；仅使用训练集原始质量归一化。
        residual = (
            (predicted[:, 1:] - predicted[:, :1]) - (masses[:, 1:] - masses[:, :1])
        ) / masses[:, :1]
        response_loss, dg = huber(residual)
        loss += response_weight * response_loss
        derivative[:, 1:] += response_weight * dg / masses[:, :1]
        derivative[:, :1] -= response_weight * dg.sum(1, keepdims=True) / masses[:, :1]
    gradient = mass_backward(derivative, cache)
    probability = cache["probability"]
    truth = np.eye(4)[construction.astype(int)]
    loss -= CLASS_WEIGHT * float(
        np.log(np.maximum(probability[truth > 0], 1e-300)).mean()
    )
    gradient[:, :4] += CLASS_WEIGHT * (probability - truth) / len(logits)
    return loss, gradient


def fit_shared(
    features: Array,
    priors: Array,
    volumes: Array,
    masses: Array,
    construction: Array,
    seed: int,
    epochs: int,
    hidden: int,
    response_weight: float,
) -> tuple[Network, list[dict[str, Any]]]:
    network = Network(features.shape[1], 8, hidden, seed)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        output, cache = network.forward(features)
        loss, gradient = state_loss(
            output, priors, volumes, masses, construction, response_weight
        )
        network.update(network.backward(gradient, cache), DEFAULT_RATE)
        if epoch % 10 == 0 or epoch == epochs - 1:
            history.append(
                {"epoch": epoch, "loss": loss, "seconds": time.perf_counter() - start}
            )
    return network, history


def direct_features(features: Array, edits: tuple[tuple[int, float], ...]) -> Array:
    codes = np.zeros((len(edits) + 1, 3))
    for index, (phase, density) in enumerate(edits, 1):
        if phase not in (0, 1) or density <= 0 or not np.isfinite(density):
            raise ValueError("直接回归编辑参数非法")
        codes[index] = (1, phase, np.log(density / 1000))
    return np.column_stack(
        (np.repeat(features, len(codes), axis=0), np.tile(codes, (len(features), 1)))
    )


def direct_predict(
    network: Network,
    features: Array,
    reference: Array,
    edits: tuple[tuple[int, float], ...] = TRAIN_EDITS,
) -> Array:
    output, _ = network.forward(direct_features(features, edits))
    log_mass = output[:, 0].reshape(len(features), len(edits) + 1)
    if np.any(np.abs(log_mass) > 100):
        raise ValueError("直接回归数值发散")
    return np.exp(log_mass) * reference[:, None]


def fit_direct(
    features: Array,
    reference: Array,
    masses: Array,
    construction: Array,
    seed: int,
    epochs: int,
    hidden: int,
) -> tuple[Network, list[dict[str, Any]]]:
    expanded = direct_features(features, TRAIN_EDITS)
    target = np.log(masses / reference[:, None]).ravel()
    classes = np.eye(4)[np.repeat(construction.astype(int), len(TRAIN_EDITS) + 1)]
    network = Network(expanded.shape[1], 5, hidden, seed)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        output, cache = network.forward(expanded)
        loss, gradient_mass = huber(output[:, 0] - target)
        probability = np.exp(output[:, 1:] - output[:, 1:].max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
        loss -= CLASS_WEIGHT * float(
            np.log(np.maximum(probability[classes > 0], 1e-300)).mean()
        )
        gradient = np.column_stack(
            (gradient_mass, CLASS_WEIGHT * (probability - classes) / len(classes))
        )
        network.update(network.backward(gradient, cache), DEFAULT_RATE)
        if epoch % 10 == 0 or epoch == epochs - 1:
            history.append(
                {"epoch": epoch, "loss": loss, "seconds": time.perf_counter() - start}
            )
    return network, history


def train_models(
    inputs: dict[str, Array],
    labels: dict[str, Array],
    seed: int,
    epochs: int,
    hidden: int,
) -> tuple[dict[str, Network], Array, Array, list[dict[str, Any]]]:
    if epochs < 1 or hidden < 1:
        raise ValueError("epochs和hidden必须为正整数")
    index = inputs["split"] == 0
    raw = inputs["features"][index]
    mean, scale = raw.mean(0), raw.std(0)
    scale = np.maximum(scale, 1e-6)
    features = (raw - mean) / scale
    priors = inputs["priors"][index]
    volumes = candidate_volumes(inputs["pitch_m"][index])
    masses, construction = labels["mass_kg"][index], labels["construction"][index]
    models, histories = {}, []
    for method, weight in (("mass_only", 0.0), ("shared_response", RESPONSE_WEIGHT)):
        model, history = fit_shared(
            features,
            priors,
            volumes,
            masses,
            construction,
            seed,
            epochs,
            hidden,
            weight,
        )
        models[method] = model
        histories.extend({"method": method, **row} for row in history)
    reference = priors[:, 0] * volumes[:, 0, 0]
    model, history = fit_direct(
        features, reference, masses, construction, seed, epochs, hidden
    )
    models["direct_response"] = model
    histories.extend({"method": "direct_response", **row} for row in history)
    return models, mean, scale, histories
