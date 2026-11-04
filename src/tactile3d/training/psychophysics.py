from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

from tactile3d.utils.io import read_jsonl, write_json


def fit_bradley_terry(
    comparisons_path: str | Path,
    output_path: str | Path | None = None,
    *,
    l2: float = 1e-3,
) -> dict[str, Any]:
    """Fit latent human-expectation scores from pairwise judgements.

    Input rows use ``item_a``, ``item_b``, ``choice`` (``a``, ``b`` or
    ``tie``), and an optional ``split``. Only train rows fit the scale; held-out
    rows report predictive agreement and remain independent validation data.
    """

    rows = read_jsonl(comparisons_path)
    if not rows:
        raise ValueError("No psychophysical comparisons were provided")
    items = sorted({str(row[key]) for row in rows for key in ("item_a", "item_b")})
    if len(items) < 2:
        raise ValueError("At least two distinct items are required")
    index = {item: idx for idx, item in enumerate(items)}
    train_rows = [row for row in rows if str(row.get("split", "train")) == "train"]
    if not train_rows:
        raise ValueError("No rows belong to the train split")
    _check_connected(train_rows, items)

    def objective(free_scores: np.ndarray) -> float:
        scores = np.concatenate([[0.0], free_scores])
        loss = 0.5 * l2 * float(np.dot(free_scores, free_scores))
        for row in train_rows:
            delta = scores[index[str(row["item_a"])]] - scores[index[str(row["item_b"])]]
            target = _target(row.get("choice"))
            weight = float(row.get("weight", 1.0))
            loss += weight * (np.logaddexp(0.0, delta) - target * delta)
        return float(loss)

    optimized = minimize(objective, np.zeros(len(items) - 1), method="BFGS")
    if not optimized.success:
        raise RuntimeError(f"Bradley-Terry optimization failed: {optimized.message}")
    scores = np.concatenate([[0.0], optimized.x])
    scores -= scores.mean()
    scale = scores.std()
    if scale > 0:
        scores /= scale

    metrics = {}
    for split in sorted({str(row.get("split", "train")) for row in rows}):
        metrics[split] = _evaluate(
            [row for row in rows if str(row.get("split", "train")) == split], scores, index
        )
    result = {
        "schema_version": 1,
        "model": "Bradley-Terry",
        "score_semantics": "higher means more of the named perceptual attribute",
        "num_items": len(items),
        "num_train_comparisons": len(train_rows),
        "scores": {item: float(scores[idx]) for item, idx in index.items()},
        "metrics": metrics,
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def _target(choice: Any) -> float:
    normalized = str(choice).strip().lower()
    if normalized in {"a", "item_a", "left", "1"}:
        return 1.0
    if normalized in {"b", "item_b", "right", "0"}:
        return 0.0
    if normalized in {"tie", "equal", "0.5"}:
        return 0.5
    raise ValueError(f"Unsupported pairwise choice: {choice!r}")


def _evaluate(rows: list[dict[str, Any]], scores: np.ndarray, index: dict[str, int]) -> dict:
    if not rows:
        return {"count": 0, "nll": None, "accuracy_without_ties": None}
    nll = 0.0
    correct = 0
    comparable = 0
    for row in rows:
        delta = scores[index[str(row["item_a"])]] - scores[index[str(row["item_b"])]]
        probability = 1.0 / (1.0 + math.exp(-float(np.clip(delta, -40, 40))))
        target = _target(row.get("choice"))
        nll -= target * math.log(max(probability, 1e-12))
        nll -= (1.0 - target) * math.log(max(1.0 - probability, 1e-12))
        if target != 0.5:
            comparable += 1
            correct += int((probability >= 0.5) == (target == 1.0))
    return {
        "count": len(rows),
        "nll": nll / len(rows),
        "accuracy_without_ties": correct / comparable if comparable else None,
    }


def _check_connected(rows: list[dict[str, Any]], items: list[str]) -> None:
    graph: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        left, right = str(row["item_a"]), str(row["item_b"])
        graph[left].add(right)
        graph[right].add(left)
    visited = set()
    stack = [items[0]]
    while stack:
        item = stack.pop()
        if item in visited:
            continue
        visited.add(item)
        stack.extend(graph[item] - visited)
    missing = sorted(set(items) - visited)
    if missing:
        raise ValueError(f"Training comparison graph is disconnected; unreachable items: {missing}")
