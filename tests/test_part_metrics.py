from __future__ import annotations

import numpy as np
import pytest

from tactile3d.vision.part_metrics import part_mask_metrics


def test_perfect_mask_and_small_missing_part_are_distinguished() -> None:
    truth = np.zeros((32, 32), np.uint8)
    truth[3:27, 3:27] = 1
    truth[12:16, 12:16] = 2
    exact = part_mask_metrics(truth, truth, 2)
    assert exact["mean_iou"] == 1
    assert exact["mean_boundary_f1"] == 1
    missing = truth.copy()
    missing[missing == 2] = 1
    failed = part_mask_metrics(missing, truth, 2)
    assert failed["classes"][1]["iou"] == 0
    assert failed["mean_iou"] < 0.5


def test_unlabeled_pixels_are_never_used_as_background_truth() -> None:
    truth = np.full((32, 32), 255, np.uint8)
    with pytest.raises(ValueError, match="没有人工标注"):
        part_mask_metrics(np.zeros_like(truth), truth, 2)
    truth[10:20, 10:20] = 1
    prediction = np.full((32, 32), 2, np.uint8)
    prediction[10:20, 10:20] = 1
    result = part_mask_metrics(prediction, truth, 2)
    assert result["mean_iou"] == 1
    assert result["valid_pixels"] == 100
