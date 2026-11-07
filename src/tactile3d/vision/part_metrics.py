"""与独立人工标签比较部件掩码；255是未标注，不作背景真值。"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

IGNORE_LABEL = 255
BOUNDARY_TOLERANCE_PIXELS = 2


def part_mask_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
    classes: int,
    boundary_tolerance: int = BOUNDARY_TOLERANCE_PIXELS,
) -> dict[str, Any]:
    """计算同分辨率部件IoU和边界F1，不将模型提议作为真值。

    Args:
        prediction: 自动预测，0为背景/拒绝标签，1到classes为部件。
        reference: 人工标签，0为背景，255为未标注。
        classes: 冻结标签字典的部件数量。
        boundary_tolerance: 参考图像像素单位的边界匹配容差。

    Returns:
        每类交并计数与边界计数、mIoU、有效像素数；全未标注输入拒绝。
    """
    pred, truth = np.asarray(prediction), np.asarray(reference)
    if (
        pred.ndim != 2
        or pred.shape != truth.shape
        or not pred.size
        or pred.dtype.kind not in "iu"
        or truth.dtype.kind not in "iu"
    ):
        raise ValueError("预测与人工标签必须为同尺寸的非空二维整数图")
    if type(classes) is not int or not 1 <= classes < IGNORE_LABEL:
        raise ValueError("部件数量必须是1到254的整数")
    if type(boundary_tolerance) is not int or boundary_tolerance < 0:
        raise ValueError("边界容差必须为非负整数")
    valid = truth != IGNORE_LABEL
    if not valid.any():
        raise ValueError("没有人工标注像素，不能报告分割准确率")
    if (
        pred.min() < 0
        or pred.max() > classes
        or np.any((truth[valid] < 0) | (truth[valid] > classes))
    ):
        raise ValueError("掩码包含标签字典之外的类别")
    reliable_boundary = valid.copy()
    if not valid.all():
        reliable_boundary &= distance_transform_edt(valid) > boundary_tolerance + 1
    rows = []
    for label in range(1, classes + 1):
        p, t = (pred == label) & valid, (truth == label) & valid
        intersection, union = int((p & t).sum()), int((p | t).sum())
        pb = (p & ~binary_erosion(p)) & reliable_boundary
        tb = (t & ~binary_erosion(t)) & reliable_boundary
        pc, tc = int(pb.sum()), int(tb.sum())
        matched_pred = (
            int((pb & (distance_transform_edt(~tb) <= boundary_tolerance)).sum())
            if tc
            else 0
        )
        matched_truth = (
            int((tb & (distance_transform_edt(~pb) <= boundary_tolerance)).sum())
            if pc
            else 0
        )
        precision = matched_pred / pc if pc else 0.0
        recall = matched_truth / tc if tc else 0.0
        rows.append(
            {
                "label": label,
                "reference_pixels": int(t.sum()),
                "intersection": intersection,
                "union": union,
                "iou": intersection / union if union else None,
                "predicted_boundary_pixels": pc,
                "reference_boundary_pixels": tc,
                "matched_predicted_boundary_pixels": matched_pred,
                "matched_reference_boundary_pixels": matched_truth,
                "boundary_f1": (
                    2 * precision * recall / (precision + recall)
                    if precision + recall
                    else (None if not pc and not tc else 0.0)
                ),
            }
        )
    ious = [r["iou"] for r in rows if r["iou"] is not None]
    boundaries = [r["boundary_f1"] for r in rows if r["boundary_f1"] is not None]
    foreground = valid & (truth > 0)
    return {
        "classes": rows,
        "valid_pixels": int(valid.sum()),
        "unlabeled_pixels": int((~valid).sum()),
        "mean_iou": float(np.mean(ious)) if ious else None,
        "mean_boundary_f1": float(np.mean(boundaries)) if boundaries else None,
        "foreground_label_coverage": (
            float(((pred > 0) & foreground).sum() / foreground.sum())
            if foreground.any()
            else None
        ),
        "boundary_tolerance_pixels": boundary_tolerance,
    }
