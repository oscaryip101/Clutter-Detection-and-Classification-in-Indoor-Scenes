# src/metrics.py
from typing import List, Tuple
import numpy as np

# (x, y, w, h)
Box = Tuple[int, int, int, int]

def box_iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union == 0:
        return 0.0
    return inter_area / union


def compute_prf(
    preds: List[List[Box]],
    gts: List[List[Box]],
    iou_thresh: float = 0.5,
):
    """
    preds / gts: A list of bboxes organized by image, for example:
      preds = [[(x,y,w,h), ...],  # image 1
               [(x,y,w,h), ...],  # image 2
               ...]
    """
    assert len(preds) == len(gts)
    TP = 0
    FP = 0
    FN = 0

    for pred_boxes, gt_boxes in zip(preds, gts):
        gt_used = [False] * len(gt_boxes)

        # For each prediction box, find the ground truth (GT) with the largest IoU.
        for pb in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for i, gb in enumerate(gt_boxes):
                iou = box_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= iou_thresh and best_idx >= 0 and not gt_used[best_idx]:
                TP += 1
                gt_used[best_idx] = True
            else:
                FP += 1

        # Unmatched GTs are considered FNs.
        FN += sum(1 for used in gt_used if not used)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return dict(TP=TP, FP=FP, FN=FN,
                precision=precision, recall=recall, f1=f1)
