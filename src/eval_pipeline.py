# src/eval_pipeline.py
import os
from typing import List, Tuple
from .pipeline import run_pipeline
from .metrics import compute_prf

GROUND_TRUTH = {
    "1.png": [(100, 120, 60, 80), (250, 200, 40, 50)],
    # "2.png": [...],
}

def eval_once(tidy_dir: str, cluttered_dir: str):
    pred_all: List[List[Tuple[int,int,int,int]]] = []
    gt_all: List[List[Tuple[int,int,int,int]]] = []

    for fname, gt_boxes in GROUND_TRUTH.items():
        tidy_path = os.path.join(tidy_dir, fname)
        cluttered_path = os.path.join(cluttered_dir, fname)

        result = run_pipeline(tidy_path, cluttered_path)
        # Using filtered_boxes as Clutter prediction
        pred_boxes = [(x, y, w, h) for (x, y, w, h, cls_id, conf) in result["filtered_boxes"]]

        pred_all.append(pred_boxes)
        gt_all.append(gt_boxes)

    stats = compute_prf(pred_all, gt_all, iou_thresh=0.5)
    print("Eval results:", stats)


if __name__ == "__main__":
    eval_once("../data/tidy", "../data/cluttered")
