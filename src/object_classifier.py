# src/object_classifier.py
from ultralytics import YOLO
import cv2
import numpy as np
from typing import Tuple, List


def load_object_classifier(
    model_path: str = r"runs/classify/train_full/weights/best.pt",
):
    """
    Load the trained YOLO classification model.
    """
    model = YOLO(model_path)
    return model


def classify_patch(model, patch_bgr: np.ndarray) -> Tuple[int, float, str]:
    """
    Classify a single object patch (BGR image).

    Returns
    -------
    cls_id : int
        Predicted class index.
    conf : float
        Confidence score.
    cls_name : str
        Class name string.
    """
    if patch_bgr is None or patch_bgr.size == 0:
        return -1, 0.0, "invalid"

    patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
    results = model(patch_rgb, verbose=False)
    r = results[0]
    probs = r.probs

    cls_id = int(probs.top1)
    conf = float(probs.top1conf)

    names = model.names
    if isinstance(names, dict):
        cls_name = names.get(cls_id, str(cls_id))
    else:
        cls_name = names[cls_id] if 0 <= cls_id < len(names) else str(cls_id)

    return cls_id, conf, cls_name


def classify_patches_batch(
    model,
    patches_bgr: List[np.ndarray],
    conf_thresh: float = 0.0,
):
    """
    Classify a list of patches. Returns list of (cls_id, conf, cls_name).
    """
    results_list = []
    for patch in patches_bgr:
        cls_id, conf, cls_name = classify_patch(model, patch)
        if conf_thresh > 0.0 and conf < conf_thresh:
            results_list.append((-1, conf, "low_conf"))
        else:
            results_list.append((cls_id, conf, cls_name))
    return results_list
