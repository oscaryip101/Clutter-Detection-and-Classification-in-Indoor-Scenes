# src/detection.py
from ultralytics import YOLO
import numpy as np


def load_yolo(
    model_path: str = "yolo12m.pt",
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    """
    Load a YOLOv8 model from Ultralytics.

    Parameters
    ----------
    model_path : str
        Path or preset name for the YOLO model weights.
    conf_threshold : float
        Confidence threshold for predictions.
    iou_threshold : float
        IoU threshold for NMS inside YOLO.

    Returns
    -------
    model : YOLO
        Ultralytics YOLO model with stored conf / iou attributes.
    """
    model = YOLO(model_path)
    # Store thresholds on the model so run_yolo can reuse them
    model.conf = conf_threshold
    model.iou = iou_threshold
    return model


def run_yolo(model, image):
    """
    Run YOLO detection on a BGR image and return boxes with class info.

    Parameters
    ----------
    model : YOLO
        Ultralytics YOLO model loaded by load_yolo().
    image : np.ndarray
        BGR image (H, W, 3).

    Returns
    -------
    boxes : list of tuples
        Each element is (x, y, w, h, cls_id, conf, cls_name):
        - (x, y, w, h) in pixel coordinates
        - cls_id : int, COCO class index
        - conf   : float, confidence score
        - cls_name : str, human-readable class name (e.g., "bottle")
    """
    if image is None:
        return []

    # Run model prediction
    results = model.predict(
        source=image,
        conf=getattr(model, "conf", 0.25),
        iou=getattr(model, "iou", 0.45),
        verbose=False,
    )

    if not results:
        return []

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return []

    boxes_xyxy = r.boxes.xyxy.cpu().numpy()  # (N, 4)
    cls_ids = r.boxes.cls.cpu().numpy()      # (N,)
    confs = r.boxes.conf.cpu().numpy()       # (N,)
    names = r.names                          # dict: id -> name

    out_boxes = []
    for (x1, y1, x2, y2), c, p in zip(boxes_xyxy, cls_ids, confs):
        x1 = float(x1)
        y1 = float(y1)
        x2 = float(x2)
        y2 = float(y2)
        w = x2 - x1
        h = y2 - y1

        cls_id = int(c)
        conf = float(p)
        if isinstance(names, dict):
            cls_name = names.get(cls_id, str(cls_id))
        else:
            cls_name = str(cls_id)

        out_boxes.append((x1, y1, w, h, cls_id, conf, cls_name))

    return out_boxes
