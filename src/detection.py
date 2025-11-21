# src/detection.py
"""
YOLO detection module using ultralytics YOLOv8 (PyTorch backend).

We keep the same interface:
    - load_yolo() -> model
    - run_yolo(model, image_bgr) -> list of (x, y, w, h, cls_id, conf)
"""

import cv2
import numpy as np
from ultralytics import YOLO


def load_yolo(
    model_path: str = "yolov8n.pt",
    conf_threshold: float = 0.05,
    iou_threshold: float = 0.45,
):
    """
    Load a pretrained YOLOv8 model.

    Parameters
    ----------
    model_path : str
        Path or name of YOLOv8 model. "yolov8n.pt" will automatically download the official nano copyrighted version from the internet.
    conf_threshold : float
        Confidence threshold for predictions.
    iou_threshold : float
        IoU threshold for NMS.

    Returns
    -------
    model : ultralytics.YOLO
        YOLO model instance.
    """
    model = YOLO(model_path)

    model.overrides["conf"] = conf_threshold
    model.overrides["iou"] = iou_threshold
    model.overrides["verbose"] = False

    return model


def run_yolo(model, image_bgr):
    """
    Run YOLOv8 on a BGR OpenCV image and return a list of detections.

    Parameters
    ----------
    model : ultralytics.YOLO
        Model returned by load_yolo().
    image_bgr : np.ndarray
        Input image in BGR (OpenCV) format.

    Returns
    -------
    boxes : list of tuples
        Each tuple is (x, y, w, h, cls_id, conf) in ORIGINAL image coordinates.
    """
    # YOLOv8 RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = model(image_rgb)[0]  # Retrieve the result of the 0th image in the batch.

    # results.boxes:
    #   .xyxy -> (N, 4): x1, y1, x2, y2
    #   .conf -> (N,)
    #   .cls  -> (N,)
    if results.boxes is None:
        return []

    boxes_xyxy = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    clses = results.boxes.cls.cpu().numpy().astype(int)

    detections = []
    for (x1, y1, x2, y2), conf, cls_id in zip(boxes_xyxy, confs, clses):
        # Convert to int xywh format, consistent with that in pipeline.py.
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        w = max(0, x2 - x1)
        h = max(0, y2 - y1)

        detections.append((x1, y1, w, h, int(cls_id), float(conf)))

    return detections
