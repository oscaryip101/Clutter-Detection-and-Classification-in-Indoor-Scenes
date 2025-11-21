from .alignment import align_images
from .differencing import (
    compute_difference_mask,
    threshold_mask,
    clean_mask,
    extract_bounding_boxes
)
from .detection import load_yolo, run_yolo
from .utils import load_image, save_image, draw_boxes
import numpy as np


def run_pipeline(
    tidy_path,
    cluttered_path,
    save_path=None,
    diff_thresh: int = 20,           # NEW: Differential threshold
    yolo_conf: float = 0.25,         # NEW: YOLO Confidence threshold
    yolo_iou: float = 0.45,          # NEW: YOLO NMS IoU threshold
    overlap_ratio_thresh: float = 0.0  # NEW: The overlap ratio threshold between the differential and the detection box
):
    """
    Full clutter detection pipeline.

    Steps:
    1. Load images
    2. Align tidy → cluttered
    3. Compute difference mask
    4. Threshold + clean mask
    5. Extract difference bounding boxes
    6. Run YOLO on cluttered
    7. Filter YOLO boxes using difference mask
    8. Draw boxes and optionally save output

    Returns
    -------
    result : dict
        {
            "aligned": aligned image,
            "diff_gray": raw abs-diff map,
            "mask": thresholded mask,
            "cleaned": cleaned mask,
            "diff_boxes": boxes from differencing,
            "yolo_boxes": YOLO detections,
            "filtered_boxes": YOLO detections that overlap changes,
            "output": drawn output image
        }
    """
    # ------------------------------
    # 1. Load images
    # ------------------------------
    tidy = load_image(tidy_path)
    cluttered = load_image(cluttered_path)

    # ------------------------------
    # 2. Align tidy → cluttered
    # ------------------------------
    aligned, H, matches = align_images(tidy, cluttered)
    if aligned is None:
        print("Alignment failed. Exiting pipeline.")
        return {"error": "homography_failed"}

    # ------------------------------
    # 3. Compute difference mask
    # ------------------------------
    diff_gray = compute_difference_mask(aligned, cluttered)
    mask = threshold_mask(diff_gray, thresh=diff_thresh)  # NEW: 使用传入的阈值
    cleaned = clean_mask(mask)

    # ------------------------------
    # 4. Extract bounding boxes
    # ------------------------------
    diff_boxes = extract_bounding_boxes(cleaned)

    # ------------------------------
    # 5. Run YOLO detection on cluttered image
    # ------------------------------
    yolo_model = load_yolo(
        conf_threshold=yolo_conf,
        iou_threshold=yolo_iou
    )
    yolo_boxes = run_yolo(yolo_model, cluttered)

    # ------------------------------
    # 6. Filter YOLO boxes using differencing mask
    # (Keep only YOLO detections overlapping changed regions)
    # ------------------------------
    filtered_boxes = []
    for box in yolo_boxes:
        x, y, w, h, cls_id, conf = box

        # crop mask
        submask = cleaned[y:y + h, x:x + w]
        if submask.size == 0:
            continue

        if overlap_ratio_thresh <= 0.0:
            if np.any(submask > 0):
                filtered_boxes.append(box)
        else:
            overlap_ratio = np.count_nonzero(submask > 0) / submask.size
            if overlap_ratio >= overlap_ratio_thresh:
                filtered_boxes.append(box)

    # ------------------------------
    # 7. Draw boxes
    # ------------------------------
    output = cluttered.copy()
    output = draw_boxes(output, filtered_boxes, color=(0, 255, 0))  # green clutter boxes
    output = draw_boxes(output, diff_boxes, color=(255, 0, 0))      # blue diff regions

    # ------------------------------
    # 8. Save image (optional)
    # ------------------------------
    if save_path is not None:
        save_image(save_path, output)

    # ------------------------------
    # Return all data for debugging
    # ------------------------------
    return {
        "aligned": aligned,
        "diff_gray": diff_gray,
        "mask": mask,
        "cleaned": cleaned,
        "diff_boxes": diff_boxes,
        "yolo_boxes": yolo_boxes,
        "filtered_boxes": filtered_boxes,
        "output": output
    }
