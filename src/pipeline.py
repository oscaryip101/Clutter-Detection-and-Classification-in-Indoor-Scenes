# src/pipeline.py
import numpy as np

from src.alignment import align_images
from src.differencing import (
    compute_difference_mask,
    threshold_mask,
    clean_mask,
    extract_bounding_boxes,
)
from src.detection import load_yolo, run_yolo
from src.lighting import correct_lighting
from src.utils import load_image, save_image, draw_boxes


def box_iou_xywh(box_a, box_b):
    """
    Compute IoU between two boxes given in (x, y, w, h) format.
    """
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def remove_static_objects(clutter_boxes, tidy_boxes, static_iou_thresh=0.5):
    """
    Remove boxes in clutter_boxes that have high IoU with any box in tidy_boxes
    and share the same class id.

    Parameters
    ----------
    clutter_boxes : list[(x, y, w, h, cls_id, conf, cls_name)]
    tidy_boxes    : list[(x, y, w, h, cls_id, conf, cls_name)]
    static_iou_thresh : float
        IoU threshold above which a clutter box is considered a "pre-existing" object.

    Returns
    -------
    final_boxes : list[(x, y, w, h, cls_id, conf, cls_name)]
        Boxes that are likely to correspond to newly added objects.
    """
    final_boxes = []

    # 提取 tidy 里的 (box, cls_id)
    tidy_items = [ (tuple(b[0:4]), int(b[4])) for b in tidy_boxes ]

    for box in clutter_boxes:
        c_xywh = tuple(box[0:4])
        c_cls  = int(box[4])

        is_static = False
        for t_xywh, t_cls in tidy_items:
            if c_cls != t_cls:
                continue

            iou = box_iou_xywh(c_xywh, t_xywh)
            if iou >= static_iou_thresh:
                is_static = True
                break

        if not is_static:
            final_boxes.append(box)

    return final_boxes


def run_pipeline(
    tidy_path,
    cluttered_path,
    save_path=None,
    diff_thresh=20,
    yolo_conf=0.25,
    yolo_iou=0.45,
    overlap_ratio_thresh=0.0,
    static_iou_thresh=0.5,
    use_lighting=True,
):
    """
    Full clutter detection pipeline.

    Steps:
    1. Load images
    2. Align tidy → cluttered
    3. (Optional) Adjust lighting of aligned tidy to match cluttered
    4. Compute difference mask (absdiff → threshold → clean)
    5. Extract difference bounding boxes (blue boxes)
    6. Run YOLO on both cluttered image and aligned tidy image
    7. Filter clutter YOLO boxes using difference mask (overlap ratio)
    8. Remove boxes that also appear in tidy YOLO detections (static objects)
    9. Draw boxes and optionally save output
    """
    # 1. Load
    tidy = load_image(tidy_path)
    cluttered = load_image(cluttered_path)

    # 2. Align
    aligned, H, matches = align_images(tidy, cluttered)
    if aligned is None:
        print("Alignment failed. Exiting pipeline.")
        return {"error": "homography_failed"}

    # 3. Lighting correction
    if use_lighting:
        aligned = correct_lighting(aligned, cluttered)

    # 4. Differencing
    diff_gray = compute_difference_mask(aligned, cluttered)
    mask = threshold_mask(diff_gray, thresh=diff_thresh)
    cleaned = clean_mask(mask)

    # 5. Diff boxes (blue)
    diff_boxes = extract_bounding_boxes(cleaned)

    # 6. YOLO on cluttered + aligned tidy
    yolo_model = load_yolo(
        conf_threshold=yolo_conf,
        iou_threshold=yolo_iou,
    )

    yolo_boxes_clutter = run_yolo(yolo_model, cluttered)
    yolo_boxes_tidy = run_yolo(yolo_model, aligned)

    # 7. Filter clutter YOLO boxes by diff mask overlap
    filtered_boxes = []
    h_mask, w_mask = cleaned.shape[:2]

    for box in yolo_boxes_clutter:
        x, y, w, h = box[0:4]

        x0 = max(0, int(x))
        y0 = max(0, int(y))
        x1 = min(w_mask, int(x + w))
        y1 = min(h_mask, int(y + h))

        if x1 <= x0 or y1 <= y0:
            continue

        submask = cleaned[y0:y1, x0:x1]
        if submask.size == 0:
            continue

        overlap_pixels = np.count_nonzero(submask > 0)
        ratio = overlap_pixels / float(submask.size)

        if overlap_ratio_thresh <= 0.0:
            if overlap_pixels > 0:
                filtered_boxes.append(box)
        else:
            if ratio >= overlap_ratio_thresh:
                filtered_boxes.append(box)

    # 8. Remove static objects (appear in tidy YOLO detections)
    final_boxes = remove_static_objects(
        filtered_boxes, yolo_boxes_tidy, static_iou_thresh=static_iou_thresh
    )

    # 9. Draw: green = final clutter (带标签), blue = diff regions
    output = cluttered.copy()
    output = draw_boxes(output, final_boxes, color=(0, 255, 0))
    output = draw_boxes(output, diff_boxes, color=(255, 0, 0))

    # 10. Save
    if save_path is not None:
        save_image(save_path, output)

    return {
        "aligned": aligned,
        "diff_gray": diff_gray,
        "mask": mask,
        "cleaned": cleaned,
        "diff_boxes": diff_boxes,
        "yolo_boxes_clutter": yolo_boxes_clutter,
        "yolo_boxes_tidy": yolo_boxes_tidy,
        "filtered_boxes": filtered_boxes,
        "final_boxes": final_boxes,
        "output": output,
    }
