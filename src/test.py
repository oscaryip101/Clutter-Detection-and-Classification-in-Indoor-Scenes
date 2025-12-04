# src/run_diff_demo.py
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.lighting import correct_lighting

from src.differencing import (
    compute_difference_mask,
    threshold_mask,
    clean_mask,
    extract_bounding_boxes,
)


def visualize_diff_pipeline(
    tidy_path,
    cluttered_path,
    diff_thresh=20,
    clean_kernel_size=5,
    iou_thresh=0.3,
    save_path=None,
):
    """
    Run and visualize the image differencing pipeline step-by-step.

    Parameters
    ----------
    tidy_path : str
        Path to tidy (baseline) image.
    cluttered_path : str
        Path to cluttered image.
    save_path : str, optional
        Path to save the visualization figure. If None, the figure is shown
        with plt.show() instead of being saved.
    diff_thresh : int, optional
        Threshold value used in threshold_mask to binarize the difference image.
    clean_kernel_size : int, optional
        Kernel size for morphological cleaning (passed to clean_mask).
    iou_thresh : float, optional
        IoU threshold for NMS in extract_bounding_boxes.
    """
    # --- Load images ---
    tidy_bgr = cv2.imread(tidy_path)
    cluttered_bgr = cv2.imread(cluttered_path)

    if tidy_bgr is None or cluttered_bgr is None:
        raise FileNotFoundError("Could not load tidy or cluttered image.")

    # Adjust tidy image lighting to better match the cluttered image
    tidy_bgr = correct_lighting(tidy_bgr, cluttered_bgr)

    # Convert BGR -> RGB for plotting
    tidy_rgb = cv2.cvtColor(tidy_bgr, cv2.COLOR_BGR2RGB)
    cluttered_rgb = cv2.cvtColor(cluttered_bgr, cv2.COLOR_BGR2RGB)

    # --- Step 1: compute grayscale absolute difference map ---
    diff_gray = compute_difference_mask(tidy_rgb, cluttered_rgb)

    # --- Step 2: threshold to obtain a raw binary mask ---
    diff_mask_raw = threshold_mask(diff_gray, thresh=diff_thresh)

    # For visualization, we can show the grayscale difference image
    abs_diff_gray = diff_gray

    # --- Step 3: clean the mask (morphological operations) ---
    cleaned_mask = clean_mask(diff_mask_raw, kernel_size=clean_kernel_size)

    # --- Step 4: extract bounding boxes (with NMS inside) ---
    boxes = extract_bounding_boxes(cleaned_mask, iou_thresh=iou_thresh)

    # --- Step 5: draw boxes on cluttered image ---
    cluttered_with_boxes = cluttered_rgb.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(
            cluttered_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2
        )  # red boxes

    # --- Plot everything ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    axes[0].imshow(tidy_rgb)
    axes[0].set_title("Tidy image")
    axes[0].axis("off")

    axes[1].imshow(cluttered_rgb)
    axes[1].set_title("Cluttered image")
    axes[1].axis("off")

    axes[2].imshow(abs_diff_gray, cmap="gray")
    axes[2].set_title("Absolute difference (grayscale)")
    axes[2].axis("off")

    axes[3].imshow(diff_mask_raw, cmap="gray")
    axes[3].set_title(f"Raw diff mask (thresh={diff_thresh})")
    axes[3].axis("off")

    axes[4].imshow(cleaned_mask, cmap="gray")
    axes[4].set_title(f"Cleaned mask (kernel={clean_kernel_size})")
    axes[4].axis("off")

    axes[5].imshow(cluttered_with_boxes)
    axes[5].set_title(f"Final boxes (N={len(boxes)}, IoU={iou_thresh})")
    axes[5].axis("off")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    print(f"  -> num boxes after NMS: {len(boxes)}")

    return {
        "boxes": boxes,
        "diff_gray": diff_gray,
        "diff_mask_raw": diff_mask_raw,
        "cleaned_mask": cleaned_mask,
    }


def main():
    # Construct paths relative to the project root (one level above src)
    project_root = os.path.dirname(os.path.dirname(__file__))
    tidy_path = os.path.join(project_root, "data", "tidy", "10.png")
    cluttered_path = os.path.join(project_root, "data", "cluttered", "10.png")

    os.makedirs(os.path.join(project_root, "data", "output"), exist_ok=True)

    # Hyperparameter combinations for differencing only
    configs = [
        {
            "name": "diff1",
            "diff_thresh": 15,
            "clean_kernel_size": 3,
            "iou_thresh": 0.3,
        },
        {
            "name": "diff2",
            "diff_thresh": 25,
            "clean_kernel_size": 5,
            "iou_thresh": 0.3,
        },
        {
            "name": "diff3",
            "diff_thresh": 30,
            "clean_kernel_size": 7,
            "iou_thresh": 0.5,
        },
    ]

    for cfg in configs:
        print(f"\n=== Running differencing config {cfg['name']} ===")

        result = visualize_diff_pipeline(
            tidy_path=tidy_path,
            cluttered_path=cluttered_path,
            diff_thresh=cfg["diff_thresh"],
            clean_kernel_size=cfg["clean_kernel_size"],
            iou_thresh=cfg["iou_thresh"],
            save_path=None,  # show plots instead of saving
        )

        print("  -> boxes:", len(result["boxes"]))


if __name__ == "__main__":
    main()