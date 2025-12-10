# src/run_diff_demo.py
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from src.lighting import correct_lighting
from src.alignment import align_images

# Global toggle: If True, save all figures instead of showing them
SAVE_IMAGES = True  # Set to True to save all figures instead of showing them


from src.differencing import (
    compute_difference_mask,
    threshold_mask,
    clean_mask,
    extract_bounding_boxes,
    crop_patches_from_boxes,
)


from src.object_classifier import (
    load_object_classifier,
    classify_patches_batch,
)

# --- Helper: Refine low-confidence, large boxes with a second-pass differencing ---
def refine_boxes_with_second_pass(
    tidy_bgr,
    cluttered_bgr,
    boxes,
    predictions,
    classifier_model,
    conf_low_thresh=0.2,
    noise_box_area_frac=0.05,
    big_box_area_frac=0.1,
    refine_diff_thresh=30,
    refine_kernel_size=1,
    refine_iou_thresh=0.5,
    debug_save_dir=None,
    debug_prefix="",
):
    """
    Refine low-confidence detections by re-running differencing on large boxes.

    For each box:
      - If box area is smaller than noise_box_area_frac of the full image: drop the box (noise), regardless of confidence.
      - Else if confidence >= conf_low_thresh: keep the box and its prediction.
      - Else if confidence < conf_low_thresh and the box area is smaller than big_box_area_frac of the full image: drop the box.
      - Else (confidence < conf_low_thresh and box area >= big_box_area_frac): run a second-pass differencing inside that box with a (potentially) different configuration, then classify the resulting sub-boxes.

    Parameters
    ----------
    tidy_bgr : np.ndarray
        Tidy image in BGR format (already lighting-corrected).
    cluttered_bgr : np.ndarray
        Cluttered image in BGR format.
    boxes : list of (x, y, w, h)
        Original bounding boxes from the first differencing pass.
    predictions : list of (cls_id, conf, cls_name)
        Classification predictions for each box.
    classifier_model : Any
        Loaded object classifier model used by classify_patches_batch.
    conf_low_thresh : float, optional
        Confidence threshold below which a detection is considered low-confidence.
    noise_box_area_frac : float, optional
        Minimum fraction of the full image area for a box to NOT be considered noise. Boxes smaller than this are always dropped.
    big_box_area_frac : float, optional
        Minimum fraction of the full image area for a box to be considered "big" and eligible for second-pass refinement.
    refine_diff_thresh : int, optional
        Threshold for the second-pass differencing inside a big, low-confidence box.
    refine_kernel_size : int, optional
        Kernel size for morphological cleaning in the second-pass.
    refine_iou_thresh : float, optional
        IoU threshold for NMS in the second-pass.
    debug_save_dir : str or None, optional
        If not None, directory where debug figures for refinement will be saved instead of shown.
    debug_prefix : str, optional
        Prefix to use when naming debug image files for refinement steps.

    Returns
    -------
    refined_boxes : list of (x, y, w, h)
        Final list of boxes after refinement.
    refined_predictions : list of (cls_id, conf, cls_name)
        Corresponding predictions for the refined boxes.
    """
    dropped_count = 0
    second_pass_count = 0
    dropped_boxes_global = []

    if not boxes:
        return [], []

    img_h, img_w = cluttered_bgr.shape[:2]
    full_area = float(img_h * img_w)

    refined_boxes = []
    refined_preds = []

    for idx, (x, y, w, h) in enumerate(boxes):
        # If we don't have a prediction for this box, just keep it as-is.
        if idx >= len(predictions):
            refined_boxes.append((x, y, w, h))
            refined_preds.append(None)
            continue

        cls_id, conf, cls_name = predictions[idx]

        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(img_w, x + w)
        y1 = min(img_h, y + h)

        box_area = float(w * h)
        area_frac = box_area / full_area if full_area > 0 else 0.0

        # Case 0: very small box (noise) → always drop, ignore confidence
        if area_frac < noise_box_area_frac:
            print(
                f"[Refine] Dropped NOISE box idx={idx}, conf={conf:.3f}, area_frac={area_frac:.3f}"
            )
            # Accumulate for global dropped-box visualization
            dropped_boxes_global.append((x0, y0, x1, y1))
            dropped_count += 1
            continue

        # Case 1: confidence is high enough → keep as-is.
        if conf >= conf_low_thresh:
            refined_boxes.append((x, y, w, h))
            refined_preds.append((cls_id, conf, cls_name))
            continue

        # Now we know: conf < conf_low_thresh (low-confidence) and box is at least noise_box_area_frac

        # Case 2: not big enough to refine (mid-size, low-confidence) → drop
        if area_frac < big_box_area_frac:
            print(
                f"[Refine] Dropped LOW-CONF mid-size box idx={idx}, "
                f"conf={conf:.3f}, area_frac={area_frac:.3f}"
            )
            # Accumulate for global dropped-box visualization
            dropped_boxes_global.append((x0, y0, x1, y1))
            dropped_count += 1
            continue

        # Case 3: big AND low-confidence → second-pass differencing
        tidy_patch = tidy_bgr[y0:y1, x0:x1, :]
        clutter_patch = cluttered_bgr[y0:y1, x0:x1, :]

        # Second-pass differencing on the patch with different hyperparameters
        diff_gray_local = compute_difference_mask(tidy_patch, clutter_patch)
        diff_mask_local = threshold_mask(diff_gray_local, thresh=refine_diff_thresh)
        cleaned_local = clean_mask(diff_mask_local, kernel_size=refine_kernel_size)
        sub_boxes = extract_bounding_boxes(cleaned_local, iou_thresh=refine_iou_thresh)

        if not sub_boxes:
            # Nothing found in second pass; effectively drop this region.
            continue

        # Plot differencing steps for this refined patch to help tuning
        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.imshow(diff_gray_local, cmap="gray")
        plt.title("Local diff_gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(diff_mask_local, cmap="gray")
        plt.title(f"Local mask (th={refine_diff_thresh})")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cleaned_local, cmap="gray")
        plt.title(f"Local cleaned (k={refine_kernel_size})")
        plt.axis("off")

        plt.tight_layout()
        if debug_save_dir is not None:
            fname = f"{debug_prefix}_box{idx}_secondpass_diffsteps.png"
            out_path = os.path.join(debug_save_dir, fname)
            plt.savefig(out_path, dpi=150)
            plt.close()
        else:
            plt.show()

        # Log that we are doing a second-pass differencing for this box
        print(
            f"[Refine] Second-pass differencing for box idx={idx}, "
            f"conf={conf:.3f}, area_frac={area_frac:.3f}, sub_boxes={len(sub_boxes)}"
        )

        # Debug plot: show the clutter patch with second-pass sub-boxes overlaid
        debug_img = clutter_patch.copy()
        for (sx, sy, sw, sh) in sub_boxes:
            cv2.rectangle(debug_img, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)

        plt.figure()
        plt.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Second-pass refinement for box idx={idx}")
        plt.axis("off")
        if debug_save_dir is not None:
            fname = f"{debug_prefix}_box{idx}_secondpass_subboxes.png"
            out_path = os.path.join(debug_save_dir, fname)
            plt.savefig(out_path, dpi=150)
            plt.close()
        else:
            plt.show()

        second_pass_count += 1

        # Classify each sub-box patch
        sub_patches_bgr = crop_patches_from_boxes(
            clutter_patch,
            sub_boxes,
            margin=2,
            resize_to=None,
        )
        sub_preds = classify_patches_batch(
            classifier_model, sub_patches_bgr, conf_thresh=0.0
        )

        # Map local sub-box coordinates back to global image coordinates,
        # but drop tiny sub-boxes that are below the noise_box_area_frac threshold.
        for (sx, sy, sw, sh), sub_pred in zip(sub_boxes, sub_preds):
            gx = x0 + sx
            gy = y0 + sy

            sub_area = float(sw * sh)
            sub_area_frac = sub_area / full_area if full_area > 0 else 0.0

            # Treat very small sub-boxes as noise as well
            if sub_area_frac < noise_box_area_frac:
                print(
                    f"[Refine] Dropped NOISE sub-box from second-pass, "
                    f"area_frac={sub_area_frac:.3f}"
                )
                dropped_boxes_global.append((gx, gy, gx + sw, gy + sh))
                dropped_count += 1
                continue

            refined_boxes.append((gx, gy, sw, sh))
            refined_preds.append(sub_pred)

    # If we dropped any boxes, visualize them all on one image
    if dropped_boxes_global:
        dropped_vis = cluttered_bgr.copy()
        for (dx0, dy0, dx1, dy1) in dropped_boxes_global:
            cv2.rectangle(dropped_vis, (dx0, dy0), (dx1, dy1), (0, 0, 255), 2)
        plt.figure()
        plt.imshow(cv2.cvtColor(dropped_vis, cv2.COLOR_BGR2RGB))
        plt.title(f"All dropped boxes (count={len(dropped_boxes_global)})")
        plt.axis("off")
        plt.tight_layout()
        if debug_save_dir is not None:
            fname = f"{debug_prefix}_dropped_boxes.png"
            out_path = os.path.join(debug_save_dir, fname)
            plt.savefig(out_path, dpi=150)
            plt.close()
        else:
            plt.show()

    if dropped_count > 0 or second_pass_count > 0:
        print(f"[Refine] Summary: dropped {dropped_count}, second-pass: {second_pass_count}")

    return refined_boxes, refined_preds


def visualize_diff_pipeline(
    tidy_path,
    cluttered_path,
    diff_thresh=20,
    clean_kernel_size=5,
    iou_thresh=0.3,
    save_path=None,
    classifier_model=None,
    debug_save_dir=None,
    debug_prefix="",
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
    debug_save_dir : str or None, optional
        If not None, directory where intermediate refinement figures will be saved instead of shown.
    debug_prefix : str, optional
        Prefix to use when naming debug image files for this image pair/config.
    """
    # --- Load images ---
    tidy_bgr = cv2.imread(tidy_path)
    cluttered_bgr = cv2.imread(cluttered_path)

    if tidy_bgr is None or cluttered_bgr is None:
        raise FileNotFoundError("Could not load tidy or cluttered image.")

    # --- Geometric alignment: warp tidy into the cluttered frame ---
    aligned_tidy_bgr, H, matches = align_images(tidy_bgr, cluttered_bgr)
    if aligned_tidy_bgr is None:
        print("[Align] Homography estimation failed. Proceeding without geometric alignment.")
        tidy_aligned_bgr = tidy_bgr
    else:
        tidy_aligned_bgr = aligned_tidy_bgr

    # --- Photometric alignment: adjust tidy lighting to match cluttered ---
    tidy_aligned_bgr = correct_lighting(tidy_aligned_bgr, cluttered_bgr)

    # Convert BGR -> RGB for plotting
    tidy_rgb = cv2.cvtColor(tidy_aligned_bgr, cv2.COLOR_BGR2RGB)
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

    # --- Optional: classify each detected region using the trained classifier ---
    predictions = []
    if classifier_model is not None and len(boxes) > 0:
        # Crop BGR patches from the cluttered image; YOLO classifier will handle resizing.
        patches_bgr = crop_patches_from_boxes(
            cluttered_bgr,
            boxes,
            margin=5,
            resize_to=None,
        )
        predictions = classify_patches_batch(classifier_model, patches_bgr, conf_thresh=0.0)

    # --- Optional: refine large, low-confidence boxes with a second-pass differencing ---
    if classifier_model is not None and len(boxes) > 0:
        boxes, predictions = refine_boxes_with_second_pass(
            tidy_bgr=tidy_aligned_bgr,
            cluttered_bgr=cluttered_bgr,
            boxes=boxes,
            predictions=predictions,
            classifier_model=classifier_model,
            conf_low_thresh=0.4,
            noise_box_area_frac=0.005,  # drop boxes < 0.5% of image area as noise
            big_box_area_frac=0.03,     # boxes >= 3% of image area are eligible for second-pass
            refine_diff_thresh=50,
            refine_kernel_size=1,
            refine_iou_thresh=0.5,
            debug_save_dir=debug_save_dir,
            debug_prefix=debug_prefix,
        )

    # --- Global filter: remove boxes that are too large (> 50% of image area) ---
    if len(boxes) > 0:
        img_h, img_w = cluttered_bgr.shape[:2]
        full_area = float(img_h * img_w) if img_h > 0 and img_w > 0 else 0.0
        max_area_frac = 0.5  # drop boxes larger than 50% of the image

        filtered_boxes = []
        filtered_predictions = []

        for idx, (x, y, w, h) in enumerate(boxes):
            box_area = float(w * h)
            area_frac = box_area / full_area if full_area > 0 else 0.0

            if area_frac > max_area_frac:
                print(
                    f"[Filter] Dropped TOO-LARGE box idx={idx}, "
                    f"area_frac={area_frac:.3f} (> {max_area_frac:.2f})"
                )
                continue

            filtered_boxes.append((x, y, w, h))
            if idx < len(predictions):
                filtered_predictions.append(predictions[idx])

        boxes = filtered_boxes
        predictions = filtered_predictions

    # --- Step 5: draw boxes on cluttered image ---
    cluttered_with_boxes = cluttered_rgb.copy()
    for idx, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(
            cluttered_with_boxes, (x, y), (x + w, y + h), (255, 0, 0), 2
        )  # red boxes
        # If we have predictions, overlay class name and confidence
        if idx < len(predictions):
            cls_id, conf, cls_name = predictions[idx]
            label = f"{cls_name} ({conf:.2f})"
            cv2.putText(
                cluttered_with_boxes,
                label,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

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
        if SAVE_IMAGES and debug_save_dir is not None:
            # Fallback: save main figure with a generic name if requested
            fname = f"{debug_prefix}_pipeline.png" if debug_prefix else "pipeline.png"
            out_path = os.path.join(debug_save_dir, fname)
            plt.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved visualization to {out_path}")
        else:
            plt.show()

    print(f"  -> num boxes after NMS: {len(boxes)}")

    return {
        "boxes": boxes,
        "diff_gray": diff_gray,
        "diff_mask_raw": diff_mask_raw,
        "cleaned_mask": cleaned_mask,
        "predictions": predictions,
    }


def main():
    # Construct project root (one level above src)
    project_root = os.path.dirname(os.path.dirname(__file__))

    os.makedirs(os.path.join(project_root, "data", "output"), exist_ok=True)
    final_images_dir = os.path.join(project_root, "data", "final_images")
    if SAVE_IMAGES:
        os.makedirs(final_images_dir, exist_ok=True)

    # List of test image numbers to evaluate.
    # You can add/remove entries here to control which tidy/cluttered pairs are tested.
    test_nums_single = ["21"]
    test_nums_display = ["2", "6", "8", "13", "15", "18", "21"]
    test_nums_all = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    test_nums = test_nums_single

    # Hyperparameter combinations for differencing only
    configs = [
        {
            "name": "diff1",
            "diff_thresh": 30,
            "clean_kernel_size": 3,
            "iou_thresh": 0.8,
        }
    ]

    # 🔹 Load trained classifier model ONCE
    classifier_model = load_object_classifier()

    for num in test_nums:
        print(f"\n======================")
        print(f"Testing image pair #{num}")
        print(f"======================")

        tidy_path = os.path.join(project_root, "data", "tidy", f"{num}.png")
        cluttered_path = os.path.join(project_root, "data", "cluttered", f"{num}.png")

        for cfg in configs:
            print(f"\n=== Running differencing config {cfg['name']} ===")

            prefix = f"{num}_{cfg['name']}"
            save_path = None
            debug_dir = None
            if SAVE_IMAGES:
                debug_dir = final_images_dir
                save_path = os.path.join(final_images_dir, f"{prefix}_pipeline.png")

            result = visualize_diff_pipeline(
                tidy_path=tidy_path,
                cluttered_path=cluttered_path,
                diff_thresh=cfg["diff_thresh"],
                clean_kernel_size=cfg["clean_kernel_size"],
                iou_thresh=cfg["iou_thresh"],
                save_path=save_path,
                classifier_model=classifier_model,
                debug_save_dir=debug_dir,
                debug_prefix=prefix,
            )

            print("  -> boxes:", len(result["boxes"]))
            preds = result.get("predictions", [])
            for (box, pred) in zip(result["boxes"], preds):
                x, y, w, h = box
                cls_id, conf, cls_name = pred
                print(f"     box=({x},{y},{w},{h}) -> class={cls_name}, conf={conf:.3f}")


if __name__ == "__main__":
    main()