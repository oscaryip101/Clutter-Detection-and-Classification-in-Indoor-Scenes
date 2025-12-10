import cv2
import numpy as np

def compute_difference_mask(aligned_tidy, cluttered):
    """
        Compute the raw pixel-wise difference between the aligned tidy image
        and the cluttered image.

        Parameters
        ----------
        aligned_tidy : np.ndarray
            Tidy image warped into cluttered coordinate frame.
        cluttered : np.ndarray
            Original cluttered image.

        Returns
        -------
        diff_gray : np.ndarray
            Grayscale absolute-difference map highlighting changed regions.
        """
    diff = cv2.absdiff(aligned_tidy, cluttered)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_gray = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    return diff_gray

def threshold_mask(diff_gray, thresh=20):
    """
    Threshold the grayscale difference map to obtain a binary foreground mask.

    Parameters
    ----------
    diff_gray : np.ndarray
        Raw grayscale difference image.
    thresh : int, optional
        Threshold for binarization.

    Returns
    -------
    mask : np.ndarray
        Binary mask where changed pixels = 255.
    """
    _, mask = cv2.threshold(diff_gray, thresh, 255, cv2.THRESH_BINARY)
    return mask

def clean_mask(mask, kernel_size=5):
    """
    Refine the binary difference mask using morphological operations.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask from thresholding.
    kernel_size : int, optional
        Size of the square structuring element used for morphological operations.

    Returns
    -------
    cleaned : np.ndarray
        Noise-reduced and hole-filled mask suitable for region extraction.

    Notes
    -----
    Uses morphological opening to remove small noise
    and closing to merge fragmented regions.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

def extract_bounding_boxes(mask,iou_thresh=0.3):
    """
    Extract bounding boxes around connected components in the mask and apply
    IoU-based Non-Maximum Suppression (NMS) using OpenCV.

    Parameters
    ----------
    mask : np.ndarray
        Cleaned binary mask representing changed regions.
    iou_thresh : float, optional
        IoU / overlap threshold used for non-maximum suppression. Boxes with IoU greater
        than this value are suppressed.

    Returns
    -------
    boxes : list of tuples
        List of bounding boxes in the format (x, y, w, h) for each region that survives NMS.

    Notes
    -----
    Uses cv2.dnn.NMSBoxes to eliminate overlapping noisy detections and keep
    the strongest clutter regions, where box area is used as a simple
    confidence score proxy.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    raw_boxes = []
    scores = []

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # cv2.dnn.NMSBoxes expects boxes as [x, y, w, h]
        raw_boxes.append([int(x), int(y), int(w), int(h)])
        # Use box area as a simple proxy for confidence: larger regions are more likely
        # to correspond to true clutter instead of noise.
        scores.append(float(w * h))

    if not raw_boxes:
        return []

    # score_threshold is set to 0.0 so that all boxes are considered, and only
    # the IoU-based NMS (nms_threshold) determines which ones are kept.
    indices = cv2.dnn.NMSBoxes(raw_boxes, scores, score_threshold=0.0, nms_threshold=iou_thresh)

    kept_boxes = []
    if len(indices) > 0:
        # indices can be a list of lists (e.g., [[0], [2]]) or a flat list; flatten to be safe
        for i in np.array(indices).flatten():
            x, y, w, h = raw_boxes[int(i)]
            kept_boxes.append((x, y, w, h))

    return kept_boxes

def crop_patches_from_boxes(
    cluttered_bgr,
    boxes,
    margin=5,
    resize_to=None,
):
    """
    Crop image patches from the cluttered image given bounding boxes.

    Parameters
    ----------
    cluttered_bgr : np.ndarray
        Original cluttered image in BGR format.
    boxes : list of tuples
        Bounding boxes as (x, y, w, h).
    margin : int, optional
        Extra pixels to pad around each box on all sides.
    resize_to : tuple or None, optional
        If not None, resize each patch to (width, height).

    Returns
    -------
    patches_bgr : list of np.ndarray
        List of cropped (and optionally resized) BGR patches.
    """
    h_img, w_img = cluttered_bgr.shape[:2]
    patches = []

    for (x, y, w, h) in boxes:
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(w_img, x + w + margin)
        y1 = min(h_img, y + h + margin)

        patch = cluttered_bgr[y0:y1, x0:x1, :]

        if resize_to is not None:
            patch = cv2.resize(patch, resize_to, interpolation=cv2.INTER_LINEAR)

        patches.append(patch)

    return patches