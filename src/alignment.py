import cv2
import numpy as np

def extract_sift_features(image):
    """
    Extract SIFT keypoints and descriptors from an input image.

    Parameters
    ----------
    image : np.ndarray
        Input BGR image.

    Returns
    -------
    kp : list of cv2.KeyPoint
        Detected SIFT keypoints.
    desc : np.ndarray
        Corresponding SIFT descriptors of shape (N, 128),
        or None if no descriptors are found.

    Notes
    -----
    The image is converted to grayscale before feature extraction.
    """
    # SIFT keypoints + descriptors
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, desc = sift.detectAndCompute(gray, None)
    return kp, desc


def match_features(desc1, desc2, ratio=0.75):
    """
    Match SIFT descriptors using FLANN-based k-NN matching and
    Lowe's ratio test for filtering ambiguous matches.

    Parameters
    ----------
    desc1 : np.ndarray
        Descriptors from the first (tidy) image.
    desc2 : np.ndarray
        Descriptors from the second (cluttered) image.
    ratio : float, optional
        Lowe's ratio threshold for rejecting ambiguous matches.

    Returns
    -------
    good : list of cv2.DMatch
        Filtered list of reliable feature matches.
    """
    
    if desc1 is None or desc2 is None:
        return []
    
    # FLANN + Lowe’s ratio test
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in matches if m.distance < ratio * n.distance]
    return good


def estimate_homography(kp1, kp2, matches, thresh=4.0):
    """
    Estimate a projective homography between two images from
    matched SIFT keypoints using RANSAC.

    Parameters
    ----------
    kp1 : list of cv2.KeyPoint
        Keypoints from the tidy image.
    kp2 : list of cv2.KeyPoint
        Keypoints from the cluttered image.
    matches : list of cv2.DMatch
        Descriptor matches filtered by ratio test.
    thresh : float, optional
        RANSAC reprojection error threshold.

    Returns
    -------
    H : np.ndarray or None
        Estimated 3x3 homography matrix mapping points
        from tidy → cluttered coordinates.
    mask : np.ndarray or None
        RANSAC inlier mask (1 for inliers, 0 for outliers).

    Notes
    -----
    At least 4 matches are required to compute homography.
    """
    # Compute homography with RANSAC
    if len(matches) < 4:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, thresh)
    return H, mask


def align_images(tidy_image, cluttered_image):
    """
    Align the tidy reference image to the cluttered image using
    SIFT feature matching and RANSAC-based homography estimation.

    Parameters
    ----------
    tidy_image : np.ndarray
        Reference tidy-room image.
    cluttered_image : np.ndarray
        Target cluttered-room image.

    Returns
    -------
    aligned : np.ndarray or None
        The tidy image warped into the cluttered image's coordinate frame.
    H : np.ndarray or None
        Estimated homography matrix.
    matches : list of cv2.DMatch
        Feature matches used for homography estimation.

    Notes
    -----
    If homography estimation fails, the function returns (None, None, matches).
    """
    # Full alignment pipeline
    kp1, desc1 = extract_sift_features(tidy_image)
    kp2, desc2 = extract_sift_features(cluttered_image)

    matches = match_features(desc1, desc2)
    H, mask = estimate_homography(kp1, kp2, matches)

    if H is None:
        print("Homography failed.")
        return None, None, matches

    h, w = cluttered_image.shape[:2]
    aligned = cv2.warpPerspective(tidy_image, H, (w, h))
    return aligned, H, matches