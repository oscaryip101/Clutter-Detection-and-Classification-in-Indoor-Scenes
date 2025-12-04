import numpy as np
import cv2

def correct_lighting(
    aligned_image: np.ndarray,
    cluttered_image: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Adjust the lighting of the aligned tidy image to better match the cluttered image.

    This function is meant to be called *after* geometric alignment, when
    `aligned_image` and `cluttered_image` are roughly in the same coordinate frame.

    It performs a simple per-channel mean/std normalization:
    for each channel c,
        aligned_corrected[c] = (aligned[c] - mean_aligned[c]) * (std_cluttered[c] / std_aligned[c]) + mean_cluttered[c]

    This compensates for global brightness and contrast differences between the
    two images, making simple differencing more robust to lighting changes.

    Parameters
    ----------
    aligned_image : np.ndarray
        The aligned tidy image in BGR format.
    cluttered_image : np.ndarray
        The cluttered image in BGR format (reference lighting).
    eps : float, optional
        Small constant to avoid division by zero if a channel has near-zero variance.

    Returns
    -------
    corrected : np.ndarray
        A new BGR image with the same shape as `aligned_image`, where the global
        lighting has been adjusted to better match `cluttered_image`.
    """
    if aligned_image is None or cluttered_image is None:
        raise ValueError("correct_lighting: one or both input images are None")

    if aligned_image.shape != cluttered_image.shape:
        raise ValueError(
            f"correct_lighting: shape mismatch {aligned_image.shape} vs {cluttered_image.shape}"
        )

    # Work in float for math, keep a copy to avoid modifying inputs in-place
    aligned = aligned_image.astype(np.float32)
    cluttered = cluttered_image.astype(np.float32)

    corrected = np.empty_like(aligned, dtype=np.float32)

    # Handle both grayscale and 3-channel BGR images
    if aligned.ndim == 2 or aligned.shape[2] == 1:
        # Single-channel
        m_a, s_a = float(aligned.mean()), float(aligned.std())
        m_c, s_c = float(cluttered.mean()), float(cluttered.std())

        if s_a < eps:
            s_a = eps

        corrected_gray = (aligned - m_a) * (s_c / s_a) + m_c
        corrected = np.clip(corrected_gray, 0, 255)
    else:
        # Per-channel normalization for 3-channel images
        for c in range(3):
            chan_a = aligned[..., c]
            chan_c = cluttered[..., c]

            m_a, s_a = float(chan_a.mean()), float(chan_a.std())
            m_c, s_c = float(chan_c.mean()), float(chan_c.std())

            if s_a < eps:
                s_a = eps

            chan_corr = (chan_a - m_a) * (s_c / s_a) + m_c
            corrected[..., c] = chan_corr

        corrected = np.clip(corrected, 0, 255)

    return corrected.astype(np.uint8)
