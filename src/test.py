from utils import load_image, show_image
from alignment import align_images
from differencing import compute_difference_mask, threshold_mask, clean_mask

tidy = load_image("data/tidy/1.png")
cluttered = load_image("data/cluttered/1.png")

aligned, H, matches = align_images(tidy, cluttered)
show_image("Aligned Image", aligned)

diff_gray = compute_difference_mask(aligned, cluttered)
show_image("Difference Gray", diff_gray)

mask = threshold_mask(diff_gray)
show_image("Threshold Mask", mask)

cleaned = clean_mask(mask)
show_image("Cleaned Mask", cleaned)