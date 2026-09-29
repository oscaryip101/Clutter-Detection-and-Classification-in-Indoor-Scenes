# Clutter Detection and Classification in Indoor Scenes

**Detect what changed in a room—and classify the objects in those regions.**

A computer vision pipeline that compares a tidy reference image with a cluttered image of the same indoor scene. It combines classical image processing with a trained YOLO11s classification model to locate candidate clutter regions, predict object categories, and visualize each stage of the process.

Developed as a team project for **CSCI 5561 at the University of Minnesota**.

**Built with:** Python · OpenCV · NumPy · Matplotlib · Ultralytics YOLO

## The Problem

Recognizing objects alone does not tell us which objects represent clutter. This project uses a tidy image as a baseline to identify changes in the scene, then classifies the objects within those changed regions.

The main challenges are camera movement, lighting differences, noisy change masks, and nearby objects merging into a single region. The pipeline addresses these through alignment, lighting normalization, mask cleanup, and a second refinement pass guided by classification confidence.

## How It Works

1. **Align the images.** Extract SIFT features, match them with FLANN and Lowe's ratio test, and estimate a homography with RANSAC to map the tidy image into the cluttered image's coordinate frame.
2. **Normalize lighting.** Adjust the reference image's per-channel mean and standard deviation to match the target image's overall brightness and contrast.
3. **Detect changed regions.** Compute absolute image differences, apply Gaussian smoothing and thresholding, and clean the binary mask with morphological opening and closing.
4. **Extract candidate boxes.** Find external contours and apply non-maximum suppression to overlapping bounding boxes.
5. **Classify object crops.** Pass each candidate region to the trained YOLO11s classifier to obtain an object label and confidence score.
6. **Refine uncertain regions.** Re-run local differencing inside large, low-confidence boxes, classify the resulting subregions, and filter small or excessively large boxes.

The output includes bounding boxes, class predictions, confidence scores, and intermediate visualizations for inspecting the pipeline.

## Classifier Results

The final recorded training run continued from the `train5` checkpoint for **10 epochs**, using **224 × 224** inputs and a **batch size of 64**.

| Metric | Final epoch validation result |
| --- | ---: |
| Top-1 accuracy | **86.91%** |
| Top-5 accuracy | **98.48%** |

Source: [`train_full/results.csv`](src/runs/classify/train_full/results.csv). These metrics measure object classification on the configured validation split; they do not measure the accuracy of the complete clutter detection pipeline.

## Getting Started

### 1. Clone and install dependencies

```bash
git clone https://github.com/oscaryip101/Clutter-Detection-and-Classification-in-Indoor-Scenes.git
cd Clutter-Detection-and-Classification-in-Indoor-Scenes
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
python -m pip install numpy opencv-python matplotlib ultralytics
```

These dependencies are inferred from the source imports. The repository does not currently include a pinned environment file.

### 2. Run an example

Save the following as `demo.py` in the repository root, then run `python demo.py`:

```python
from pathlib import Path

from src.object_classifier import load_object_classifier
from src.test import visualize_diff_pipeline

output_dir = Path("data/demo_output")
output_dir.mkdir(parents=True, exist_ok=True)

model = load_object_classifier(
    "src/runs/classify/train_full/weights/best.pt"
)

result = visualize_diff_pipeline(
    tidy_path="data/tidy/23.png",
    cluttered_path="data/cluttered/23.png",
    diff_thresh=30,
    clean_kernel_size=3,
    iou_thresh=0.8,
    classifier_model=model,
    save_path=str(output_dir / "23_pipeline.png"),
    debug_save_dir=str(output_dir),
    debug_prefix="23",
)

for box, prediction in zip(result["boxes"], result["predictions"]):
    class_id, confidence, label = prediction
    print(f"{label}: {confidence:.3f} | box={box}")
```

This saves the pipeline figure and any refinement visualizations to `data/demo_output/`. Change both input paths to try another matching image pair.

**Checkpoint path:** The example explicitly supplies the checkpoint's location under `src/runs/`. The current default in `load_object_classifier()` points to `runs/`, so it needs an explicit path when called from the repository root.

## Repository Structure

```text
data/
├── tidy/                 # Tidy reference images
├── cluttered/            # Matching cluttered images
├── final_images/         # Saved pipeline and refinement figures
└── output/               # Additional saved visualizations
src/
├── alignment.py          # SIFT matching and homography estimation
├── lighting.py           # Brightness and contrast normalization
├── differencing.py       # Change masks, bounding boxes, and crops
├── object_classifier.py  # Classifier loading and inference
├── test.py               # Demo, visualization, and second-pass refinement
├── runs/classify/        # Training configurations, results, and weights
└── unused/               # Earlier experimental code
```

## Scope and Limitations

- Requires a tidy reference image of the same scene. Detected changes are treated as candidate clutter; the system does not independently determine whether an object belongs in a room.
- Large viewpoint changes, parallax, shadows, and local lighting changes can produce false detections.
- Thresholds and area filters may need tuning for different scenes. Small objects can be filtered out, while adjacent objects can merge into one region.
- The classifier training dataset referenced by the saved configurations is not included in the repository. Saved checkpoints support inference, but reproducing training requires that dataset.

## Team and Contributions

| Contributor | Contributions |
| --- | --- |
| **[Oscar Yip](https://github.com/oscaryip101)** | Project design and planning; image differencing and refinement after classification; code integration, testing, and debugging; presentation development and final video production. |
| **Shaojie Lan** | Dataset sourcing and classifier training; code integration and testing; project presentations. |
| **Aryan Sood** | Image alignment; final presentation. |
