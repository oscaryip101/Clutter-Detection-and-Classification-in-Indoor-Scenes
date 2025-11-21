# src/run_demo.py
import os
from .pipeline import run_pipeline


def main():
    tidy_path = "data/tidy/1.png"
    cluttered_path = "data/cluttered/1.png"

    os.makedirs("data/output", exist_ok=True)

    # Hyperparameter combination
    configs = [
        {
            "name": "test1",
            "diff_thresh": 20,
            "yolo_conf": 0.25,
            "yolo_iou": 0.45,
            "overlap_ratio_thresh": 0.0,
        },
        {
            "name": "test2",
            "diff_thresh": 20,
            "yolo_conf": 0.15,
            "yolo_iou": 0.45,
            "overlap_ratio_thresh": 0.0,
        },
        {
            "name": "test3",
            "diff_thresh": 20,
            "yolo_conf": 0.05,
            "yolo_iou": 0.8,
            "overlap_ratio_thresh": 0.1,
        },
        {
            "name": "test4",
            "diff_thresh": 30,
            "yolo_conf": 0.05,
            "yolo_iou": 0.9,
            "overlap_ratio_thresh": 0.5,
        },
    ]

    for cfg in configs:
        save_path = os.path.join("data", "output", f"demo_{cfg['name']}.png")
        print(f"\n=== Running config {cfg['name']} ===")
        print(f"Saving to: {save_path}")

        result = run_pipeline(
            tidy_path,
            cluttered_path,
            save_path=save_path,
            diff_thresh=cfg["diff_thresh"],
            yolo_conf=cfg["yolo_conf"],
            yolo_iou=cfg["yolo_iou"],
            overlap_ratio_thresh=cfg["overlap_ratio_thresh"],
        )

        if "error" in result:
            print("  -> ERROR:", result["error"])
        else:
            print(
                "  -> diff_boxes:", len(result["diff_boxes"]),
                "| yolo:", len(result["yolo_boxes"]),
                "| filtered:", len(result["filtered_boxes"]),
            )


if __name__ == "__main__":
    main()
