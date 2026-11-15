"""Fine-tune a small public detector (Ultralytics YOLO11n, COCO-pretrained) to find the ball on
640x640 full-resolution tiles of the tactical video, using the ATD-projected labels from
scripts/video_dataset.py. M4 experiment, NOT used by run.py: the detector was not accurate enough
(TECHNICAL.md), and its weights stay in cache/ because they are trained on the match video.

    python scripts/video_train.py [--epochs 40]
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config  # noqa: E402

DS = os.path.join(config.CACHE_DIR, "video_ds_vis", "ds.yaml")
RUNS = os.path.join(config.CACHE_DIR, "video_runs")
OUT = os.path.join(config.CACHE_DIR, "weights", "ball_yolo.pt")


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--base", default="yolo11n.pt")
    args = ap.parse_args()
    model = YOLO(args.base)
    model.train(data=DS, imgsz=640, epochs=args.epochs, batch=32, device=0, workers=4,
                project=RUNS, name="ball_vis", exist_ok=True, seed=0, deterministic=True,
                single_cls=True, mosaic=0.5, scale=0.2, fliplr=0.5, degrees=0.0,
                hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, translate=0.1,
                close_mosaic=5, patience=20, plots=False, verbose=False)
    best = os.path.join(RUNS, "ball_vis", "weights", "best.pt")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    shutil.copyfile(best, OUT)
    print("saved", OUT)


if __name__ == "__main__":
    main()
