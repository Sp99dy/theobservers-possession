"""M4 experiment (NOT used by run.py): run the ball detector on tactical video frames.

Each frame becomes a motion image (possession.video.motion_image), cut into 21 overlapping
320x320 full-resolution tiles that the detector scales to DET_IMGSZ. Detections from all tiles
are mapped back to frame pixels, merged when closer than MERGE_PX (keeping the most confident),
and the TOP_K most confident are kept per frame.

Reproducibility: outputs are QUANTISED (pixels to 0.5 px, confidence to 0.01) before anything
downstream uses them, so tiny floating-point differences between GPUs or CPUs cannot change the
submission. Results are cached in cache/ (derived from the match video, never committed).
"""

import os

import numpy as np

from . import config, video

# Trained on the match video, so kept out of the public repo (cache/ is gitignored).
MODEL_PATH = os.path.join(config.CACHE_DIR, "weights", "ball_yolo.pt")
MIN_CONF = 0.05          # [D] keep weak candidates; the acceptance threshold is applied later
TOP_K = 3
MERGE_PX = 8.0
BATCH_FRAMES = 8
DET_IMGSZ = 640


def _quantise(u, v, c):
    return np.round(u * 2) / 2, np.round(v * 2) / 2, np.round(c, 2)


def _merge(dets):
    """dets: list of (u, v, conf) in frame pixels -> top-K after greedy merge."""
    dets = sorted(dets, key=lambda d: -d[2])
    kept = []
    for u, v, c in dets:
        if all((u - a) ** 2 + (v - b) ** 2 > MERGE_PX ** 2 for a, b, _ in kept):
            kept.append((u, v, c))
        if len(kept) == TOP_K:
            break
    return kept


def detect(frames, device=None, log=None):
    """{frame: [(u, v, conf), ...]} for the given video frame indices (top-K, quantised)."""
    import torch
    from ultralytics import YOLO
    device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
    model = YOLO(MODEL_PATH)
    tiles = video.tiles()
    out = {}
    buf_f, buf_img = [], []

    def flush():
        crops = [img[y:y + video.TILE, x:x + video.TILE] for img in buf_img for x, y in tiles]
        res = model.predict(crops, imgsz=DET_IMGSZ, conf=MIN_CONF, device=device, half=False,
                            verbose=False, max_det=5)
        for i, f in enumerate(buf_f):
            dets = []
            for j, (x0, y0) in enumerate(tiles):
                r = res[i * len(tiles) + j]
                if r.boxes is None or not len(r.boxes):
                    continue
                xywh = r.boxes.xywh.cpu().numpy().astype(np.float64)
                conf = r.boxes.conf.cpu().numpy().astype(np.float64)
                for (bx, by, _, _), c in zip(xywh, conf):
                    dets.append((bx + x0, by + y0, c))
            kept = _merge(dets)
            out[f] = [tuple(float(z) for z in _quantise(*d)) for d in kept]
        buf_f.clear()
        buf_img.clear()

    for n, (f, img) in enumerate(video.iter_motion(frames)):
        buf_f.append(f)
        buf_img.append(img)
        if len(buf_f) == BATCH_FRAMES:
            flush()
        if log and n and n % 5000 == 0:
            log("  detected {} / {} frames".format(n, len(frames)))
    if buf_f:
        flush()
    return out


def to_arrays(dets):
    """{frame: [(u, v, c)...]} -> (frame, rank, u, v, conf) arrays, sorted."""
    rows = [(f, k, u, v, c) for f in sorted(dets) for k, (u, v, c) in enumerate(dets[f])]
    if not rows:
        return {k: np.zeros(0) for k in ("frame", "rank", "u", "v", "conf")}
    a = np.array(rows, dtype=np.float64)
    return {"frame": a[:, 0].astype(np.int64), "rank": a[:, 1].astype(np.int8), "u": a[:, 2],
            "v": a[:, 3], "conf": a[:, 4]}
