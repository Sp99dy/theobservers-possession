"""Measure the ball detector with aggregates only, then cache its detections on the ATD's gaps.

    python scripts/video_eval.py [--n-val 3000]

1. Held-out minutes (never trained on) where the ATD sees the ball: is the top detection on the
   ATD ball (within HIT_PX)? Coverage and precision by confidence threshold.
2. All graded frames where the ATD has no ball: share with a confident detection, and, next to
   each gap edge (ATD ball seen within EDGE frames), agreement with that neighbouring ATD ball.
Detections on the gap frames are cached to cache/ball_det_gaps_v1.npz for the pipeline.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, video, video_detect  # noqa: E402
from scripts.video_dataset import is_val  # noqa: E402

HIT_PX = 12.0
EDGE = 3
THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
GAP_CACHE = os.path.join(config.CACHE_DIR, "ball_det_gaps_v1.npz")


def top1(dets, frames):
    u = np.full(len(frames), np.nan)
    v = np.full(len(frames), np.nan)
    c = np.zeros(len(frames))
    for i, f in enumerate(frames):
        d = dets.get(int(f), [])
        if d:
            u[i], v[i], c[i] = d[0]
    return u, v, c


def atd_pixels(H, ball, frames):
    uv = np.array([[float(z[0]) for z in video.project(H[f], ball[f, 0], ball[f, 1])]
                   for f in frames])
    return uv[:, 0], uv[:, 1]


def report(name, dist, conf):
    print(name)
    for t in THRESHOLDS:
        sel = conf >= t
        print("  conf >= {:.1f}: coverage {:.3f}  precision (top-1 within {:.0f} px) {}".format(
            t, sel.mean(), HIT_PX,
            "{:.3f}".format(np.mean(dist[sel] <= HIT_PX)) if sel.any() else "-"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-val", type=int, default=3000)
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(1)
    H, _, _ = video.load_homographies()
    ball, graded = video.atd_ball_video()
    seen = graded & ~np.isnan(ball[:, 0])
    val = np.flatnonzero(seen)
    val = np.sort(rng.choice(val[is_val(val)], args.n_val, replace=False))

    dets = video_detect.detect(val)
    u, v, c = top1(dets, val)
    au, av = atd_pixels(H, ball, val)
    dist = np.hypot(u - au, v - av)
    dist[np.isnan(dist)] = np.inf
    report("held-out minutes, ATD sees the ball (n {}):".format(len(val)), dist, c)
    ok = np.isfinite(dist)
    print("  top-1 distance to ATD ball px: median {:.1f}  p75 {:.1f}".format(
        *np.percentile(dist[ok], [50, 75])))
    print("  ({:.0f}s)".format(time.time() - t0), flush=True)

    gaps = np.flatnonzero(graded & np.isnan(ball[:, 0]))
    if os.path.isfile(GAP_CACHE):
        z = np.load(GAP_CACHE)
        gd = {}
        for f, uu, vv, cc in zip(z["frame"], z["u"], z["v"], z["conf"]):
            gd.setdefault(int(f), []).append((uu, vv, cc))
    else:
        gd = video_detect.detect(gaps, log=print)
        np.savez(GAP_CACHE, **video_detect.to_arrays(gd))
    gu, gv, gc = top1(gd, gaps)
    print("gap frames (ATD has no ball): n {}".format(len(gaps)))
    for t in THRESHOLDS:
        print("  conf >= {:.1f}: share with a detection {:.3f}".format(t, np.mean(gc >= t)))

    # Gap edges: compare with the ATD ball just outside the gap.
    is_gap = np.zeros(len(ball), bool)
    is_gap[gaps] = True
    pos = {int(f): i for i, f in enumerate(gaps)}
    ed, ec = [], []
    for f in gaps:
        for k in range(1, EDGE + 1):
            for nb in (f - k, f + k):
                if 0 <= nb < len(ball) and seen[nb] and abs(nb - f) == k:
                    i = pos[int(f)]
                    if np.isnan(gu[i]):
                        ed.append(np.inf)
                    else:
                        x, y = video.project(H[nb], ball[nb, 0], ball[nb, 1])
                        ed.append(np.hypot(gu[i] - x[0], gv[i] - y[0]))
                    ec.append(gc[i])
                    break
            else:
                continue
            break
    ed, ec = np.asarray(ed), np.asarray(ec)
    report("gap frames within {} frames of an ATD ball (n {}), vs that ATD ball:".format(
        EDGE, len(ed)), ed, ec)
    print("total {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
