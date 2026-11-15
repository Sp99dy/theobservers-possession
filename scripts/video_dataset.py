"""Build the ball-detector training set from the tactical video, labelled automatically by the ATD.

Inputs are motion images (possession.video.motion_image) cut into 320x320 tiles.

    python scripts/video_dataset.py [--n-train 9000] [--n-val 1500]

Positives: tiles (640x640, full resolution) cut around the projected ATD ball at a random offset,
on graded frames where the ATD sees the ball inside the pitch. Negatives: tiles from the same
frames that do not contain the projected ball. Validation frames come from separate one-minute
blocks (every 5th minute), never used for training, so recall is measured on unseen footage.

Output: cache/video_ds/{images,labels}/{train,val} + ds.yaml (YOLO format). cache/ is gitignored;
nothing here is ever looked at by a person, and only counts are printed.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, organisers, video  # noqa: E402

OUT = os.path.join(config.CACHE_DIR, "video_ds_vis")
NEG_SHARE = 0.3
BOX_MIN_PX = 10.0        # [D] pseudo-labels are a few px off; a 10 px box tolerates that
# Visibility filter [M]. The ATD ball is a clear blob on only ~30% of frames (local grey contrast
# > 30 at the projected point vs 2% at random points), so positives are kept only where the ball
# is visible in grey OR in motion; otherwise "ball here" over an empty patch is label noise.
VIS_GREY = 25.0
VIS_MOTION = 80.0
VAL_EVERY_MIN = 5


def is_val(frame):
    return (frame // (25 * 60)) % VAL_EVERY_MIN == VAL_EVERY_MIN - 1


_YY, _XX = np.mgrid[-10:11, -10:11]
_R = np.hypot(_YY, _XX)
_RING = (_R >= 6) & (_R <= 10)


def visibility(img, u, v):
    """(grey contrast, motion contrast) at pixel (u, v) of a motion image; NaN near the edge."""
    u, v = int(round(u)), int(round(v))
    if not (12 <= u < img.shape[1] - 12 and 12 <= v < img.shape[0] - 12):
        return np.nan, np.nan
    p = img[v - 10:v + 11, u - 10:u + 11].astype(np.float32)
    grey = p[8:13, 8:13, 0].max() - p[..., 0][_RING].mean()
    mot = p[..., 1:].max(axis=2)
    return grey, mot[8:13, 8:13].max() - mot[_RING].mean()


def main():
    import cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=24000)
    ap.add_argument("--n-val", type=int, default=4000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    H, _, _ = video.load_homographies()
    ball, graded = video.atd_ball_video()
    length = organisers.games().GAMES["BAR-ATH"]["pitch"][0]
    ok = graded & ~np.isnan(ball[:, 0]) & (ball[:, 0] >= 0) & (ball[:, 0] <= 1) \
        & (ball[:, 1] >= 0) & (ball[:, 1] <= 1)
    cand = np.flatnonzero(ok)
    val_c = cand[is_val(cand)]
    tr_c = cand[~is_val(cand)]
    picks = {"train": np.sort(rng.choice(tr_c, args.n_train, replace=False)),
             "val": np.sort(rng.choice(val_c, args.n_val, replace=False))}
    split_of = {int(f): s for s, fs in picks.items() for f in fs}
    for s in picks:
        os.makedirs(os.path.join(OUT, "images", s), exist_ok=True)
        os.makedirs(os.path.join(OUT, "labels", s), exist_ok=True)

    counts = {"train": [0, 0], "val": [0, 0]}
    sizes = []
    vis_at_ball, vis_random, dropped = [], [], 0
    for f, img in video.iter_motion(split_of):
        s = split_of[f]
        u, v = video.project(H[f], ball[f, 0], ball[f, 1])
        u, v = float(u[0]), float(v[0])
        u1, v1 = video.project(H[f], ball[f, 0] + video.BALL_DIAMETER_M / length, ball[f, 1])
        d = float(np.hypot(u1[0] - u, v1[0] - v))
        box = max(BOX_MIN_PX, 1.5 * d)
        sizes.append(box)
        if not (0 <= u < video.W and 0 <= v < video.HGT):
            continue
        vis = visibility(img, u, v)
        vis_at_ball.append(vis)
        vis_random.append(visibility(img, rng.uniform(40, video.W - 40),
                                     rng.uniform(200, 900)))
        neg = rng.random() < NEG_SHARE
        if not neg and not (vis[0] >= VIS_GREY or vis[1] >= VIS_MOTION):
            dropped += 1
            continue
        if neg:
            options = [(x, y) for x, y in video.tiles()
                       if not (x - box <= u < x + video.TILE + box and
                               y - box <= v < y + video.TILE + box)]
            if not options:
                continue
            x0, y0 = options[rng.integers(len(options))]
        else:
            m = 16
            lo_x, hi_x = max(0, int(u) - video.TILE + m), min(video.W - video.TILE, int(u) - m)
            lo_y, hi_y = max(0, int(v) - video.TILE + m), min(video.HGT - video.TILE, int(v) - m)
            x0 = int(rng.integers(lo_x, max(lo_x, hi_x) + 1))
            y0 = int(rng.integers(lo_y, max(lo_y, hi_y) + 1))
        crop = img[y0:y0 + video.TILE, x0:x0 + video.TILE]
        name = "f{:06d}_{}_{}".format(f, x0, y0)
        cv2.imwrite(os.path.join(OUT, "images", s, name + ".jpg"), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        with open(os.path.join(OUT, "labels", s, name + ".txt"), "w") as h:
            if not neg:
                h.write("0 {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                    (u - x0) / video.TILE, (v - y0) / video.TILE, box / video.TILE,
                    box / video.TILE))
        counts[s][int(neg)] += 1
    with open(os.path.join(OUT, "ds.yaml"), "w") as h:
        h.write("path: {}\ntrain: images/train\nval: images/val\nnames:\n  0: ball\n".format(
            OUT.replace("\\", "/")))
    for name, arr in (("at ATD ball", vis_at_ball), ("at random points", vis_random)):
        a = np.asarray(arr, dtype=np.float64)
        a = a[~np.isnan(a).any(axis=1)]
        print("visible {:<17} grey>={:.0f} {:.3f}  motion>={:.0f} {:.3f}  either {:.3f}".format(
            name, VIS_GREY, np.mean(a[:, 0] >= VIS_GREY), VIS_MOTION,
            np.mean(a[:, 1] >= VIS_MOTION),
            np.mean((a[:, 0] >= VIS_GREY) | (a[:, 1] >= VIS_MOTION))))
    print("positives dropped as not visible: {}".format(dropped))
    for s, (p, n) in counts.items():
        print("{}: {} positive tiles, {} negative tiles".format(s, p, n))
    print("label box px: median {:.1f}  p95 {:.1f}".format(*np.percentile(sizes, [50, 95])))


if __name__ == "__main__":
    main()
