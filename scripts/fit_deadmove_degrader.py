"""v6: degrader step 8 (degrade.dead_ball_moves), fitted label-free to BAR-ATH aggregates.

    POSSESSION_CACHE_DIR=cache_deadmove python scripts/fit_deadmove_degrader.py --base cache \
        --csv out/v5_run1.csv

Separate cache dir (feature caches are not keyed by degrader version): links the clean-match and
BAR-ATH caches and copies the base degrader params with a "deadmove" block. Target statistic: the
share of frames with the ball tracked and moving (> 2 m/s over 0.2 s) in the interior (2 s trimmed at
each end) of dead stretches >= 4 s: on BAR-ATH the stretches v5 predicts D (our own CSV, no
labels), on degraded IDSSE the true dead stretches of the offset-0 copies. Unconditional, not
"moving given tracked": the degrader keeps the ball visible on ~30% of dead frames vs BAR-ATH's
13%, so matching the conditional share would show the network ~3x BAR-ATH's moving dead ball. p_stretch is bisected
to match it; kickabouts stay off unless p_stretch saturates.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, degrade, io_atd, io_idsse, model  # noqa: E402
from scripts.diag_deadtime_ball import MOVE, ball_speed, runs  # noqa: E402

PARAMS = "degrader_params.json"
TRIM_S = 2.0


def interior_moving(xy, dead, fps):
    v = ball_speed(xy, fps)
    tr = ~np.isnan(xy[:, 0])
    a, b = runs(dead)
    k = int(TRIM_S * fps)
    sel = np.zeros(len(dead), dtype=bool)
    for s, e in zip(a, b):
        if e - s >= degrade.DM_MIN_S * fps:
            sel[s + k:e - k] = True
    return float((v[sel & tr] > MOVE).sum()), float((sel & tr).sum()), float(sel.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(config.REPO_DIR, "cache"))
    ap.add_argument("--csv", default=os.path.join(config.REPO_DIR, "out", "v5_run1.csv"))
    args = ap.parse_args()
    dst = config.CACHE_DIR
    assert os.path.abspath(dst) != os.path.abspath(args.base), "set POSSESSION_CACHE_DIR"
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(args.base, "idsse_J*_v1.npz")) + \
            glob.glob(os.path.join(args.base, "BAR-ATH_*.npz")):
        link = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(link):
            os.symlink(os.path.realpath(f), link)
    with open(os.path.join(args.base, PARAMS)) as h:
        params = json.load(h)
    params.pop("deadmove", None)
    log = lambda s: print(s, flush=True)

    atd = io_atd.load_match()
    letters = np.loadtxt(args.csv, delimiter=",", skiprows=1, usecols=1, dtype="<U1")
    dead = (letters == "D") & atd.graded
    m, t, n = interior_moving(atd.ball_xy, dead, atd.fps)
    target = m / n
    log("BAR-ATH v5-pred D interior: tracked+moving {:.3f} of frames ({:.0f} tracked / {:.0f} frames)"
        .format(target, t, n))

    clean = {mid: io_idsse.load_labelled(mid) for mid in io_idsse.MATCH_IDS}

    def simulate(sp):
        p = dict(params, deadmove=sp) if sp else params
        M = T = 0.0
        for i, (mid, c) in enumerate(clean.items()):
            deg = degrade.degrade(c, p, seed=model.copy_seed(i, 0))
            a, _, b = interior_moving(deg.ball_xy, (c.labels == 2) & c.graded, c.fps)
            M, T = M + a, T + b
        return M / T

    base = simulate(None)
    log("degrader v1 (offset 0): {:.3f}".format(base))
    sp = {"p_stretch": 1.0, "kick_every_s": 0.0}
    top = simulate(sp)
    log("p_stretch 1.0: {:.3f}".format(top))
    if top > target:
        lo, hi = 0.0, 1.0
        for _ in range(8):
            mid_ = (lo + hi) / 2
            v = simulate(dict(sp, p_stretch=mid_))
            log("  p_stretch {:.4f}: {:.3f} (target {:.3f})".format(mid_, v, target))
            lo, hi = (mid_, hi) if v < target else (lo, mid_)
        sp["p_stretch"] = (lo + hi) / 2
    else:
        lo, hi = 2.0, 120.0           # kickabout every X s: smaller X -> more movement
        for _ in range(8):
            mid_ = (lo * hi) ** 0.5
            v = simulate(dict(sp, kick_every_s=mid_))
            log("  kick_every_s {:.1f}: {:.3f} (target {:.3f})".format(mid_, v, target))
            lo, hi = (lo, mid_) if v < target else (mid_, hi)
        sp["kick_every_s"] = (lo * hi) ** 0.5
    fit = simulate(sp)
    sp.update(target_moving_share=target, base_moving_share=base, fit_moving_share=fit)
    params["deadmove"] = sp
    params["degrader_version"] = "1+deadmove"
    degrade.save(params, os.path.join(dst, PARAMS))
    log("fitted {} -> {:.3f} (target {:.3f}) -> {}".format(sp, fit, target,
                                                         os.path.join(dst, PARAMS)))


if __name__ == "__main__":
    main()
