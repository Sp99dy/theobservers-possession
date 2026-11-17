"""Measure the broadcast -> tactical frame lag per half, label-free (possession/ballfill.LAG).

    python scripts/measure_broadcast_lag.py

Coarse search (step 50 frames, +-80k) then fine search (+-25) of the lag minimising the median
distance between the tactical ball at t and the broadcast ball at t - lag. Aggregates only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import ballfill, io_atd  # noqa: E402


def median_dist(tb, bb, s, e, lag, step):
    t = np.arange(s, e, step)
    j = t - lag
    ok = (j >= 0) & (j < len(bb))
    d = np.linalg.norm(tb[t[ok]] - bb[j[ok]], axis=1)
    d = d[np.isfinite(d)]
    return (float(np.median(d)), len(d)) if len(d) > 200 else (99.0, len(d))


def main():
    m = io_atd.load_match()
    bb, _ = ballfill.broadcast_ball(m.pitch)
    for h, s, e in m.half_blocks():
        coarse = min((median_dist(m.ball_xy, bb, s, e, lag, 25) + (lag,)
                      for lag in range(-80000, 80001, 50)), key=lambda r: r[0])
        fine = min((median_dist(m.ball_xy, bb, s, e, lag, 1) + (lag,)
                    for lag in range(coarse[2] - 50, coarse[2] + 51)), key=lambda r: r[0])
        print("half {}: coarse lag {} ({:.2f} m), fine lag {} (median {:.3f} m over {} frames); "
              "ballfill.LAG = {}".format(h, coarse[2], coarse[0], fine[2], fine[0], fine[1],
                                         ballfill.LAG[h]), flush=True)


if __name__ == "__main__":
    main()
