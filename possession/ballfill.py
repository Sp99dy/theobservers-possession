"""BAR-ATH input repair: fill tactical ball gaps with the broadcast ATD's ball (inference only).

Both ATD feeds are in pitch coordinates but on different video timelines. The broadcast recording
is NOT aligned to the tactical one (games.py); the offset is constant within a half:
tactical video frame t <-> broadcast video frame t - LAG[half]. LAG was measured label-free by
scripts/measure_broadcast_lag.py: over a +-80k-frame search the median tactical-vs-broadcast ball
distance is minimal at 4014 (H1, 0.80 m) and 26920 (H2, 0.85 m), the values other teams report.
Rule: a graded frame whose tactical ball is missing takes the broadcast ball at t - LAG when that
ball is present and the broadcast homography is valid. A filled run is kept only if it joins the
tactical ball: at each end that has a tactical ball within 1 s, the gap is <= JOIN_M, and at least
one end has one. Unjoined or jumping runs are what replays and broadcast misdetections look like
(before this check, 28% of filled frames sat in runs jumping > 5 m at an end). Nothing else changes; IDSSE has no
broadcast feed, so training and CV are untouched.
"""

import os
from dataclasses import replace

import numpy as np

from . import config, io_atd
from .canonical import normalised_to_metres

LAG = {1: 4014, 2: 26920}          # [M] measured, see module docstring
N_BROADCAST = 164488
JOIN_M = 5.0                       # [D] a ball cannot be 5 m away from itself across a gap edge
JOIN_S = 1.0                       # [D] look this far for a tactical ball at each end


def broadcast_ball(pitch, data_dir=None):
    d = os.path.join(data_dir or config.MTAH_DATA_DIR, "broadcast")
    raw = io_atd.load_raw(os.path.join(d, "BAR-ATH_broadcast_atd.xml"),
                          os.path.join(d, "BAR-ATH_broadcast_atd.txt"))
    _, ball = io_atd.raw_to_video_index(raw, N_BROADCAST)
    bx, by = normalised_to_metres(ball[:, 0], ball[:, 1], pitch)
    hom = np.loadtxt(os.path.join(d, "BAR-ATH_broadcast_homography.csv"), delimiter=",",
                     skiprows=1, usecols=(0, 11))
    valid = np.zeros(N_BROADCAST, dtype=bool)
    valid[hom[:, 0].astype(int)] = hom[:, 1] > 0
    return np.stack([bx, by], axis=1), valid


def fill(match, data_dir=None):
    """(match with filled ball_xy, boolean mask of filled frames). Works on the unoriented match."""
    bb, valid = broadcast_ball(match.pitch, data_dir)
    ball = match.ball_xy.copy()
    filled = np.zeros(match.n_frames, dtype=bool)
    for h, s, e in match.half_blocks():
        t = np.arange(s, e)
        j = t - LAG[h]
        ok = (j >= 0) & (j < len(bb))
        t, j = t[ok], j[ok]
        take = np.isnan(ball[t, 0]) & ~np.isnan(bb[j, 0]) & valid[j]
        ball[t[take]] = bb[j[take]]
        filled[t[take]] = True
    keep = _joined_runs(match.ball_xy, ball, filled, int(round(JOIN_S * match.fps)))
    ball[filled & ~keep] = np.nan
    return replace(match, ball_xy=ball), keep


def _joined_runs(tactical, ball, filled, look):
    keep = np.zeros_like(filled)
    m = np.r_[False, filled, False].astype(np.int8)
    d = np.diff(m)
    T = len(filled)
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        gaps = []
        for ref, step in ((s, -1), (e - 1, 1)):
            for k in range(1, look + 1):
                u = ref + step * k
                if 0 <= u < T and not np.isnan(tactical[u, 0]):
                    gaps.append(float(np.linalg.norm(tactical[u] - ball[ref])))
                    break
        if gaps and max(gaps) <= JOIN_M:
            keep[s:e] = True
    return keep
