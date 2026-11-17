"""BAR-ATH input repair: fill short tactical ball gaps with the broadcast ATD's ball (inference only).

Both ATD feeds are in pitch coordinates but on different video timelines. The broadcast recording
is NOT aligned to the tactical one (games.py); the offset is constant within a half:
tactical video frame t <-> broadcast video frame t - LAG[half]. LAG was measured label-free by
scripts/measure_broadcast_lag.py: over a +-80k-frame search the median tactical-vs-broadcast ball
distance is minimal at 4014 (H1, 0.80 m) and 26920 (H2, 0.85 m), values reported in public sources.

The broadcast ball is noisy (where both feeds see it, only 70-74% agree within 2 m), so the rule is
conservative and all-or-nothing per gap. A gap = a run of graded frames without a tactical ball.
It is filled with the broadcast ball only if
  - it lasts at most MAX_GAP_S,
  - at both edges the broadcast ball agrees with the tactical ball within EDGE_M on each of the
    EDGE_N tactical-visible frames next to the gap (the gap must have a tactical ball on both sides),
  - on every frame of the gap the broadcast ball is present and its homography valid, and
  - no step from the last tactical ball, through the gap, to the next tactical ball exceeds STEP_M.
Feature treatment: features have no "interpolated" state (ball_present is simply ~isnan), so a
filled ball counts as present, exactly like any tracked ball; the degraded IDSSE training copies
likewise only know present or missing. IDSSE has no broadcast feed: training and CV are untouched.
"""

import os
from dataclasses import replace

import numpy as np

from . import config, io_atd
from .canonical import normalised_to_metres

LAG = {1: 4014, 2: 26920}          # [M] measured, see module docstring
N_BROADCAST = 164488
MAX_GAP_S = 3.0                    # [D] peer review: short gaps only
EDGE_N = 5                         # [D] tactical-visible frames checked at each edge
EDGE_M = 2.0                       # [D] broadcast must agree with tactical within this at the edges
STEP_M = 3.0                       # [D] no per-frame jump above this (75 m/s) through the gap


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


def aligned(match, bb, valid):
    """Broadcast ball and validity on the tactical timeline (NaN / False outside the halves)."""
    out = np.full((match.n_frames, 2), np.nan)
    ok_v = np.zeros(match.n_frames, dtype=bool)
    for h, s, e in match.half_blocks():
        t = np.arange(s, e)
        j = t - LAG[h]
        ok = (j >= 0) & (j < len(bb))
        out[t[ok]] = bb[j[ok]]
        ok_v[t[ok]] = valid[j[ok]]
    return out, ok_v


def gap_fill(tac, bro, valid, graded, fps):
    """Boolean mask of frames to fill (rules in the module docstring)."""
    T = len(tac)
    miss = np.isnan(tac[:, 0]) & graded
    m = np.r_[False, miss, False].astype(np.int8)
    d = np.diff(m)
    fill = np.zeros(T, dtype=bool)
    vis = np.flatnonzero(~np.isnan(tac[:, 0]))
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if e - s > MAX_GAP_S * fps or s == 0 or e >= T:
            continue
        if not (graded[s - 1] and graded[e]):
            continue
        if np.isnan(bro[s:e, 0]).any() or not valid[s:e].all():
            continue
        i = np.searchsorted(vis, s)
        before, after = vis[max(0, i - EDGE_N):i], vis[i:i + EDGE_N]
        if len(before) < EDGE_N or len(after) < EDGE_N:
            continue
        edges = np.r_[before, after]
        de = np.linalg.norm(tac[edges] - bro[edges], axis=1)
        if not np.all(de <= EDGE_M):
            continue
        path = np.vstack([tac[s - 1], bro[s:e], tac[e]])
        if np.any(np.linalg.norm(np.diff(path, axis=0), axis=1) > STEP_M):
            continue
        fill[s:e] = True
    return fill


def fill(match, data_dir=None):
    """(match with filled ball_xy, boolean mask of filled frames). Works on the unoriented match."""
    bb, valid = broadcast_ball(match.pitch, data_dir)
    bro, ok = aligned(match, bb, valid)
    mask = gap_fill(match.ball_xy, bro, ok, match.graded, match.fps)
    ball = match.ball_xy.copy()
    ball[mask] = bro[mask]
    return replace(match, ball_xy=ball), mask
