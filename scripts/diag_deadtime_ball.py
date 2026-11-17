"""v6 step 1: does the ball move during dead time? IDSSE truth vs BAR-ATH v5-predicted D.

    python scripts/diag_deadtime_ball.py --csv out/v5_run1.csv

Label-free on BAR-ATH: only the tracking and our own predicted CSV are read; aggregates only.
IDSSE: true dead stretches (labels), on the clean match and on the degraded evaluation copy.
Per frame: ball tracked; ball moving (speed over 0.2 s > 2 m/s). Per stretch: does the ball
travel between players (leaves within 1.5 m of one player, ends within 1.5 m of another, >= 5 m).
Short spells: H/A runs <= 4 s that touch a D stretch on either side.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import io_atd, io_idsse, model  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH  # noqa: E402

D = 2
MOVE = 2.0
NEAR = 1.5


def runs(mask):
    m = np.r_[False, mask, False].astype(np.int8)
    d = np.diff(m)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def ball_speed(xy, fps):
    k = max(1, int(round(0.2 * fps)))
    v = np.full(len(xy), np.nan)
    v[k:] = np.linalg.norm(xy[k:] - xy[:-k], axis=1) * fps / k
    return v


def passes(xy, pxy, s, e, fps):
    """Ball hand-overs inside [s, e): ball within NEAR of player i, later within NEAR of j != i,
    >= 5 m apart."""
    n = 0
    last = None
    for t in range(s, e, max(1, fps // 5)):
        b = xy[t]
        if np.isnan(b[0]):
            continue
        d = np.linalg.norm(pxy[t] - b, axis=1)
        if np.all(np.isnan(d)):
            continue
        j = int(np.nanargmin(d))
        if d[j] > NEAR:
            continue
        if last is not None and last[0] != j and np.linalg.norm(b - last[1]) >= 5.0:
            n += 1
        last = (j, b.copy())
    return n


def stats(name, xy, pxy, fps, dead, graded, labels=None):
    v = ball_speed(xy, fps)
    tracked = ~np.isnan(xy[:, 0])
    out = {}
    for tag, mask in (("dead", dead & graded), ("live", ~dead & graded)):
        n = mask.sum()
        out[tag] = {"minutes": round(n / fps / 60, 1),
                    "tracked": round(float(tracked[mask].mean()), 3),
                    "moving": round(float((v[mask] > MOVE).mean()), 3),
                    "moving_given_tracked": round(float((v[mask & tracked] > MOVE).mean()), 3)}
    a, b = runs(dead & graded)
    long_ = [(s, e) for s, e in zip(a, b) if e - s >= 2 * fps]
    mv = [np.nanmax(np.where(np.isnan(v[s:e]), -1, v[s:e])) > MOVE for s, e in long_]
    hand = [passes(xy, pxy, s, e, fps) for s, e in long_]
    out["stretches"] = {"n": len(long_), "share_with_moving_ball": round(float(np.mean(mv)), 3),
                        "share_with_handover": round(float(np.mean([h > 0 for h in hand])), 3),
                        "handovers_per_stretch": round(float(np.mean(hand)), 3)}
    if labels is not None:
        out["short_spells"] = short_spells(labels, graded, v, fps)
    print(name, json.dumps(out), flush=True)
    return out


def short_spells(lab, graded, v, fps, max_s=4.0):
    """H/A runs <= max_s with a D run immediately before or after (and both sides graded)."""
    lab = np.where(graded, lab, -1)
    b = np.flatnonzero(np.diff(lab)) + 1
    st, en = np.r_[0, b], np.r_[b, len(lab)]
    n = between = 0
    moving = []
    for i, (s, e) in enumerate(zip(st, en)):
        if lab[s] not in (0, 1) or e - s > max_s * fps:
            continue
        prev = lab[st[i - 1]] if i else -1
        nxt = lab[st[i + 1]] if i + 1 < len(st) else -1
        if D in (prev, nxt):
            n += 1
            between += prev == D and nxt == D
            moving.append(float((v[s:e] > MOVE).mean()))
    return {"n": n, "between_two_D": int(between),
            "moving_share": round(float(np.mean(moving)), 3) if moving else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="out/v5_run1.csv")
    args = ap.parse_args()
    with open(PARAMS_PATH) as h:
        params = json.load(h)
    for mid in io_idsse.MATCH_IDS:
        m = io_idsse.load_labelled(mid)
        dead = m.labels == D
        stats("idsse_clean " + mid, m.ball_xy, m.players_xy, m.fps, dead, m.graded, m.labels)
        c = model.make_copy(mid, params, *model.EVAL_COPY)
        stats("idsse_eval9 " + mid, c.ball_xy, c.players_xy, c.fps, c.labels == D, c.graded)
    atd = io_atd.load_match()
    letters = np.loadtxt(args.csv, delimiter=",", skiprows=1, usecols=1, dtype="<U1")
    lab = np.searchsorted(np.array(["A", "D", "H", "X"]), letters)
    lab = np.array([1, 2, 0, 3])[lab]          # H=0 A=1 D=2 X=3
    assert len(lab) == atd.n_frames
    stats("barath_pred " + os.path.basename(args.csv), atd.ball_xy, atd.players_xy, atd.fps,
          lab == D, atd.graded, lab)


if __name__ == "__main__":
    main()
