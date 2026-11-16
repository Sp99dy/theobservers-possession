"""Label-free tracking-error statistics: BAR-ATH ATD vs degraded IDSSE (aggregates only).

    python scripts/diag_tracking_errors.py [--degrader-version N]

Error types seen in BAR-ATH clips (patterns only):
1. wrong-team tracklets near duels: a player's track ends and, at the same spot, a track of the
   OTHER team starts (or a slot jumps onto an opponent's position), usually within ~2 m of an
   opponent; it lasts seconds to tens of seconds.
2. keeper out of view: a stationary ghost dot on a goal line, and/or the keeper shown in the other
   team.
3. crowds during stoppages (12+ per team): measured only as counts.

Never reads labels: every statistic uses tracking positions within the halves.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, io_atd, io_idsse, model  # noqa: E402
from possession.canonical import orient  # noqa: E402

HANDOFF_WIN = 12        # frames either side of a track end
HANDOFF_DIST = 2.0      # m between the end and the new start
CONTACT_DIST = 2.0      # m to the nearest opponent at the end
JUMP_M = 0.8            # one-frame step > 20 m/s at 25 fps = identity jump
GOAL_DEPTH = 3.0        # m from the goal line
GOAL_HALF_W = 12.0      # |y| for "on the goal line"
STILL_S = 20.0          # stationary for this long (a real keeper moves within 20 s)
STILL_M = 0.5


def _runs_mask(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def _starts_ends(p):
    """Per slot: start frames (first present) and end frames (first absent after present)."""
    d = np.diff(np.vstack([np.zeros((1, p.shape[1]), bool), p, np.zeros((1, p.shape[1]), bool)])
                .astype(np.int8), axis=0)
    return d == 1, d == -1          # (T+1, S): start at t, end at t (last present t-1)


def stats(m, n_real=None):
    """n_real: only the first n_real slots take part in handoff/jump/ghost statistics (the
    degrader's extra duplicate slots re-pick their source player every frame)."""
    m = orient(m)
    fps = m.fps
    g = m.graded
    xy = m.players_xy
    team = m.team_of_slot
    p = ~np.isnan(xy[..., 0]) & g[:, None]
    T, S = p.shape
    minutes = g.sum() / fps / 60
    L, W = m.pitch
    nh, na = p[:, team == 0].sum(1), p[:, team == 1].sum(1)
    out = {"minutes": float(minutes),
           "share_any12": float(np.mean(((nh >= 12) | (na >= 12))[g])),
           "share_12_vs_10": float(np.mean((((nh >= 12) & (na <= 10)) |
                                            ((na >= 12) & (nh <= 10)))[g]))}

    a12, b12 = _runs_mask((((nh >= 12) & (na <= 10)) | ((na >= 12) & (nh <= 10))) & g)
    out["run_12_vs_10_s"] = (np.percentile((b12 - a12) / fps, [50, 90]).tolist()
                             if len(a12) else [0, 0])
    if n_real is not None:
        p = p.copy()
        p[:, n_real:] = False
    st, en = _starts_ends(p)
    run_len = {}
    for j in range(S):
        a, b = np.flatnonzero(st[:, j]), np.flatnonzero(en[:, j])
        for s, e in zip(a, b):
            run_len[(j, s)] = e - s
    cross, same, contact, durs, at_goal = 0, 0, 0, [], 0
    ends = np.argwhere(en[1:])          # (t, j) with t = end frame (last present t - 1 + 1)
    for t, j in ends:
        t = t + 1
        last = xy[t - 1, j]
        lo, hi = max(0, t - HANDOFF_WIN), min(T, t + HANDOFF_WIN + 1)
        cand = np.argwhere(st[lo:hi])
        best = None
        for dt, k in cand:
            if k == j:
                continue
            f = lo + dt
            if f >= T or np.isnan(xy[f, k, 0]):
                continue
            d = np.linalg.norm(xy[f, k] - last)
            if d < HANDOFF_DIST and (best is None or d < best[0]):
                best = (d, k, f)
        if best is None:
            continue
        _, k, f = best
        if team[k] == team[j]:
            same += 1
            continue
        cross += 1
        opp = (team != team[j]) & p[t - 1] & (np.arange(S) != k)
        if opp.any():
            dd = np.linalg.norm(xy[t - 1, opp] - last, axis=1)
            contact += int(np.nanmin(dd) < CONTACT_DIST)
        durs.append(run_len.get((k, f), 0) / fps)
        at_goal += int(abs(last[0]) > L / 2 - 16.5 and abs(last[1]) < 20.2)

    # one-frame identity jumps landing on an opponent's previous position
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
    jumps = np.argwhere((step > JUMP_M) & p[1:] & p[:-1])
    jump_cross = 0
    for t, j in jumps:
        opp = (team != team[j]) & p[t]
        if opp.any() and np.nanmin(np.linalg.norm(xy[t, opp] - xy[t + 1, j], axis=1)) < 1.0:
            jump_cross += 1

    # stationary dots on a goal line, by whether the slot's team defends that goal
    w = int(STILL_S * fps)
    if n_real is not None:
        S = n_real
    ghost_own, ghost_opp = np.zeros(T, bool), np.zeros(T, bool)
    for j in range(S):
        x, y = xy[:, j, 0], xy[:, j, 1]
        near = p[:, j] & (np.abs(x) > L / 2 - GOAL_DEPTH) & (np.abs(y) < GOAL_HALF_W)
        if near.sum() < w:
            continue
        disp = np.full(T, np.inf)
        disp[w:] = np.linalg.norm(xy[w:, j] - xy[:-w, j], axis=1)
        still = near & np.r_[np.zeros(w, bool), near[:-w]] & (disp < STILL_M)
        own_goal_sign = -1 if team[j] == 0 else 1          # oriented: home defends -x
        own = still & (np.sign(x) == own_goal_sign)
        ghost_own |= own
        ghost_opp |= still & ~own
    out.update({
        "cross_handoffs_per_min": cross / minutes,
        "same_handoffs_per_min": same / minutes,
        "cross_handoff_contact_share": contact / max(cross, 1),
        "cross_handoff_dur_s": np.percentile(durs, [25, 50, 75, 90]).tolist() if durs else [],
        "cross_handoff_durs": durs,
        "cross_handoff_in_box_share": at_goal / max(cross, 1),
        "jumps_per_min": len(jumps) / minutes,
        "jump_onto_opponent_per_min": jump_cross / minutes,
        "share_ghost_own_goal": float(ghost_own[g].mean()),
        "share_ghost_opp_goal": float(ghost_opp[g].mean()),
    })
    return out


def summarise(name, rows):
    mins = sum(r["minutes"] for r in rows)
    wavg = lambda k: sum(r[k] * r["minutes"] for r in rows) / mins
    durs = np.concatenate([r["cross_handoff_durs"] for r in rows]) if rows else np.array([])
    q = np.percentile(durs, [25, 50, 75, 90]) if len(durs) else [np.nan] * 4
    print("{:<18} 12+:{:.3f} 12v10:{:.3f} (run p50/p90 {:.1f}/{:.1f}s) in-box {:.0%} | cross-team handoff {:.2f}/min (contact {:.0%}, dur "
          "p25/50/75/90 {:.1f}/{:.1f}/{:.1f}/{:.1f}s) same-team {:.2f}/min | jumps {:.2f}/min, "
          "onto opponent {:.2f}/min | goal-line ghost own {:.3f} opp {:.3f}".format(
              name, wavg("share_any12"), wavg("share_12_vs_10"),
              *np.mean([r["run_12_vs_10_s"] for r in rows], axis=0),
              wavg("cross_handoff_in_box_share"), wavg("cross_handoffs_per_min"),
              sum(r["cross_handoff_contact_share"] * r["cross_handoffs_per_min"] * r["minutes"]
                  for r in rows) / max(1e-9, sum(r["cross_handoffs_per_min"] * r["minutes"]
                                                 for r in rows)),
              *q, wavg("same_handoffs_per_min"), wavg("jumps_per_min"),
              wavg("jump_onto_opponent_per_min"), wavg("share_ghost_own_goal"),
              wavg("share_ghost_opp_goal")), flush=True)
    return durs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-atd", action="store_true")
    ap.add_argument("--out", default=os.path.join(config.CACHE_DIR, "tracking_error_stats.json"))
    args = ap.parse_args()
    res = {}
    if not args.skip_atd:
        r = stats(io_atd.load_match())
        res["atd"] = r
        summarise("BAR-ATH ATD", [r])
    with open(os.path.join(config.CACHE_DIR, "degrader_params.json")) as h:
        params = json.load(h)
    rows = []
    for mid in io_idsse.MATCH_IDS:
        n_real = len(io_idsse.load_match(mid).team_of_slot)
        rows.append(stats(model.make_copy(mid, params, *model.EVAL_COPY), n_real))
    summarise("IDSSE degraded", rows)
    res["idsse"] = rows
    with open(args.out, "w") as h:
        json.dump(res, h)


if __name__ == "__main__":
    main()
