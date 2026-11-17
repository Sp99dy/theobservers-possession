"""Feature version 3 = version 2 + near-ball pressure, control and relative-speed columns.

New module only: features.design_matrix (v1/v2) is untouched, so v2 stays bit-for-bit identical.
All new columns are team-agnostic (they describe the player nearest to the ball and his opponents,
whichever team he is on), so they are invariant under the H/A swap and the mirror; NaN where the
ball or the nearest player is not tracked. No label-dependent input.
"""

import numpy as np

DWELL_CAP_S = 5.0
CLIP_D = 30.0
MIN_SPEED = 0.5                       # m/s; the angle is undefined below this


def _vel(xy, fps):
    """Centred velocity (m/s), NaN at gaps. xy (T, ..., 2)."""
    v = np.full(xy.shape, np.nan)
    v[1:-1] = (xy[2:] - xy[:-2]) * fps / 2.0
    return v


def extra_columns(match):
    fps = match.fps
    P = match.players_xy                                   # (T, S, 2)
    team = np.asarray(match.team_of_slot)
    ball = match.ball_xy
    T = len(ball)
    d = np.linalg.norm(P - ball[:, None, :], axis=-1)
    field = np.isin(team, (0, 1))
    d = np.where(np.isnan(d) | ~field[None, :], np.inf, d)
    j = d.argmin(1)
    ok = np.isfinite(d[np.arange(T), j]) & ~np.isnan(ball[:, 0])
    pj = P[np.arange(T), j]                                # nearest player's position
    tj = team[j]
    dd = np.linalg.norm(P - pj[:, None, :], axis=-1)       # distances to every slot
    opp = (team[None, :] != tj[:, None]) & field[None, :]
    opp &= np.isfinite(dd)
    ddo = np.where(opp, dd, np.inf)
    cols = {}
    cols["p3_opp_in3"] = np.where(ok, (ddo < 3.0).sum(1), np.nan)
    cols["p3_opp_in5"] = np.where(ok, (ddo < 5.0).sum(1), np.nan)
    nd = ddo.min(1)
    cols["p3_opp_near_d"] = np.where(ok & np.isfinite(nd), np.minimum(nd, CLIP_D), np.nan)

    vb = _vel(ball, fps)
    vp = _vel(P, fps)[np.arange(T), j]
    # the centred difference is only valid if j is also nearest-tracked on both neighbours
    sb = np.linalg.norm(vb, axis=-1)
    sp = np.linalg.norm(vp, axis=-1)
    rel = np.linalg.norm(vb - vp, axis=-1)
    cols["p3_rel_speed"] = np.where(ok, np.minimum(rel, 40.0), np.nan)
    cols["p3_speed_diff"] = np.where(ok, np.clip(sb - sp, -40.0, 40.0), np.nan)
    cosang = np.sum(vb * vp, -1) / np.maximum(sb * sp, 1e-9)
    cols["p3_cos_angle"] = np.where(ok & (sb > MIN_SPEED) & (sp > MIN_SPEED),
                                    np.clip(cosang, -1, 1), np.nan)

    dwell = np.zeros(T)
    cap = DWELL_CAP_S * fps
    same = np.r_[False, (j[1:] == j[:-1]) & ok[1:] & ok[:-1]]
    run = 0
    for t in range(T):                                    # plain loop: ~150k frames, cheap
        run = run + 1 if same[t] else 0
        dwell[t] = min(run, cap)
    cols["p3_dwell_s"] = np.where(ok, dwell / fps, np.nan)
    return cols


def design_matrix(match, base=None):
    from possession import features
    X2, names = features.design_matrix(match, base=base, version=2)
    cols = extra_columns(match)
    X = np.concatenate([X2, np.column_stack([cols[k] for k in cols]).astype(np.float32)], axis=1)
    return X, names + list(cols)
