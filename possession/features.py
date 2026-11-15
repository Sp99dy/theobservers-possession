"""Per-frame, identity-free features from a canonical, oriented Match.

One code path for every source (BAR-ATH now, degraded IDSSE in M2). Nothing here reads labels.
Slots are tracklets, so everything is a team-level aggregate: nearest distances, counts, means.
Speeds are computed from positions (the provided ball speed channel is unusable on BAR-ATH: its
maximum is 7 m/s while position displacement gives a realistic p95 of 17 m/s).
"""

import numpy as np

from . import config as C
from .canonical import AWAY, HOME


def rolling_mean(x, half_width):
    """Centred mean over +/- half_width frames, ignoring NaN; NaN where the window has no data."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    v = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - half_width)
    hi = np.minimum(len(x), i + half_width + 1)
    count = n[hi] - n[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (v[hi] - v[lo]) / count
    out[count == 0] = np.nan
    return out


def rolling_fraction(mask, half_width):
    return rolling_mean(np.asarray(mask, dtype=np.float64), half_width)


def _nearest(players, ball):
    d = np.linalg.norm(players - ball[:, None, :], axis=-1)
    d = np.where(np.isnan(d), np.inf, d)
    return d.min(axis=1)


def displacement_speed(xy, fps):
    """Speed (m/s) from consecutive positions, assigned to the later frame; NaN at gaps."""
    step = np.full(xy.shape[:-1], np.nan)
    step[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=-1) * fps
    return step


def compute(match):
    """Dict of (T,) float arrays. Requires an oriented match (home attacks +x)."""
    if not match.oriented:
        raise ValueError("orient() the match first")
    fps = match.fps
    length, width = match.pitch
    f = {}

    ball = match.ball_xy
    present = match.ball_present
    over_x = np.abs(ball[:, 0]) - length / 2
    over_y = np.abs(ball[:, 1]) - width / 2
    over = np.fmax(over_x, over_y)                       # >0 means outside the pitch box
    f["ball_present"] = present.astype(np.float64)
    f["ball_out"] = (present & (over > C.BALL_OUT_MARGIN_M)).astype(np.float64)
    f["ball_in"] = (present & (over <= C.BALL_OUT_MARGIN_M)).astype(np.float64)

    # Ball speed from positions, median-free but averaged over +/-0.5 s to beat jitter.
    ball_speed = displacement_speed(ball, fps)
    ball_speed = np.minimum(ball_speed, C.BALL_SPEED_CAP)
    f["ball_speed"] = rolling_mean(ball_speed, fps // 2)

    home = match.players_xy[:, match.team_of_slot == HOME]
    away = match.players_xy[:, match.team_of_slot == AWAY]
    f["d_home"] = _nearest(home, ball)
    f["d_away"] = _nearest(away, ball)

    # Collective speed: mean over tracked slots of displacement speed, capped against tracklet
    # switches, then smoothed over +/-1 s.
    pspeed = np.minimum(displacement_speed(match.players_xy, fps), C.PLAYER_SPEED_CAP)
    with np.errstate(invalid="ignore"):
        frame_mean = np.nanmean(pspeed, axis=1) if pspeed.size else np.full(match.n_frames, np.nan)
    f["player_speed"] = rolling_mean(frame_mean, fps)

    hx, ax = home[..., 0], away[..., 0]
    nh = (~np.isnan(hx)).sum(axis=1)
    na = (~np.isnan(ax)).sum(axis=1)
    f["n_home"] = nh.astype(np.float64)
    f["n_away"] = na.astype(np.float64)
    tol = C.KICKOFF_HALFWAY_TOL_M
    with np.errstate(invalid="ignore", divide="ignore"):
        f["home_own_half"] = np.where(nh > 0, (hx < tol).sum(axis=1) / nh, np.nan)
        f["away_own_half"] = np.where(na > 0, (ax > -tol).sum(axis=1) / na, np.nan)
    allx = match.players_xy[..., 0]
    with np.errstate(invalid="ignore"):
        f["centroid_x"] = np.nanmean(allx, axis=1) if allx.size else np.full(match.n_frames, np.nan)
    d_centre = np.linalg.norm(match.players_xy, axis=-1)
    f["d_centre_home"] = np.where(np.isnan(d_centre[:, match.team_of_slot == HOME]), np.inf,
                                  d_centre[:, match.team_of_slot == HOME]).min(axis=1)
    f["d_centre_away"] = np.where(np.isnan(d_centre[:, match.team_of_slot == AWAY]), np.inf,
                                  d_centre[:, match.team_of_slot == AWAY]).min(axis=1)
    return f


# -- v2: design matrix for the learned classifier --------------------------------------------------

def _team_stats(xy, fps):
    """Per-frame mean x, mean y, std x, std y and mean x-velocity of one team (NaN-aware)."""
    x, y = xy[..., 0], xy[..., 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cx = np.nanmean(x, axis=1)
            cy = np.nanmean(y, axis=1)
            sx = np.nanstd(x, axis=1)
            sy = np.nanstd(y, axis=1)
            sp = np.nanmean(np.minimum(displacement_speed(xy, fps), C.PLAYER_SPEED_CAP), axis=1)
            q_back = np.nanpercentile(x, 10, axis=1)   # defensive line (oriented frame)
            q_front = np.nanpercentile(x, 90, axis=1)
    vx = np.full(len(cx), np.nan)
    smooth = rolling_mean(cx, fps // 2)
    vx[fps:-fps] = (smooth[2 * fps:] - smooth[:-2 * fps]) / 2.0     # m/s over +/-1 s
    return {"cx": cx, "cy": cy, "sx": sx, "sy": sy, "speed": sp, "vx": vx,
            "back": q_back, "front": q_front}


def _last_next(values, valid):
    """For each frame: value at the last valid frame at or before it, and at the next at or after,
    plus the distance (frames) to each. NaN / large where none."""
    T = len(values)
    idx = np.arange(T)
    last = np.where(valid, idx, -1)
    last = np.maximum.accumulate(last)
    nxt = np.where(valid, idx, T)
    nxt = np.minimum.accumulate(nxt[::-1])[::-1]
    lv = np.where(last >= 0, values[np.clip(last, 0, T - 1)], np.nan)
    nv = np.where(nxt < T, values[np.clip(nxt, 0, T - 1)], np.nan)
    since = np.where(last >= 0, idx - last, 10 ** 6).astype(np.float64)
    until = np.where(nxt < T, nxt - idx, 10 ** 6).astype(np.float64)
    return lv, nv, since, until


FEATURE_VERSION = 1                  # the shipped v2/v3 design matrix (their models expect it)
LATEST_FEATURE_VERSION = 2           # v1 + touch detection + near-ball slowdown (v4)
ROLL_WINDOWS_S = (0.5, 2.0, 5.0)     # [D] SPEC 5.3: centred +/-0.5 s, 2 s, 5 s
GAP_CAP_S = 30.0                     # [D] beyond 30 s the exact gap length carries no information

# -- touch detection (feature version 2) ---------------------------------------------------------
# The ATD ball (and the degrader that mimics it) is smoothed over ~1 s, so a kick or an interception
# shows up as a change of ball velocity spread over ~0.5 s either side. We compare the mean
# velocity over [t-20, t-8] with [t+8, t+20] frames and call a local maximum of that change a touch.
TOUCH_GAP = 8                # [M] frames skipped either side: ~ half the degrader's ball smoothing
TOUCH_SPAN = 12              # [D] frames averaged on each side (~0.5 s)
TOUCH_MIN_DV = 3.0           # [F] m/s: a pass, a trap or a deflection; jitter is ~0.5 m/s
TOUCH_MAX_DIST = 3.0         # [F/M] the toucher is within 3 m of the smoothed ball at the touch
TOUCH_SUPPRESS = 12          # [D] one touch per ~0.5 s at most (local maximum of the change)
SLOW_NEAR_K = 4              # [F] "players close to the ball": the four nearest
SLOW_WINDOWS_S = (3.0, 6.0, 12.0)    # [F] a stoppage lasts tens of seconds; play resumes in seconds
SLOW_WALK = 2.0              # [F] m/s: walking/jogging pace


def _causal_mean(x, lo, hi):
    """Mean of x over the window [t+lo, t+hi) for every t (lo<hi, may be negative); NaN-aware."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    v = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    n = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    a = np.clip(i + lo, 0, len(x))
    b = np.clip(i + hi, 0, len(x))
    count = n[b] - n[a]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (v[b] - v[a]) / count
    out[count == 0] = np.nan
    return out, count


def detect_touches(match, fps):
    """[(frame, team, dv, speed_after)] ball contacts inferred from the ball trajectory.

    team is +1 for home, -1 for away (signed, so swap_teams flips it). Only frames where the ball
    is tracked on both sides of the contact can yield a touch.
    """
    ball = match.ball_xy
    vel = np.full_like(ball, np.nan)
    vel[3:-3] = (ball[6:] - ball[:-6]) * fps / 6.0
    before, nb = zip(*[_causal_mean(vel[:, k], -TOUCH_GAP - TOUCH_SPAN, -TOUCH_GAP) for k in (0, 1)])
    after, na = zip(*[_causal_mean(vel[:, k], TOUCH_GAP + 1, TOUCH_GAP + 1 + TOUCH_SPAN)
                      for k in (0, 1)])
    vb, va = np.stack(before, 1), np.stack(after, 1)
    enough = (nb[0] >= TOUCH_SPAN // 2) & (na[0] >= TOUCH_SPAN // 2) & match.ball_present
    dv = np.where(enough, np.linalg.norm(va - vb, axis=1), 0.0)
    dv = np.nan_to_num(dv)
    # local maximum within +/- TOUCH_SUPPRESS frames
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(dv, TOUCH_SUPPRESS, constant_values=-1.0)
    local_max = sliding_window_view(pad, 2 * TOUCH_SUPPRESS + 1).max(axis=1)
    cand = np.flatnonzero((dv >= TOUCH_MIN_DV) & (dv >= local_max) & (match.half > 0))
    home = match.players_xy[:, match.team_of_slot == HOME]
    away = match.players_xy[:, match.team_of_slot == AWAY]
    dh = _nearest(home[cand], ball[cand])
    da = _nearest(away[cand], ball[cand])
    speed_after = np.linalg.norm(va[cand], axis=1)
    out = []
    last = -10 ** 9
    for t, h, a, d, s in zip(cand, dh, da, dv[cand], speed_after):
        if min(h, a) > TOUCH_MAX_DIST or t - last <= TOUCH_SUPPRESS:
            continue
        out.append((int(t), 1 if h <= a else -1, float(d), float(s)))
        last = t
    return out


def touch_features(match, fps):
    """Who touched the ball last / next, how long ago, and the touch balance per team."""
    T = match.n_frames
    touches = detect_touches(match, fps)
    cols = {}
    team = np.zeros(T)
    dv = np.full(T, np.nan)
    sp = np.full(T, np.nan)
    present = np.zeros(T, dtype=bool)
    for t, k, d, s in touches:
        team[t], dv[t], sp[t], present[t] = k, d, s, True
    cap = GAP_CAP_S * fps
    lt, nt, since, until = _last_next(team, present)
    cols["touch_last_team"] = np.nan_to_num(lt)
    cols["touch_next_team"] = np.nan_to_num(nt)
    cols["touch_since_s"] = np.minimum(since, cap) / fps
    cols["touch_until_s"] = np.minimum(until, cap) / fps
    ld, nd, _, _ = _last_next(np.nan_to_num(dv), present)
    ls, ns, _, _ = _last_next(np.nan_to_num(sp), present)
    cols["touch_last_dv"] = ld
    cols["touch_last_speed_after"] = ls
    cols["touch_next_dv"] = nd
    # team of the touch before the last one: two touches by the same team = control
    prev = np.zeros(T)
    if len(touches) > 1:
        ts = np.array([t for t, _, _, _ in touches])
        ks = np.array([k for _, k, _, _ in touches], dtype=float)
        j = np.searchsorted(ts, np.arange(T), side="right") - 2       # index of second-to-last
        prev = np.where(j >= 0, ks[np.clip(j, 0, len(ks) - 1)], 0.0)
    cols["touch_prev_team"] = prev
    for w in (2.0, 5.0):
        n = int(round(w * fps))
        m, c = _causal_mean(team, -n, 1)
        cols["touch_bal_prev{:g}".format(w)] = np.nan_to_num(m) * c
        m, c = _causal_mean(team, 0, n + 1)
        cols["touch_bal_next{:g}".format(w)] = np.nan_to_num(m) * c
    return cols


def speed_series(match, fps):
    """Per-frame mean speed of all players, of the SLOW_NEAR_K players nearest the ball (or where
    it was last seen), and the share of players slower than SLOW_WALK."""
    pspeed = np.minimum(displacement_speed(match.players_xy, fps), C.PLAYER_SPEED_CAP)
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            all_speed = np.nanmean(pspeed, axis=1)
            slow_share = np.nanmean(np.where(np.isnan(pspeed), np.nan, pspeed < SLOW_WALK), axis=1)
    # reference point: the ball, or where it was last seen
    ref = match.ball_xy.copy()
    ok = match.ball_present
    idx = np.where(ok, np.arange(len(ok)), -1)
    idx = np.maximum.accumulate(idx)
    ref = np.where((idx >= 0)[:, None], ref[np.clip(idx, 0, None)], np.nan)
    d = np.linalg.norm(match.players_xy - ref[:, None, :], axis=-1)
    d = np.where(np.isnan(d) | np.isnan(pspeed), np.inf, d)
    k = min(SLOW_NEAR_K, d.shape[1])
    near = np.argpartition(d, k - 1, axis=1)[:, :k]
    nd = np.take_along_axis(d, near, 1)
    ns = np.take_along_axis(np.nan_to_num(pspeed), near, 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        near_speed = np.where(np.isfinite(nd), ns, 0).sum(1) / np.isfinite(nd).sum(1)
    return {"all": all_speed, "near": near_speed, "slowshare": slow_share}


def slowdown_features(match, fps):
    """Collective and near-ball player speed before vs after each frame (stoppage detection)."""
    cols = {}
    for name, x in speed_series(match, fps).items():
        for w in SLOW_WINDOWS_S:
            n = int(round(w * fps))
            prev = _causal_mean(x, -n, 0)[0]
            nxt = _causal_mean(x, 0, n)[0]
            cols["slow_{}_next{:g}".format(name, w)] = nxt
            if w <= 6.0:
                cols["slow_{}_prev{:g}".format(name, w)] = prev
                cols["slow_{}_drop{:g}".format(name, w)] = prev - nxt
    return cols


def design_matrix(match, base=None, version=FEATURE_VERSION):
    """(T, F) float32 feature matrix and the list of feature names. Oriented match required.

    Identity-free, orientation-normalised (home attacks +x), and team-symmetric by construction:
    swapping the teams (swap_teams) maps every home feature to the matching away feature.
    version 1 is the shipped design; version 2 appends touch and slowdown features.
    """
    f = base if base is not None else compute(match)
    fps = match.fps
    length, width = match.pitch
    cols = {}

    ball = match.ball_xy
    ball_in = f["ball_in"] > 0
    cols["ball_present"] = f["ball_present"]
    cols["ball_out"] = f["ball_out"]
    cols["ball_x"] = np.where(match.ball_present, ball[:, 0], np.nan)
    cols["ball_absy"] = np.where(match.ball_present, np.abs(ball[:, 1]), np.nan)
    edge = np.fmin(length / 2 - np.abs(ball[:, 0]), width / 2 - np.abs(ball[:, 1]))
    cols["ball_edge"] = np.where(match.ball_present, edge, np.nan)
    cols["ball_speed"] = f["ball_speed"]
    near = np.minimum(f["d_home"], f["d_away"])
    margin = np.clip(f["d_away"] - f["d_home"], -15, 15)
    cols["near_dist"] = np.where(ball_in & np.isfinite(near), np.minimum(near, 30), np.nan)
    cols["margin"] = np.where(ball_in & np.isfinite(margin), margin, np.nan)
    cols["d_home"] = np.where(ball_in, np.minimum(f["d_home"], 30), np.nan)
    cols["d_away"] = np.where(ball_in, np.minimum(f["d_away"], 30), np.nan)

    # Gap features: who was in control when the ball was last / next seen, and how long ago.
    close = ball_in & (near < C.CONTROL_DISTANCE_M)
    lm, nm, since, until = _last_next(np.nan_to_num(margin), close)
    cap = GAP_CAP_S * fps
    cols["last_close_margin"] = lm
    cols["next_close_margin"] = nm
    cols["since_close_s"] = np.minimum(since, cap) / fps
    cols["until_close_s"] = np.minimum(until, cap) / fps
    _, _, since_b, until_b = _last_next(np.zeros(match.n_frames), match.ball_present)
    cols["since_ball_s"] = np.minimum(since_b, cap) / fps
    cols["until_ball_s"] = np.minimum(until_b, cap) / fps
    lx, nx, _, _ = _last_next(np.nan_to_num(ball[:, 0]), match.ball_present)
    cols["last_ball_x"] = lx
    cols["next_ball_x"] = nx
    le, ne, _, _ = _last_next(np.nan_to_num(edge), match.ball_present)
    cols["last_ball_edge"] = le
    cols["next_ball_edge"] = ne

    home = _team_stats(match.home_xy, fps)
    away = _team_stats(match.away_xy, fps)
    for key in ("cx", "sx", "sy", "speed", "vx", "back", "front"):
        cols["home_" + key] = home[key]
        cols["away_" + key] = away[key]
    cols["home_cy_abs"] = np.abs(home["cy"])
    cols["away_cy_abs"] = np.abs(away["cy"])
    cols["player_speed"] = f["player_speed"]
    cols["n_home"] = f["n_home"]
    cols["n_away"] = f["n_away"]
    cols["home_own_half"] = f["home_own_half"]
    cols["away_own_half"] = f["away_own_half"]
    cols["centroid_x"] = f["centroid_x"]
    cols["d_centre_home"] = np.minimum(f["d_centre_home"], 60)
    cols["d_centre_away"] = np.minimum(f["d_centre_away"], 60)

    # Centred rolling means of the time-varying cues.
    rolled = ("ball_present", "ball_out", "ball_speed", "margin", "near_dist", "player_speed",
              "home_vx", "away_vx", "home_speed", "away_speed", "centroid_x",
              "home_own_half", "away_own_half", "home_sx", "away_sx")
    for name in rolled:
        for w in ROLL_WINDOWS_S:
            cols["{}_m{:g}".format(name, w)] = rolling_mean(cols[name], int(round(w * fps)))

    if version >= 2:
        cols.update(touch_features(match, fps))
        cols.update(slowdown_features(match, fps))

    names = list(cols)
    X = np.empty((match.n_frames, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = cols[name]
    return X, names
