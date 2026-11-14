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


FEATURE_VERSION = 1                  # bump when the design matrix changes (invalidates caches)
ROLL_WINDOWS_S = (0.5, 2.0, 5.0)     # [D] SPEC 5.3: centred +/-0.5 s, 2 s, 5 s
GAP_CAP_S = 30.0                     # [D] beyond 30 s the exact gap length carries no information


def design_matrix(match, base=None):
    """(T, F) float32 feature matrix and the list of feature names. Oriented match required.

    Identity-free, orientation-normalised (home attacks +x), and team-symmetric by construction:
    swapping the teams (swap_teams) maps every home feature to the matching away feature.
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

    names = list(cols)
    X = np.empty((match.n_frames, len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = cols[name]
    return X, names
