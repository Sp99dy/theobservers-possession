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
