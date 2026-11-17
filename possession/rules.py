"""v1 frame evidence: hand-set, logistic-style scores with signs and sizes argued from football.

Two per-frame scores, both in nats (natural-log likelihood ratios per frame, before the decoder's
time aggregation):

    z  > 0  favours a dead ball,      z < 0 favours live play
    y  > 0  favours home possession,  y < 0 favours away possession (only meaningful when live)

turned into per-state log-likelihoods  D: +z/2,  H: -z/2 + y/2,  A: -z/2 - y/2.
Nothing in this module was fitted to any ground truth; config.py says where each number came from.
"""

import numpy as np

from . import config as C
from .canonical import AWAY, HOME


def dead_score(f, speed_threshold):
    """z: evidence for a dead ball."""
    z = np.zeros(len(f["ball_present"]))
    # Players walk while the ball is dead. Linear in the 1 s mean player speed, clipped.
    speed = np.nan_to_num(f["player_speed"], nan=speed_threshold)
    z += np.clip(-C.SPEED_SLOPE * (speed - speed_threshold), -C.SPEED_CLIP, C.SPEED_CLIP)
    # A ball outside the pitch box is out of play.
    z += C.BALL_OUT_LLR * f["ball_out"]
    # The tracker loses the ball more often when it is out of play or held.
    z += C.BALL_MISSING_LLR * (1.0 - f["ball_present"])
    # On the pitch, a ball moving fast is being played; a ball standing still is being placed.
    bs = f["ball_speed"]
    in_play = f["ball_in"] > 0
    with np.errstate(invalid="ignore"):
        z -= C.BALL_MOVING_LLR * (in_play & (bs > C.BALL_MOVING_SPEED))
        z += C.BALL_STILL_LLR * (in_play & (bs < C.BALL_STILL_SPEED))
    return z


def team_score(f):
    """y: evidence for home vs away possession from who is nearest the ball."""
    dh, da = f["d_home"], f["d_away"]
    ok = (f["ball_in"] > 0) & np.isfinite(dh) & np.isfinite(da)
    y = np.zeros(len(dh))
    nearest = np.minimum(dh, da)
    # Closer control is stronger evidence; a ball in flight (nobody near) says little.
    weight = 1.0 / (1.0 + (nearest / C.CONTROL_DISTANCE_M) ** 2)
    margin = np.clip((da - dh) / C.TEAM_MARGIN_SCALE_M, -C.TEAM_MARGIN_CLIP, C.TEAM_MARGIN_CLIP)
    y[ok] = (weight * margin)[ok]
    # Territory: the team in possession usually plays in the opponent's half. Weak, and the only
    # team cue available while the ball is missing.
    terr = np.clip(np.nan_to_num(f["centroid_x"]) / C.TERRITORY_SCALE_M, -1.0, 1.0)
    y += C.TERRITORY_LLR * terr
    return y


def kickoff_evidence(n, kickoffs, pre_spans, fps=25):
    """(dz, dy) from detected kickoffs: the line-up is dead, the kicking team then has the ball."""
    dz = np.zeros(n)
    dy = np.zeros(n)
    for k in kickoffs:
        dz[k.start:k.end] += C.KICKOFF_DEAD_LLR
        stop = min(n, k.end + int(C.KICKOFF_TEAM_WINDOW_S * fps))
        dz[k.end:stop] -= C.KICKOFF_LIVE_LLR
        dy[k.end:stop] += C.KICKOFF_TEAM_LLR * (1 if k.kicking_team == HOME else -1)
    for start, stop in pre_spans:
        dz[start:stop] += C.KICKOFF_DEAD_LLR
    return dz, dy


def emissions(z, y):
    """(T, 3) per-frame log-likelihoods for H, A, D."""
    return np.stack([-z / 2 + y / 2, -z / 2 - y / 2, z / 2], axis=1)


LAW9_MIN_FRAMES = 5      # [L] 0.2 s at 25 fps: a ball over the line for that long is out of play
LAW9_NATS = 30.0         # [D] H/A log-lik set this far below D: a hard constraint in practice


def law9_mask(ball_out, min_frames=LAW9_MIN_FRAMES):
    """Law 9: frames in runs of >= min_frames where the tracked ball is more than
    BALL_OUT_MARGIN_M outside the touchline or goal line (features' ball_out)."""
    out = np.asarray(ball_out) > 0
    m = np.r_[False, out, False].astype(np.int8)
    d = np.diff(m)
    mask = np.zeros(len(out), dtype=bool)
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a >= min_frames:
            mask[a:b] = True
    return mask


def law9_override(ll, mask):
    """Force D on masked frames as an emission override (the decoder's minimum durations and
    switch costs still apply): H and A log-likelihoods are put LAW9_NATS below D's."""
    out = ll.copy()
    out[mask, 0] = out[mask, 2] - LAW9_NATS
    out[mask, 1] = out[mask, 2] - LAW9_NATS
    return out
