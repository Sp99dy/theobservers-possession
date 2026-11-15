"""Metrica Sports sample games 1 and 2 (public, github.com/metrica-sports/sample-data) -> Match.

Secondary validation only (SPEC 6). The games have clean tracking and Metrica's own event data,
but no per-frame possession, so H/A/D labels are DERIVED from the events by the rule below. These
labels are an approximation of the challenge's convention, which is why this check can only show
that a model selected on IDSSE does not collapse on Metrica's tracking, not fine differences.

Event rule (documented in TECHNICAL.md):
  * team: the team of the latest possession event (PASS, SET PIECE, SHOT, RECOVERY, BALL LOST,
    which is logged by the team that had the ball) owns the ball from that event's start frame. A
    loose ball stays with the last team, as in the challenge's definition.
  * dead: from the end of a stoppage event (BALL OUT; a SHOT ending OUT or GOAL; BALL LOST by
    OFFSIDE / HAND BALL) or the start of a FAULT RECEIVED, until the next restart (SET PIECE, or a
    PASS of subtype GOAL KICK). A stoppage with any live event before the restart, or no restart
    within MAX_DEAD_S, is ignored (advantage played, or a logging gap).
  * graded: each period from its first to its last event.
Then the same 2 s controlled-possession rule as for IDSSE (io_idsse.relabel_controlled).
"""

import os

import numpy as np
import pandas as pd

from . import config
from .canonical import A, D, H, X, Match
from .io_idsse import relabel_controlled

GAMES = (1, 2)
PITCH = (105.0, 68.0)          # Metrica sample data convention
MAX_DEAD_S = 90.0              # [D] a stoppage longer than this without a logged restart is a gap
POSSESSION_TYPES = ("PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST")
RESTART = lambda t, s: t == "SET PIECE" or (t == "PASS" and s == "GOAL KICK")


def _path(game, kind):
    return os.path.join(config.METRICA_DIR, "Sample_Game_{}_{}.csv".format(game, kind))


def _tracking(game, side):
    df = pd.read_csv(_path(game, "RawTrackingData_{}_Team".format(side)), skiprows=2)
    cols = list(df.columns)
    players = []
    for i, c in enumerate(cols):
        if c.startswith("Player"):
            players.append(df.iloc[:, [i, i + 1]].to_numpy(np.float64))
    b = cols.index("Ball")
    ball = df.iloc[:, [b, b + 1]].to_numpy(np.float64)
    return df["Period"].to_numpy(), df["Frame"].to_numpy(), np.stack(players, axis=1), ball


def _to_metres(xy):
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] - 0.5) * PITCH[0]
    out[..., 1] = (0.5 - xy[..., 1]) * PITCH[1]      # Metrica y points down; ours points up
    return out


def event_labels(events, n_frames, fps=25):
    """(T,) int8 H/A/D/X labels (frame index = Metrica frame - 1) and (T,) half array."""
    ev = events.sort_values(["Start Frame", "End Frame"], kind="stable").reset_index(drop=True)
    team = ev["Team"].map({"Home": H, "Away": A}).to_numpy()
    typ = ev["Type"].to_numpy()
    sub = ev["Subtype"].fillna("").to_numpy()
    start = ev["Start Frame"].to_numpy(np.int64) - 1
    end = np.maximum(ev["End Frame"].to_numpy(np.int64) - 1, start)
    period = ev["Period"].to_numpy()

    labels = np.full(n_frames, X, dtype=np.int8)
    half = np.zeros(n_frames, dtype=np.int8)
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        s, e = int(start[idx].min()), int(end[idx].max()) + 1
        half[s:e] = p
        owner = None
        for i in idx:
            if typ[i] in POSSESSION_TYPES and not np.isnan(team[i]):
                if owner is not None:
                    labels[owner[1]:start[i]] = owner[0]
                owner = (int(team[i]), int(start[i]))
        if owner is not None:
            labels[owner[1]:e] = owner[0]
        labels[s:e][labels[s:e] == X] = H if team[idx[0]] == H else A   # before the first owner

        # Dead-ball stretches.
        for k, i in enumerate(idx):
            stop_at = None
            if typ[i] == "BALL OUT":
                stop_at = end[i]
            elif typ[i] == "SHOT" and ("OUT" in sub[i] or "GOAL" in sub[i]):
                stop_at = end[i]
            elif typ[i] == "BALL LOST" and ("OFFSIDE" in sub[i] or "HAND BALL" in sub[i]):
                stop_at = end[i]
            elif typ[i] == "FAULT RECEIVED":
                stop_at = start[i]
            if stop_at is None:
                continue
            restart = None
            for j in idx[k + 1:]:
                if RESTART(typ[j], sub[j]):
                    restart = start[j]
                    break
                if typ[j] in ("PASS", "SHOT", "RECOVERY") and start[j] > stop_at + fps:
                    break                       # play went on: advantage, or a mislogged stop
            if restart is not None and 0 < restart - stop_at <= MAX_DEAD_S * fps:
                labels[stop_at:restart] = D
    labels[half == 0] = X
    return labels, half


def load_match(game, control=True):
    per_h, fr_h, home, ball = _tracking(game, "Home")
    per_a, fr_a, away, _ = _tracking(game, "Away")
    assert np.array_equal(fr_h, fr_a), "home and away tracking frames differ"
    assert np.array_equal(fr_h, np.arange(1, len(fr_h) + 1)), "frames are not 1..N"
    players = _to_metres(np.concatenate([home, away], axis=1))
    team_of_slot = np.r_[np.zeros(home.shape[1]), np.ones(away.shape[1])].astype(np.int8)
    events = pd.read_csv(_path(game, "RawEventsData"))
    labels, half = event_labels(events, len(fr_h))
    m = Match(name="metrica_{}".format(game), fps=25, pitch=PITCH, players_xy=players,
              players_speed=np.full(players.shape[:2], np.nan), team_of_slot=team_of_slot,
              ball_xy=_to_metres(ball), ball_speed=np.full(len(ball), np.nan), half=half,
              labels=labels)
    if control:
        from dataclasses import replace
        m = replace(m, labels=relabel_controlled(m.labels, m.half_blocks(),
                                                 int(round(config.CONTROL_MIN_S * m.fps))))
    return m
