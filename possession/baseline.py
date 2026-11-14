"""The organisers' nearest-player baseline, recomputed through our loader and writer.

Its only purpose is a pipeline check: if our ATD parsing, slot-to-team mapping, frame alignment and
CSV writer are all correct, the result is byte-identical to the organisers' committed
evaluation/results/baseline_*.csv. The smoothing rule is imported from their starter/baseline.py.

The distance arithmetic mirrors theirs operation for operation (x*L - bx*L, not (x-bx)*L), on the
same correctly rounded float64 values, so even exact distance ties break the same way: first slot
wins.
"""

import numpy as np

from . import config, io_atd, organisers

NOBODY = -1


def nearest_player_codes(players, ball, team_of_slot, pitch):
    """(T,) int8: team (0 home / 1 away) of the slot nearest the ball, NOBODY where undefined.

    players: (T, S, 3) normalised x, y, speed; ball: (T, 3). NaN = missing.
    """
    length, width = pitch
    out = np.full(len(ball), NOBODY, dtype=np.int8)
    has_ball = ~np.isnan(ball[:, 0])
    if not has_ball.any():
        return out
    bx = ball[has_ball, 0] * length
    by = ball[has_ball, 1] * width
    dx = players[has_ball, :, 0] * length - bx[:, None]
    dy = players[has_ball, :, 1] * width - by[:, None]
    distance = dx * dx + dy * dy
    valid = ~np.isnan(players[has_ball, :, 0])
    distance = np.where(valid, distance, np.inf)
    best = np.argmin(distance, axis=1)          # first minimum, like the strict '<' in theirs
    anyone = valid.any(axis=1)
    codes = np.where(anyone, team_of_slot[best], NOBODY).astype(np.int8)
    out[has_ball] = codes
    return out


def baseline_codes(data_dir=None):
    g = organisers.games()
    meta_path, raw_path = g.require(["atd_meta", "atd_raw"], g.GAME,
                                    data_dir or config.MTAH_DATA_DIR)
    raw = io_atd.load_raw(meta_path, raw_path)
    n_video = g.source_frames("tactical", g.GAME)
    players, ball = io_atd.raw_to_video_index(raw, n_video)
    return nearest_player_codes(players, ball, raw["team_of_slot"], g.GAMES[g.GAME]["pitch"])


def baseline_labels(codes, window):
    """Letters from the organisers' own majority-vote smoother."""
    return organisers.baseline().smooth(codes, window)
