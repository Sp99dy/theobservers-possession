"""Algorithmic kickoff detection. No frame constants: everything comes from player geometry.

Laws of the Game, law 8: at a kickoff every player is in his own half. So a kickoff is preceded by
a stretch where (oriented) nearly all home players have x < 0 and nearly all away players have
x > 0, while the players are almost standing still. Measured on BAR-ATH tracking (no ground truth),
that pattern held for >= 3 s exactly six times in graded play: the second-half kickoff and four
others at the times of the four tagged goals. Open play never produced it.

The kickoff itself is the moment that formation breaks up.
"""

from dataclasses import dataclass

import numpy as np

from . import config as C
from .canonical import AWAY, HOME
from .features import rolling_fraction


@dataclass
class Kickoff:
    start: int          # first frame of the line-up formation
    end: int            # first frame after it: the kickoff instant
    half: int
    kicking_team: int   # HOME / AWAY: the side with a player nearest the centre spot at the end


def _runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def formation_mask(f, half):
    """Frames where both teams are lined up in their own halves (majority over +/-0.5 s)."""
    with np.errstate(invalid="ignore"):
        raw = ((f["n_home"] >= C.KICKOFF_MIN_PLAYERS) & (f["n_away"] >= C.KICKOFF_MIN_PLAYERS)
               & (f["home_own_half"] >= C.KICKOFF_OWN_HALF_FRAC)
               & (f["away_own_half"] >= C.KICKOFF_OWN_HALF_FRAC) & (half > 0))
    return (rolling_fraction(raw, C.FPS // 2) > 0.5) & (half > 0)


def detect(f, half, fps=25):
    """[Kickoff] in time order."""
    form = formation_mask(f, half)
    out = []
    for start, end in _runs(form):
        if end - start < C.KICKOFF_MIN_DURATION_S * fps:
            continue
        if not np.nanmean(f["player_speed"][start:end]) < C.KICKOFF_MAX_SPEED:
            continue
        if half[start] != half[end - 1]:
            continue
        tail = slice(max(start, end - fps), end)
        home_c = np.min(f["d_centre_home"][tail])
        away_c = np.min(f["d_centre_away"][tail])
        out.append(Kickoff(start, end, int(half[start]), HOME if home_c <= away_c else AWAY))
    return out


def pre_kickoff_spans(kickoffs, half_blocks, fps=25):
    """[(start, stop)] stretches from a half's first graded frame to its opening kickoff.

    Every half begins with a kickoff. If a kickoff line-up starts within the first few minutes
    of a graded half, the graded half began before the kickoff and everything up to it is a dead
    ball. If none does, the half was graded from the kickoff itself and nothing is added.
    """
    spans = []
    for h, start, stop in half_blocks:
        first = [k for k in kickoffs if k.half == h]
        if first and first[0].start - start <= C.PRE_KICKOFF_WINDOW_S * fps:
            spans.append((start, first[0].end))
    return spans
