"""The one representation every data source is converted into.

Coordinates are metres, pitch-centred, x along the length and y across it with y pointing "up"
(the usual maths convention): x in [-L/2, L/2], y in [-W/2, W/2]. Missing observations are NaN.

Slots are tracklets, not people (BAR-ATH has 46 slots for 22 players), so downstream features must
be permutation-invariant within a team: centroids, spreads, nearest-player distances, counts.

Label codes follow the organisers' scorer: H=0, A=1, D=2, X=3.
"""

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

H, A, D, X = 0, 1, 2, 3
LETTERS = np.array(["H", "A", "D", "X"])
HOME, AWAY = 0, 1


def normalised_to_metres(x, y, pitch):
    """Metrica/EPTS normalised coordinates (origin top-left, y down, both 0..1) -> canonical metres.

    x_m = (x - 0.5) * length          left touchline-to-touchline axis becomes centred
    y_m = (0.5 - y) * width           flipped so that y points up
    """
    length, width = pitch
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return (x - 0.5) * length, (0.5 - y) * width


@dataclass
class Match:
    """Per-frame arrays for one match, indexed by the frame we label (video frame for BAR-ATH)."""
    name: str
    fps: int
    pitch: tuple                      # (length, width) in metres
    players_xy: np.ndarray            # (T, S, 2) float64 metres, NaN where the slot is empty
    players_speed: np.ndarray         # (T, S)    float64 as provided by the source (may be NaN)
    team_of_slot: np.ndarray          # (S,)      int8, HOME=0 / AWAY=1
    ball_xy: np.ndarray               # (T, 2)    float64 metres, NaN where the ball is not tracked
    ball_speed: np.ndarray            # (T,)      float64 as provided (do not trust it for ATD)
    half: np.ndarray                  # (T,)      int8, 0 = not in a half, else 1 or 2
    labels: Optional[np.ndarray] = None   # (T,) int8 H/A/D/X codes, only for labelled sources
    oriented: bool = False            # True once orient() has made home attack +x in every half
    attack_sign: dict = field(default_factory=dict)  # half -> +1/-1 as measured BEFORE orienting

    @property
    def n_frames(self):
        return len(self.half)

    @property
    def graded(self):
        """Frames inside a half. For BAR-ATH these are exactly the frames the organisers score."""
        return self.half > 0

    @property
    def ball_present(self):
        return ~np.isnan(self.ball_xy[:, 0])

    @property
    def player_present(self):
        return ~np.isnan(self.players_xy[..., 0])

    @property
    def home_xy(self):
        return self.players_xy[:, self.team_of_slot == HOME]

    @property
    def away_xy(self):
        return self.players_xy[:, self.team_of_slot == AWAY]

    def half_blocks(self):
        """[(half, start, stop)] half-open frame spans of each half, in play order."""
        out = []
        for h in sorted(set(np.unique(self.half).tolist()) - {0}):
            idx = np.flatnonzero(self.half == h)
            if len(idx) != idx[-1] - idx[0] + 1:
                raise ValueError("half {} is not contiguous".format(h))
            out.append((h, int(idx[0]), int(idx[-1]) + 1))
        return out


def team_centroid_x(match, team, min_players=5):
    """Per-frame mean x of one team's tracked players; NaN where fewer than min_players are seen."""
    xy = match.players_xy[:, match.team_of_slot == team, 0]
    seen = ~np.isnan(xy)
    count = seen.sum(axis=1)
    total = np.where(seen, xy, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        centroid = total / count
    centroid[count < min_players] = np.nan
    return centroid


def measure_attack_sign(match):
    """{half: +1 if the home team attacks +x in that half, else -1}, measured from the data.

    A team defends the goal on its own side, so the side whose players sit at lower x on average
    attacks +x. We compare the two teams' median centroid x over the whole half: teams change ends
    only at half-time, and the median over tens of thousands of frames is immune to the few minutes
    where one side is pinned in its own half.
    """
    signs = {}
    home_cx = team_centroid_x(match, HOME)
    away_cx = team_centroid_x(match, AWAY)
    for h, start, stop in match.half_blocks():
        hx = home_cx[start:stop]
        ax = away_cx[start:stop]
        ok = ~np.isnan(hx) & ~np.isnan(ax)
        if ok.sum() == 0:
            raise ValueError("half {}: no frames with both teams tracked".format(h))
        gap = np.median(ax[ok]) - np.median(hx[ok])
        if gap == 0:
            raise ValueError("half {}: cannot tell attacking direction".format(h))
        signs[h] = 1 if gap > 0 else -1
    return signs


def orient(match):
    """Copy of the match rotated 180 degrees in every half where home attacks -x.

    A 180-degree rotation (x -> -x, y -> -y) is exactly what swapping ends does physically, so left
    and right wings keep their meaning. Frames outside both halves are left as they are.
    """
    if match.oriented:
        return match
    signs = measure_attack_sign(match)
    flip = np.ones(match.n_frames)
    for h, sign in signs.items():
        flip[match.half == h] = sign
    players = match.players_xy * flip[:, None, None]
    ball = match.ball_xy * flip[:, None]
    return replace(match, players_xy=players, ball_xy=ball, oriented=True, attack_sign=signs)
