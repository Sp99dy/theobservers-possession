"""Smart Tagging as soft evidence (BAR-ATH only: IDSSE has no tags, so this sits outside the
classifier and is added to the decoder's log-likelihoods).

The tags are automatic, phase-level and loosely timed, so every term is small next to a confident
ball observation, and their combined strength is one scalar, TAG_WEIGHT, set by reasoning (config.py).
"""

import numpy as np

from . import config as C
from .canonical import HOME
from .io_tagging import ATTACK_CODES, DEFEND_CODES, GOAL_CODE, SET_PIECE_CODE


def evidence(n, tags, kickoffs, fps=25):
    """(dz, dy) per frame, already multiplied by TAG_WEIGHT."""
    dz = np.zeros(n)
    team_votes = np.zeros(n)
    for t in tags:
        a, b = max(0, t.start), min(n, t.end)
        if t.code in ATTACK_CODES or t.code in DEFEND_CODES:
            if t.team is None:
                continue
            sign = 1.0 if t.team == HOME else -1.0
            if t.code in DEFEND_CODES:
                sign = -sign
            team_votes[a:b] += sign
        elif t.code == SET_PIECE_CODE:
            # A restart happened near the tag start: dead ball shortly before it, and the taking
            # team (when tagged) has the ball right after it. Kick-offs are left to kickoff.py.
            if "kick off" in t.type:
                continue
            lo = max(0, t.start - int(C.SET_PIECE_DEAD_BEFORE_S * fps))
            hi = min(n, t.start + int(C.SET_PIECE_DEAD_AFTER_S * fps))
            dz[lo:hi] += C.SET_PIECE_DEAD_LLR
            if t.team is not None:
                stop = min(n, t.start + int(C.SET_PIECE_TEAM_S * fps))
                team_votes[t.start:stop] += (1.0 if t.team == HOME else -1.0)
        elif t.code == GOAL_CODE:
            # After a goal the ball is dead until the next kickoff.
            nxt = [k for k in kickoffs if k.start >= t.end - fps
                   and k.start - t.end <= C.GOAL_TO_KICKOFF_MAX_S * fps]
            if nxt:
                dz[min(n, t.end):nxt[0].start] += C.GOAL_DEAD_LLR
    dy = C.TAG_TEAM_LLR * np.clip(team_votes, -1.0, 1.0)
    return C.TAG_WEIGHT * dz, C.TAG_WEIGHT * dy
