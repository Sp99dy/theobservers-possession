"""v5.4: broadcast replay evidence for BAR-ATH (inference only, label-free).

Broadcasters cut to replays and close-ups mostly while play is stopped. On those frames the
broadcast ATD's homography is invalid. replay[t] = the broadcast homography is invalid at broadcast
frame t - ballfill.LAG[half] (the per-half lag measured label-free, 4014 / 26920 frames); graded
frames only (100% of graded frames have a lagged broadcast frame).

Evidence: replay frames get REPLAY_LLR (config "replay_llr") added to the D log-likelihood after
v5.2's 1e-3 rounding, then are rounded again, before v5.2's decoder. The value is the label-free
log(P(replay | D) / P(replay | live)), estimated by hard EM on BAR-ATH from v5.2's own decoding
(scripts/replay_evidence.py, bridge session): iteration 0 / 1 / 2 gave +2.020 / +2.179 / +2.195
(P(r|D) 0.659, P(r|live) 0.073 at the last); ball-present frames only +2.066, soft +1.910. It is
FROZEN at the last iterate, 2.195 (2.19494 rounded to the 1e-3 grid, which leaves every rounded
log-likelihood unchanged), so reproduction never re-runs EM. Weight 1.0. IDSSE has no broadcast,
so this cannot be scored by CV: adopted on Fadel's decision with label-free evidence.
"""

import numpy as np

from . import ballfill


def replay_mask(match, data_dir=None):
    """(T,) bool: graded tactical frames whose lagged broadcast frame has an invalid homography."""
    _, valid = ballfill.broadcast_ball(match.pitch, data_dir)
    out = np.zeros(match.n_frames, dtype=bool)
    for h, s, e in match.half_blocks():
        t = np.arange(s, e)
        j = t - ballfill.LAG[h]
        ok = (j >= 0) & (j < len(valid))
        out[t[ok]] = ~valid[j[ok]]
    return out & match.graded
