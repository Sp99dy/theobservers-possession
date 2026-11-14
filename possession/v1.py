"""v1: rule-based evidence + Smart Tagging soft layer + kickoff detection + segmental Viterbi."""

import numpy as np

from . import config as C
from . import decode, features, kickoff, rules, tagging_prior
from .canonical import X, orient


def predict(match, tags=None, use_tags=True, return_debug=False):
    """(T,) int8 codes: H/A/D on every frame inside a half, X elsewhere."""
    m = orient(match)
    f = features.compute(m)
    g = m.graded
    speed_threshold = float(np.nanquantile(f["player_speed"][g], C.DEAD_SHARE_PRIOR))

    z = rules.dead_score(f, speed_threshold)
    y = rules.team_score(f)

    kos = kickoff.detect(f, m.half, m.fps)
    pre = kickoff.pre_kickoff_spans(kos, m.half_blocks(), m.fps)
    dz, dy = rules.kickoff_evidence(m.n_frames, kos, pre, m.fps)
    z += dz
    y += dy

    if use_tags and tags:
        dz, dy = tagging_prior.evidence(m.n_frames, tags, kos, m.fps)
        z += dz
        y += dy

    # The scores were computed in the oriented frame, where "home" is still home: orientation
    # only rotates coordinates, so y needs no mapping back.
    ll = rules.emissions(z, y)
    blocks = [(s, e) for _, s, e in m.half_blocks()]
    codes = decode.decode_blocks(ll, blocks, C.STEP, C.STEP_WEIGHT, C.MIN_STEPS,
                                 C.SWITCH_COST, fill=X)
    if return_debug:
        return codes, {"kickoffs": kos, "pre_kickoff": pre, "speed_threshold": speed_threshold,
                       "attack_sign": m.attack_sign}
    return codes
