"""v3 (M3): v2 with the refinements that IDSSE nested CV confirmed.

* a larger emission classifier (HGB_V3 below);
* a v1/v2 ensemble: the v1 rule evidence (dead-ball score z, home-vs-away score y) is added at weight
  `rule_weight`, in classifier units (the same label-free EVIDENCE_SCALE as the kickoff/tag layers);
* the kickoff and Smart Tagging layers exactly as in v2.

Every choice was made on IDSSE leave-one-match-out CV (scripts/cv_idsse.py,
scripts/decoder_experiments.py); none on BAR-ATH labels.
"""

import json
import os

import numpy as np

from . import config as C
from . import cv, features, kickoff, model, rules, tagging_prior
from .canonical import X, orient
from .v2 import evidence_scale

MODEL_PATH = os.path.join(C.REPO_DIR, "models", "emission_v3.joblib")
DECODER_PATH = os.path.join(C.REPO_DIR, "models", "decoder_v3.json")
HGB_V3 = dict(max_iter=1000, max_leaf_nodes=127, learning_rate=0.06)   # [M] chosen on IDSSE nested CV


def load_decoder(path=DECODER_PATH):
    with open(path) as handle:
        d = json.load(handle)
    return d["decoder"], float(d.get("rule_weight", 0.0))


def predict(match, tags=None, use_tags=True, return_debug=False, mdl=None, decoder=None,
            rule_weight=None):
    m = orient(match)
    mdl = mdl or model.load(MODEL_PATH)
    if decoder is None or rule_weight is None:
        dec, rw = load_decoder()
        decoder = decoder or dec
        rule_weight = rw if rule_weight is None else rule_weight
    base = features.compute(m)
    Xf, _ = features.design_matrix(m, base)
    ll = model.emission_ll(mdl, Xf)
    scale = evidence_scale(ll, base)

    g = m.graded
    speed_threshold = float(np.nanquantile(base["player_speed"][g], C.DEAD_SHARE_PRIOR))
    z = rules.dead_score(base, speed_threshold)
    y = rules.team_score(base)

    kos = kickoff.detect(base, m.half, m.fps)
    pre = kickoff.pre_kickoff_spans(kos, m.half_blocks(), m.fps)
    dz, dy = rules.kickoff_evidence(m.n_frames, kos, pre, m.fps)
    if use_tags and tags:
        tz, ty = tagging_prior.evidence(m.n_frames, tags, kos, m.fps)
        dz, dy = dz + tz, dy + ty
    ll = ll + scale * (rule_weight * rules.emissions(z, y) + rules.emissions(dz, dy))

    blocks = [(s, e) for _, s, e in m.half_blocks()]
    codes = cv.decode_ll(ll, blocks, decoder, fill=X)
    if return_debug:
        return codes, {"kickoffs": kos, "pre_kickoff": pre, "evidence_scale": scale,
                       "attack_sign": m.attack_sign, "speed_threshold": speed_threshold,
                       "rule_weight": rule_weight}
    return codes
