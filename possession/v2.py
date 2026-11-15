"""v2: learned emissions (HGB trained on degraded IDSSE) + kickoff and Smart Tagging layers +
the segmental decoder with costs tuned by IDSSE leave-one-match-out CV.

Nothing about BAR-ATH's labels enters: the classifier and the decoder settings come from IDSSE,
and the extra BAR-ATH layers (kickoffs, tags) are the v1 terms, re-expressed in the classifier's
evidence units by a label-free measurement (EVIDENCE_SCALE below).
"""

import json
import os

import numpy as np

from . import config as C
from . import cv, features, kickoff, model, rules, tagging_prior
from .canonical import X, orient

MODEL_PATH = os.path.join(C.REPO_DIR, "models", "emission_v2.joblib")
DECODER_PATH = os.path.join(C.REPO_DIR, "models", "decoder_v2.json")


def evidence_scale(ll, f):
    """Nats of classifier evidence per nat of v1 evidence, measured on this match without labels.

    v1's tag and kickoff terms were sized against a clear nearest-player call (median 1.2 nats/frame
    of H-vs-A evidence on close-control frames). We measure the classifier's median |H - A| log-
    likelihood gap on the same kind of frames and rescale the v1 terms by the ratio, so a tag keeps
    the same weight relative to a clear ball observation as in v1.
    """
    near = np.minimum(f["d_home"], f["d_away"])
    close = (f["ball_in"] > 0) & (near < 1.5)
    if close.sum() < 100:
        return 1.0
    return float(np.median(np.abs(ll[close, 0] - ll[close, 1])) / C.V1_CLEAR_CALL_NATS)


def predict(match, tags=None, use_tags=True, return_debug=False, mdl=None, decoder=None,
            feature_version=features.FEATURE_VERSION):
    m = orient(match)
    mdl = mdl or model.load(MODEL_PATH)
    if decoder is None:
        with open(DECODER_PATH) as handle:
            decoder = json.load(handle)["decoder"]
    base = features.compute(m)
    Xf, _ = features.design_matrix(m, base, version=feature_version)
    ll = model.emission_ll(mdl, Xf)
    scale = evidence_scale(ll, base)

    kos = kickoff.detect(base, m.half, m.fps)
    pre = kickoff.pre_kickoff_spans(kos, m.half_blocks(), m.fps)
    dz, dy = rules.kickoff_evidence(m.n_frames, kos, pre, m.fps)
    if use_tags and tags:
        tz, ty = tagging_prior.evidence(m.n_frames, tags, kos, m.fps)
        dz, dy = dz + tz, dy + ty
    ll = ll + scale * rules.emissions(dz, dy)

    blocks = [(s, e) for _, s, e in m.half_blocks()]
    codes = cv.decode_ll(ll, blocks, decoder, fill=X)
    if return_debug:
        return codes, {"kickoffs": kos, "pre_kickoff": pre, "evidence_scale": scale,
                       "attack_sign": m.attack_sign, "speed_threshold": float("nan")}
    return codes
