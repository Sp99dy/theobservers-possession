"""v4 (M3b refinement, the default): v2's pipeline on feature version 2 (touch detection + near-ball/collective slowdown).

* the emission classifier sees the v2 features plus `features.touch_features` (ball contacts
  inferred from the ball trajectory: who touched last/next, how long ago, touch balance) and
  `features.slowdown_features` (mean speed of all players and of the 4 players nearest the ball
  over the previous/next seconds: stoppages look like a collective slowdown);
* the kickoff and Smart Tagging layers and the segmental decoder exactly as in v2, with the decoder
  costs re-tuned on IDSSE leave-one-match-out CV for this classifier.

Every choice was made on IDSSE nested CV (scripts/cv_idsse.py --feature-version 2); none on BAR-ATH
labels. No v1-rule blend: on top of a larger classifier it was within noise (experiments.md).
"""

import json
import os

from . import config as C
from . import model, v2

FEATURE_VERSION = 2
MODEL_PATH = os.path.join(C.REPO_DIR, "models", "emission_v4.joblib")
DECODER_PATH = os.path.join(C.REPO_DIR, "models", "decoder_v4.json")
HGB_V4 = {}          # default HGB_PARAMS: 1000 trees / 127 leaves gave +0.006 nested, within noise


def load_decoder(path=DECODER_PATH):
    with open(path) as handle:
        return json.load(handle)["decoder"]


def predict(match, tags=None, use_tags=True, return_debug=False, mdl=None, decoder=None):
    mdl = mdl or model.load(MODEL_PATH)
    decoder = decoder or load_decoder()
    return v2.predict(match, tags, use_tags=use_tags, return_debug=return_debug, mdl=mdl,
                      decoder=decoder, feature_version=FEATURE_VERSION)
