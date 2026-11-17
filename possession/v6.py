"""v6 candidate: v5 with members trained on the dead-time ball degrader (degrade step 8).

Same code path as v5 (possession/v5.py); only the weights differ: the NN members are trained by
scripts/train_nn.py on copies from a cache whose degrader params carry "deadmove" (synthetic set-up
passes and kickabouts inside true IDSSE dead stretches, fitted to BAR-ATH tracking aggregates by
scripts/fit_deadmove_degrader.py). Models live in POSSESSION_V6_DIR or out/models_v6, never in
v5's directory.
"""

import os

from . import config as C
from . import v5

MODEL_DIR = os.environ.get("POSSESSION_V6_DIR") or os.path.join(C.OUT_DIR, "models_v6")


def predict(match, tags=None, use_tags=True, return_debug=False):
    return v5.predict(match, tags, use_tags=use_tags, return_debug=return_debug,
                      model_dir=MODEL_DIR)
