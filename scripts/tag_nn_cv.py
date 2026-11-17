"""nn_cv.py with simulated Smart Tagging features (review idea 7): two extra columns per frame,
seconds since the last / to the next simulated set-piece tag (scripts/tag_simulate.py).

    POSSESSION_CONTROL_MIN_S=0.5 python scripts/tag_nn_cv.py --name tag_gru05dm --arch gru \
        --train-cache cache_deadmove --eval-caches cache cache_deadmove --threads 3

Wraps model.copy_features (which nn.load_copy also uses), so training and evaluation copies both get
the tag columns, drawn per copy with the copy's own seed. Always runs with --folds = all seven
matches, so nn_cv writes only out/nn_<name>/ and never an oof file into a cache directory.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import io_idsse, model  # noqa: E402
from scripts import nn_cv, tag_simulate  # noqa: E402

_orig = model.copy_features


def copy_features_tagged(match_id, params, offset, swapped, version, use_cache=True):
    X, y, g, names = _orig(match_id, params, offset, swapped, version, use_cache)
    F = tag_simulate.features_for_copy(match_id, offset, swapped, len(X))
    return (np.concatenate([X, F.astype(X.dtype)], axis=1), y, g,
            list(names) + ["tag_since_s", "tag_to_s"])


model.copy_features = copy_features_tagged

if __name__ == "__main__":
    sys.argv += ["--folds", *io_idsse.MATCH_IDS]
    nn_cv.main()
