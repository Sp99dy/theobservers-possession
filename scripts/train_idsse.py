"""Train the final v2 emission model on all 7 IDSSE matches -> models/emission_v2.joblib, and
store the CV-tuned decoder settings -> models/decoder_v2.json.

    python scripts/train_idsse.py [--tag v2]

Deterministic: fixed degrader seeds, fixed HGB random_state, no early stopping. Needs the IDSSE
files (scripts/download_idsse.py) and, for the degrader statistics, the local BAR-ATH tracking
(never its labels).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, features, io_idsse, model, v2  # noqa: E402
from build_idsse_features import degrader_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2", help="which CV run's decoder settings to ship")
    args = ap.parse_args()
    params = degrader_params()
    X, y, names = model.training_rows(io_idsse.MATCH_IDS, params, features.FEATURE_VERSION)
    mdl = model.train(X, y)
    mdl["feature_names"] = names
    model.save(mdl, v2.MODEL_PATH)
    print("trained on {} rows x {} features -> {}".format(len(y), X.shape[1], v2.MODEL_PATH))
    with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.tag))) as handle:
        cvres = json.load(handle)
    with open(v2.DECODER_PATH, "w") as handle:
        json.dump({"decoder": cvres["decoder"], "source": "IDSSE LOMO CV, run '{}'".format(args.tag),
                   "cv_pooled": cvres["pooled"]}, handle, indent=1, sort_keys=True)
    print("decoder settings -> {}".format(v2.DECODER_PATH))


if __name__ == "__main__":
    main()
