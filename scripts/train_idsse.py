"""Train the final v2 emission model on all 7 IDSSE matches -> models/emission_v2.joblib, and
store the CV-tuned decoder settings -> models/decoder_v2.json.

    python scripts/train_idsse.py [--tag v2]
    python scripts/train_idsse.py --version v3 --experiment cache/decoder_experiments_<tag>_ens_v1.json --key ens_v1_<w>
    python scripts/train_idsse.py --version v4 --tag f2_all      # feature version 2

Deterministic: fixed degrader seeds, fixed HGB random_state, no early stopping. Needs the IDSSE
files (scripts/download_idsse.py) and, for the degrader statistics, the local BAR-ATH tracking
(never its labels).
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, features, io_idsse, model, v2, v3, v4  # noqa: E402
from build_idsse_features import degrader_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2", help="which CV run's decoder settings to ship")
    ap.add_argument("--version", default="v2", choices=["v2", "v3", "v4"])
    ap.add_argument("--experiment", help="v3: decoder_experiments JSON holding the chosen variant")
    ap.add_argument("--key", help="v3: variant name in that JSON, e.g. ens_v1_0.5")
    args = ap.parse_args()
    mod = {"v2": v2, "v3": v3, "v4": v4}[args.version]
    hgb = {"v2": {}, "v3": v3.HGB_V3, "v4": v4.HGB_V4}[args.version]
    fver = v4.FEATURE_VERSION if args.version == "v4" else features.FEATURE_VERSION
    params = degrader_params()
    X, y, names = model.training_rows(io_idsse.MATCH_IDS, params, fver)
    mdl = model.train(X, y, **hgb)
    mdl["feature_names"] = names
    model.save(mdl, mod.MODEL_PATH)
    print("trained on {} rows x {} features -> {}".format(len(y), X.shape[1], mod.MODEL_PATH))
    if args.version in ("v2", "v4"):
        with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.tag))) as h:
            cvres = json.load(h)
        out = {"decoder": cvres["decoder"], "source": "IDSSE LOMO CV, run '{}'".format(args.tag),
               "cv_pooled": cvres["pooled"], "hgb": hgb, "feature_version": fver}
    else:
        with open(args.experiment) as h:
            res = json.load(h)[args.key]
        out = {"decoder": res["decoder"], "rule_weight": float(args.key.rsplit("_", 1)[1]),
               "hgb": hgb, "source": "IDSSE LOMO CV, {} {}".format(
                   os.path.basename(args.experiment), args.key),
               "cv_pooled": res["tuned"], "cv_nested": res["nested"]}
    with open(mod.DECODER_PATH, "w") as handle:
        json.dump(out, handle, indent=1, sort_keys=True)
    print("decoder settings -> {}".format(mod.DECODER_PATH))


if __name__ == "__main__":
    main()
