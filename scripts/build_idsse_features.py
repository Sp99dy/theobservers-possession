"""Fit the degrader to BAR-ATH tracking statistics and build the corrupted IDSSE feature sets.

    python scripts/build_idsse_features.py [--jobs 7]

Writes cache/degrader_params.json and cache/idsse_feat_*.npz (all gitignored). The degrader
statistics come from the BAR-ATH tracking only (no labels); IDSSE itself is public data.
"""

import argparse
import json
import os
import sys
import time

from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, degrade, features, io_atd, io_idsse, model  # noqa: E402

PARAMS_PATH = os.path.join(config.CACHE_DIR, "degrader_params.json")


def degrader_params(refit=False):
    if os.path.isfile(PARAMS_PATH) and not refit:
        with open(PARAMS_PATH) as handle:
            return json.load(handle)
    clean = [io_idsse.load_labelled(m) for m in io_idsse.MATCH_IDS]
    params = degrade.fit(io_atd.load_match(), clean)
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    degrade.save(params, PARAMS_PATH)
    return params


def _one(mid, params, offset, swapped, version=features.FEATURE_VERSION):
    t = time.time()
    X, y, g, names = model.copy_features(mid, params, offset, swapped, version)
    return mid, offset, swapped, X.shape, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=7)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--feature-version", type=int, default=features.FEATURE_VERSION)
    args = ap.parse_args()
    params = degrader_params(args.refit)
    print("degrader: h_live {h_live:.2e} h_dead {h_dead:.2e} beta {beta} smooth players "
          "{player_smooth} ball {ball_smooth}  fit error {fit_error:.4f}".format(**params))
    for mid in io_idsse.MATCH_IDS:          # warm the clean-match cache serially
        io_idsse.load_match(mid)
    jobs = [(mid, off, sw) for mid in io_idsse.MATCH_IDS
            for off, sw in model.TRAIN_COPIES + (model.EVAL_COPY,)]
    out = Parallel(n_jobs=args.jobs, verbose=0)(
        delayed(_one)(mid, params, off, sw, args.feature_version) for mid, off, sw in jobs)
    for mid, off, sw, shape, sec in out:
        print("{} copy {}{}  X {}  {:.0f}s".format(mid, off, "s" if sw else " ", shape, sec))


if __name__ == "__main__":
    main()
