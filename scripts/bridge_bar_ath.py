"""BAR-ATH aggregates (no labels) for v4-style models trained at another label bridge.

    POSSESSION_CONTROL_MIN_S=1.0 python scripts/bridge_bar_ath.py --tag b10_f2
    python scripts/bridge_bar_ath.py --shipped          # v4 as shipped (models/)

Trains on all 7 IDSSE matches at the configured bridge (as train_idsse.py --version v4, but saved
to cache/, never models/), decodes BAR-ATH with that CV run's decoder through v4.predict, and prints
possessions, mean length and dead-ball share. Never reads the BAR-ATH ground truth.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, io_atd, io_idsse, io_tagging, model, organisers, v4  # noqa: E402
from possession.canonical import D, X  # noqa: E402
from build_idsse_features import degrader_params  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag")
    ap.add_argument("--shipped", action="store_true")
    args = ap.parse_args()
    if args.shipped:
        mdl, dec, name = None, None, "v4 shipped (bridge 2.0s)"
    else:
        path = os.path.join(config.CACHE_DIR, "emission_{}.joblib".format(args.tag))
        if os.path.isfile(path):
            mdl = model.load(path)
        else:
            X_, y, names = model.training_rows(io_idsse.MATCH_IDS, degrader_params(),
                                               v4.FEATURE_VERSION)
            mdl = model.train(X_, y, **v4.HGB_V4)
            mdl["feature_names"] = names
            model.save(mdl, path)
        with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.tag))) as h:
            dec = json.load(h)["decoder"]
        name = "{} (bridge {:.1f}s)".format(args.tag, config.CONTROL_MIN_S)
    match = io_atd.load_match()
    codes = v4.predict(match, io_tagging.load(), mdl=mdl, decoder=dec)
    pm = organisers.pm()
    st = pm.possession_stats(codes, pm.valid_runs(codes))
    live = codes != X
    minutes = live.sum() / match.fps / 60
    print("{}: possessions {}  per labelled minute {:.2f}  mean {:.1f}s  D share {:.3f}".format(
        name, st["n_spells"], st["n_spells"] / minutes, st["mean_duration_s"],
        np.mean(codes[live] == D)))


if __name__ == "__main__":
    main()
