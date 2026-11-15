"""Possession-change head: P(the ball changes team near this frame), IDSSE LOMO out-of-fold.

    python scripts/change_head.py [--version 2] [--radius 10] [--tag chg_f2]

A binary gradient-boosting classifier on the same features as the emission model (feature
version 2 includes the touch features). Target: a direct H <-> A switch of the label within
+-radius frames. Out-of-fold log-odds for the held-out corruption of each match are cached to
cache/oof_{tag}.npz, for the touch-aware decoder (scripts/change_decoder_experiments.py).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, io_idsse, model  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH  # noqa: E402

CHANGE_PARAMS = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
                     l2_regularization=1.0, early_stopping=False, random_state=0)   # [D] as HGB_PARAMS, unweighted


def change_target(y, radius):
    """1 where a direct H<->A label switch lies within +-radius frames."""
    y = np.asarray(y)
    sw = np.zeros(len(y), dtype=bool)
    sw[1:] = (y[1:] != y[:-1]) & np.isin(y[1:], (0, 1)) & np.isin(y[:-1], (0, 1))
    k = np.ones(2 * radius + 1)
    return (np.convolve(sw.astype(float), k, mode="same") > 0).astype(np.int8)


def rows(match_ids, params, version, radius):
    Xs, ts = [], []
    for mid in match_ids:
        for offset, swapped in model.TRAIN_COPIES:
            X, y, g, _ = model.copy_features(mid, params, offset, swapped, version)
            idx = np.flatnonzero(g)[::model.SUBSAMPLE]
            Xs.append(X[idx])
            ts.append(change_target(y, radius)[idx])
    return np.vstack(Xs), np.concatenate(ts)


def train(X, t):
    return HistGradientBoostingClassifier(**CHANGE_PARAMS).fit(X, t)


def log_odds(clf, X):
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, default=2)
    ap.add_argument("--radius", type=int, default=10)
    ap.add_argument("--tag", default="chg_f2")
    args = ap.parse_args()
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    t0 = time.time()
    oof, aucs, rates = {}, [], []
    for mid in io_idsse.MATCH_IDS:
        others = [m for m in io_idsse.MATCH_IDS if m != mid]
        X, t = rows(others, params, args.version, args.radius)
        clf = train(X, t)
        Xe, ye, ge, _ = model.copy_features(mid, params, *model.EVAL_COPY, args.version)
        lo = log_odds(clf, Xe).astype(np.float32)
        te = change_target(ye, args.radius)
        aucs.append(roc_auc_score(te[ge], lo[ge]))
        rates.append(t.mean())
        oof[mid] = lo
        print("fold {}: {} rows, positive rate {:.3f}, held-out AUC {:.3f}  ({:.0f}s)".format(
            mid, len(t), t.mean(), aucs[-1], time.time() - t0), flush=True)
    np.savez(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(args.tag)), **oof)
    print("mean held-out AUC {:.3f}; base rate {:.3f}".format(np.mean(aucs), np.mean(rates)))


if __name__ == "__main__":
    main()
