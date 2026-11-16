"""Fully nested stacking (HANDOFF 4.3): stage 2 over +-8 s of stage-1 outputs, with no leak.

    POSSESSION_CONTROL_MIN_S=0.5 python scripts/stack_nested.py --tag b05_f2 [--jobs 2]

For each held-out match i and each other match j, a stage-1 HGB (as cv_idsse.py) is trained on the
five matches other than i and j and predicts j's held-out corruption: 42 fits, cached one file each
(resumable). Stage 2 for match i is trained on those inner out-of-fold outputs of the six matches
j != i (context features as possession.stack, without the change head), then applied to match i's
ordinary out-of-fold stage-1 output (cache/oof_<tag>.npz, from a model that never saw i).

Writes cache/oof_stackN_<tag>.npz and cache/cv_stackN_<tag>_decoder.json (the stage-1 decoder as
the tuning start), so scripts/bridge_cross_eval.py can score it nested against any label bridge.
Deviation from stack_experiments.py: no change-head channel (it would need 42 more fits); its
columns are constant zero.
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, io_idsse, model, stack  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH  # noqa: E402

IDS = io_idsse.MATCH_IDS


def _inner_path(tag, i, j):
    return os.path.join(config.CACHE_DIR, "stackN_inner_{}_{}_{}.npy".format(tag, i, j))


def inner_fit(tag, i, j, params, train_cache):
    path = _inner_path(tag, i, j)
    if os.path.isfile(path):
        return i, j, 0.0
    t0 = time.time()
    others = [m for m in IDS if m not in (i, j)]
    eval_cache = config.CACHE_DIR
    if train_cache:
        config.CACHE_DIR = train_cache
    Xtr, ytr, _ = model.training_rows(others, params, 2)
    config.CACHE_DIR = eval_cache
    mdl = model.train(Xtr, ytr)
    X, _, _, _ = model.copy_features(j, params, *model.EVAL_COPY, 2)
    ll = model.emission_ll(mdl, X).astype(np.float32)
    np.save(path + ".tmp.npy", ll)
    os.replace(path + ".tmp.npy", path)
    return i, j, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="stage-1 run: cache/oof_<tag>.npz")
    ap.add_argument("--train-cache", help="cache dir of the training copies (as cv_idsse.py)")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    with open(os.path.join(args.train_cache or config.CACHE_DIR, "degrader_params.json")) as h:
        tparams = json.load(h)
    with open(PARAMS_PATH) as h:
        params = json.load(h)

    pairs = [(i, j) for i in IDS for j in IDS if i != j]
    log("bridge {:.1f}s, {} inner stage-1 fits ({} cached)".format(
        config.CONTROL_MIN_S, len(pairs), sum(os.path.isfile(_inner_path(args.tag, i, j))
                                              for i, j in pairs)))
    for i, j, sec in Parallel(n_jobs=args.jobs, verbose=0)(
            delayed(inner_fit)(args.tag, i, j, tparams, args.train_cache) for i, j in pairs):
        if sec:
            log("  inner {} -> {}: {:.0f}s".format(i, j, sec))
    log("inner fits done ({:.0f}s)".format(time.time() - t0))

    z = np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(args.tag)))
    ev = {m: model.copy_features(m, params, *model.EVAL_COPY, 2) for m in IDS}
    oof2 = {}
    for i in IDS:
        Xs, ys = [], []
        for j in IDS:
            if j == i:
                continue
            _, y, g, _ = ev[j]
            ll = np.load(_inner_path(args.tag, i, j))
            C, _ = stack.context_features(ll, np.zeros(len(ll)), g)
            idx = np.flatnonzero(g)[::model.SUBSAMPLE]
            Xs.append(C[idx])
            ys.append(y[idx])
        mdl = model.train(np.vstack(Xs), np.concatenate(ys))
        C, _ = stack.context_features(z[i], np.zeros(len(z[i])), ev[i][2])
        oof2[i] = model.emission_ll(mdl, C).astype(np.float32)
        acc1 = np.mean(np.argmax(z[i][ev[i][2]], 1) == ev[i][1][ev[i][2]])
        acc2 = np.mean(np.argmax(oof2[i][ev[i][2]], 1) == ev[i][1][ev[i][2]])
        log("stage 2 for {}: frame acc stage 1 {:.4f} -> stage 2 {:.4f}".format(i, acc1, acc2))
    tag2 = "stackN_" + args.tag
    np.savez(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(tag2)), **oof2)
    shutil.copy(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.tag)),
                os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(tag2)))
    log("wrote oof_{} ({:.0f}s)".format(tag2, time.time() - t0))


if __name__ == "__main__":
    main()
