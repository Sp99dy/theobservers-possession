"""Second-stage (stacked) emission model on IDSSE LOMO CV.

    python scripts/stack_experiments.py [--variant ctx|ctx_raw] [--tag f2_all] [--change chg_f2]

For each held-out match i, stage 2 is trained on the other six matches' held-out corruptions:
inputs are the stage-1 out-of-fold outputs (emission log-probs from cv_idsse --tag TAG and the
change head's log-odds from change_head.py) read over the surrounding seconds
(possession.stack.context_features), optionally plus the raw feature-v2 columns; target H/A/D.
Its predictions for match i replace the stage-1 emissions; the decoder is re-tuned and scored
NESTED, as every other version, and compared match by match with the plain v4 decoder.

Approximation, stated: the stage-1 out-of-fold outputs of match j came from a model that saw
match i. A fully nested stack needs 42 stage-1 fits; the leak is second-order (it reaches
stage 2 only through how stage 1 behaves on other matches).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, cv, io_idsse, metrics, model, stack  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH, blocks_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="ctx", choices=("ctx", "ctx_raw"))
    ap.add_argument("--tag", default="f2_all")
    ap.add_argument("--change", default="chg_f2")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.tag))) as handle:
        base_dp = json.load(handle)["decoder"]
    z = np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(args.tag)))
    zc = np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(args.change)))

    data = {}
    for mid in io_idsse.MATCH_IDS:
        X, y, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, 2)
        C, _ = stack.context_features(z[mid], zc[mid], g)
        F = np.hstack([C, X]) if args.variant == "ctx_raw" else C
        data[mid] = (F, y, g)
    log("stage-2 inputs: {} columns ({:.0f}s)".format(data[io_idsse.MATCH_IDS[0]][0].shape[1],
                                                       time.time() - t0))

    oof2 = {}
    for mid in io_idsse.MATCH_IDS:
        rows = [(F[np.flatnonzero(g)[::model.SUBSAMPLE]], y[np.flatnonzero(g)[::model.SUBSAMPLE]])
                for m2, (F, y, g) in data.items() if m2 != mid]
        Xtr = np.vstack([r[0] for r in rows])
        ytr = np.concatenate([r[1] for r in rows])
        mdl = model.train(Xtr, ytr)
        oof2[mid] = model.emission_ll(mdl, data[mid][0]).astype(np.float32)
        log("fold {}: {} rows ({:.0f}s)".format(mid, len(ytr), time.time() - t0))
    tag2 = "stack_{}_{}".format(args.variant, args.tag)
    np.savez(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(tag2)), **oof2)

    items = [(oof2[mid].astype(np.float64), data[mid][1], blocks_of(data[mid][2]))
             for mid in io_idsse.MATCH_IDS]
    acc1 = np.mean(np.concatenate([np.argmax(z[mid][g], 1) == y[g]
                                   for mid, (_, y, g) in data.items()]))
    acc2 = np.mean(np.concatenate([np.argmax(oof2[mid][g], 1) == y[g]
                                   for mid, (_, y, g) in data.items()]))
    log("frame argmax accuracy: stage 1 {:.4f}, stage 2 {:.4f}".format(acc1, acc2))

    log("v4 decoder  " + metrics.format_summary(cv.pooled_score(items, base_dp)))
    dp, best = cv.tune_decoder(items, start=base_dp, log=log)
    log("tuned       " + metrics.format_summary(best))
    log("decoder     " + json.dumps(dp))
    preds = []
    for i, mid in enumerate(io_idsse.MATCH_IDS):
        rest = [it for j, it in enumerate(items) if j != i]
        dpi, _ = cv.tune_decoder(rest, start=dp, sweeps=1)
        preds.append(cv.decode_ll(items[i][0], items[i][2], dpi, fill=3))
    ns = cv.pooled_score(items, preds=preds)
    log("NESTED      " + metrics.format_summary(ns))
    with np.load(os.path.join(config.CACHE_DIR, "nested_preds_{}.npz".format(args.tag))) as zb:
        base = [zb[mid] for mid in io_idsse.MATCH_IDS]
    b = cv.pooled_score(items, preds=base)["spell_f1_mean"]
    diffs = [cv.pooled_score([it], preds=[p])["spell_f1_mean"] -
             cv.pooled_score([it], preds=[q])["spell_f1_mean"]
             for it, p, q in zip(items, preds, base)]
    log("vs v4 (nested {:.4f}): {:+.4f}; per match {}; wins {}/7".format(
        b, ns["spell_f1_mean"] - b, " ".join("{:+.3f}".format(d) for d in diffs),
        sum(d > 0 for d in diffs)))
    with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(tag2)), "w") as handle:
        json.dump({"decoder": dp, "pooled": best, "nested": ns}, handle, indent=1)
    log("total {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
