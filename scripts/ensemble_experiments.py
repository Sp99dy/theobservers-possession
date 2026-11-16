"""Emission ensembles on IDSSE LOMO CV: the mean of several models' out-of-fold log-probs.

    python scripts/ensemble_experiments.py --members f2_all f2_cap1000 [stack_ctx_f2_all]

Each member is cache/oof_{name}.npz (log P(H), log P(A), log P(D) per frame of the held-out
corruption). The ensemble emission is the equal-weight mean of the members' log-probs (a geometric
mean of probabilities; no weights are fitted). The decoder is tuned on pooled out-of-fold spellF1
and scored NESTED, then compared match by match with v4's nested predictions.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, cv, io_idsse, metrics, model  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH, blocks_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--base", default="f2_all")
    ap.add_argument("--eval-bridge", type=float, default=None,
                    help="score against labels at this bridge (default: POSSESSION_CONTROL_MIN_S)")
    ap.add_argument("--save-nested", action="store_true",
                    help="also write cache/nested_preds_<ensemble name>.npz (usable as --base)")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.base))) as handle:
        base_dp = json.load(handle)["decoder"]
    zs = [np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(m))) for m in args.members]
    items = []
    for mid in io_idsse.MATCH_IDS:
        _, y, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, 2)   # labels only; v2 cache
        if args.eval_bridge is not None:
            y = model.copy_labels(mid, model.EVAL_COPY[1], args.eval_bridge)
        ll = np.mean([z[mid].astype(np.float64) for z in zs], axis=0)
        items.append((ll, y, blocks_of(g)))
    log("members " + " + ".join(args.members))
    dp, best = cv.tune_decoder(items, start=base_dp)
    log("tuned       " + metrics.format_summary(best))
    log("decoder     " + json.dumps(dp))
    preds = []
    for i in range(len(items)):
        rest = [it for j, it in enumerate(items) if j != i]
        dpi, _ = cv.tune_decoder(rest, start=dp, sweeps=1)
        preds.append(cv.decode_ll(items[i][0], items[i][2], dpi, fill=3))
    ns = cv.pooled_score(items, preds=preds)
    log("NESTED      " + metrics.format_summary(ns))
    name = "ens_" + "+".join(args.members)
    if args.eval_bridge is not None:
        name += "_e{:g}".format(args.eval_bridge)
    if args.save_nested:
        np.savez(os.path.join(config.CACHE_DIR, "nested_preds_{}.npz".format(name)),
                 **dict(zip(io_idsse.MATCH_IDS, preds)))
    per = [cv.pooled_score([it], preds=[p])["spell_f1_mean"] for it, p in zip(items, preds)]
    log("per match   " + " ".join("{:.4f}".format(v) for v in per))
    base_path = os.path.join(config.CACHE_DIR, "nested_preds_{}.npz".format(args.base))
    if args.eval_bridge is not None:
        tagged = os.path.join(config.CACHE_DIR, "nested_preds_{}_e{:g}.npz".format(
            args.base, args.eval_bridge))
        if os.path.isfile(tagged) or args.eval_bridge != config.CONTROL_MIN_S:
            base_path = tagged
    if not os.path.isfile(base_path):
        log("no {}: per-match comparison skipped".format(os.path.basename(base_path)))
        return
    with np.load(base_path) as zb:
        base = [zb[mid] for mid in io_idsse.MATCH_IDS]
    b = cv.pooled_score(items, preds=base)["spell_f1_mean"]
    diffs = [cv.pooled_score([it], preds=[p])["spell_f1_mean"] -
             cv.pooled_score([it], preds=[q])["spell_f1_mean"]
             for it, p, q in zip(items, preds, base)]
    log("vs {} (nested {:.4f}): {:+.4f}; per match {}; wins {}/7".format(
        args.base, b, ns["spell_f1_mean"] - b, " ".join("{:+.3f}".format(d) for d in diffs),
        sum(d > 0 for d in diffs)))
    with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(name)), "w") as handle:
        json.dump({"members": args.members, "decoder": dp, "pooled": best, "nested": ns,
                   "per_match_vs_v4": diffs}, handle, indent=1)
    log("total {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
