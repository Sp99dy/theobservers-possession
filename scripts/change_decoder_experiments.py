"""Touch-aware decoder on IDSSE LOMO CV: emissions + the possession-change head.

    python scripts/change_decoder_experiments.py [--tag f2_all] [--change chg_f2] [--variant full]

Variants: full (switch-cost bonus + fast starts), bonus (fast starts off), fast (bonus off).
The decoder, including the change settings, is tuned by coordinate descent on pooled out-of-fold
spellF1, then scored NESTED (re-tuned without the held-out match), as every other version.
Compared match by match with the plain decoder's nested predictions (postfilter_experiments.py).
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

FIXED = {"full": (), "bonus": ("fast_thr", "fast_steps"), "fast": ("chg_w", "chg_off")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="f2_all")
    ap.add_argument("--change", default="chg_f2")
    ap.add_argument("--variant", default="full", choices=sorted(FIXED))
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.tag))) as handle:
        base_dp = json.load(handle)["decoder"]
    z = np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(args.tag)))
    zc = np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(args.change)))
    items = []
    for mid in io_idsse.MATCH_IDS:
        _, y, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, 2)   # labels only; v2 cache
        items.append((z[mid].astype(np.float64), y, blocks_of(g), zc[mid].astype(np.float64)))

    start = dict(base_dp, chg_w=0.0, chg_off=0.0, fast_thr=99.0, fast_steps=4)
    log("start     " + metrics.format_summary(cv.pooled_score(items, start)))
    dp, best = cv.tune_decoder(items, start=start, log=log, fixed=FIXED[args.variant])
    log("tuned     " + metrics.format_summary(best))
    log("decoder   " + json.dumps(dp))

    preds = []
    for i, mid in enumerate(io_idsse.MATCH_IDS):
        rest = [it for j, it in enumerate(items) if j != i]
        dpi, _ = cv.tune_decoder(rest, start=dp, sweeps=1, fixed=FIXED[args.variant])
        preds.append(cv.decode_ll(items[i][0], items[i][2], dpi, fill=3, change=items[i][3]))
        log("  nested {} ({:.0f}s): {}".format(mid, time.time() - t0, json.dumps(dpi)))
    ns = cv.pooled_score(items, preds=preds)
    log("NESTED    " + metrics.format_summary(ns))

    base_path = os.path.join(config.CACHE_DIR, "nested_preds_{}.npz".format(args.tag))
    if os.path.isfile(base_path):
        with np.load(base_path) as zb:
            base = [zb[mid] for mid in io_idsse.MATCH_IDS]
        b = cv.pooled_score(items, preds=base)["spell_f1_mean"]
        diffs = [cv.pooled_score([it], preds=[p])["spell_f1_mean"] -
                 cv.pooled_score([it], preds=[q])["spell_f1_mean"]
                 for it, p, q in zip(items, preds, base)]
        log("vs plain decoder (nested {:.4f}): {:+.4f}; per match {}; wins {}/7".format(
            b, ns["spell_f1_mean"] - b, " ".join("{:+.3f}".format(d) for d in diffs),
            sum(d > 0 for d in diffs)))
    out = os.path.join(config.CACHE_DIR, "change_decoder_{}_{}.json".format(args.tag, args.variant))
    with open(out, "w") as handle:
        json.dump({"decoder": dp, "pooled": best, "nested": ns}, handle, indent=1)
    np.savez(os.path.join(config.CACHE_DIR, "nested_preds_{}_{}.npz".format(args.tag, args.variant)),
             **dict(zip(io_idsse.MATCH_IDS, preds)))
    log("total {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
