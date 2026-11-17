"""Decoder tuned on the MEAN nested spellF1 over the 2 s, 1 s and 0.5 s label conventions.

    python scripts/mean_decoder.py --members nn_gru05 nn_gru05s1 nn_tcn05 --law9 --quantize 0.001 \
        --base v51 --tag v52

BAR-ATH's label convention is unknown, so one decoder must do well under all three. Objective: the
mean over the three bridges of the pooled spellF1 (one decoder for all). Grid: cv.decoder_grid plus
min_team=10. Start: the ensemble's 1 s-tuned decoder (--start). Nested: each held-out match is decoded
with a decoder tuned on the other six; per-match wins are on the per-match mean over conventions,
against --base's nested predictions (cache/nested_preds_<base>_e{2,1,0.5}.npz). Writes
cache/cv_<tag>_mean_decoder.json (tuned on all 7, the shipping decoder) and
cache/nested_preds_<tag>_mean_e{b}.npz (the same nested predictions, one copy per bridge, so it can
serve as --base for the next experiment).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, cv, io_idsse, model, rules  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH, blocks_of  # noqa: E402
from scripts.ensemble_experiments import law9_eval_mask  # noqa: E402

BRIDGES = (2.0, 1.0, 0.5)
EXTRA = {"min_team": (10,)}


def grid(dp):
    g = cv.decoder_grid(dp)
    for k, v in EXTRA.items():
        g[k] = tuple(sorted(set(g[k]) | set(v)))
    return g


def mean_score(sub, preds):
    """sub: [(ll, {bridge: labels}, blocks)] -> (mean, [per-bridge spellF1])"""
    per = [cv.pooled_score([(ll, labs[b], bl) for ll, labs, bl in sub], preds=preds)["spell_f1_mean"]
           for b in BRIDGES]
    return float(np.mean(per)), per


def decode(sub, dp):
    return [cv.decode_ll(ll, bl, dp, fill=3) for ll, _, bl in sub]


def tune(sub, start, sweeps=2):
    dp = dict(start)
    best = mean_score(sub, decode(sub, dp))[0]
    for _ in range(sweeps):
        changed = False
        for name, values in grid(dp).items():
            for v in values:
                if v == dp[name]:
                    continue
                trial = dict(dp, **{name: v})
                s = mean_score(sub, decode(sub, trial))[0]
                if s > best + 1e-9:
                    best, dp, changed = s, trial, True
        if not changed:
            break
    return dp, best


def load_items(members, law9, quantize, swap_tta=None):
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    zs = [np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(m))) for m in members]
    items = []
    for mid in io_idsse.MATCH_IDS:
        _, _, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, 2)
        labs = {b: model.copy_labels(mid, model.EVAL_COPY[1], b) for b in BRIDGES}
        ll = np.mean([z[mid].astype(np.float64) for z in zs], axis=0)
        if swap_tta is not None:
            ll = swap_tta(mid, ll)
        if law9:
            ll = rules.law9_override(ll, law9_eval_mask(mid, params))
        if quantize:
            ll = np.round(ll / quantize) * quantize
        items.append((ll, labs, blocks_of(g)))
    return items


def report(items, preds, base_preds, log, name):
    m, per = mean_score(items, preds)
    pm = np.array([mean_score([it], [p])[0] for it, p in zip(items, preds)])
    line = "{:<10} mean {:.4f}  2s {:.4f}  1s {:.4f}  0.5s {:.4f}".format(name, m, *per)
    if base_preds is not None:
        bm_b = [cv.pooled_score([(ll, labs[b], bl) for ll, labs, bl in items],
                                preds=base_preds[b])["spell_f1_mean"] for b in BRIDGES]
        bmean = float(np.mean(bm_b))
        bpm = np.array([np.mean([cv.pooled_score([(it[0], it[1][b], it[2])],
                                                 preds=[base_preds[b][i]])["spell_f1_mean"]
                                 for b in BRIDGES]) for i, it in enumerate(items)])
        d = pm - bpm
        log("base       mean {:.4f}  2s {:.4f}  1s {:.4f}  0.5s {:.4f}".format(bmean, *bm_b))
        line += "   vs base {:+.4f}  wins {}/7 [{}]".format(
            m - bmean, int((d > 0).sum()), " ".join("{:+.3f}".format(x) for x in d))
    log(line)
    return m, pm


def main(swap_tta=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True)
    ap.add_argument("--law9", action="store_true")
    ap.add_argument("--quantize", type=float, default=None)
    ap.add_argument("--start", default=None, help="decoder json to start from (default: the "
                    "members' 1 s-tuned ensemble decoder, else cv_v51_decoder.json)")
    ap.add_argument("--base", default=None, help="nested_preds_<base>_e{b}.npz to compare against")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    items = load_items(args.members, args.law9, args.quantize, swap_tta)
    start_path = args.start or os.path.join(config.CACHE_DIR, "cv_ens_{}_e1_decoder.json".format(
        "+".join(args.members)))
    if not os.path.isfile(start_path):
        start_path = os.path.join(config.CACHE_DIR, "cv_v51_decoder.json")
    with open(start_path) as h:
        start = json.load(h)["decoder"]
    log("members {} law9 {} quantize {} start {}".format(
        "+".join(args.members), args.law9, args.quantize, os.path.basename(start_path)))
    dp, best = tune(items, start)
    log("tuned all 7: mean {:.4f} {}  {:.0f}s".format(best, json.dumps(dp), time.time() - t0))
    with open(os.path.join(config.CACHE_DIR, "cv_{}_mean_decoder.json".format(args.tag)), "w") as h:
        json.dump({"decoder": dp, "objective": "mean spellF1 over 2/1/0.5 s",
                   "members": args.members, "law9": args.law9, "quantize": args.quantize}, h,
                  indent=1)
    preds = []
    for i in range(len(items)):
        rest = [it for j, it in enumerate(items) if j != i]
        dpi, _ = tune(rest, dp, sweeps=1)
        preds.append(cv.decode_ll(items[i][0], items[i][2], dpi, fill=3))
        log("  fold {} {}  {:.0f}s".format(i, json.dumps(dpi), time.time() - t0))
    for b in BRIDGES:
        np.savez(os.path.join(config.CACHE_DIR, "nested_preds_{}_mean_e{:g}.npz".format(args.tag, b)),
                 **dict(zip(io_idsse.MATCH_IDS, preds)))
    base = None
    if args.base:
        base = {}
        for b in BRIDGES:
            with np.load(os.path.join(config.CACHE_DIR, "nested_preds_{}_e{:g}.npz".format(
                    args.base, b))) as z:
                base[b] = [z[mid] for mid in io_idsse.MATCH_IDS]
    report(items, preds, base, log, "NESTED")
    log("total {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
