"""Post-filter experiments: short possessions sandwiched between stoppages (IDSSE LOMO CV).

    python scripts/postfilter_experiments.py [--tag f2_all]

Prompted by a free kick in the BAR-ATH public window: after a foul the goalkeeper hands the ball
back to the other team, and the decoder reports a short possession for the goalkeeper's team inside
what is one stoppage. Candidate rule: a team segment shorter than L seconds that starts right after
a dead ball is relabelled dead when

  dd        it also ends in a dead ball                         (D, T, D)
  dd_other  it ends in a dead ball and the next possession is the other team's
            (the short team is not the one taking the restart)  (D, T, D, ..., not-T)
  other     the next possession, directly or after a dead ball, is the other team's
                                                                (D, T, [D], not-T)

Decoded with the nested decoder (re-tuned without the held-out match) on the cached out-of-fold
emissions of `cv_idsse.py --tag TAG`. The filter (variant, L) is itself chosen nested: on the other
six matches, then applied to the held-out one.
"""

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, cv, io_idsse, metrics, model  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH, blocks_of  # noqa: E402

H, A, D = 0, 1, 2
VARIANTS = ("dd", "dd_other", "other")
LENGTHS_S = (3, 4, 5, 6, 8, 10)


def runs_of(c):
    b = np.flatnonzero(c[1:] != c[:-1]) + 1
    starts = np.r_[0, b]
    ends = np.r_[b, len(c)]
    return [(int(s), int(e), int(c[s])) for s, e in zip(starts, ends)]


def postfilter(codes, blocks, variant, max_len):
    """Relabel short team segments between stoppages as dead (see module doc). max_len in frames."""
    out = codes.copy()
    for s0, e0 in blocks:
        r = runs_of(codes[s0:e0])
        for i in range(1, len(r) - 1):
            s, e, c = r[i]
            if c not in (H, A) or e - s >= max_len or r[i - 1][2] != D:
                continue
            nxt = r[i + 1][2]
            later = next((rc for _, _, rc in r[i + 1:] if rc in (H, A)), None)
            if variant == "dd":
                hit = nxt == D
            elif variant == "dd_other":
                hit = nxt == D and later is not None and later != c
            else:
                hit = later is not None and later != c and nxt in (D, 1 - c)
            if hit:
                out[s0 + s:s0 + e] = D
        # (relabelling uses the original runs, so one pass; neighbouring hits merge naturally)
    return out


def score(items, preds):
    return cv.pooled_score(items, preds=preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="f2_all")
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.tag))) as handle:
        base_dp = json.load(handle)["decoder"]
    z = np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(args.tag)))
    items = []
    for mid in io_idsse.MATCH_IDS:
        _, y, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, 2)   # labels only; v2 cache
        items.append((z[mid].astype(np.float64), y, blocks_of(g)))

    cache = os.path.join(config.CACHE_DIR, "nested_preds_{}.npz".format(args.tag))
    if os.path.isfile(cache):
        with np.load(cache) as zz:
            preds = [zz[mid] for mid in io_idsse.MATCH_IDS]
    else:
        preds = []
        for i in range(len(items)):
            rest = [it for j, it in enumerate(items) if j != i]
            dpi, _ = cv.tune_decoder(rest, start=base_dp, sweeps=1)
            preds.append(cv.decode_ll(items[i][0], items[i][2], dpi, fill=3))
            log("nested decoder for {} ({:.0f}s)".format(io_idsse.MATCH_IDS[i], time.time() - t0))
        np.savez(cache, **dict(zip(io_idsse.MATCH_IDS, preds)))

    fps = 25
    base = score(items, preds)
    log("base (nested)          " + metrics.format_summary(base))

    # how much of the dead ball truth is in the shape the filter targets
    results = {}
    for variant, L in itertools.product(VARIANTS, LENGTHS_S):
        fp = [postfilter(p, it[2], variant, L * fps) for p, it in zip(preds, items)]
        changed = sum(int((a != b).sum()) for a, b in zip(fp, preds))
        right = sum(int(((a != b) & (it[1] == D)).sum()) for a, b, it in zip(fp, preds, items))
        res = score(items, fp)
        per = [score([it], [p])["spell_f1_mean"] - score([it], [q])["spell_f1_mean"]
               for it, p, q in zip(items, fp, preds)]
        results[(variant, L)] = (fp, res)
        log("{:<9} L={:>2}s  spellF1 {:.4f} ({:+.4f})  macroF1 {:.4f}  poss {}  frames changed {} "
            "({:.0%} truly dead)  match wins {}/{}".format(
                variant, L, res["spell_f1_mean"], res["spell_f1_mean"] - base["spell_f1_mean"],
                res["macro_f1"], res["n_pred_spells"], changed, right / max(changed, 1),
                sum(d > 0 for d in per), sum(d != 0 for d in per)))

    # nested choice of (variant, L): pick on six matches, apply to the seventh
    nested = []
    picks = []
    for i in range(len(items)):
        rest = [j for j in range(len(items)) if j != i]
        best = max([(None, 0)] + list(results),
                   key=lambda k: score([items[j] for j in rest],
                                       [preds[j] if k[0] is None else results[k][0][j]
                                        for j in rest])["spell_f1_mean"])
        picks.append(best)
        nested.append(preds[i] if best[0] is None else results[best][0][i])
    ns = score(items, nested)
    log("filter chosen nested: picks {}".format(picks))
    log("NESTED with filter     " + metrics.format_summary(ns) +
        "  ({:+.4f})".format(ns["spell_f1_mean"] - base["spell_f1_mean"]))
    log("total {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
