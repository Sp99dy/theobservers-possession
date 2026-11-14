"""Leave-one-match-out CV on degraded IDSSE (7 matches): v1 rules vs v2 learned emissions.

    python scripts/cv_idsse.py [--tag NAME] [--features ball|ball+shape|all] [--nested]

1. For each match: train HGB on the other six (4 corrupted/mirrored copies each), predict that
   match's held-out corruption -> out-of-fold per-frame log-likelihoods.
2. Score v1 (rule-based, no tags: IDSSE has none) on the same held-out corruptions.
3. Tune the decoder by coordinate descent on pooled out-of-fold spellF1. With --nested, also
   re-tune it leaving each match out, which gives the honest (unbiased) CV estimate.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, cv, features, io_idsse, metrics, model, v1  # noqa: E402

PARAMS_PATH = os.path.join(config.CACHE_DIR, "degrader_params.json")
FEATURE_SETS = {
    "ball": ("ball_", "near_dist", "margin", "d_home", "d_away", "last_", "next_", "since_",
             "until_"),
    "ball+shape": ("ball_", "near_dist", "margin", "d_home", "d_away", "last_", "next_",
                   "since_", "until_", "home_c", "away_c", "home_s", "away_s", "home_b", "away_b",
                   "home_f", "away_f", "centroid", "n_", "home_own", "away_own", "d_centre"),
}


def blocks_of(g):
    e = np.diff(np.concatenate(([0], g.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(e == 1).tolist(), np.flatnonzero(e == -1).tolist()))


def column_subset(names, which):
    if which == "all":
        return None
    prefixes = FEATURE_SETS[which]
    # "speed"/"vx" columns are motion; excluded from the ball and ball+shape sets.
    cols = [j for j, n in enumerate(names) if n.startswith(prefixes)
            and not any(k in n for k in ("speed", "_vx")) or n in ("ball_speed",)]
    return sorted(set(cols))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--features", default="all", choices=["all", "ball", "ball+shape"])
    ap.add_argument("--nested", action="store_true")
    ap.add_argument("--skip-v1", action="store_true")
    args = ap.parse_args()
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    ver = features.FEATURE_VERSION
    t0 = time.time()

    oof_path = os.path.join(config.CACHE_DIR, "oof_{}.npz".format(args.tag))
    evals = {}
    for mid in io_idsse.MATCH_IDS:
        X, y, g, names = model.copy_features(mid, params, *model.EVAL_COPY, ver)
        evals[mid] = (X, y, g)
    cols = column_subset(names, args.features)
    if os.path.isfile(oof_path):
        z = np.load(oof_path)
        oof = {mid: z[mid] for mid in io_idsse.MATCH_IDS}
    else:
        oof = {}
        for mid in io_idsse.MATCH_IDS:
            others = [m for m in io_idsse.MATCH_IDS if m != mid]
            Xtr, ytr, _ = model.training_rows(others, params, ver)
            mdl = model.train(Xtr, ytr, cols)
            oof[mid] = model.emission_ll(mdl, evals[mid][0]).astype(np.float32)
            print("fold {}: trained on {} rows, {:.0f}s".format(mid, len(ytr), time.time() - t0),
                  flush=True)
        np.savez(oof_path, **oof)

    items = [(oof[mid].astype(np.float64), evals[mid][1], blocks_of(evals[mid][2]))
             for mid in io_idsse.MATCH_IDS]

    # Frame-level classifier quality (argmax, no decoder) for reference.
    acc = np.mean(np.concatenate([np.argmax(ll[g], 1) == y[g]
                                  for (ll, y, _), (_, _, g) in zip(items, evals.values())]))
    print("\nOOF frame argmax accuracy {:.3f}".format(acc))

    if not args.skip_v1:
        preds = []
        for mid in io_idsse.MATCH_IDS:
            m = model.make_copy(mid, params, *model.EVAL_COPY)
            preds.append(v1.predict(m, None, use_tags=False))
        s = cv.pooled_score(items, preds=preds)
        print("v1 rules (no tags)        " + metrics.format_summary(s), flush=True)

    s = cv.pooled_score(items, cv.DEFAULT_DECODER)
    print("v2, v1 decoder settings   " + metrics.format_summary(s), flush=True)
    dp, best = cv.tune_decoder(items, log=print)
    print("v2, tuned decoder (pooled, in-sample for decoder)  " + metrics.format_summary(best))
    print("decoder: " + json.dumps(dp))

    per = []
    for it, mid in zip(items, io_idsse.MATCH_IDS):
        s = cv.pooled_score([it], dp)
        per.append(s["spell_f1_mean"])
        print("   {}  {}".format(mid, metrics.format_summary(s)))
    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(per, len(per))) for _ in range(2000)]
    print("per-match spellF1 mean {:.4f}  90% bootstrap CI [{:.4f}, {:.4f}]".format(
        np.mean(per), *np.percentile(boots, [5, 95])))

    if args.nested:
        preds = []
        for i, mid in enumerate(io_idsse.MATCH_IDS):
            rest = [it for j, it in enumerate(items) if j != i]
            dpi, _ = cv.tune_decoder(rest, start=dp, sweeps=1)
            preds.append(cv.decode_ll(items[i][0], items[i][2], dpi, fill=3))
        s = cv.pooled_score(items, preds=preds)
        print("v2 NESTED (decoder re-tuned without the held-out match)  " + metrics.format_summary(s))

    out = os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.tag))
    with open(out, "w") as handle:
        json.dump({"decoder": dp, "pooled": best, "per_match": per}, handle, indent=1)
    print("total {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
