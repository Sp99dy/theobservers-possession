"""M3 decoder experiments on the cached out-of-fold v2 emissions (IDSSE LOMO CV).

    python scripts/decoder_experiments.py [--tag v2] [--only NAME ...]

Every variant re-tunes the decoder by coordinate descent on pooled out-of-fold spellF1 and is then
scored NESTED (decoder re-tuned with the held-out match left out), so variants are compared on the
same honest footing as the shipped v2 (nested 0.6427).

Variants:
  hmm        no minimum durations (plain HMM with switch costs): the HMM-vs-HSMM ablation
  split_d    D_after_H / D_after_A states (restart team can depend on who had the ball)
  kickoff    + the BAR-ATH kickoff layer (v1 kickoff detector, rescaled), tested on IDSSE
  ens_v1     + w * v1 rule emissions + the kickoff layer (v1/v2 ensemble, SPEC 5.4)
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, cv, features, io_idsse, kickoff, metrics, model, rules, v1, v2  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH, blocks_of  # noqa: E402

ENS_WEIGHTS = (0.25, 0.5, 1.0)


def extras(mid, params, oof_ll):
    """v1 rule emissions and kickoff evidence on the held-out corruption, cached."""
    path = os.path.join(config.CACHE_DIR, "idsse_extras_{}_v{}.npz".format(
        mid, features.FEATURE_VERSION))
    if os.path.isfile(path):
        with np.load(path) as z:
            return dict(z)
    m = model.make_copy(mid, params, *model.EVAL_COPY)
    _, dbg = v1.predict(m, None, use_tags=False, return_debug=True)
    f = features.compute(m)
    kos = kickoff.detect(f, m.half, m.fps)
    pre = kickoff.pre_kickoff_spans(kos, m.half_blocks(), m.fps)
    dz, dy = rules.kickoff_evidence(m.n_frames, kos, pre, m.fps)
    out = dict(v1_ll=dbg["ll"].astype(np.float32),
               ko_ll=rules.emissions(dz, dy).astype(np.float32),
               scale=np.float64(v2.evidence_scale(oof_ll, f)),
               n_kickoffs=np.int64(len(kos)))
    np.savez(path, **out)
    return out


def nested(items, dp, fixed=()):
    preds = []
    for i in range(len(items)):
        rest = [it for j, it in enumerate(items) if j != i]
        dpi, _ = cv.tune_decoder(rest, start=dp, sweeps=1, fixed=fixed)
        preds.append(cv.decode_ll(items[i][0], items[i][2], dpi, fill=3))
    return cv.pooled_score(items, preds=preds)


def run(name, items, start, log, fixed=()):
    t0 = time.time()
    dp, best = cv.tune_decoder(items, start=start, fixed=fixed)
    ns = nested(items, dp, fixed)
    log("{:<10} tuned {}".format(name, metrics.format_summary(best)))
    log("{:<10} NESTED {}".format(name, metrics.format_summary(ns)))
    log("{:<10} decoder {}  ({:.0f}s)".format(name, json.dumps(dp), time.time() - t0))
    return dp, best, ns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--only", nargs="*", default=["hmm", "split_d", "kickoff", "ens_v1"])
    ap.add_argument("--weights", nargs="*", type=float, default=list(ENS_WEIGHTS))
    args = ap.parse_args()
    log = lambda s: print(s, flush=True)
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    with open(os.path.join(config.CACHE_DIR, "cv_{}_decoder.json".format(args.tag))) as handle:
        base_dp = json.load(handle)["decoder"]
    z = np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(args.tag)))
    items, ext = [], []
    for mid in io_idsse.MATCH_IDS:
        _, y, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, features.FEATURE_VERSION)
        ll = z[mid].astype(np.float64)
        items.append((ll, y, blocks_of(g)))
        ext.append(extras(mid, params, ll))
    log("kickoffs detected per held-out match: {}".format([int(e["n_kickoffs"]) for e in ext]))
    log("evidence scale per match: {}".format([round(float(e["scale"]), 2) for e in ext]))
    log("base      tuned " + metrics.format_summary(cv.pooled_score(items, base_dp)))
    results = {}

    if "hmm" in args.only:
        start = dict(base_dp, min_team=1, min_d=1)
        results["hmm"] = run("hmm", items, start, log, fixed=("min_team", "min_d"))
    if "split_d" in args.only:
        start = {k: v for k, v in base_dp.items() if k != "leave_d"}
        start.update(leave_same=base_dp["leave_d"], leave_other=base_dp["leave_d"])
        results["split_d"] = run("split_d", items, start, log)
    if "kickoff" in args.only:
        kitems = [(ll + e["scale"] * e["ko_ll"], y, b) for (ll, y, b), e in zip(items, ext)]
        results["kickoff"] = run("kickoff", kitems, base_dp, log)
    if "ens_v1" in args.only:
        for w in args.weights:
            # Exactly the BAR-ATH pipeline form: v1's rule evidence (its kickoff term removed;
            # emissions are linear in z, y) at weight w, plus the kickoff layer at full weight.
            eitems = [(ll + e["scale"] * (w * (e["v1_ll"] - e["ko_ll"]) + e["ko_ll"]), y, b)
                      for (ll, y, b), e in zip(items, ext)]
            results["ens_v1_{}".format(w)] = run("ens_v1 {}".format(w), eitems, base_dp, log)

    out = os.path.join(config.CACHE_DIR, "decoder_experiments_{}_{}.json".format(
        args.tag, "_".join(args.only)))
    with open(out, "w") as handle:
        json.dump({k: {"decoder": r[0], "tuned": r[1], "nested": r[2]} for k, r in results.items()},
                  handle, indent=1)


if __name__ == "__main__":
    main()
