"""Sanity check of feature version 2 (touches, slowdown) on one degraded IDSSE copy per match.

    python scripts/diag_features_v2.py [--matches 2]

Label-aware checks on IDSSE only (public data): touch rate, how often the last detected touch's
team equals the true possession team (vs. the nearest-player rule), on all live frames and on the
possessions v2 misses; and how well the slowdown features separate dead from live frames.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from possession import config, cv, features, io_idsse, model  # noqa: E402
import diag_deadball as dd  # noqa: E402
import diag_liveplay as dl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=7)
    args = ap.parse_args()
    with open(os.path.join(REPO, "models", "decoder_v2.json")) as fh:
        dp = json.load(fh)["decoder"]
    with open(os.path.join(config.CACHE_DIR, "degrader_params.json")) as fh:
        params = json.load(fh)
    oof = np.load(os.path.join(config.CACHE_DIR, "oof_v2.npz"))
    acc = {k: [0, 0, 0] for k in ("live", "missed", "matched")}
    slow_pos, slow_neg, rates = {}, {}, []
    for mid in io_idsse.MATCH_IDS[:args.matches]:
        t0 = time.time()
        m = model.make_copy(mid, params, *model.EVAL_COPY)
        tf = features.touch_features(m, m.fps)
        sf = features.slowdown_features(m, m.fps)
        base = features.compute(m)
        y, g = m.labels, m.graded
        n_t = len(features.detect_touches(m, m.fps))
        rates.append(n_t / (g.sum() / m.fps / 60))
        blocks = dl.blocks_from(g)
        pred = cv.decode_ll(oof[mid].astype(np.float64), blocks, dp, fill=3)
        gs, ps = dl.spells(y, blocks), dl.spells(pred, blocks)
        mg, _ = dl.greedy(gs, ps)
        c = dl.classify(gs, ps, mg)
        last = tf["touch_last_team"]
        lt = np.where(last > 0, 0, np.where(last < 0, 1, -1))
        prox = np.where(base["d_home"] <= base["d_away"], 0, 1)
        live = g & (y < 2)
        acc["live"][0] += int(np.sum(lt[live] == y[live]))
        acc["live"][1] += int(np.sum(prox[live] == y[live]))
        acc["live"][2] += int(live.sum())
        for i, sp in enumerate(gs):
            k = c[i][0]
            if k in acc:
                sl = slice(sp[0], sp[1])
                acc[k][0] += int(np.sum(lt[sl] == sp[2]))
                acc[k][1] += int(np.sum(prox[sl] == sp[2]))
                acc[k][2] += sp[1] - sp[0]
        dead = g & (y == 2)
        for name, x in sf.items():
            slow_pos.setdefault(name, []).append(x[dead][::5])
            slow_neg.setdefault(name, []).append(x[live][::5])
        print("{}  {} touches ({:.1f}/min)  {:.0f}s".format(mid, n_t, rates[-1], time.time() - t0),
              flush=True)
    print("\nTeam agreement with the true possession team (share of frames)")
    print("   {:8s} {:>11s} {:>15s}".format("", "last touch", "nearest player"))
    for k, (a, b, n) in acc.items():
        print("   {:8s} {:11.1%} {:15.1%}   ({} frames)".format(k, a / n, b / n, n))
    print("\nSlowdown features: AUC dead vs live (0.5 = useless; direction-free)")
    res = []
    for name in slow_pos:
        a = dd.auc(np.concatenate(slow_pos[name]), np.concatenate(slow_neg[name]))
        res.append((max(a, 1 - a), name))
    for a, name in sorted(res, reverse=True):
        print("   {:26s} {:.3f}".format(name, a))


if __name__ == "__main__":
    main()
