"""Gate experiment under the mean-of-conventions metric (mean nested spellF1 over 2 / 1 / 0.5 s).

    python scripts/gate_mean.py [--jobs 2]

Two arms, both fully nested on v5's emissions (nn_gru05 + nn_gru05s1 + nn_tcn05):
  nogate : every decoder parameter tuned on the six other matches with the gate fixed closed
  gate   : the same plus the gate (cv.TOUCH_GRID) on the margin+touch signal of gate_short_spells
The objective is the mean of the pooled spellF1 against the 2, 1 and 0.5 s labels (one decoder for
all three). Start: v5's 1 s decoder (closed gate). Per-match wins are on the per-match mean.
Also re-scores gate_short_spells' 2 s-tuned folds and v5's 2 s nested preds on all three labels.
Writes only out/gate_mean_* (resumable per fold).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config, cv  # noqa: E402
from scripts import gate_short_spells as G  # noqa: E402

OUT = G.OUT
GATE_KEYS = tuple(cv.TOUCH_GRID)


def log(s):
    line = "{} {}".format(time.strftime("%H:%M:%S"), s)
    print(line, flush=True)
    with open(os.path.join(OUT, "gate_mean_progress.log"), "a") as h:
        h.write(line + "\n")


def load_all():
    """per match: (ll, blocks, sig, {btag: labels})"""
    per = {b: G.build_items("margin+touch", b) for b in G.BRIDGES}
    out = []
    for i in range(len(G.IDS)):
        ll, _, blocks, sig = per["e2"][i]
        out.append((ll, blocks, sig, {b: per[b][i][1] for b in G.BRIDGES}))
    return out


def scores(sub, preds):
    """mean spellF1 over bridges (pooled over sub), plus per bridge."""
    res = {}
    for b in G.BRIDGES:
        items = [(ll, labs[b], blocks, sig) for (ll, blocks, sig, labs) in sub]
        res[b] = cv.pooled_score(items, preds=preds)
    return np.mean([r["spell_f1_mean"] for r in res.values()]), res


def objective(sub, dp):
    preds = [cv.decode_ll(ll, blocks, dp, fill=3, change=sig) for ll, blocks, sig, _ in sub]
    return scores(sub, preds)


def tune(sub, start, fixed, sweeps=2):
    dp = dict(start)
    best, _ = objective(sub, dp)
    for _ in range(sweeps):
        changed = False
        for name, values in cv.decoder_grid(dp, fixed).items():
            for v in values:
                if v == dp[name]:
                    continue
                trial = dict(dp, **{name: v})
                s, _ = objective(sub, trial)
                if s > best + 1e-9:
                    best, dp, changed = s, trial, True
        if not changed:
            break
    return dp, best


def fold_path(arm, i):
    return os.path.join(OUT, "gate_mean_fold_{}_{}.npy".format(arm, i))


def run(arm):
    data = load_all()
    with open(os.path.join(config.CACHE_DIR, "cv_{}_e1_decoder.json".format(G.ENS))) as h:
        start = dict(json.load(h)["decoder"], chg_w=0.0, chg_off=0.0, fast_thr=99.0, fast_steps=4)
    fixed = GATE_KEYS if arm == "nogate" else ()
    for i, mid in enumerate(G.IDS):
        fp = fold_path(arm, i)
        if os.path.isfile(fp):
            continue
        t0 = time.time()
        rest = [d for j, d in enumerate(data) if j != i]
        dpi, s = tune(rest, start, fixed)
        ll, blocks, sig, _ = data[i]
        np.save(fp, cv.decode_ll(ll, blocks, dpi, fill=3, change=sig))
        json.dump(dpi, open(fp + ".json", "w"))
        log("{} fold {} {}: inner mean {:.4f}, {:.0f}s  {}".format(arm, i, mid, s, time.time() - t0,
                                                                  json.dumps(dpi)))
    return arm


def table(data, name, preds, ref=None):
    m, res = scores(data, preds)
    per = np.array([scores([d], [p])[0] for d, p in zip(data, preds)])
    line = "{:<30} mean {:.4f}  2s {:.4f}  1s {:.4f}  0.5s {:.4f}".format(
        name, m, res["e2"]["spell_f1_mean"], res["e1"]["spell_f1_mean"],
        res["e0.5"]["spell_f1_mean"])
    if ref is not None:
        d = per - ref[1]
        line += "   vs ref {:+.4f}  wins {}/7 [{}]".format(
            m - ref[0], int((d > 0).sum()), " ".join("{:+.3f}".format(x) for x in d))
    return line, (m, per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.jobs) as ex:
        list(ex.map(run, ["nogate", "gate"]))
    data = load_all()
    lines = []
    zb = np.load(os.path.join(config.CACHE_DIR, "nested_preds_{}_e2.npz".format(G.ENS)))
    l, ref2 = table(data, "v5 2s-tuned (shipped style)", [zb[m] for m in G.IDS])
    lines.append(l)
    for v in G.VARIANTS:
        p = [np.load(G._fold_path(v, "e2", i)) for i in range(len(G.IDS))]
        lines.append(table(data, "gate {} 2s-tuned".format(v), p, ref2)[0])
    pn = [np.load(fold_path("nogate", i)) for i in range(len(G.IDS))]
    l, refm = table(data, "mean-tuned, no gate", pn, ref2)
    lines.append(l)
    pg = [np.load(fold_path("gate", i)) for i in range(len(G.IDS))]
    lines.append(table(data, "mean-tuned + gate (vs no gate)", pg, refm)[0])
    for name, p in (("v5 2s-tuned", [zb[m] for m in G.IDS]), ("mean-tuned no gate", pn),
                    ("mean-tuned + gate", pg)):
        sp = {b: G.short_pr([(ll, labs[b], blocks, sig) for ll, blocks, sig, labs in data], p)
              for b in G.BRIDGES}
        lines.append("short spells {:<20} ".format(name) + "  ".join(
            "{} R {:.3f} P {:.3f} (n pred {})".format(b, s["short_recall"], s["short_precision"],
                                                     s["short_pred"]) for b, s in sp.items()))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "gate_mean_report.txt"), "w") as h:
        h.write(txt + "\n")


if __name__ == "__main__":
    main()
