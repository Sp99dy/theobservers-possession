"""Score the Metrica-augmented ensembles vs v5.1 (v5 members + Law 9), nested, mean of conventions.

    python scripts/metrica_score.py [--jobs 2]

Arms (3-member ensembles gru05 / gru05s1 / tcn05, mean of out-of-fold log-probs, Law-9 override
on the standard evaluation copy as in v5.1):
  v5.1    : cache/oof_nn_{gru05,gru05s1,tcn05}.npz
  w1.0    : out/nn_metrica_w10_<member>/<match>.npy
  w0.5    : out/nn_metrica_w05_<member>/<match>.npy
Per held-out match one decoder tuned on the six others on the mean spellF1 over the 2 / 1 / 0.5 s
labels (gate_mean.tune, gate closed), from v5.1's 2 s Law-9 decoder. Wins on the per-match mean vs
v5.1. Writes only out/metrica_score_*.
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
from possession import config, cv, model, rules  # noqa: E402
from scripts import gate_mean as GM  # noqa: E402
from scripts import gate_short_spells as G  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH, blocks_of  # noqa: E402

OUT = G.OUT
MEMBERS = ("gru05", "gru05s1", "tcn05")
ARMS = ("v5.1", "w1.0", "w0.5")
LAW9_DEC = "cv_ens_nn_gru05+nn_gru05s1+nn_tcn05_law9_e2_decoder.json"


def member_ll(arm, mem, mid):
    if arm == "v5.1":
        with np.load(os.path.join(config.CACHE_DIR, "oof_nn_{}.npz".format(mem))) as z:
            return z[mid]
    tag = {"w1.0": "w10", "w0.5": "w05"}[arm]
    return np.load(os.path.join(OUT, "nn_metrica_{}_{}".format(tag, mem), mid + ".npy"))


def load(arm):
    with open(PARAMS_PATH) as h:
        params = json.load(h)
    data = []
    for mid in G.IDS:
        _, _, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, 2)
        labs = {b: model.copy_labels(mid, model.EVAL_COPY[1], br) for b, br in G.BRIDGES.items()}
        ll = np.mean([member_ll(arm, m, mid).astype(np.float64) for m in MEMBERS], 0)
        mask = np.load(os.path.join(config.CACHE_DIR, "law9_{}_9.npy".format(mid)))
        ll = rules.law9_override(ll, mask)
        data.append((ll, blocks_of(g), np.zeros(len(ll)), labs))
    return data


def _path(arm):
    return os.path.join(OUT, "metrica_score_preds_{}.npz".format(arm))


def run(arm):
    if os.path.isfile(_path(arm)):
        return arm
    data = load(arm)
    with open(os.path.join(config.CACHE_DIR, LAW9_DEC)) as h:
        start = dict(json.load(h)["decoder"], chg_w=0.0, chg_off=0.0, fast_thr=99.0, fast_steps=4)
    preds = {}
    for i, mid in enumerate(G.IDS):
        t0 = time.time()
        rest = [d for j, d in enumerate(data) if j != i]
        dpi, s = GM.tune(rest, start, GM.GATE_KEYS)
        ll, blocks, sig, _ = data[i]
        preds[mid] = cv.decode_ll(ll, blocks, dpi, fill=3, change=sig)
        print("{} fold {}: inner {:.4f} {:.0f}s {}".format(arm, mid, s, time.time() - t0,
                                                         json.dumps(dpi)), flush=True)
    np.savez(_path(arm), **preds)
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.jobs) as ex:
        list(ex.map(run, ARMS))
    res = {a: (load(a), [np.load(_path(a))[m] for m in G.IDS]) for a in ARMS}
    l, ref = GM.table(res["v5.1"][0], "v5.1 (mean-tuned, Law 9)", res["v5.1"][1])
    lines = [l] + [GM.table(res[a][0], "metrica {} vs v5.1".format(a), res[a][1], ref)[0]
                   for a in ARMS[1:]]
    for a in ARMS:
        sp = {b: G.short_pr([(ll, labs[b], bl, s) for ll, bl, s, labs in res[a][0]], res[a][1])
              for b in G.BRIDGES}
        lines.append("short spells {:<6} ".format(a) + "  ".join(
            "{} R {:.3f} P {:.3f}".format(b, s["short_recall"], s["short_precision"])
            for b, s in sp.items()))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "metrica_score_report.txt"), "w") as h:
        h.write(txt + "\n")


if __name__ == "__main__":
    main()
