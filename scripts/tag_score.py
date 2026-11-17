"""Score simulated-Smart-Tagging members vs v6 and v5 on the standard and the dead-move eval copy.

    python scripts/tag_score.py [--jobs 2]

Arms (3-member ensembles, mean of out-of-fold log-probs):
  v5      : gru05 / gru05s1 / tcn05 (retrained "r" oofs for the dead-move copy)
  v6      : gru05dm / gru05dms1 / tcn05dm
  v6+tags : out/nn_tag_{gru05dm,gru05dms1,tcn05dm}/<match>__<cache>.npy
For each arm and eval copy: one decoder per held-out match, tuned on the six others on the mean
spellF1 over the 2 / 1 / 0.5 s labels (gate_mean.tune, gate closed), starting from v5's 1 s
decoder. Reports the mean and per-convention scores; per-match wins on the per-match mean vs v6.
Writes only out/tag_score_*.
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
from possession import config, cv, model  # noqa: E402
from scripts import gate_mean as GM  # noqa: E402
from scripts import gate_short_spells as G  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH, blocks_of  # noqa: E402

OUT = G.OUT
EVALS = ("cache", "cache_deadmove")
ARMS = {
    "v5": {"cache": ["nn_gru05", "nn_gru05s1", "nn_tcn05"],
           "cache_deadmove": ["nn_gru05r", "nn_gru05s1r", "nn_tcn05r"]},
    "v6": {e: ["nn_gru05dm", "nn_gru05dms1", "nn_tcn05dm"] for e in EVALS},
    "v6+tags": {e: ["tag_gru05dm", "tag_gru05dms1", "tag_tcn05dm"] for e in EVALS},
}


def member_ll(arm, ev, name, mid):
    if arm == "v6+tags":
        return np.load(os.path.join(OUT, "nn_" + name, "{}__{}.npy".format(mid, ev)))
    with np.load(os.path.join(REPO, ev, "oof_{}.npz".format(name))) as z:
        return z[mid]


def load(arm, ev):
    with open(os.path.join(REPO, ev, "degrader_params.json")) as h:
        params = json.load(h)
    data = []
    home = config.CACHE_DIR
    for mid in G.IDS:
        config.CACHE_DIR = os.path.join(REPO, ev)
        _, _, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, 2)
        config.CACHE_DIR = home
        labs = {b: model.copy_labels(mid, model.EVAL_COPY[1], br) for b, br in G.BRIDGES.items()}
        ll = np.mean([member_ll(arm, ev, n, mid).astype(np.float64) for n in ARMS[arm][ev]], 0)
        data.append((ll, blocks_of(g), np.zeros(len(ll)), labs))
    return data


def run(task):
    arm, ev = task
    path = os.path.join(OUT, "tag_score_preds_{}_{}.npz".format(arm.replace("+", "_"), ev))
    if os.path.isfile(path):
        return task
    data = load(arm, ev)
    with open(os.path.join(config.CACHE_DIR, "cv_{}_e1_decoder.json".format(G.ENS))) as h:
        start = dict(json.load(h)["decoder"], chg_w=0.0, chg_off=0.0, fast_thr=99.0, fast_steps=4)
    preds = {}
    for i, mid in enumerate(G.IDS):
        t0 = time.time()
        rest = [d for j, d in enumerate(data) if j != i]
        dpi, s = GM.tune(rest, start, GM.GATE_KEYS)
        ll, blocks, sig, _ = data[i]
        preds[mid] = cv.decode_ll(ll, blocks, dpi, fill=3, change=sig)
        print("{} {} fold {}: inner {:.4f} {:.0f}s".format(arm, ev, mid, s, time.time() - t0),
              flush=True)
    np.savez(path, **preds)
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    tasks = [(a, e) for a in ARMS for e in EVALS]
    with ProcessPoolExecutor(args.jobs) as ex:
        list(ex.map(run, tasks))
    lines = []
    for ev in EVALS:
        res = {}
        for arm in ARMS:
            z = np.load(os.path.join(OUT, "tag_score_preds_{}_{}.npz".format(arm.replace("+", "_"), ev)))
            res[arm] = (load(arm, ev), [z[m] for m in G.IDS])
        l, ref = GM.table(*res["v6"][:1], "v6 [{}]".format(ev), res["v6"][1])
        lines.append(l)
        lines.append(GM.table(res["v5"][0], "v5 [{}] vs v6".format(ev), res["v5"][1], ref)[0])
        lines.append(GM.table(res["v6+tags"][0], "v6+tags [{}] vs v6".format(ev),
                              res["v6+tags"][1], ref)[0])
        _, r5 = GM.table(res["v5"][0], "", res["v5"][1])
        lines.append(GM.table(res["v6+tags"][0], "v6+tags [{}] vs v5".format(ev),
                              res["v6+tags"][1], r5)[0])
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "tag_score_report.txt"), "w") as h:
        h.write(txt + "\n")


if __name__ == "__main__":
    main()
