"""Boundary refinement on v5.2's nested predictions (decode-only, IDSSE, nested choice).

    python scripts/refine_bounds.py

Every predicted H<->A boundary t (first frame of the new team b after team a; D boundaries never
move) may move within +-w s, staying inside the a-then-b stretch (each side keeps >= 1 frame):
 v1  log-odds: f = 5-frame mean of ll[b] - ll[a] (ensemble emissions, Law 9, 1e-3); move to the
     upward zero crossing (f[k-1] <= 0 < f[k]) closest to t, else to the steepest rise of f
 v2  touch: move to the nearest detected touch by team b (features.detect_touches on the eval
     copy), else keep
Candidates off + {v1, v2} x w {0.4, 0.8, 1.2}; per held-out match the best on the other six (mean
spellF1 over 2 / 1 / 0.5 s). Reference: cache/nested_preds_v52_mean_e2.npz. Writes out/refine_*.
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import features, model  # noqa: E402
from scripts import gate_cd as CD  # noqa: E402
from scripts import mean_decoder as MD  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH  # noqa: E402

IDS = MD.io_idsse.MATCH_IDS
WS = (0.4, 0.8, 1.2)
OUT = os.path.join(REPO, "out")


def touches():
    path = os.path.join(OUT, "refine_touches.npz")
    if os.path.isfile(path):
        z = np.load(path)
        return {m: z[m] for m in IDS}
    params = json.load(open(PARAMS_PATH))
    res = {}
    for mid in IDS:
        m = model.make_copy(mid, params, *model.EVAL_COPY)
        res[mid] = np.array([(t, 0 if s > 0 else 1) for t, s, *_ in features.detect_touches(m, m.fps)],
                            dtype=np.int64).reshape(-1, 2)
    np.savez(path, **res)
    return res


def refine(pred, ll, tch, var, w):
    w = int(round(w * 25))
    out = pred.copy()
    cut = np.flatnonzero(np.diff(pred)) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(pred)]
    for i in range(1, len(starts)):
        t = starts[i]
        a, b = pred[t - 1], pred[t]
        if not (a in (0, 1) and b in (0, 1) and a != b):
            continue
        lo = max(starts[i - 1] + 1, t - w)
        hi = min(ends[i] - 1, t + w)
        if hi <= lo:
            continue
        new = t
        if var == 1:
            f = np.convolve(ll[:, b] - ll[:, a], np.ones(5) / 5, mode="same")
            k = np.arange(lo, hi + 1)
            k = k[k >= 1]
            up = k[(f[k - 1] <= 0) & (f[k] > 0)]
            if len(up):
                new = int(up[np.argmin(np.abs(up - t))])
            elif len(k):
                new = int(k[np.argmax(f[k] - f[k - 1])])
        else:
            c = tch[(tch[:, 1] == b) & (tch[:, 0] >= lo) & (tch[:, 0] <= hi), 0] if len(tch) else []
            if len(c):
                new = int(c[np.argmin(np.abs(c - t))])
        if new > t:
            out[t:new] = a
        elif new < t:
            out[new:t] = b
    return out


def main():
    items = MD.load_items(["nn_gru05", "nn_gru05s1", "nn_tcn05"], True, 0.001)
    z = np.load(os.path.join(REPO, "cache", "nested_preds_v52_mean_e2.npz"))
    base = [z[m] for m in IDS]
    tch = touches()
    cands = {"off": base}
    for var in (1, 2):
        for w in WS:
            cands[(var, w)] = [refine(p, it[0], tch[m], var, w) for p, it, m in zip(base, items, IDS)]
    l, ref = CD.table(items, "v5.2 nested (reference)", base)
    lines = [l]
    for k, v in cands.items():
        if k != "off":
            lines.append(CD.table(items, "fixed v{} w {}".format(*k), v, ref)[0])
    nested, chosen = [], []
    for i in range(len(IDS)):
        rest = [j for j in range(len(IDS)) if j != i]
        k = max(cands, key=lambda k: MD.mean_score([items[j] for j in rest], [cands[k][j] for j in rest])[0])
        chosen.append(k)
        nested.append(cands[k][i])
    lines.append(CD.table(items, "NESTED {}".format(chosen), nested, ref)[0])
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "refine_report.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
