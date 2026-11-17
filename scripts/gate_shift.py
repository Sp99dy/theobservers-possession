"""Switch delay (idea from a public repo): move every decoded H<->A boundary later by d seconds.

    python scripts/gate_shift.py

Reference: v5.2 nested predictions (cache/nested_preds_v52_mean_e2.npz; Law 9, rounding, mean-tuned). Only H<->A boundaries move (the new team starts d later, never past
its own end); boundaries with D are untouched. d is chosen per held-out match on the six others
(mean spellF1 over the 2 / 1 / 0.5 s labels, d = 0 allowed), then applied to the held-out match.
Writes only out/gate_shift_report.txt.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from scripts import gate_mean as GM  # noqa: E402
from scripts import gate_short_spells as G  # noqa: E402
from scripts import metrica_score as MS  # noqa: E402

DS = (0.0, 0.2, 0.4, 0.6)


def shift(pred, d_frames):
    if d_frames <= 0:
        return pred
    out = pred.copy()
    cut = np.flatnonzero(np.diff(pred)) + 1
    ends = np.r_[cut[1:], len(pred)]
    for t, e in zip(cut, ends):
        a, b = pred[t - 1], pred[t]
        if a in (0, 1) and b in (0, 1) and a != b:
            out[t:min(t + d_frames, e - 1)] = a
    return out


def main():
    from scripts import gate_cd as CD
    from scripts import mean_decoder as MD
    data = MD.load_items(["nn_gru05", "nn_gru05s1", "nn_tcn05"], True, 0.001)
    z = np.load(os.path.join(REPO, "cache", "nested_preds_v52_mean_e2.npz"))
    base = [z[m] for m in G.IDS]
    GM.table = CD.table
    GM.scores = lambda its, ps: MD.mean_score(its, ps)
    cand = {d: [shift(p, int(round(d * 25))) for p in base] for d in DS}
    lines = []
    l, ref = GM.table(data, "v5.2 nested (ref, d=0)", base)
    lines.append(l)
    for d in DS[1:]:
        lines.append(GM.table(data, "fixed d = {:.1f} s".format(d), cand[d], ref)[0])
    nested, chosen = [], []
    for i in range(len(G.IDS)):
        rest = [j for j in range(len(G.IDS)) if j != i]
        best = max(DS, key=lambda d: GM.scores([data[j] for j in rest], [cand[d][j] for j in rest])[0])
        chosen.append(best)
        nested.append(cand[best][i])
    lines.append(GM.table(data, "nested d (chosen {})".format(chosen), nested, ref)[0])
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(G.OUT, "gate_shift_report.txt"), "w") as h:
        h.write(txt + "\n")


if __name__ == "__main__":
    main()
