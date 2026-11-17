"""Oracle decomposition of v5.2's nested IDSSE predictions (diagnostic, decode-only).

    python scripts/oracle_decomp.py

Base: cache/nested_preds_v52_mean_e2.npz (v5.2: Law 9, 1e-3, mean-tuned decoder, nested).
Each oracle is built per label convention (2 / 1 / 0.5 s) from that convention's truth, scored with
the pooled spellF1, then averaged over the three conventions. All edits are inside scored blocks.
 a  perfect alive/dead: D exactly where truth is D; on truth-alive frames keep our team, and where
    we said D the team is our nearest own team label (previous, else next; no truth team used)
 b  perfect team given alive: where both we and truth are alive, take the true team
 c  perfect boundaries: every predicted change point moves to the nearest true change point
    within +-1 s (25 frames); segment labels kept, segments that collapse are dropped
 d  2 s cap: prediction = truth with true H/A spells shorter than 2 s (50 frames) merged into the
    previous spell (next at a block start); also v5.2 scored against that merged truth
Writes out/oracle_decomp_report.txt.
"""

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import cv, io_idsse  # noqa: E402
from scripts import mean_decoder as MD  # noqa: E402

IDS = io_idsse.MATCH_IDS
FPS = 25


def runs(x):
    cut = np.flatnonzero(np.diff(x)) + 1
    return np.r_[0, cut], np.r_[cut, len(x)]


def fill_team(p):
    """team for every frame: our own label if H/A, else nearest previous (else next) H/A."""
    out = p.copy()
    idx = np.where(np.isin(p, (0, 1)), np.arange(len(p)), -1)
    prev = np.maximum.accumulate(idx)
    idx2 = np.where(np.isin(p, (0, 1)), np.arange(len(p)), len(p))
    nxt = np.minimum.accumulate(idx2[::-1])[::-1]
    for t in np.flatnonzero(~np.isin(p, (0, 1))):
        j = prev[t] if prev[t] >= 0 else (nxt[t] if nxt[t] < len(p) else -1)
        out[t] = p[j] if j >= 0 else 0
    return out


def oracle_a(p, y):
    o = fill_team(p)
    o[y == 2] = 2
    o[~np.isin(y, (0, 1, 2))] = p[~np.isin(y, (0, 1, 2))]
    return o


def oracle_b(p, y):
    o = p.copy()
    m = np.isin(p, (0, 1)) & np.isin(y, (0, 1))
    o[m] = y[m]
    return o


def oracle_c(p, y, w=FPS):
    a, b = runs(p)
    tb = runs(y)[0][1:]
    starts = [0]
    for s in a[1:]:
        if len(tb):
            k = np.argmin(np.abs(tb - s))
            s2 = tb[k] if abs(tb[k] - s) <= w else s
        else:
            s2 = s
        starts.append(int(s2))
    o = np.empty_like(p)
    labs = p[a]
    # apply in order; later segments overwrite from their (snapped) start; collapsed ones vanish
    bounds = list(zip(starts, labs))
    for i, (s, l) in enumerate(bounds):
        e = bounds[i + 1][0] if i + 1 < len(bounds) else len(p)
        if e > s:
            o[s:e] = l
    return o


def merge_short(y, n=2 * FPS):
    o = y.copy()
    a, b = runs(y)
    for i, (s, e) in enumerate(zip(a, b)):
        if y[s] in (0, 1) and e - s < n:
            if i > 0:
                o[s:e] = o[s - 1]
            elif i + 1 < len(a):
                o[s:e] = y[b[i]]
    return o


def per_block(fn, p, y, blocks):
    o = p.copy()
    for s, e in blocks:
        o[s:e] = fn(p[s:e], y[s:e])
    return o


def score(items, b, preds, labs=None):
    return cv.pooled_score([(ll, (labs[i] if labs else lb[b]), bl)
                            for i, (ll, lb, bl) in enumerate(items)], preds=preds)["spell_f1_mean"]


def main():
    items = MD.load_items(["nn_gru05", "nn_gru05s1", "nn_tcn05"], True, 0.001)
    z = np.load(os.path.join(REPO, "cache", "nested_preds_v52_mean_e2.npz"))
    base = [z[m] for m in IDS]
    rows = {k: [] for k in ("v5.2", "a alive/dead", "b team|alive", "c boundaries +-1s",
                            "d truth, <2s merged", "d' v5.2 vs merged truth")}
    for b in MD.BRIDGES:
        ys = [it[1][b] for it in items]
        bl = [it[2] for it in items]
        rows["v5.2"].append(score(items, b, base))
        for k, fn in (("a alive/dead", oracle_a), ("b team|alive", oracle_b),
                      ("c boundaries +-1s", oracle_c)):
            rows[k].append(score(items, b, [per_block(fn, p, y, B) for p, y, B in zip(base, ys, bl)]))
        merged = [per_block(lambda p, y: merge_short(y), y, y, B) for y, B in zip(ys, bl)]
        rows["d truth, <2s merged"].append(score(items, b, merged))
        rows["d' v5.2 vs merged truth"].append(score(items, b, base, labs=merged))
    ref = np.mean(rows["v5.2"])
    lines = []
    for k, v in rows.items():
        lines.append("{:<26} mean {:.4f}  gain {:+.4f}   2s {:.4f}  1s {:.4f}  0.5s {:.4f}".format(
            k, np.mean(v), np.mean(v) - ref, *v))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(REPO, "out", "oracle_decomp_report.txt"), "w") as h:
        h.write(txt + "\n")


if __name__ == "__main__":
    main()
