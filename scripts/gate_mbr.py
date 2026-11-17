"""Consensus / minimum-Bayes-risk decoding (review idea 3). Decoder only, no fitted parameters.

    python scripts/gate_mbr.py

Candidates per held-out match i (all nested: every decoder was tuned without match i):
  emissions {ensemble, nn_gru05, nn_gru05s1, nn_tcn05}
  x decoders {2 s-, 1 s-, 0.5 s-tuned fold decoders of gate_short_spells (margin+touch)}
  x gate {on (as tuned), off (chg_w 0, fast_thr 99)}
  + gate_mean's two mean-tuned folds (no gate / gate), ensemble emissions.
Each half block is cut at stoppages shared by all candidates (every candidate says D). Inside each
piece, the candidate with the highest mean spellF1 against all other candidates is kept.
Scored on the mean nested spellF1 over the 2 / 1 / 0.5 s labels; per-match wins on the per-match
mean, vs the mean-tuned no-gate decoder and vs the best single candidate.
Writes only out/gate_mbr_*.
"""

import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config, cv  # noqa: E402
from scripts import gate_mean as GM  # noqa: E402
from scripts import gate_short_spells as G  # noqa: E402

IOUS = (0.3, 0.4, 0.5, 0.6, 0.7)
OUT = G.OUT


def spells(c):
    cut = np.flatnonzero(np.diff(c)) + 1
    a, b = np.r_[0, cut], np.r_[cut, len(c)]
    return [(x, y, int(c[x])) for x, y in zip(a, b) if c[x] in (0, 1)]


def spell_f1(sa, sb):
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    ov = []
    for i, (a0, a1, ka) in enumerate(sa):
        for j, (b0, b1, kb) in enumerate(sb):
            if ka != kb or b1 <= a0 or a1 <= b0:
                continue
            ov.append(((min(a1, b1) - max(a0, b0)) / (max(a1, b1) - min(a0, b0)), i, j))
    ov.sort(key=lambda t: (-t[0], t[1], t[2]))
    tot = 0.0
    for th in IOUS:
        ui, uj, m = set(), set(), 0
        for iou, i, j in ov:
            if iou < th:
                break
            if i in ui or j in uj:
                continue
            ui.add(i)
            uj.add(j)
            m += 1
        tot += 2 * m / (len(sa) + len(sb))
    return tot / len(IOUS)


def consensus(cands, blocks):
    K = len(cands)
    out = cands[0].copy()
    picks = np.zeros(K, int)
    for s, e in blocks:
        alld = np.all([c[s:e] == 2 for c in cands], axis=0)
        cut = np.flatnonzero(np.diff(alld.astype(np.int8))) + 1
        a, b = np.r_[0, cut], np.r_[cut, e - s]
        for x, y in zip(a, b):
            if alld[x]:
                continue
            sp = [spells(c[s + x:s + y]) for c in cands]
            keys = {}
            for k in range(K):
                keys.setdefault(tuple(sp[k]), []).append(k)
            uniq = list(keys)
            if len(uniq) == 1:
                continue
            w = np.array([len(keys[u]) for u in uniq], float)
            F = np.ones((len(uniq), len(uniq)))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    F[i, j] = F[j, i] = spell_f1(list(uniq[i]), list(uniq[j]))
            # mean F1 against the other K-1 candidates (duplicates count, self excluded)
            score = (F @ w - 1.0) / (K - 1)
            best = keys[uniq[int(np.argmax(score))]][0]
            picks[best] += 1
            out[s + x:s + y] = cands[best][s + x:s + y]
    return out, picks


def main():
    t0 = time.time()
    data = GM.load_all()
    zs = {m: np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(m))) for m in G.MEMBERS}
    names, per_match_cands = None, []
    for i, mid in enumerate(G.IDS):
        ll_ens, blocks, sig, _ = data[i]
        lls = {"ens": ll_ens}
        lls.update({m: zs[m][mid].astype(np.float64) for m in G.MEMBERS})
        cands, nm = [], []
        for b in G.BRIDGES:
            dp = json.load(open(G._fold_path("margin+touch", b, i) + ".json"))
            off = dict(dp, chg_w=0.0, fast_thr=99.0)
            for gate, d in (("on", dp), ("off", off)):
                for en, ll in lls.items():
                    cands.append(cv.decode_ll(ll, blocks, d, fill=3, change=sig))
                    nm.append("{}|{}|gate {}".format(en, b, gate))
        for arm in ("nogate", "gate"):
            cands.append(np.load(GM.fold_path(arm, i)))
            nm.append("ens|mean-tuned|{}".format(arm))
        names = nm
        per_match_cands.append(cands)
        print("{} candidates decoded for {} ({:.0f}s)".format(len(cands), mid, time.time() - t0),
              flush=True)

    mbr, picks = [], np.zeros(len(names), int)
    for i, mid in enumerate(G.IDS):
        out, pk = consensus(per_match_cands[i], data[i][1])
        mbr.append(out)
        picks += pk
        print("consensus {} done ({:.0f}s)".format(mid, time.time() - t0), flush=True)

    ref_line, ref = GM.table(data, "mean-tuned, no gate (ref)", [c[-2] for c in per_match_cands])
    singles = []
    for k, n in enumerate(names):
        m, _ = GM.scores(data, [c[k] for c in per_match_cands])
        singles.append((m, k))
    singles.sort(reverse=True)
    bk = singles[0][1]
    best_line, best = GM.table(data, "best single: " + names[bk], [c[bk] for c in per_match_cands], ref)
    lines = [ref_line, best_line,
             GM.table(data, "MBR consensus vs ref", mbr, ref)[0],
             GM.table(data, "MBR consensus vs best single", mbr, best)[0],
             "top singles (mean): " + "; ".join("{} {:.4f}".format(names[k], m) for m, k in singles[:6]),
             "pieces won per candidate: " + "; ".join(
                 "{} {}".format(names[k], int(picks[k])) for k in np.argsort(-picks)[:8])]
    sp = {b: G.short_pr([(ll, labs[b], blocks, s) for ll, blocks, s, labs in data], mbr)
          for b in G.BRIDGES}
    lines.append("short spells MBR " + "  ".join(
        "{} R {:.3f} P {:.3f}".format(b, s["short_recall"], s["short_precision"]) for b, s in sp.items()))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "gate_mbr_report.txt"), "w") as h:
        h.write(txt + "\n")
    np.savez(os.path.join(OUT, "gate_mbr_preds.npz"), **dict(zip(G.IDS, mbr)))


if __name__ == "__main__":
    main()
