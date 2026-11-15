"""Diagnostic: where does v2 lose spellF1 in live play?

    python scripts/diag_liveplay.py

IDSSE open data only. v2 out-of-fold emissions (cache/oof_v2.npz) on the CV evaluation corruption,
decoded with the shipped v2 decoder, scored with the organisers' code.

1. What happened to every true possession that is NOT matched at IoU 0.5: missed entirely,
   merged with a neighbour, split, or found with a misplaced boundary.
2. Oracles: paste the truth over one kind of error at a time and see how much spellF1 comes back.
3. For possessions we miss: was the evidence there (the classifier preferred the right team) and
   the decoder overruled it, or did the classifier not see it either?
"""

import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config, cv, features, io_idsse, model  # noqa: E402

FPS = 25
H, A, D, X = 0, 1, 2, 3
TOL = 3 * FPS            # a predicted switch within 3 s of a true one counts as "the same switch"


def blocks_from(g):
    idx = np.flatnonzero(g)
    b = np.flatnonzero(np.diff(idx) > 1)
    return [(int(s), int(e) + 1) for s, e in zip(np.r_[idx[0], idx[b + 1]], np.r_[idx[b], idx[-1]])]


def runs(codes, blocks):
    out = []
    for s, e in blocks:
        c = codes[s:e]
        b = np.flatnonzero(np.diff(c)) + 1
        for a, z in zip(np.r_[0, b], np.r_[b, len(c)]):
            out.append((s + int(a), s + int(z), int(c[a])))
    return out


def spells(codes, blocks):
    return [r for r in runs(codes, blocks) if r[2] in (H, A)]


def inter(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def iou(a, b):
    i = inter(a, b)
    return i / (max(a[1], b[1]) - min(a[0], b[0])) if i else 0.0


def greedy(gt_sp, pr_sp, thr=0.5):
    ov = []
    for i, g in enumerate(gt_sp):
        for j, p in enumerate(pr_sp):
            if g[2] == p[2] and inter(g, p):
                ov.append((iou(g, p), i, j))
    ov.sort(key=lambda t: (-t[0], t[1], t[2]))
    mg, mp = {}, {}
    for v, i, j in ov:
        if v < thr:
            break
        if i in mg or j in mp:
            continue
        mg[i], mp[j] = j, i
    return mg, mp


def classify(gt_sp, pr_sp, mg):
    """Category for every unmatched true possession, plus detail."""
    out = {}
    for i, g in enumerate(gt_sp):
        if i in mg:
            out[i] = ("matched", None)
            continue
        L = g[1] - g[0]
        same = [(j, p, inter(g, p)) for j, p in enumerate(pr_sp) if p[2] == g[2] and inter(g, p)]
        cover = sum(x[2] for x in same) / L
        if cover < 0.5:
            out[i] = ("missed", None)
            continue
        big = [x for x in same if x[2] >= 0.2 * L]
        if len(big) >= 2:
            out[i] = ("split", None)
            continue
        j, p, _ = max(same, key=lambda x: x[2])
        # Does this predicted spell also cover a different true possession of the same team?
        others = [k for k, h in enumerate(gt_sp) if k != i and h[2] == g[2] and inter(h, p) >= 0.2 * (h[1] - h[0])]
        if others:
            out[i] = ("merged", None)
        else:
            out[i] = ("boundary", (p[0] - g[0], p[1] - g[1]))
    return out


def transitions(codes, blocks):
    out = []
    for s, e in blocks:
        c = codes[s:e]
        for k in (np.flatnonzero(np.diff(c)) + 1):
            out.append((s + int(k), int(c[k - 1]), int(c[k])))
    return out


def main():
    with open(os.path.join(REPO, "models", "decoder_v2.json")) as fh:
        dp = json.load(fh)["decoder"]
    with open(os.path.join(config.CACHE_DIR, "degrader_params.json")) as fh:
        params = json.load(fh)
    oof = np.load(os.path.join(config.CACHE_DIR, "oof_v2.npz"))

    items, preds, gts = [], [], []
    for mid in io_idsse.MATCH_IDS:
        _, y, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, features.FEATURE_VERSION)
        blocks = blocks_from(g)
        ll = oof[mid].astype(np.float64)
        items.append((ll, y, blocks))
        preds.append(cv.decode_ll(ll, blocks, dp, fill=X))
        gts.append(y)
    base = cv.pooled_score(items, preds=preds)
    print("v2 OOF, shipped decoder: spellF1 {:.4f}, {} predicted vs {} true possessions".format(
        base["spell_f1_mean"], base["n_pred_spells"], base["n_gt_spells"]))

    # -- 1. what happened to each true possession ----------------------------------------------
    cats, gt_len, details, spurious, n_pred = [], [], [], 0, 0
    per_match = []
    for (ll, y, blocks), pred in zip(items, preds):
        gs, ps = spells(y, blocks), spells(pred, blocks)
        mg, mp = greedy(gs, ps)
        c = classify(gs, ps, mg)
        per_match.append((gs, ps, mg, mp, c))
        for i, g in enumerate(gs):
            cats.append(c[i][0])
            gt_len.append((g[1] - g[0]) / FPS)
            details.append(c[i][1])
        n_pred += len(ps)
        for j, p in enumerate(ps):
            if j in mp:
                continue
            same = sum(inter(g, p) for g in gs if g[2] == p[2])
            if same < 0.5 * (p[1] - p[0]):
                spurious += 1
    cats, gt_len = np.array(cats), np.array(gt_len)
    print("\n1. True possessions (n={}) at IoU 0.5".format(len(cats)))
    print("   {:10s} {:>6s} {:>7s} {:>10s}".format("outcome", "n", "share", "median s"))
    for k in ("matched", "missed", "merged", "boundary", "split"):
        m = cats == k
        print("   {:10s} {:6d} {:7.1%} {:10.1f}".format(k, m.sum(), m.mean(),
                                                     np.median(gt_len[m]) if m.any() else np.nan))
    print("   predicted possessions with no real counterpart (spurious): {} of {}".format(
        spurious, n_pred))

    print("\n   Recall by true possession length:")
    edges = [0, 2.4, 4, 6, 10, 20, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (gt_len >= lo) & (gt_len < hi)
        if m.any():
            row = "   {:>5.1f}-{:<5s} n={:4d} ({:4.0%})  matched {:4.0%}".format(
                lo, "inf" if hi > 1e8 else "{:.1f}s".format(hi), m.sum(), m.mean(),
                np.mean(cats[m] == "matched"))
            row += "  missed {:4.0%}  merged {:4.0%}  boundary {:4.0%}  split {:4.0%}".format(
                *[np.mean(cats[m] == k) for k in ("missed", "merged", "boundary", "split")])
            print(row)

    b = np.array([d for d, k in zip(details, cats) if k == "boundary"], dtype=float)
    if len(b):
        print("\n   Boundary errors (pred - true, seconds): start median {:+.1f} (|.| {:.1f}), "
              "end median {:+.1f} (|.| {:.1f})".format(np.median(b[:, 0]) / FPS,
                                                      np.median(np.abs(b[:, 0])) / FPS,
                                                      np.median(b[:, 1]) / FPS,
                                                      np.median(np.abs(b[:, 1])) / FPS))

    # -- 2. oracles ----------------------------------------------------------------------------
    def score(fixer):
        fixed = [fixer(k, pred.copy()) for k, pred in enumerate(preds)]
        return cv.pooled_score(items, preds=fixed)

    def fill_missed(k, p, max_len=None):
        y, (gs, ps, mg, mp, c) = gts[k], per_match[k]
        for i, g in enumerate(gs):
            if c[i][0] == "missed" and (max_len is None or g[1] - g[0] < max_len * FPS):
                p[g[0]:g[1]] = y[g[0]:g[1]]
        return p

    def fix_merged(k, p):
        # Paste the truth over the stretch between the two true possessions a predicted spell merged.
        y, (gs, ps, mg, mp, c) = gts[k], per_match[k]
        for i, g in enumerate(gs):
            if c[i][0] == "merged":
                p[g[0]:g[1]] = y[g[0]:g[1]]
                for h in gs:
                    if h is not g and h[2] != g[2] and inter(h, (g[0] - 5 * FPS, g[1] + 5 * FPS, 0)):
                        p[h[0]:h[1]] = y[h[0]:h[1]]
        return p

    def drop_spurious(k, p):
        y, (gs, ps, mg, mp, c) = gts[k], per_match[k]
        for j, q in enumerate(ps):
            if j in mp:
                continue
            same = sum(inter(g, q) for g in gs if g[2] == q[2])
            if same < 0.5 * (q[1] - q[0]):
                p[q[0]:q[1]] = y[q[0]:q[1]]
        return p

    def snap_switches(k, p, kinds=("team",)):
        # Every predicted switch within 3 s of a true switch of the same kind is moved onto it.
        y, blocks = gts[k], items[k][2]
        tt = transitions(y, blocks)
        pt = transitions(p, blocks)
        used = set()
        for t, a, b in tt:
            team = {a, b} == {H, A}
            if ("team" in kinds and team) or ("dead" in kinds and not team):
                best = None
                for u, a2, b2 in pt:
                    if (a2, b2) == (a, b) and abs(u - t) <= TOL and u not in used:
                        if best is None or abs(u - t) < abs(best - t):
                            best = u
                if best is not None:
                    used.add(best)
                    lo, hi = min(t, best), max(t, best)
                    p[lo:hi] = y[lo:hi]
        return p

    def dead_all(k, p):
        y = gts[k]
        p[y == D] = D
        return p

    print("\n2. Oracles (base {:.4f}): paste the truth over one kind of error".format(
        base["spell_f1_mean"]))
    oracles = [
        ("fill missed possessions < 4 s", lambda k, p: fill_missed(k, p, 4)),
        ("fill missed possessions < 10 s", lambda k, p: fill_missed(k, p, 10)),
        ("fill ALL missed possessions", fill_missed),
        ("un-merge merged possessions", fix_merged),
        ("remove spurious possessions", drop_spurious),
        ("snap H<->A switches within 3 s", snap_switches),
        ("snap alive<->dead switches within 3 s", lambda k, p: snap_switches(k, p, ("dead",))),
        ("all dead balls perfect", dead_all),
        ("missed + merged + spurious + team snap",
         lambda k, p: snap_switches(k, drop_spurious(k, fix_merged(k, fill_missed(k, p))))),
    ]
    for name, fn in oracles:
        s = score(fn)
        print("   {:40s} spellF1 {:.4f}  (+{:.4f})  possessions {}".format(
            name, s["spell_f1_mean"], s["spell_f1_mean"] - base["spell_f1_mean"], s["n_pred_spells"]))

    # -- 3. missed possessions: was the evidence there? ------------------------------------------
    print("\n3. Missed true possessions: what did the classifier (before the decoder) say?")
    rows = []
    for k, (ll, y, blocks) in enumerate(items):
        gs, ps, mg, mp, c = per_match[k]
        for i, g in enumerate(gs):
            if c[i][0] != "missed":
                continue
            seg = ll[g[0]:g[1]]
            arg = np.argmax(seg, axis=1)
            pr = np.exp(seg)
            rows.append(((g[1] - g[0]) / FPS, np.mean(arg == g[2]), np.mean(arg == 1 - g[2]),
                         np.mean(arg == D), np.mean(pr[:, g[2]])))
    r = np.array(rows)
    print("   n={}  median length {:.1f}s".format(len(r), np.median(r[:, 0])))
    print("   classifier argmax on those frames: right team {:.0%}, other team {:.0%}, dead {:.0%}"
          "  (mean P(right team) {:.2f})".format(*r[:, 1:5].mean(axis=0)))
    for lo, hi in ((0, 4), (4, 10), (10, 1e9)):
        m = (r[:, 0] >= lo) & (r[:, 0] < hi)
        if m.any():
            print("   {:>4.0f}-{:<4s} n={:3d}: right team {:.0%}, other {:.0%}, dead {:.0%};"
                  " share with right-team majority {:.0%}".format(
                      lo, "inf" if hi > 1e8 else "{:.0f}s".format(hi), m.sum(),
                      *r[m, 1:4].mean(axis=0), np.mean(r[m, 1] > 0.5)))


if __name__ == "__main__" and "--ball" not in sys.argv:
    main()


def ball_check():
    """4. Missed possessions vs matched ones: how often is the (degraded) ball tracked, and does
    the classifier get the team right on frames where it is?"""
    import diag_liveplay as me  # noqa
    with open(os.path.join(REPO, "models", "decoder_v2.json")) as fh:
        dp = json.load(fh)["decoder"]
    with open(os.path.join(config.CACHE_DIR, "degrader_params.json")) as fh:
        params = json.load(fh)
    oof = np.load(os.path.join(config.CACHE_DIR, "oof_v2.npz"))
    acc = {k: {"n": 0, "present": [], "right_present": [], "right_absent": [], "near": []}
           for k in ("matched", "missed", "merged")}
    for mid in io_idsse.MATCH_IDS:
        Xf, y, g, names = model.copy_features(mid, params, *model.EVAL_COPY, features.FEATURE_VERSION)
        col = {n: j for j, n in enumerate(names)}
        blocks = blocks_from(g)
        ll = oof[mid].astype(np.float64)
        pred = cv.decode_ll(ll, blocks, dp, fill=X)
        gs, ps = spells(y, blocks), spells(pred, blocks)
        mg, _ = greedy(gs, ps)
        c = classify(gs, ps, mg)
        pres = Xf[:, col["ball_present"]] > 0
        near = Xf[:, col["near_dist"]]
        arg = np.argmax(ll, axis=1)
        for i, sp in enumerate(gs):
            k = c[i][0]
            if k not in acc:
                continue
            sl = slice(sp[0], sp[1])
            p, r = pres[sl], arg[sl] == sp[2]
            acc[k]["n"] += 1
            acc[k]["present"].append(p.mean())
            if p.any():
                acc[k]["right_present"].append(r[p].mean())
                acc[k]["near"].append(np.nanmedian(near[sl][p]) if np.isfinite(near[sl][p]).any() else np.nan)
            if (~p).any():
                acc[k]["right_absent"].append(r[~p].mean())
    print("\n4. Ball tracking inside true possessions (degraded IDSSE, as on BAR-ATH)")
    for k, a in acc.items():
        print("   {:8s} n={:4d}  ball tracked {:4.0%} of frames | classifier right team: "
              "ball tracked {:4.0%}, ball missing {:4.0%} | median nearest-player dist {:.1f} m".format(
                  k, a["n"], np.mean(a["present"]), np.mean(a["right_present"]),
                  np.mean(a["right_absent"]), np.nanmedian(a["near"])))


if __name__ == "__main__" and "--ball" in sys.argv:
    ball_check()
