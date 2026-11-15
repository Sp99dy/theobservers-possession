"""Diagnostic: which dead balls does v2 miss, what are they worth, and do long speed windows help?

    python scripts/diag_deadball.py

IDSSE / DFL open data only (CC-BY). Every DFL restart event (free kick, throw-in, corner, goal
kick, kickoff, penalty) carries the whistle time (DecisionTimestamp) and the restart time
(EventTime), so each ground-truth dead-ball stretch can be labelled with its cause.

Uses the v2 out-of-fold emissions (cache/oof_v2.npz) on the CV evaluation corruption and the
shipped v2 decoder, i.e. exactly what the v2 CV numbers were computed from.
"""

import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config, cv, features, io_idsse, metrics, model  # noqa: E402

FPS = 25
RESTARTS = ("FreeKick", "ThrowIn", "CornerKick", "GoalKick", "KickOff", "Penalty")
D = 2


def ts(s):
    return datetime.fromisoformat(s).timestamp()


def half_clock(match_id):
    """{section: (N0, t0)}: first BALL frame number of each half and its wall-clock time."""
    path = io_idsse.local_path(match_id, "tracking")
    with open(path, "rb") as fh:
        data = fh.read()
    out = {}
    for head in io_idsse._FRAMESET.finditer(data):
        attrs = dict((k.decode(), v.decode()) for k, v in io_idsse._ATTR.findall(head.group(1)))
        if attrs.get("TeamId") != "BALL" or attrs.get("GameSection") not in io_idsse.SECTIONS:
            continue
        m = re.compile(rb'<Frame N="(\d+)" T="([^"]+)"').search(data, head.end())
        out[attrs["GameSection"]] = (int(m.group(1)), ts(m.group(2).decode()))
    return out


def restart_events(match_id):
    """[(type, whistle_time or None, restart_time, foul_before)] in wall-clock seconds."""
    root = ET.parse(io_idsse.local_path(match_id, "event")).getroot()
    out, fouls = [], []
    for ev in root.iter("Event"):
        t = ts(ev.get("EventTime"))
        for child in ev:
            if child.tag == "Foul":
                fouls.append(t)
            if child.tag in RESTARTS:
                dec = child.get("DecisionTimestamp")
                out.append([child.tag, ts(dec) if dec else None, t])
    fouls = np.array(sorted(fouls))
    for r in out:
        w = r[1] if r[1] is not None else r[2]
        r.append(bool(len(fouls) and np.any(np.abs(fouls - w) < 5.0)))
    return out


def to_index(t, clock, starts, n_frames):
    """Wall-clock time -> frame index in the concatenated match (None if outside both halves)."""
    for (section, (n0, t0)), start in zip(sorted(clock.items(), key=lambda kv: kv[1][1]), starts):
        i = start + int(round((t - t0) * FPS))
        if start <= i < n_frames:
            return i, section
    return None, None


def d_stretches(y, blocks):
    out = []
    for s, e in blocks:
        c = y[s:e]
        b = np.flatnonzero(np.diff(c)) + 1
        for a, z in zip(np.r_[0, b], np.r_[b, len(c)]):
            if c[a] == D:
                out.append((s + a, s + z))
    return out


def windows(x, past, fut):
    """Mean of x over [t-past, t) and [t, t+fut) frames (NaN-aware, truncated at the edges)."""
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    v = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
    n = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(x))
    res = []
    for lo, hi in ((np.maximum(0, i - past), i), (i, np.minimum(len(x), i + fut))):
        with np.errstate(invalid="ignore", divide="ignore"):
            res.append((v[hi] - v[lo]) / (n[hi] - n[lo]))
    return res


def auc(pos, neg):
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    allv = np.r_[pos, neg]
    ranks = np.empty(len(allv))
    order = np.argsort(allv, kind="mergesort")
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, ranks)
    ranks = (sums / cnt)[inv]
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    with open(os.path.join(REPO, "models", "decoder_v2.json")) as fh:
        dp = json.load(fh)["decoder"]
    with open(os.path.join(config.CACHE_DIR, "degrader_params.json")) as fh:
        params = json.load(fh)
    oof = np.load(os.path.join(config.CACHE_DIR, "oof_v2.npz"))

    rows, items, preds, feats = [], [], [], []
    unmatched_restarts = 0
    for mid in io_idsse.MATCH_IDS:
        X, y, g, names = model.copy_features(mid, params, *model.EVAL_COPY, features.FEATURE_VERSION)
        idx = np.flatnonzero(g)
        b = np.flatnonzero(np.diff(idx) > 1)
        blocks = [(int(s), int(e) + 1) for s, e in zip(np.r_[idx[0], idx[b + 1]], np.r_[idx[b], idx[-1]])]
        ll = oof[mid].astype(np.float64)
        pred = cv.decode_ll(ll, blocks, dp, fill=3)
        items.append((ll, y, blocks))
        preds.append(pred)

        clock = half_clock(mid)
        starts = [s for s, _ in blocks]
        evs = []
        for typ, whistle, restart, foul in restart_events(mid):
            ri, _ = to_index(restart, clock, starts, len(y))
            wi = to_index(whistle, clock, starts, len(y))[0] if whistle is not None else None
            if ri is not None:
                evs.append((typ, wi, ri, foul))

        col = {n: j for j, n in enumerate(names)}
        speed = np.nanmean(np.stack([X[:, col["home_speed"]], X[:, col["away_speed"]]]), axis=0)
        f = {"centred ±5 s (in v2)": X[:, col["player_speed_m5"]].astype(np.float64)}
        p5, f5 = windows(speed, 5 * FPS, 5 * FPS)
        p10, f10 = windows(speed, 10 * FPS, 10 * FPS)
        _, f20 = windows(speed, 1, 20 * FPS)
        f["next 5 s"], f["next 10 s"], f["next 20 s"] = f5, f10, f20
        f["drop: prev 5 s − next 10 s"] = p5 - f10
        f["ratio: next 10 s / prev 5 s"] = f10 / p5
        bspd = X[:, col["ball_speed"]].astype(np.float64)
        feats.append((y, pred, f, g))

        for s, e in d_stretches(y, blocks):
            cand = [ev for ev in evs if s - 3 * FPS <= ev[2] <= e + 3 * FPS]
            if cand:
                ev = min(cand, key=lambda ev: abs(ev[2] - e))
                typ = ev[0] + (" (after foul)" if ev[0] == "FreeKick" and ev[3] else "")
                whistle_lag = None if ev[1] is None else (s - ev[1]) / FPS
            else:
                typ, whistle_lag = "no restart event", None
                unmatched_restarts += 1
            seg = pred[s:e]
            dpos = np.flatnonzero(seg == D)
            onset = (dpos[0] if len(dpos) else e - s) / FPS
            rows.append((mid, typ, s, e, (e - s) / FPS, float(np.mean(seg == D)), onset, whistle_lag))

    # -- 1. misses by cause --------------------------------------------------------------------
    print("\n1. Ground-truth dead-ball stretches by cause (7 IDSSE matches, v2 OOF + shipped decoder)")
    print("{:26s} {:>5s} {:>8s} {:>9s} {:>10s} {:>11s} {:>12s}".format(
        "cause", "n", "mean s", "D recall", "onset lag", "missed<20%", "|whistle-GT|"))
    types = sorted(set(r[1] for r in rows), key=lambda t: -sum(r[1] == t for r in rows))
    for t in types:
        rs = [r for r in rows if r[1] == t]
        L = np.array([r[4] for r in rs])
        rec = np.sum([r[5] * r[4] for r in rs]) / L.sum()
        lag = np.median([r[6] for r in rs])
        miss = np.mean([r[5] < 0.2 for r in rs])
        wl = [abs(r[7]) for r in rs if r[7] is not None]
        print("{:26s} {:5d} {:8.1f} {:9.2f} {:9.1f}s {:11.0%} {:>12s}".format(
            t, len(rs), L.mean(), rec, lag, miss, "{:.1f}s".format(np.median(wl)) if wl else "–"))

    # -- 2. oracle gains -----------------------------------------------------------------------
    base = cv.pooled_score(items, preds=preds)["spell_f1_mean"]
    print("\n2. Oracle: spellF1 if the dead balls of one cause were detected perfectly "
          "(base {:.4f})".format(base))
    for t in types + ["ALL"]:
        fixed = []
        for (mid, pred) in zip(io_idsse.MATCH_IDS, preds):
            p = pred.copy()
            for r in rows:
                if r[0] == mid and (t == "ALL" or r[1] == t):
                    p[r[2]:r[3]] = D
            fixed.append(p)
        s = cv.pooled_score(items, preds=fixed)
        print("   {:26s} spellF1 {:.4f}  (+{:.4f})  possessions {}".format(
            t, s["spell_f1_mean"], s["spell_f1_mean"] - base, s["n_pred_spells"]))

    # -- 3. do long speed windows separate the missed foul stoppages from live play? ------------
    print("\n3. AUC of mean player speed (lower = dead) — positives: frames of foul free-kick "
          "stoppages the decoder called alive; negatives: live frames")
    pos_mask, neg_mask = [], []
    for mid, (y, pred, f, g) in zip(io_idsse.MATCH_IDS, feats):
        pm = np.zeros(len(y), bool)
        for r in rows:
            if r[0] == mid and r[1] == "FreeKick (after foul)":
                pm[r[2]:r[3]] = True
        pos_mask.append(pm & (pred != D))
        neg_mask.append(g & (y != D) & (y != 3))
    print("   positives {} frames, negatives {} frames".format(
        sum(int(p.sum()) for p in pos_mask), sum(int(n.sum()) for n in neg_mask)))
    for name in feats[0][2]:
        pos = np.concatenate([f[name][p] for (_, _, f, _), p in zip(feats, pos_mask)])
        neg = np.concatenate([f[name][n] for (_, _, f, _), n in zip(feats, neg_mask)])
        a = auc(pos, neg)
        if not name.startswith("drop"):
            a = 1 - a    # low speed means dead; report AUC as "how well it separates", >= 0.5
        print("   {:28s} AUC {:.3f}   median dead {:.2f} vs live {:.2f}".format(
            name, a, np.nanmedian(pos), np.nanmedian(neg)))
    print("\nrestart-less dead stretches: {}".format(unmatched_restarts))

    # -- 4. where do the DFL dead-ball labels start relative to the whistle? ---------------------
    print("\n4. GT dead-ball start minus whistle (DecisionTimestamp), seconds: + = label starts later")
    for t in types:
        wl = np.array([r[7] for r in rows if r[1] == t and r[7] is not None])
        if len(wl):
            q = np.percentile(wl, [10, 25, 50, 75, 90])
            print("   {:26s} n={:3d}  p10 {:+.1f}  p25 {:+.1f}  median {:+.1f}  p75 {:+.1f}  "
                  "p90 {:+.1f}".format(t, len(wl), *q))


if __name__ == "__main__":
    main()
