"""Emission-gated short possessions in v5's decoder (decoder only, on v5's out-of-fold emissions).

    python scripts/gate_short_spells.py [--jobs 2]

v5 = mean log-probs of nn_gru05 + nn_gru05s1 + nn_tcn05. The touch-aware path of the decoder
(cv.TOUCH_GRID: chg_w, chg_off, fast_thr, fast_steps) is fed a per-frame gate signal g instead of
the change head:
  margin : g = |log P(H) - log P(A)| - 3              (the emission's own team confidence)
  margin+touch : max(margin g, 3 within +-5 frames of a detected ball touch, else -10)
Where the step max of g >= fast_thr, a switch may start a possession of only fast_steps steps;
chg_w * max(g - chg_off, 0) is taken off the H <-> A switch cost.

Fully nested per held-out match i: start from v5's decoder for that eval bridge plus a closed gate
(chg_w 0, fast_thr 99), tune every decoder parameter including the gate on the six other matches
(2 sweeps), decode match i. Compared match by match with v5's nested predictions at the same eval
bridge. Writes only out/gate_* (resumable per fold).
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
from possession import config, cv, features, io_idsse, model, organisers  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH, blocks_of  # noqa: E402

MEMBERS = ["nn_gru05", "nn_gru05s1", "nn_tcn05"]
ENS = "ens_" + "+".join(MEMBERS)
BRIDGES = {"e2": 2.0, "e1": 1.0, "e0.5": 0.5}
VARIANTS = ("margin", "margin+touch")
OUT = os.path.join(REPO, "out")
IDS = io_idsse.MATCH_IDS
SHORT = 75
MARGIN_OFF = 3.0
TOUCH_WIN = 5


def log(s):
    line = "{} {}".format(time.strftime("%H:%M:%S"), s)
    print(line, flush=True)
    with open(os.path.join(OUT, "gate_progress.log"), "a") as h:
        h.write(line + "\n")


def touch_masks():
    path = os.path.join(OUT, "gate_touch.npz")
    if os.path.isfile(path):
        return dict(np.load(path))
    with open(PARAMS_PATH) as h:
        params = json.load(h)
    res = {}
    for mid in IDS:
        deg = model.make_copy(mid, params, *model.EVAL_COPY)
        m = np.zeros(deg.n_frames, bool)
        for t, *_ in features.detect_touches(deg, deg.fps):
            m[max(0, t - TOUCH_WIN): t + TOUCH_WIN + 1] = True
        res[mid] = m
    np.savez(path, **res)
    return res


def build_items(variant, btag):
    with open(PARAMS_PATH) as h:
        params = json.load(h)
    zs = [np.load(os.path.join(config.CACHE_DIR, "oof_{}.npz".format(m))) for m in MEMBERS]
    tm = touch_masks() if "touch" in variant else None
    items = []
    for mid in IDS:
        _, _, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, 2)
        y = model.copy_labels(mid, model.EVAL_COPY[1], BRIDGES[btag])
        ll = np.mean([z[mid].astype(np.float64) for z in zs], axis=0)
        sig = np.abs(ll[:, 0] - ll[:, 1]) - MARGIN_OFF
        if tm is not None:
            sig = np.maximum(sig, np.where(tm[mid], 3.0, -10.0))
        items.append((ll, y, blocks_of(g), sig))
    return items


def _fold_path(variant, btag, i):
    return os.path.join(OUT, "gate_fold_{}_{}_{}.npy".format(variant, btag, i))


def run(task):
    variant, btag = task
    items = build_items(variant, btag)
    with open(os.path.join(config.CACHE_DIR, "cv_{}_{}_decoder.json".format(ENS, btag))) as h:
        start = dict(json.load(h)["decoder"], chg_w=0.0, chg_off=0.0, fast_thr=99.0, fast_steps=4)
    chosen = {}
    for i, mid in enumerate(IDS):
        fp = _fold_path(variant, btag, i)
        if os.path.isfile(fp) and os.path.isfile(fp + ".json"):
            chosen[mid] = json.load(open(fp + ".json"))
            continue
        t0 = time.time()
        rest = [it for j, it in enumerate(items) if j != i]
        dpi, _ = cv.tune_decoder(rest, start=start, sweeps=2)
        pred = cv.decode_ll(items[i][0], items[i][2], dpi, fill=3, change=items[i][3])
        np.save(fp, pred)
        json.dump(dpi, open(fp + ".json", "w"))
        chosen[mid] = dpi
        log("{} {} fold {} {}: {:.0f}s  {}".format(variant, btag, i, mid, time.time() - t0,
                                                   json.dumps(dpi)))
    return variant, btag, chosen


def short_pr(items, preds):
    pm = organisers.pm()
    ng = mg = npd = mp = 0
    for (_, y, blocks, _), pr in zip(items, preds):
        for s, e in blocks:
            def sp(c):
                cut = np.flatnonzero(np.diff(c[s:e])) + 1
                a, b = np.r_[0, cut], np.r_[cut, e - s]
                return [(s + x, s + z, int(c[s + x])) for x, z in zip(a, b) if c[s + x] in (0, 1)]
            gs, ps = sp(y), sp(pr)
            ov = pm.spell_overlaps(gs, ps)
            fg = pm.matched_gt_flags(ov, len(gs), 0.3)
            used, taken = set(), set()
            for iou, gi, pj in ov:                       # same greedy matching, pred side
                if iou < 0.3:
                    break
                if gi in used or pj in taken:
                    continue
                used.add(gi)
                taken.add(pj)
            for k, (a, b, _) in enumerate(gs):
                if b - a <= SHORT:
                    ng += 1
                    mg += int(fg[k])
            for k, (a, b, _) in enumerate(ps):
                if b - a <= SHORT:
                    npd += 1
                    mp += int(k in taken)
    return dict(short_gt=ng, short_recall=mg / max(ng, 1), short_pred=npd,
                short_precision=mp / max(npd, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    touch_masks()
    tasks = [(v, b) for v in VARIANTS for b in BRIDGES]
    with ProcessPoolExecutor(args.jobs) as ex:
        done = list(ex.map(run, tasks))
    rep, res = [], {}
    for variant, btag, chosen in done:
        items = build_items(variant, btag)
        preds = [np.load(_fold_path(variant, btag, i)) for i in range(len(IDS))]
        zb = np.load(os.path.join(config.CACHE_DIR, "nested_preds_{}_{}.npz".format(ENS, btag)))
        base = [zb[m] for m in IDS]
        s_new = cv.pooled_score(items, preds=preds)
        s_old = cv.pooled_score(items, preds=base)
        per_new = [cv.pooled_score([it], preds=[p])["spell_f1_mean"] for it, p in zip(items, preds)]
        per_old = [cv.pooled_score([it], preds=[p])["spell_f1_mean"] for it, p in zip(items, base)]
        d = np.array(per_new) - np.array(per_old)
        sp_new, sp_old = short_pr(items, preds), short_pr(items, base)
        res[(variant + "|" + btag)] = dict(nested=s_new, v5=s_old, per_diff=d.tolist(),
                                           short_new=sp_new, short_v5=sp_old, decoders=chosen)
        rep.append("{:<13} eval {:<4}: v5 {:.4f} -> {:.4f} ({:+.4f})  wins {}/7  [{}]  "
                   "possessions {} -> {}  short recall {:.3f} -> {:.3f}  short precision {:.3f} -> "
                   "{:.3f}  (short pred {} -> {})".format(
                       variant, btag, s_old["spell_f1_mean"], s_new["spell_f1_mean"],
                       s_new["spell_f1_mean"] - s_old["spell_f1_mean"], int((d > 0).sum()),
                       " ".join("{:+.3f}".format(x) for x in d), s_old["n_pred_spells"],
                       s_new["n_pred_spells"], sp_old["short_recall"], sp_new["short_recall"],
                       sp_old["short_precision"], sp_new["short_precision"], sp_old["short_pred"],
                       sp_new["short_pred"]))
    with open(os.path.join(OUT, "gate_results.json"), "w") as h:
        json.dump(res, h, indent=1, default=float)
    with open(os.path.join(OUT, "gate_report.txt"), "w") as h:
        h.write("\n".join(rep) + "\n")
    print("\n".join(rep))


if __name__ == "__main__":
    main()
