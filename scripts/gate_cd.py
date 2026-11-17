"""Two decoder-only ideas on v5.2 (survey of public repos, pass 2), nested, mean spellF1 over 2/1/0.5 s.

    python scripts/gate_cd.py [--only C|D]

Base: v5.2 (members gru05 + gru05s1 + tcn05, Law 9, log-likelihoods rounded to 1e-3, decoder
cache/cv_v52_mean_decoder.json); reference = its nested predictions
cache/nested_preds_v52_mean_e2.npz (the same predictions serve all three bridges).

C. Snap stoppage starts to the ball's exit. On the evaluation copy, an exit is the last frame with
   the tracked ball inside the pitch before a Law-9 out-of-play run (rules.law9_mask). A decoded D
   segment that starts 1..k s after an exit (and follows a team) is moved back to start at exit+1.
   k in {0 (off), 1, 2, 3} s chosen per held-out match on the six others.
D. Dead states that carry the restarting team. 4-state Viterbi H, A, D_H, D_A (both D states emit
   D's log-likelihood; D_x additionally gets +w where the restarting team is x = the team that did
   NOT make the latest detected touch, features.detect_touches on the evaluation copy). Leaving
   D_x for x costs leave_d, for the other team leave_d + pen. D_H <-> D_A is blocked. Other settings
   from v5.2. (pen, w) tuned per held-out match on the six others over a fixed grid, (0, 0) = v5.2.
   Simulated Smart Tagging teams are not used (IDSSE tag sim keeps no team).
Writes only out/gate_cd_*.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config, cv, decode, features, model, rules  # noqa: E402
from scripts import mean_decoder as MD  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH  # noqa: E402

OUT = os.path.join(REPO, "out")
IDS = MD.io_idsse.MATCH_IDS
KS = (0, 1, 2, 3)
PENS = (0.0, 0.5, 1.5, 3.0, 5.0)
WS = (0.0, 0.25, 0.5, 1.0, 2.0)


def log(s):
    line = "{} {}".format(time.strftime("%H:%M:%S"), s)
    print(line, flush=True)
    with open(os.path.join(OUT, "gate_cd_progress.log"), "a") as h:
        h.write(line + "\n")


def per_match_mean(item, pred):
    return MD.mean_score([item], [pred])[0]


def table(items, name, preds, ref=None):
    m, per = MD.mean_score(items, preds)
    pm = np.array([per_match_mean(it, p) for it, p in zip(items, preds)])
    line = "{:<34} mean {:.4f}  2s {:.4f}  1s {:.4f}  0.5s {:.4f}".format(name, m, *per)
    if ref is not None:
        d = pm - ref[1]
        line += "   vs v5.2 {:+.4f}  wins {}/7 [{}]".format(
            m - ref[0], int((d > 0).sum()), " ".join("{:+.3f}".format(x) for x in d))
    return line, (m, pm)


def eval_side_info():
    path = os.path.join(OUT, "gate_cd_side.npz")
    if os.path.isfile(path):
        z = np.load(path)
        return ({m: z["exit_" + m] for m in IDS}, {m: z["restart_" + m] for m in IDS})
    with open(PARAMS_PATH) as h:
        params = json.load(h)
    exits, restart, save = {}, {}, {}
    for mid in IDS:
        m = model.make_copy(mid, params, *model.EVAL_COPY)
        L, W = m.pitch
        b = m.ball_xy
        inside = (np.abs(b[:, 0]) <= L / 2) & (np.abs(b[:, 1]) <= W / 2)
        mask = np.load(os.path.join(config.CACHE_DIR, "law9_{}_9.npy".format(mid)))
        on = np.flatnonzero(np.diff(np.r_[False, mask].astype(np.int8)) == 1)
        ex = []
        for a in on:
            idx = np.flatnonzero(inside[max(0, a - 50):a])
            if len(idx):
                ex.append(max(0, a - 50) + int(idx[-1]))
        exits[mid] = np.array(ex, dtype=np.int64)
        r = np.full(m.n_frames, -1, np.int8)            # restarting team: not the last toucher
        last = -1
        tch = {t: (0 if s > 0 else 1) for t, s, *_ in features.detect_touches(m, m.fps)}
        for t in range(m.n_frames):
            if t in tch:
                last = tch[t]
            r[t] = -1 if last < 0 else 1 - last
        restart[mid] = r
        save["exit_" + mid], save["restart_" + mid] = exits[mid], r
        log("side info {}: {} exits".format(mid, len(ex)))
    np.savez(path, **save)
    return exits, restart


def snap(pred, exits, k):
    if k <= 0:
        return pred
    out = pred.copy()
    starts = np.flatnonzero((pred[1:] == 2) & np.isin(pred[:-1], (0, 1))) + 1
    for s in starts:
        cand = exits[(exits < s) & (s - exits <= k * 25)]
        if len(cand):
            x = int(cand.max())
            if np.all(np.isin(pred[x + 1:s], (0, 1))):
                out[x + 1:s] = 2
    return out


def run_c(items, base, ref, exits):
    cand = {k: [snap(p, exits[m], k) for p, m in zip(base, IDS)] for k in KS}
    lines = [table(items, "C fixed k = {} s".format(k), cand[k], ref)[0] for k in KS[1:]]
    nested, chosen = [], []
    for i in range(len(IDS)):
        rest = [j for j in range(len(IDS)) if j != i]
        k = max(KS, key=lambda k: MD.mean_score([items[j] for j in rest], [cand[k][j] for j in rest])[0])
        chosen.append(k)
        nested.append(cand[k][i])
    lines.append(table(items, "C nested k {}".format(chosen), nested, ref)[0])
    return lines


def decode4(ll, blocks, dp, r, pen, w):
    ll4 = np.column_stack([ll[:, 0], ll[:, 1], ll[:, 2] + w * (r == 0), ll[:, 2] + w * (r == 1)])
    ll4[:, 2:] += dp["bias_d"]
    t, e, lv, B = dp["team_cost"], dp["enter_d"], dp["leave_d"], cv.BLOCKED
    cost = ((0.0, t, e, e), (t, 0.0, e, e), (lv, lv + pen, 0.0, B), (lv + pen, lv, B, 0.0))
    out = decode.decode_blocks(ll4, blocks, cv.STEP, dp["step_weight"],
                               (dp["min_team"], dp["min_team"], dp["min_d"], dp["min_d"]), cost,
                               fill=-1)
    outside = out == -1
    out[out == 3] = 2
    out[outside] = 3
    return out


def run_d(items, ref, restart, dp):
    grid = [(p, w) for p in PENS for w in WS]
    preds = {}
    t0 = time.time()
    for g in grid:
        preds[g] = [decode4(it[0], it[2], dp, restart[m], *g) for it, m in zip(items, IDS)]
    log("D: decoded {} grid points ({:.0f}s)".format(len(grid), time.time() - t0))
    lines = []
    sanity = table(items, "D (pen 0, w 0) = v5.2 check", preds[(0.0, 0.0)], ref)[0]
    lines.append(sanity)
    _, ref00 = table(items, "", preds[(0.0, 0.0)])
    nested, chosen = [], []
    for i in range(len(IDS)):
        rest = [j for j in range(len(IDS)) if j != i]
        g = max(grid, key=lambda g: MD.mean_score([items[j] for j in rest], [preds[g][j] for j in rest])[0])
        chosen.append(g)
        nested.append(preds[g][i])
    lines.append(table(items, "D nested (pen, w) {}".format(chosen), nested, ref)[0])
    lines.append(table(items, "D nested vs its own (0, 0) [same decoder]", nested, ref00)[0]
                 .replace("vs v5.2", "vs (0,0)"))
    best = max(grid, key=lambda g: MD.mean_score(items, preds[g])[0])
    lines.append(table(items, "D best pooled (not nested) {}".format(best), preds[best], ref)[0])
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=("C", "D"))
    args = ap.parse_args()
    items = MD.load_items(["nn_gru05", "nn_gru05s1", "nn_tcn05"], True, 0.001)
    with open(os.path.join(config.CACHE_DIR, "cv_v52_mean_decoder.json")) as h:
        dp = json.load(h)["decoder"]
    z = np.load(os.path.join(config.CACHE_DIR, "nested_preds_v52_mean_e2.npz"))
    base = [z[m] for m in IDS]
    l, ref = table(items, "v5.2 nested (reference)", base)
    lines = [l]
    exits, restart = eval_side_info()
    if args.only in (None, "C"):
        lines += run_c(items, base, ref, exits)
        log("C done")
    if args.only in (None, "D"):
        lines += run_d(items, ref, restart, dp)
        log("D done")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "gate_cd_report.txt"), "w") as h:
        h.write(txt + "\n")


if __name__ == "__main__":
    main()
