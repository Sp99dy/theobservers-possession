"""Export the public 5-minute window for the local match viewer (viewer/index.html).

    python scripts/viewer_export.py                  # -> out/viewer/match.json (real data, LOCAL ONLY)
    python scripts/viewer_export.py --demo           # -> out/viewer/demo.json (synthetic, shareable)
    python scripts/viewer_export.py --extra v3=out/v3.csv   # add more submissions to compare

The real export contains tracking extracts and ground truth. It is written to out/ (gitignored) and
must stay on this machine: never commit it, upload it, or paste it into a cloud tool.

What it holds, per frame of the window: player and ball positions (metres, raw orientation),
the main model's internals (v4 if trained, else v2: classifier posterior, evidence after the kickoff/tag layers, the
nearest-player distances it looks at), the decoded labels of every compared submission, and the
public ground truth. Per submission: the organisers' metrics and the possession-spell matching.
"""

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config as C  # noqa: E402

OUT = os.path.join(C.OUT_DIR, "viewer")
LETTERS = "HADX"


def softmax(ll):
    z = ll - ll.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def r2(a, digits=2):
    """Rounded list with NaN/inf -> None (JSON null)."""
    a = np.asarray(a, dtype=np.float64)
    out = np.round(a, digits).tolist()
    bad = ~np.isfinite(a)
    if bad.any():
        flat = np.asarray(out, dtype=object).ravel()
        flat[bad.ravel()] = None
        out = flat.reshape(a.shape).tolist()
    return out


# -- spell matching (same rules as the organisers' scorer, but keeping the pairs) ----------------

def spells(codes, start, stop):
    out = []
    c = codes[start:stop]
    b = np.flatnonzero(c[1:] != c[:-1]) + 1
    bounds = np.r_[0, b, len(c)]
    for a, z in zip(bounds[:-1], bounds[1:]):
        if c[a] in (0, 1):
            out.append((int(start + a), int(start + z), int(c[a])))
    return out


def fill_x(codes, start, stop):
    w = codes[start:stop].copy()
    known = np.flatnonzero(w != 3)
    if len(known):
        carry = np.maximum.accumulate(np.where(w != 3, np.arange(len(w)), -1))
        carry[carry < 0] = known[0]
        w = w[carry]
    out = codes.copy()
    out[start:stop] = w
    return out


def match_pairs(gt_sp, pr_sp, ious=(0.3, 0.4, 0.5, 0.6, 0.7)):
    overlaps = []
    for i, (gs, ge, gc) in enumerate(gt_sp):
        for j, (ps, pe, pc) in enumerate(pr_sp):
            if pc != gc or pe <= gs or ge <= ps:
                continue
            inter = min(ge, pe) - max(gs, ps)
            union = max(ge, pe) - min(gs, ps)
            overlaps.append((inter / union, i, j))
    overlaps.sort(key=lambda t: (-t[0], t[1], t[2]))
    best_for_gt = {}
    for iou, i, j in overlaps:
        best_for_gt.setdefault(i, (j, iou))
    per = {}
    for thr in ious:
        used_g, used_p, pairs = set(), set(), []
        for iou, i, j in overlaps:
            if iou < thr:
                break
            if i in used_g or j in used_p:
                continue
            used_g.add(i)
            used_p.add(j)
            pairs.append([i, j, round(iou, 4)])
        per[str(thr)] = pairs
    return per, {str(i): [j, round(v, 4)] for i, (j, v) in best_for_gt.items()}


# -- real export ---------------------------------------------------------------------------------

def real(extra):
    from possession import (cv, features, io_atd, io_tagging, kickoff, metrics, model, organisers,
                            rules, tagging_prior)
    from possession import v2, v4
    from possession.canonical import X, orient

    pm = organisers.pm()
    gt = metrics.public_gt_codes()
    runs = pm.valid_runs(gt)
    assert len(runs) == 1, runs
    w0, w1 = runs[0]

    print("loading match ...", flush=True)
    match = io_atd.load_match()
    tags = io_tagging.load()

    # v2.predict / v4.predict, step by step, keeping the internals. The main method ("Our answer")
    # is v4 when its model is trained, else v2; the other one is kept as a comparison.
    m = orient(match)
    base = features.compute(m)
    kos = kickoff.detect(base, m.half, m.fps)
    pre = kickoff.pre_kickoff_spans(kos, m.half_blocks(), m.fps)
    dz, dy = rules.kickoff_evidence(m.n_frames, kos, pre, m.fps)
    tz, ty = tagging_prior.evidence(m.n_frames, tags, kos, m.fps)
    blocks = [(s, e) for _, s, e in m.half_blocks()]

    def pipeline(mod, version):
        mdl = model.load(mod.MODEL_PATH)
        with open(mod.DECODER_PATH) as handle:
            dec = json.load(handle)["decoder"]
        Xf, _ = features.design_matrix(m, base, version=version)
        clf = model.emission_ll(mdl, Xf)
        sc = v2.evidence_scale(clf, base)
        ko = sc * rules.emissions(dz, dy)
        tg = sc * rules.emissions(tz, ty)
        tot = clf + ko + tg
        return dict(ll_clf=clf, ll_ko=ko, ll_tag=tg, ll_all=tot, scale=sc, decoder=dec,
                    codes=cv.decode_ll(tot, blocks, dec, fill=X))

    runs_ = []
    if os.path.exists(v4.MODEL_PATH) and os.path.exists(v4.DECODER_PATH):
        runs_.append(("v4", v4, v4.FEATURE_VERSION))
    runs_.append(("v2 (shipped)", v2, features.FEATURE_VERSION))
    subs, piped = {}, {}
    for name, mod, version in runs_:
        print("running the {} pipeline (features, classifier, layers, decoder) ...".format(name),
              flush=True)
        piped[name] = pipeline(mod, version)
        subs[name] = piped[name]["codes"]
    main_name = runs_[0][0]
    main = piped[main_name]
    ll_clf, ll_ko, ll_tag, ll_all = main["ll_clf"], main["ll_ko"], main["ll_tag"], main["ll_all"]
    scale, decoder = main["scale"], main["decoder"]

    shipped_path = os.path.join(C.OUT_DIR, C.SUBMISSION_NAME)
    if os.path.exists(shipped_path):
        shipped = metrics.read_submission_codes(shipped_path)
        same = bool(np.array_equal(shipped[w0:w1], subs["v2 (shipped)"][w0:w1]))
        print("v2 recomputed == out/{} in the window: {}".format(C.SUBMISSION_NAME, same))
    for name, fn in (("v1", "v1.csv"), ("baseline50", "baseline_smooth50.csv")):
        p = os.path.join(C.OUT_DIR, fn)
        if os.path.exists(p):
            subs[name] = metrics.read_submission_codes(p)
    for item in extra:
        name, path = item.split("=", 1)
        subs[name] = metrics.read_submission_codes(path)

    sl = slice(w0, w1)
    gt_sp = spells(gt, w0, w1)
    methods = []
    for name, codes in subs.items():
        res = metrics.evaluate(gt, codes)
        filled = fill_x(codes, w0, w1)
        pr_sp = spells(filled, w0, w1)
        per, best = match_pairs(gt_sp, pr_sp)
        methods.append({
            "name": name,
            "labels": "".join(LETTERS[c] for c in codes[sl]),
            "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer))
                            else v) for k, v in metrics.summary(res).items()},
            "spell_f1_per_iou": {str(k): round(float(v["f1"]), 4)
                                 for k, v in res["spell_f1"].items()},
            "spells": [[s - w0, e - w0, c] for s, e, c in pr_sp],
            "pairs": {k: v for k, v in per.items()},
            "best_for_gt": best,
        })
        print("{:14s} {}".format(name, metrics.format_summary(metrics.summary(res))))

    # Feature-version-2 signals (touch detection, slowdown): inputs of v4; v2 does not see them.
    touches = [[t - w0, 0 if k > 0 else 1, round(d, 1), round(s, 1)]
               for t, k, d, s in features.detect_touches(m, m.fps) if w0 <= t < w1]
    spd = features.speed_series(m, m.fps)
    n3 = int(round(3 * m.fps))
    speed = {"all": r2(spd["all"][sl], 2), "near": r2(spd["near"][sl], 2)}
    for name in ("all", "near", "slowshare"):
        speed[name + "_next3"] = r2(features._causal_mean(spd[name], 0, n3)[0][sl], 2)
    speed["near_prev3"] = r2(features._causal_mean(spd["near"], -n3, 0)[0][sl], 2)
    print("touches in the window: {} ({:.1f}/min)".format(len(touches), len(touches) / ((w1 - w0) / m.fps / 60)))

    team = match.team_of_slot.astype(int).tolist()
    pxy = match.players_xy[sl]
    bxy = match.ball_xy[sl]
    data = {
        "kind": "real",
        "match": match.name,
        "fps": match.fps,
        "pitch": list(match.pitch),
        "first_frame": int(w0),
        "n": int(w1 - w0),
        "team_of_slot": team,
        "attack_sign": {str(k): int(v) for k, v in m.attack_sign.items()},
        "players": r2(pxy, 1),          # [frame][slot][x,y]
        "ball": r2(bxy, 1),             # [frame][x,y]
        "gt": "".join(LETTERS[c] for c in gt[sl]),
        "gt_spells": [[s - w0, e - w0, c] for s, e, c in gt_sp],
        "p_clf": r2(np.exp(ll_clf[sl]), 3),            # classifier posterior P(H,A,D | frame)
        "p_all": r2(softmax(ll_all[sl]), 3),           # after kickoff + Smart Tagging layers
        "ev_ko": r2(ll_ko[sl], 2),
        "ev_tag": r2(ll_tag[sl], 2),
        "feat": {
            "d_home": r2(np.where(np.isfinite(base["d_home"][sl]), base["d_home"][sl], np.nan), 1),
            "d_away": r2(np.where(np.isfinite(base["d_away"][sl]), base["d_away"][sl], np.nan), 1),
            "ball_in": r2(base["ball_in"][sl], 0),
            "ball_out": r2(base["ball_out"][sl], 0),
            "player_speed": r2(base["player_speed"][sl], 2),
        },
        "kickoffs": [[k.start - w0, k.end - w0, int(k.kicking_team)] for k in kos
                     if k.end > w0 and k.start < w1],
        "tags": [[max(t.start, w0) - w0, min(t.end, w1) - w0, t.code,
                  None if t.team is None else int(t.team)]
                 for t in tags if t.end > w0 and t.start < w1],
        "touches": touches,             # [frame, team 0/1, |dv| m/s, ball speed after m/s]
        "speed": speed,                 # m/s; *_next3 / *_prev3 = mean over the next / previous 3 s
        "slow_walk": features.SLOW_WALK,
        "near_k": features.SLOW_NEAR_K,
        "main_method": main_name,       # whose internals p_clf / p_all / ev_* are
        "main_feature_version": 2 if main_name == "v4" else 1,
        "decoder": decoder,
        "evidence_scale": scale,
        "methods": methods,
    }
    return data, "match.json"


# -- synthetic demo (no match data at all; safe to screenshot or share) -----------------------------

def demo():
    rng = np.random.default_rng(7)
    fps, n, w0 = 25, 7500, 11185
    L, W = 105.0, 68.0
    # Ground truth: alternating possessions with dead-ball breaks.
    gt = np.empty(n, dtype=np.int8)
    t, team = 0, 0
    while t < n:
        d = int(rng.integers(5, 30) * fps)
        gt[t:t + d] = team
        t += d
        if rng.random() < 0.3:
            k = int(rng.integers(3, 15) * fps)
            gt[t:t + k] = 2
            t += k
        team = 1 - team
    gt = gt[:n]
    # A noisy "prediction": shifted boundaries and an occasional split.
    pred = np.roll(gt, int(rng.integers(5, 40)))
    for _ in range(4):
        a = int(rng.integers(0, n - 200))
        pred[a:a + int(rng.integers(30, 150))] = 1 - pred[a] if pred[a] < 2 else 2
    base = gt.copy()
    for _ in range(60):
        a = int(rng.integers(0, n - 50))
        base[a:a + int(rng.integers(10, 50))] = int(rng.integers(0, 3))

    ball = np.cumsum(rng.normal(0, 0.4, (n, 2)), axis=0)
    ball = np.stack([np.clip(ball[:, 0] % 100 - 50, -52, 52), np.clip(ball[:, 1] % 60 - 30, -33, 33)],
                    axis=1)
    home = np.array([[-45, 0], [-30, -20], [-32, -6], [-32, 6], [-30, 20], [-15, -15], [-15, 0],
                     [-15, 15], [-3, -20], [-1, 0], [-3, 20]], float)
    away = -home
    players = np.zeros((n, 22, 2))
    for i in range(n):
        shift = ball[i] * np.array([0.35, 0.25])
        players[i, :11] = home + shift + rng.normal(0, 0.3, (11, 2))
        players[i, 11:] = away + shift + rng.normal(0, 0.3, (11, 2))
    ball[rng.random(n) < 0.15] = np.nan

    p = np.full((n, 3), 0.1)
    p[np.arange(n), pred] = 0.8
    p = p + rng.uniform(0, 0.3, (n, 3))
    p = p / p.sum(axis=1, keepdims=True)
    d_home = np.linalg.norm(players[:, :11] - ball[:, None], axis=2).min(axis=1)
    d_away = np.linalg.norm(players[:, 11:] - ball[:, None], axis=2).min(axis=1)

    gt_sp = spells(gt, 0, n)
    methods = []
    for name, codes in (("v4", pred), ("baseline50", base)):
        pr_sp = spells(codes, 0, n)
        per, best = match_pairs(gt_sp, pr_sp)
        f1s = {}
        for k, pairs in per.items():
            tp = len(pairs)
            pr = tp / max(len(pr_sp), 1)
            rc = tp / max(len(gt_sp), 1)
            f1s[k] = round(2 * pr * rc / (pr + rc), 4) if pr + rc else 0.0
        acc = float(np.mean(codes == gt))
        methods.append({
            "name": name, "labels": "".join(LETTERS[c] for c in codes),
            "metrics": {"spell_f1_mean": float(np.mean(list(f1s.values()))), "macro_f1": acc,
                        "n_pred_spells": len(pr_sp), "n_gt_spells": len(gt_sp),
                        "mean_spell_duration_s": float(np.mean([e - s for s, e, _ in pr_sp]) / fps),
                        "gt_mean_spell_duration_s": float(np.mean([e - s for s, e, _ in gt_sp]) / fps),
                        "alive_dead_f1": acc, "team_acc_both_alive": acc},
            "spell_f1_per_iou": f1s, "spells": [[s, e, c] for s, e, c in pr_sp],
            "pairs": per, "best_for_gt": best,
        })
    data = {
        "kind": "demo", "match": "DEMO (synthetic)", "fps": fps, "pitch": [L, W],
        "first_frame": w0, "n": n, "team_of_slot": [0] * 11 + [1] * 11, "attack_sign": {"1": 1},
        "players": r2(players, 1), "ball": r2(ball, 1),
        "gt": "".join(LETTERS[c] for c in gt), "gt_spells": [[s, e, c] for s, e, c in gt_sp],
        "p_clf": r2(p, 3), "p_all": r2(p, 3),
        "ev_ko": r2(np.zeros((n, 3)), 2), "ev_tag": r2(np.zeros((n, 3)), 2),
        "feat": {"d_home": r2(d_home, 1), "d_away": r2(d_away, 1),
                 "ball_in": r2(np.isfinite(ball[:, 0]).astype(float), 0),
                 "ball_out": r2(np.zeros(n), 0), "player_speed": r2(rng.uniform(1, 4, n), 2)},
        "kickoffs": [[0, 75, 0]],
        "tags": [[a, a + 500, "attacking organisation", int(i % 2)] for i, a in
                 enumerate(range(0, n - 500, 900))],
        "main_method": "v4", "main_feature_version": 2,
        "decoder": {}, "evidence_scale": 1.0, "methods": methods,
    }
    sp_all = np.where(gt == 2, 0.8, 3.0) + rng.normal(0, 0.3, n)
    sp_near = np.where(gt == 2, 0.6, 4.0) + rng.normal(0, 0.5, n)
    nxt = lambda x: np.convolve(np.r_[x, np.full(74, x[-1])], np.ones(75) / 75, "valid")  # noqa: E731
    data["speed"] = {"all": r2(sp_all, 2), "near": r2(sp_near, 2), "all_next3": r2(nxt(sp_all), 2),
                     "near_next3": r2(nxt(sp_near), 2), "near_prev3": r2(np.roll(nxt(sp_near), 75), 2),
                     "slowshare_next3": r2(nxt((gt == 2).astype(float)), 2)}
    data["touches"] = [[int(t), int(gt[t]) if gt[t] < 2 else 0, 5.0, 10.0]
                       for t in range(0, n, 60) if gt[t] < 2]
    data["slow_walk"], data["near_k"] = 2.0, 4
    return data, "demo.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demo", action="store_true", help="write synthetic data instead")
    ap.add_argument("--extra", action="append", default=[], metavar="NAME=CSV",
                    help="another submission CSV to compare (repeatable)")
    args = ap.parse_args()
    data, name = demo() if args.demo else real(args.extra)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as handle:
        json.dump(data, handle, separators=(",", ":"))
    print("wrote {} ({:.1f} MB)".format(path, os.path.getsize(path) / 1e6))


if __name__ == "__main__":
    main()
