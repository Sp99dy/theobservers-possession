"""Broadcast replay evidence for BAR-ATH, label-free (inference only, candidate for a decision).

    python scripts/replay_evidence.py [--write]

replay[t] = the broadcast homography is invalid at broadcast frame t - LAG[half] (ballfill.LAG,
measured label-free). Broadcasters cut to replays / close-ups mostly during stoppages.
P(replay | D) and P(replay | live) are estimated by hard EM on v5.2's own decoding: fit the two
rates on the current decode, add log(P(r|D)/P(r|live)) x W to the D log-likelihood on replay
frames, re-decode with v5.2's decoder, refit. Iteration 0 is also reported with soft labels
(per-frame emission P(D)) and with ball-missing frames excluded (v5.2's D partly comes from the
missing ball). No public-window labels are read. Aggregates vs v5.2 are printed; --write saves
out/replay_run1.csv and prints its sha256.
"""

import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import ballfill, config, cv, io_atd, io_tagging, v2, v5, write  # noqa: E402

W = 1.0
ITERS = 3
CAP = {}


def _capture(ll, blocks, dp, fill, change=None):
    CAP.update(ll=ll.copy(), blocks=blocks, dp=dp, fill=fill)
    return _orig(ll, blocks, dp, fill, change)


_orig = cv.decode_ll


def rates(replay, known, lab, weight=None):
    if weight is None:
        dmask, lmask = known & (lab == 2), known & np.isin(lab, (0, 1))
        return replay[dmask].mean(), replay[lmask].mean()
    w = weight[known]
    r = replay[known]
    return (r * w).sum() / w.sum(), (r * (1 - w)).sum() / (1 - w).sum()


def spells(c, g):
    cut = np.flatnonzero(np.diff(c)) + 1
    a, b = np.r_[0, cut], np.r_[cut, len(c)]
    return [(x, y, int(c[x])) for x, y in zip(a, b) if g[x]]


def aggregates(c, g, fps=25):
    sp = spells(c, g)
    live_min = np.isin(c[g], (0, 1)).sum() / fps / 60
    team = [(a, b) for a, b, k in sp if k in (0, 1)]
    short_between = 0
    for i in range(1, len(sp) - 1):
        a, b, k = sp[i]
        if k in (0, 1) and (b - a) <= 4 * fps and sp[i - 1][2] == 2 and sp[i + 1][2] == 2:
            short_between += 1
    return {"D_share": float(np.mean(c[g] == 2)), "possessions": len(team),
            "per_live_min": len(team) / live_min, "short_between_D": short_between}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    v2.cv.decode_ll = _capture
    match = io_atd.load_match()
    tags = io_tagging.load()
    model_dir = os.path.join(config.REPO_DIR, "models", "v52")
    base = v5.predict(match, tags, use_tags=True, model_dir=model_dir)
    v2.cv.decode_ll = _orig
    ll0, blocks, dp = CAP["ll"], CAP["blocks"], CAP["dp"]
    g = match.graded
    n = match.n_frames
    assert len(base) == n

    _, valid = ballfill.broadcast_ball(match.pitch)
    replay = np.zeros(n, bool)
    known = np.zeros(n, bool)
    for h, s, e in match.half_blocks():
        t = np.arange(s, e)
        j = t - ballfill.LAG[h]
        ok = (j >= 0) & (j < len(valid))
        replay[t[ok]] = ~valid[j[ok]]
        known[t[ok]] = True
    known &= g
    miss = ~match.ball_present
    print("graded frames {}, with a lagged broadcast frame {:.1%}, replay share {:.3f}".format(
        g.sum(), known.sum() / g.sum(), replay[known].mean()))

    soft = np.exp(ll0 - ll0.max(1, keepdims=True))
    soft = soft[:, 2] / soft.sum(1)
    pd_s, pl_s = rates(replay, known, None, soft)
    print("iter 0 soft (emission P(D)): P(r|D) {:.3f} P(r|live) {:.3f} log-ratio {:+.3f}".format(
        pd_s, pl_s, np.log(pd_s / pl_s)))
    pd_m, pl_m = rates(replay, known & ~miss, base)
    print("iter 0 hard, ball-present frames only: P(r|D) {:.3f} P(r|live) {:.3f} log-ratio "
          "{:+.3f}".format(pd_m, pl_m, np.log(pd_m / pl_m)))

    codes, hist = base, []
    for it in range(ITERS):
        pd_, pl_ = rates(replay, known, codes)
        pd2, pl2 = rates(replay, known & ~miss, codes)
        lr = float(np.log(pd_ / pl_))
        hist.append(lr)
        print("iter {} hard: P(r|D) {:.3f} P(r|live) {:.3f} log-ratio {:+.3f}  (ball present only "
              "{:+.3f})".format(it, pd_, pl_, lr, np.log(pd2 / pl2)))
        ll = ll0.copy()
        ll[replay & known, 2] += W * lr
        ll = np.round(ll / 1e-3) * 1e-3
        codes = cv.decode_ll(ll, blocks, dp, fill=CAP["fill"])
    a0, a1 = aggregates(base, g), aggregates(codes, g)
    print("log-ratio by iteration: {}".format(" ".join("{:+.3f}".format(x) for x in hist)))
    print("v5.2   : D share {D_share:.3f}  possessions {possessions}  per live min {per_live_min:.2f}  "
          "short H/A between two D {short_between_D}".format(**a0))
    print("replay : D share {D_share:.3f}  possessions {possessions}  per live min {per_live_min:.2f}  "
          "short H/A between two D {short_between_D}".format(**a1))
    ch = (codes != base) & g
    print("frames changed {} ({:.2%} of graded): {}".format(ch.sum(), ch.mean() * 0 + ch.sum() / g.sum(),
          {"{}->{}".format("HADX"[a], "HADX"[b]): int(((base == a) & (codes == b) & g).sum())
           for a in range(3) for b in range(3) if a != b and ((base == a) & (codes == b) & g).any()}))
    long_split = 0
    for a, b, k in spells(base, g):
        if k in (0, 1) and b - a >= 10 * 25 and np.any(codes[a:b] == 2):
            long_split += 1
    n_long = sum(1 for a, b, k in spells(base, g) if k in (0, 1) and b - a >= 10 * 25)
    short_gone = 0
    sp = spells(base, g)
    for i in range(1, len(sp) - 1):
        a, b, k = sp[i]
        if k in (0, 1) and b - a <= 100 and sp[i - 1][2] == 2 and sp[i + 1][2] == 2:
            short_gone += int(np.all(codes[a:b] == 2))
    print("short (<=4 s) H/A between two D in v5.2 that became all D: {} of {}".format(
        short_gone, a0["short_between_D"]))
    print("long (>=10 s) v5.2 live spells now containing D anywhere: {} of {}".format(long_split, n_long))
    inner, lens = 0, []
    for a, b, k in spells(base, g):
        if k in (0, 1) and b - a >= 10 * 25:
            seg = codes[a:b] == 2
            if seg.any():
                idx = np.flatnonzero(seg)
                if idx[0] > 0 and idx[-1] < b - a - 1 and not seg[0] and not seg[-1]:
                    inner += 1
                    lens += [(y - x) / 25 for x, y, kk in spells(codes[a:b], np.ones(b - a, bool)) if kk == 2]
    print("  of which split in the middle (live on both sides): {}; inserted D lengths s: {}".format(
        inner, sorted(round(x, 1) for x in lens)))
    if args.write:
        from possession.canonical import X
        out = os.path.join(config.OUT_DIR, "replay_run1.csv")
        digest = write.write_submission(out, codes)
        print("wrote {} sha256 {}".format(out, digest))


if __name__ == "__main__":
    main()
