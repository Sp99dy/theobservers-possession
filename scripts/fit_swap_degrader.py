"""Degrader version 2: add wrong-team tracklets (degrade.wrong_team_swaps), fitted to BAR-ATH.

    POSSESSION_CACHE_DIR=cache_swap python scripts/fit_swap_degrader.py [--contact] [--base cache]

Sets up a separate cache directory (the feature caches are keyed by match / copy / feature version,
not by degrader version), seeded from --base with the clean-match and BAR-ATH caches and the
version-1 degrader parameters. Then measures on the BAR-ATH tracking (no labels) the share of
frames where one team shows 12+ players and the other <= 10, and its run lengths, and calibrates
the swap hazard by bisection so that the degraded IDSSE (evaluation copies) reproduces that share.
"""

import argparse
import glob
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, degrade, io_atd, io_idsse, model  # noqa: E402

PARAMS = "degrader_params.json"


def share_12v10(m):
    g = m.graded
    p = ~np.isnan(m.players_xy[..., 0])
    nh, na = p[:, m.team_of_slot == 0].sum(1), p[:, m.team_of_slot == 1].sum(1)
    sig = (((nh >= 12) & (na <= 10)) | ((na >= 12) & (nh <= 10))) & g
    return sig, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(config.REPO_DIR, "cache"))
    ap.add_argument("--contact", action="store_true", help="swaps start only in a duel")
    args = ap.parse_args()
    dst = config.CACHE_DIR
    assert os.path.abspath(dst) != os.path.abspath(args.base), "set POSSESSION_CACHE_DIR"
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(args.base, "idsse_J*_v1.npz")) + \
            glob.glob(os.path.join(args.base, "BAR-ATH_*.npz")):
        link = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(link):
            os.symlink(os.path.abspath(f), link)
    with open(os.path.join(args.base, PARAMS)) as h:
        params = json.load(h)

    atd = io_atd.load_match()
    sig, g = share_12v10(atd)
    a, b = degrade._runs(sig)
    target = float(sig.sum() / g.sum())
    runs = sorted(int(x) for x in (b - a))
    print("BAR-ATH 12-vs-10 share {:.4f}; runs n={} median {:.1f}s p90 {:.1f}s mean {:.1f}s".format(
        target, len(runs), np.median(runs) / 25, np.percentile(runs, 90) / 25,
        np.mean(runs) / 25), flush=True)

    clean = {mid: io_idsse.load_labelled(mid) for mid in io_idsse.MATCH_IDS}

    def simulate(h):
        sp = {"hazard": float(h), "runs": runs, "contact": bool(args.contact)}
        p = dict(params, swap=sp)
        n = d = 0
        for i, (mid, m) in enumerate(clean.items()):
            deg = degrade.degrade(m, p, seed=model.copy_seed(i, model.EVAL_COPY[0]))
            s, gg = share_12v10(deg)
            n += s.sum()
            d += gg.sum()
        return n / d

    lo, hi = -7.0, -2.0                    # log10 hazard
    for _ in range(9):
        mid_ = (lo + hi) / 2
        s = simulate(10 ** mid_)
        print("  hazard 1e{:.2f}: share {:.4f}".format(mid_, s), flush=True)
        if s < target:
            lo = mid_
        else:
            hi = mid_
    h = 10 ** ((lo + hi) / 2)
    params["swap"] = {"hazard": h, "runs": runs, "contact": bool(args.contact),
                      "target_share_12v10": target, "fit_share_12v10": float(simulate(h))}
    params["degrader_version"] = 2
    degrade.save(params, os.path.join(dst, PARAMS))
    print("swap hazard {:.3e} (contact={}), fitted share {:.4f} vs target {:.4f} -> {}".format(
        h, args.contact, params["swap"]["fit_share_12v10"], target, os.path.join(dst, PARAMS)))


if __name__ == "__main__":
    main()
