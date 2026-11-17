"""HANDOFF 4.2b: add tracker identity swaps (degrade.identity_swaps), fitted to BAR-ATH.

    POSSESSION_CACHE_DIR=cache_idswap python scripts/fit_idswap_degrader.py --base cache

Sets up a separate cache dir (feature caches are keyed by match / copy / feature version, not by
degrader version) with links to the clean-match and BAR-ATH caches and the base degrader params.
Measures on the BAR-ATH tracking (no labels) one-frame identity jumps (a present track moving
> 0.8 m in one frame, > 20 m/s, within the halves) per minute, and the subset landing within 1 m
of an opponent, as scripts/diag_tracking_errors.py on fadel/bridge does. Then measures the current
degrader (evaluation copies, real slots only) and calibrates by bisection only the gap: first the
cross-team swap rate on the onto-opponent jumps, then the same-team rate on all jumps.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, degrade, io_atd, io_idsse, model  # noqa: E402

PARAMS = "degrader_params.json"
JUMP_M = 0.8
CROSS_MEAN_S = 20.0      # [F] cross-team tracklets on BAR-ATH last ~20 s (p50 21.6 s, bridge notes)


def jump_stats(m, n_real=None):
    xy = m.players_xy if n_real is None else m.players_xy[:, :n_real]
    team = m.team_of_slot if n_real is None else m.team_of_slot[:n_real]
    p = ~np.isnan(xy[..., 0]) & m.graded[:, None]
    minutes = m.graded.sum() / m.fps / 60
    step = np.linalg.norm(xy[1:] - xy[:-1], axis=-1)
    jumps = np.argwhere((step > JUMP_M) & p[1:] & p[:-1])
    onto = 0
    for t, j in jumps:
        opp = (team != team[j]) & p[t]
        if opp.any() and np.nanmin(np.linalg.norm(xy[t, opp] - xy[t + 1, j], axis=1)) < 1.0:
            onto += 1
    return len(jumps), onto, minutes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(config.REPO_DIR, "cache"))
    args = ap.parse_args()
    dst = config.CACHE_DIR
    assert os.path.abspath(dst) != os.path.abspath(args.base), "set POSSESSION_CACHE_DIR"
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(args.base, "idsse_J*_v1.npz")) + \
            glob.glob(os.path.join(args.base, "BAR-ATH_*.npz")):
        link = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(link):
            os.symlink(os.path.realpath(f), link)
    with open(os.path.join(args.base, PARAMS)) as h:
        params = json.load(h)
    params.pop("idswap", None)
    log = lambda s: print(s, flush=True)

    n, o, mins = jump_stats(io_atd.load_match())
    t_all, t_onto = n / mins, o / mins
    log("BAR-ATH: {:.3f} jumps/min, {:.3f} onto an opponent ({:.1f} min)".format(t_all, t_onto, mins))

    clean = {mid: io_idsse.load_labelled(mid) for mid in io_idsse.MATCH_IDS}
    n_real = {mid: len(m.team_of_slot) for mid, m in clean.items()}

    def simulate(sp):
        p = dict(params, idswap=sp) if sp else params
        N = O = M = 0.0
        for i, (mid, m) in enumerate(clean.items()):
            deg = degrade.degrade(m, p, seed=model.copy_seed(i, model.EVAL_COPY[0]))
            a, b, c = jump_stats(deg, n_real[mid])
            N, O, M = N + a, O + b, M + c
        return N / M, O / M

    base_all, base_onto = simulate(None)
    log("degrader v1: {:.3f} jumps/min, {:.3f} onto an opponent; gap {:+.3f} / {:+.3f}".format(
        base_all, base_onto, t_all - base_all, t_onto - base_onto))

    def bisect(key, target, which, sp):
        lo, hi = -9.0, -3.0
        for _ in range(10):
            mid_ = (lo + hi) / 2
            v = simulate(dict(sp, **{key: 10 ** mid_}))[which]
            log("  {} 1e{:.2f}: {:.3f}/min (target {:.3f})".format(key, mid_, v, target))
            lo, hi = (mid_, hi) if v < target else (lo, mid_)
        return 10 ** ((lo + hi) / 2)

    sp = {"h_same": 0.0, "h_cross": 0.0, "cross_mean_s": CROSS_MEAN_S}
    if t_onto > base_onto:
        sp["h_cross"] = bisect("h_cross", t_onto, 1, sp)
    if t_all > simulate(sp)[0]:
        sp["h_same"] = bisect("h_same", t_all, 0, sp)
    fit_all, fit_onto = simulate(sp)
    sp.update(target_jumps_per_min=t_all, target_onto_per_min=t_onto,
              base_jumps_per_min=base_all, base_onto_per_min=base_onto,
              fit_jumps_per_min=fit_all, fit_onto_per_min=fit_onto)
    params["idswap"] = sp
    params["degrader_version"] = "1+idswap"
    degrade.save(params, os.path.join(dst, PARAMS))
    log("fitted: h_same {:.3e}, h_cross {:.3e} -> {:.3f} jumps/min ({:.3f} onto) -> {}".format(
        sp["h_same"], sp["h_cross"], fit_all, fit_onto, os.path.join(dst, PARAMS)))


if __name__ == "__main__":
    main()
