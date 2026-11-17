"""Metrica sample games 1+2 as extra TRAINING matches (review idea 6): label stats and degraded
feature copies in a separate cache dir.

    POSSESSION_CONTROL_MIN_S=0.5 python scripts/metrica_build.py [--stats-only] [--jobs 2]

Labels: io_metrica's documented event rule (an approximation of the challenge convention, derived
from Metrica's events, not per-frame truth), then the configured bridge (0.5 s). Copies: the same
BAR-ATH-fitted degrader as v5's training copies (cache/degrader_params.json), offsets 0-8 each
with its team-swapped mirror, oriented, feature version 2. Written to metrica_cache/ as
metrica_feat_<game>_<offset><s>_v2.npz with X, y (configured bridge), g. Never a test fold.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config, degrade, features, io_idsse, io_metrica  # noqa: E402
from possession.canonical import orient  # noqa: E402

CACHE = os.path.join(REPO, "metrica_cache")
OFFSETS = range(9)


def feat_path(game, offset, swapped):
    return os.path.join(CACHE, "metrica_feat_{}_{}{}_v2.npz".format(game, offset, "s" if swapped else ""))


def label_stats(lab, half, fps=25):
    g = half > 0
    cut = np.flatnonzero(np.diff(lab)) + 1
    a, b = np.r_[0, cut], np.r_[cut, len(lab)]
    sp = [(y - x) / fps for x, y in zip(a, b) if lab[x] in (0, 1) and g[x]]
    mins = g.sum() / fps / 60
    return {"minutes": round(mins, 1), "possessions": len(sp), "per_min": round(len(sp) / mins, 2),
            "mean_s": round(float(np.mean(sp)), 1), "median_s": round(float(np.median(sp)), 1),
            "share_le3s": round(float(np.mean(np.array(sp) <= 3)), 2),
            "D_share": round(float(np.mean(lab[g] == 2)), 3)}


def build(task):
    game, offset, swapped = task
    path = feat_path(game, offset, swapped)
    if os.path.isfile(path):
        return task
    with open(os.path.join(REPO, "cache", "degrader_params.json")) as h:
        params = json.load(h)
    clean = io_metrica.load_match(game)
    m = degrade.degrade(clean, params, seed=50_000 + 100 * game + offset)
    if swapped:
        m = degrade.swap_teams(m)
    m = orient(m)
    X, names = features.design_matrix(m, version=2)
    np.savez(path + ".tmp.npz", X=X.astype(np.float32), y=m.labels, g=m.graded,
             names=np.array(names))
    os.replace(path + ".tmp.npz", path)
    return task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    print("bridge {} s".format(config.CONTROL_MIN_S))
    rows = {}
    for game in io_metrica.GAMES:
        m = io_metrica.load_match(game)
        s = label_stats(m.labels, m.half)
        s["ball_missing_share"] = round(float(np.mean(~m.ball_present[m.graded])), 3)
        rows["metrica_{}".format(game)] = s
    for mid in io_idsse.MATCH_IDS:
        m = io_idsse.load_labelled(mid)
        s = label_stats(m.labels, m.half)
        s["ball_missing_share"] = round(float(np.mean(~m.ball_present[m.graded])), 3)
        rows[mid] = s
    keys = list(next(iter(rows.values())))
    print("{:<11}".format("") + "".join("{:>12}".format(k) for k in keys))
    for k, s in rows.items():
        print("{:<11}".format(k) + "".join("{:>12}".format(s[c]) for c in keys))
    with open(os.path.join(REPO, "out", "metrica_label_stats.json"), "w") as h:
        json.dump(rows, h, indent=1)
    if args.stats_only:
        return
    tasks = [(g, o, s) for g in io_metrica.GAMES for o in OFFSETS for s in (False, True)]
    with ProcessPoolExecutor(args.jobs) as ex:
        for t in ex.map(build, tasks):
            print("built", t, flush=True)


if __name__ == "__main__":
    main()
