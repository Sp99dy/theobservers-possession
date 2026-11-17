"""PFF FC World Cup 2022 matches as extra TRAINING data: label/visibility stats and degraded
feature copies in cache_pff/ (never a test fold, never used for selection).

    POSSESSION_CONTROL_MIN_S=0.5 .venv/bin/python scripts/pff_build.py [--games N] [--stats-only] [--jobs 2]

Labels: io_pff's documented event rule, then the configured bridge (0.5 s). Ball: PFF's smoothed
ball with any remaining provider gaps filled by linear interpolation (held at the ends) BEFORE
degrading, so the only ball-missing pattern the model sees is the BAR-ATH-fitted degrader's
(provider dead-ball shortcut, see TECHNICAL.md). Copies: cache/degrader_params.json, offsets
COPIES (offset 0, its mirror, offset 1), oriented, feature version 2, written as
cache_pff/pff_feat_<game>_<offset><s>_v2.npz (X, y, g).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config, degrade, features, io_idsse, io_pff  # noqa: E402
from possession.canonical import orient  # noqa: E402
from scripts.metrica_build import label_stats  # noqa: E402

CACHE = os.path.join(REPO, os.environ.get("PFF_FEAT_DIR", "cache_pff"))   # feature copies
COPIES = ((0, False), (0, True), (1, False))   # [D] 3 per match (disk: 14 GB free), float16, compressed
MIN_FREE_GB = float(os.environ.get("PFF_MIN_FREE_GB", "5"))
MIN_SWAP_FREE_GB = 1.5


def swap_free_gb():
    import subprocess
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    tok = out.split()
    v = tok[tok.index("free") + 2]
    return float(v[:-1]) / (1024.0 if v.endswith("M") else 1.0)
LOG = os.path.join(REPO, "out", "pff_progress.log")


def log(s):
    line = "{} {}".format(time.strftime("%H:%M:%S"), s)
    print(line, flush=True)
    with open(LOG, "a") as h:
        h.write(line + "\n")


def feat_path(game, offset, swapped):
    return os.path.join(CACHE, "pff_feat_{}_{}{}_v2.npz".format(game, offset, "s" if swapped else ""))


def fill_ball(m):
    """Linear interpolation of the ball over provider gaps inside each half (ends held)."""
    b = m.ball_xy.copy()
    for _, s, e in m.half_blocks():
        seg = b[s:e]
        ok = ~np.isnan(seg[:, 0])
        if ok.sum() < 2:
            continue
        t = np.arange(len(seg))
        for c in (0, 1):
            seg[:, c] = np.interp(t, t[ok], seg[ok, c])
        b[s:e] = seg
    return replace(m, ball_xy=b)


def build(task):
    game, idx, offset, swapped = task
    path = feat_path(game, offset, swapped)
    if os.path.isfile(path):
        return task
    import shutil
    if shutil.disk_usage(os.path.expanduser("~")).free / 1e9 < MIN_FREE_GB:
        return task + ("SKIPPED: disk",)
    if swap_free_gb() < MIN_SWAP_FREE_GB:
        return task + ("SKIPPED: swap",)
    with open(os.path.join(REPO, "cache", "degrader_params.json")) as h:
        params = json.load(h)
    clean = fill_ball(io_pff.load_labelled(game))
    m = degrade.degrade(clean, params, seed=70_000 + 100 * idx + offset)
    if swapped:
        m = degrade.swap_teams(m)
    m = orient(m)
    X, names = features.design_matrix(m, version=2)
    np.savez_compressed(path + ".tmp.npz", X=X.astype(np.float16), y=m.labels, g=m.graded,
             names=np.array(names))
    os.replace(path + ".tmp.npz", path)
    return task


def frozen_share(m):
    """Share of graded frames where the (filled) ball moved < 1 cm since the previous frame,
    split by label D vs live: a provider that freezes the ball in dead time shows up here."""
    b = m.ball_xy
    d = np.r_[np.nan, np.hypot(*np.diff(b, axis=0).T)]
    g = m.graded & ~np.isnan(d)
    still = d < 0.01
    lab = m.labels
    return (round(float(np.mean(still[g & (lab == 2)])), 3) if (g & (lab == 2)).any() else None,
            round(float(np.mean(still[g & (lab < 2)])), 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=12)
    ap.add_argument("--stats-only", action="store_true")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--start", type=int, default=0, help="skip the first N games (already built)")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    allg = io_pff.game_ids()
    games = allg[args.start:args.games]
    log("pff build: bridge {} s, {} games {}".format(config.CONTROL_MIN_S, len(games), games))
    rows = {}
    for gid in games:
        if swap_free_gb() < MIN_SWAP_FREE_GB:
            log("STOP: swap free < {} GB".format(MIN_SWAP_FREE_GB))
            return
        t0 = time.time()
        m, raw, vis = io_pff.load_match(gid, extras=True)
        ml = io_pff.load_labelled(gid)
        bridged = io_pff.relabel_controlled(m.labels, m.half_blocks(),
                                            int(round(config.CONTROL_MIN_S * m.fps)))
        s = label_stats(bridged, m.half)
        g = ml.graded
        s["X_event"] = round(float(np.mean(bridged[g] == 3)), 3)
        s["X_masked"] = round(float(np.mean(ml.labels[g] == 3)), 3)
        lg = io_pff.long_ball_gaps(m)
        s["longgap_live"] = round(float(np.mean(lg[g & (bridged < 2)])), 3)
        s["longgap_dead"] = round(float(np.mean(lg[g & (bridged == 2)])), 3)
        s["smooth_ball_missing"] = round(float(np.mean(~m.ball_present[g])), 3)
        s["raw_ball_seen"] = round(float(np.mean(raw[g])), 3)
        live, dead = g & (bridged < 2), g & (bridged == 2)
        s["raw_seen_live"] = round(float(np.mean(raw[live])), 3)
        s["raw_seen_dead"] = round(float(np.mean(raw[dead])), 3) if dead.any() else None
        s["smooth_miss_dead"] = round(float(np.mean(~m.ball_present[dead])), 3) if dead.any() else None
        s["vis_players"] = round(float(np.mean(vis[g].sum(1))), 1)
        s["players_present"] = round(float(np.mean(m.player_present[g].sum(1))), 1)
        from dataclasses import replace
        s["still_dead"], s["still_live"] = frozen_share(fill_ball(replace(m, labels=bridged)))
        rows["pff_" + gid] = s
        log("loaded {} {:.0f}s {}".format(gid, time.time() - t0, json.dumps(s)))
    for mid in io_idsse.MATCH_IDS:
        m = io_idsse.load_labelled(mid)
        s = label_stats(m.labels, m.half)
        s["X_event"] = round(float(np.mean(m.labels[m.graded] == 3)), 3)
        s["smooth_ball_missing"] = round(float(np.mean(~m.ball_present[m.graded])), 3)
        s["still_dead"], s["still_live"] = frozen_share(m)
        rows[mid] = s
    keys = sorted({k for s in rows.values() for k in s}, key=lambda k: list(
        next(iter(rows.values()))).index(k) if k in next(iter(rows.values())) else 99)
    lines = ["{:<11}".format("") + "".join("{:>14}".format(k[:13]) for k in keys)]
    for k, s in rows.items():
        lines.append("{:<11}".format(k[:11]) + "".join("{:>14}".format(str(s.get(c, "-")))
                                                       for c in keys))
    txt = "\n".join(lines)
    print(txt)
    tag = "" if args.start == 0 else "_{}-{}".format(args.start, args.games)
    with open(os.path.join(REPO, "out", "pff_label_stats{}.txt".format(tag)), "w") as h:
        h.write(txt + "\n")
    with open(os.path.join(REPO, "out", "pff_label_stats{}.json".format(tag)), "w") as h:
        json.dump(rows, h, indent=1)
    if args.stats_only:
        return
    tasks = [(g, allg.index(g), o, s) for g in games for o, s in COPIES]
    with ProcessPoolExecutor(args.jobs) as ex:
        for t in ex.map(build, tasks):
            log("built {}".format(t))
    log("pff build done")


if __name__ == "__main__":
    main()
