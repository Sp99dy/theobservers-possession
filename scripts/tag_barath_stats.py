"""BAR-ATH Smart Tagging set-piece tags vs restarts, label-free (aggregates only).

    python scripts/tag_barath_stats.py [--pred out/v5_run1.csv]

Restarts are taken from v5's BAR-ATH prediction (no labels): the end of every predicted dead
stretch of >= 2 s that is followed by a team. For each set-piece tag: offset (s) of the tag start
from the nearest such restart. Also the share of restarts with a tag within +-15 s, per tag type.
Writes out/tag_barath_stats.json (offsets, coverage, per-type counts), read by tag_simulate.py.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import io_tagging  # noqa: E402

CODES = {"H": 0, "A": 1, "D": 2, "X": 3}
WIN_S = 15.0
MIN_DEAD_S = 2.0


def restarts(codes, fps=25):
    cut = np.flatnonzero(np.diff(codes)) + 1
    a, b = np.r_[0, cut], np.r_[cut, len(codes)]
    out = []
    for x, y in zip(a, b):
        if codes[x] == 2 and y - x >= MIN_DEAD_S * fps and y < len(codes) and codes[y] in (0, 1):
            out.append(int(y))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(REPO, "out", "v5_run1.csv"))
    args = ap.parse_args()
    codes = pd.read_csv(args.pred)["label"].map(CODES).to_numpy()
    rs = restarts(codes)
    tags = [t for t in io_tagging.load() if t.code == io_tagging.SET_PIECE_CODE]
    offs, types = [], Counter()
    for t in tags:
        d = (t.start - rs) / 25.0
        k = int(np.argmin(np.abs(d)))
        if abs(d[k]) <= WIN_S:
            offs.append(float(d[k]))
        types[t.type] += 1
    covered = sum(bool(np.any(np.abs(np.array([t.start for t in tags]) - r) <= WIN_S * 25))
                  for r in rs)
    res = {"n_tags": len(tags), "n_restarts_pred": int(len(rs)), "offsets_s": offs,
           "coverage": covered / max(1, len(rs)), "types": dict(types),
           "team_known_share": float(np.mean([t.team is not None for t in tags])),
           "tag_len_s": float(np.median([(t.end - t.start) / 25 for t in tags]))}
    o = np.array(offs)
    print("set-piece tags {}; predicted restarts {}; tags within {:.0f} s of a restart {}".format(
        len(tags), len(rs), WIN_S, len(o)))
    print("tag start - restart (s): p10/25/50/75/90 {}".format(
        np.round(np.percentile(o, [10, 25, 50, 75, 90]), 2).tolist()))
    print("restarts with a tag within {:.0f} s: {:.0%}; team known on {:.0%} of tags; types {}".format(
        WIN_S, res["coverage"], res["team_known_share"], dict(types)))
    with open(os.path.join(REPO, "out", "tag_barath_stats.json"), "w") as h:
        json.dump(res, h, indent=1)


if __name__ == "__main__":
    main()
