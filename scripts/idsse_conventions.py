"""Compare DFL's possession / dead-ball convention with the challenge's (SPEC section 6).

    python scripts/idsse_conventions.py

Prints, per IDSSE match, label shares, possessions per minute, possession and stoppage duration
distributions. The challenge's reference numbers are the organisers' published public-window
summary (17 possessions in 5 min, mean 14.2 s, 4 dead-ball stretches); these are used only to
compare definitions qualitatively, never to fit anything.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import io_idsse  # noqa: E402
from possession.canonical import D  # noqa: E402


def segment_stats(labels, blocks, fps=25):
    poss, dead = [], []
    for _, s, e in blocks:
        c = labels[s:e]
        b = np.flatnonzero(np.diff(c)) + 1
        for a, z in zip(np.r_[0, b], np.r_[b, len(c)]):
            (dead if c[a] == D else poss).append((z - a) / fps)
    return np.array(poss), np.array(dead)


def main():
    rows = []
    for mid in io_idsse.MATCH_IDS:
        m = io_idsse.load_labelled(mid) if "--controlled" in sys.argv else io_idsse.load_match(mid)
        g = m.graded
        minutes = g.sum() / m.fps / 60
        poss, dead = segment_stats(m.labels, m.half_blocks())
        rows.append((mid, minutes, np.mean(m.labels[g] == D), len(poss) / minutes, poss, dead))
        print("{}  {:5.1f} min  D share {:.3f}  possessions {:4d} ({:.2f}/min)  mean {:5.1f}s  "
              "median {:4.1f}s  <1s {:4.1%}  <2s {:4.1%}  | stoppages {:3d} ({:.2f}/min) median {:4.1f}s"
              .format(mid, minutes, rows[-1][2], len(poss), rows[-1][3], poss.mean(),
                      np.median(poss), np.mean(poss < 1), np.mean(poss < 2), len(dead),
                      len(dead) / minutes, np.median(dead)))
    allp = np.concatenate([r[4] for r in rows])
    alld = np.concatenate([r[5] for r in rows])
    mins = sum(r[1] for r in rows)
    print("\nIDSSE pooled: {:.2f} possessions/min, mean {:.1f}s, median {:.1f}s; "
          "{:.2f} stoppages/min".format(len(allp) / mins, allp.mean(), np.median(allp),
                                        len(alld) / mins))
    print("possession duration percentiles (s) p10/25/50/75/90:",
          np.round(np.percentile(allp, [10, 25, 50, 75, 90]), 1).tolist())
    print("stoppage duration percentiles (s) p10/25/50/75/90:",
          np.round(np.percentile(alld, [10, 25, 50, 75, 90]), 1).tolist())
    print("challenge public window (published): 3.40 possessions/min, mean 14.2s; "
          "0.80 stoppages/min")


if __name__ == "__main__":
    main()
