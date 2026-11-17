"""Which label bridge looks like the challenge's ground truth? A convention check, no fitting.

    python scripts/bridge_lag_curve.py

For the raw DFL labels and for v4's controlled-possession bridge (io_idsse.relabel_controlled) at
0.5 / 1 / 2 s, on all 7 clean IDSSE matches:

* the lag curve: spellF1 of the perfect labels with every change k = 1, 3, 8, 25 frames late,
  mean over matches. EVALUATION.md publishes this curve for the challenge GT on BAR-ATH
  (1.0000 / 0.9818 / 0.8933 / 0.6539). Short possessions make the curve fall faster, so it is a
  fingerprint of how choppy a labelling is;
* possessions per match and mean length (organisers' possession_stats);
* touch-to-switch lag: for each direct H<->A change of the bridged labels, frames since the raw
  DFL flag (which flips at the touch) last switched to that team. The bridge drops short spells;
  it never delays a kept switch, so this should be 0.

Reads public IDSSE labels only. Never reads BAR-ATH.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import io_idsse, organisers  # noqa: E402
from possession.canonical import A, H  # noqa: E402

PUBLISHED = (1.0000, 0.9818, 0.8933, 0.6539)
LAGS = (1, 3, 8, 25)
BRIDGES = (None, 0.5, 1.0, 2.0)


def lag_late(gt, runs, k):
    """Every label change happens k frames late (as hack2027 scratch/metric/sensitivity.py)."""
    pred = gt.copy()
    for s, e in runs:
        w = gt[s:e]
        pred[s:e] = np.concatenate((np.full(k, w[0]), w[:-k])) if k else w
    return pred


def touch_to_switch(raw, lab):
    """Frames from the raw flag's last switch into the new team to each bridged H<->A change."""
    out = []
    idx = np.flatnonzero((lab[1:] != lab[:-1]) & np.isin(lab[:-1], (H, A)) &
                         np.isin(lab[1:], (H, A))) + 1
    for f in idx:
        team = lab[f]
        j = f
        while j > 0 and not (raw[j] == team and raw[j - 1] != team):
            j -= 1
        out.append(f - j)
    return out


def main():
    pm = organisers.pm()
    clean = {mid: io_idsse.load_match(mid) for mid in io_idsse.MATCH_IDS}
    print("published BAR-ATH GT: " + "  ".join("lag{} {:.4f}".format(k, v)
                                               for k, v in zip(LAGS, PUBLISHED)))
    for b in BRIDGES:
        vals, n, dur, lags = [], [], [], []
        for mid, m in clean.items():
            raw = m.labels
            lab = raw if b is None else io_idsse.relabel_controlled(
                raw, m.half_blocks(), int(round(b * m.fps)))
            runs = pm.valid_runs(lab)
            st = pm.possession_stats(lab, runs)
            n.append(st["n_spells"])
            dur.append(st["mean_duration_s"])
            vals.append([pm.evaluate(lab, lag_late(lab, runs, k))["spell_f1_mean"] for k in LAGS])
            lags += touch_to_switch(raw, lab)
        v = np.mean(vals, axis=0)
        rmse = np.sqrt(np.mean((v - np.array(PUBLISHED)) ** 2))
        lags = np.array(lags)
        print("{:<10} poss/match {:>4.0f}  mean {:>5.1f}s  ".format(
            "raw" if b is None else "bridge {:.1f}s".format(b), np.mean(n), np.mean(dur)) +
            "  ".join("lag{} {:.4f}".format(k, x) for k, x in zip(LAGS, v)) +
            "  rmse vs published {:.4f}  touch->switch median {:.0f}f, p90 {:.0f}f, >0 {:.1%}".format(
                rmse, np.median(lags), np.percentile(lags, 90), np.mean(lags > 0)))


if __name__ == "__main__":
    main()
