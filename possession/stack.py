"""Second-stage (stacked) evidence: context features built from the first-stage outputs.

Stage 1 gives per-frame log P(H), log P(A), log P(D) (the emission classifier) and the log-odds
that the ball changes team near the frame (the change head). Stage 2 reads these over the
surrounding seconds: the team log-odds and P(D) shortly before and after, past and future window
means, and where the change signal peaks. It can learn patterns such as "a change signal here,
and the other team's evidence rises afterwards", which the frame classifier cannot see.

Every window stops at the edges of the graded half (values outside are NaN, which the
gradient-boosting model handles), so halves never leak into each other.
"""

import numpy as np

from .features import _causal_mean

OFFSETS = (10, 25, 50, 100)            # [D] frames: 0.4, 1, 2, 4 s
WINDOWS = (25, 50, 100, 200)           # [D] frames: 1, 2, 4, 8 s past / future means
PEAK_WINDOWS = (25, 50)                # [D] change-signal max over the last / next 1 and 2 s


def _shift(x, k):
    out = np.full_like(x, np.nan)
    if k > 0:
        out[:-k] = x[k:]
    elif k < 0:
        out[-k:] = x[:k]
    else:
        out[:] = x
    return out


def _rolling_max(x, lo, hi):
    """max of x over [t+lo, t+hi), NaN-aware (window lengths here are small)."""
    out = np.full(len(x), -np.inf)
    for k in range(lo, hi):
        s = _shift(x, k)
        out = np.fmax(out, s)
    out[~np.isfinite(out)] = np.nan
    return out


def context_features(ll, change, graded):
    """(T, F) float32 context features and their names. ll: (T, 3) log-probs; change: (T,)."""
    ll = np.asarray(ll, dtype=np.float64)
    g = np.asarray(graded, dtype=bool)
    team = np.where(g, ll[:, 0] - ll[:, 1], np.nan)          # home-vs-away log-odds
    dead = np.where(g, ll[:, 2], np.nan)                     # log P(D)
    chg = np.where(g, np.asarray(change, dtype=np.float64), np.nan)
    cols, names = [team, dead, chg], ["s2_team", "s2_dead", "s2_change"]
    for name, x in (("team", team), ("dead", dead), ("change", chg)):
        for k in OFFSETS:
            cols += [_shift(x, -k), _shift(x, k)]
            names += ["s2_{}_m{}".format(name, k), "s2_{}_p{}".format(name, k)]
        for w in WINDOWS:
            past = _causal_mean(x, -w, 0)[0]
            fut = _causal_mean(x, 1, w + 1)[0]
            cols += [past, fut, fut - past]
            names += ["s2_{}_past{}".format(name, w), "s2_{}_fut{}".format(name, w),
                      "s2_{}_diff{}".format(name, w)]
    for w in PEAK_WINDOWS:
        cols += [_rolling_max(chg, -w, 1), _rolling_max(chg, 0, w + 1)]
        names += ["s2_change_maxpast{}".format(w), "s2_change_maxfut{}".format(w)]
    return np.column_stack(cols).astype(np.float32), names
