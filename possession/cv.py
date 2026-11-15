"""Leave-one-match-out cross-validation on degraded IDSSE, scored with the organisers' metric.

All model selection happens here: classifier, feature set, decoder costs and durations.
"""

import numpy as np

from . import decode, metrics

DEFAULT_DECODER = dict(step_weight=1.0, team_cost=4.5, enter_d=5.0, leave_d=3.0,
                       min_team=5, min_d=10, bias_d=0.0)

# Coordinate-descent search grid for the decoder. [D] wide enough to bracket sensible values.
DECODER_GRID = dict(
    step_weight=(0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5),
    team_cost=(0.0, 0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 9.0),
    enter_d=(1.0, 2.0, 3.5, 5.0, 7.0, 10.0),
    leave_d=(0.5, 1.5, 3.0, 5.0),
    min_team=(1, 5, 8, 12, 16, 20, 25),
    min_d=(1, 5, 10, 15, 20, 25, 35),
    bias_d=(-1.0, -0.5, 0.0, 0.5, 1.0),
)
STEP = 5


# Split dead ball (SPEC 5.5): D_after_H / D_after_A remember who had the ball before the stoppage,
# so the restart can favour the same team ("leave_same") or the other ("leave_other").
SPLIT_GRID = dict(leave_same=(0.5, 1.5, 3.0, 5.0, 7.0), leave_other=(0.5, 1.5, 3.0, 5.0, 7.0))
BLOCKED = 1e9


def is_split(dp):
    return "leave_same" in dp


def decode_ll(frame_ll, blocks, dp, fill):
    ll = frame_ll.copy()
    ll[:, 2] += dp["bias_d"]
    if not is_split(dp):
        cost = ((0.0, dp["team_cost"], dp["enter_d"]),
                (dp["team_cost"], 0.0, dp["enter_d"]),
                (dp["leave_d"], dp["leave_d"], 0.0))
        return decode.decode_blocks(ll, blocks, STEP, dp["step_weight"],
                                    (dp["min_team"], dp["min_team"], dp["min_d"]), cost,
                                    fill=fill)
    # States H, A, D_after_H, D_after_A; both D states emit the D log-likelihood.
    ll4 = np.column_stack([ll[:, 0], ll[:, 1], ll[:, 2], ll[:, 2]])
    t, e, ls, lo, B = dp["team_cost"], dp["enter_d"], dp["leave_same"], dp["leave_other"], BLOCKED
    cost = ((0.0, t, e, B),
            (t, 0.0, B, e),
            (ls, lo, 0.0, B),
            (lo, ls, B, 0.0))
    out = decode.decode_blocks(ll4, blocks, STEP, dp["step_weight"],
                               (dp["min_team"], dp["min_team"], dp["min_d"], dp["min_d"]), cost,
                               fill=-1)
    outside = out == -1
    out[out == 3] = 2
    out[outside] = fill
    return out


def pooled_score(items, dp=None, preds=None):
    """items: [(frame_ll, labels, blocks)]. Pooled organisers' metric over every half."""
    gts, prs = [], []
    for i, (ll, lab, blocks) in enumerate(items):
        pred = preds[i] if preds is not None else decode_ll(ll, blocks, dp, fill=3)
        for s, e in blocks:
            gts.append(lab[s:e])
            prs.append(pred[s:e])
    gt = metrics.join_runs(gts)
    pr = metrics.join_runs(prs)
    return metrics.summary(metrics.evaluate(gt, pr))


def decoder_grid(dp, fixed=()):
    """The search grid for decoder settings dp: split-D settings replace leave_d; names in
    `fixed` are not searched (used for ablations such as 'no minimum durations')."""
    grid = dict(DECODER_GRID)
    if is_split(dp):
        del grid["leave_d"]
        grid.update(SPLIT_GRID)
    return {k: v for k, v in grid.items() if k not in fixed}


def tune_decoder(items, start=None, sweeps=2, log=None, fixed=()):
    """Coordinate descent on pooled spellF1 (macroF1 as tie-break)."""
    dp = dict(start or DEFAULT_DECODER)
    best = pooled_score(items, dp)
    key = lambda s: (round(s["spell_f1_mean"], 6), round(s["macro_f1"], 6))
    for _ in range(sweeps):
        changed = False
        for name, values in decoder_grid(dp, fixed).items():
            for v in values:
                if v == dp[name]:
                    continue
                trial = dict(dp, **{name: v})
                s = pooled_score(items, trial)
                if key(s) > key(best):
                    best, dp, changed = s, trial, True
            if log:
                log("  {:<12} -> {:<6} spellF1 {:.4f}".format(name, dp[name], best["spell_f1_mean"]))
        if not changed:
            break
    return dp, best
