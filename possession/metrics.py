"""Thin wrapper over the organisers' evaluation/possession_metrics.py (imported, not copied).

The same code scores BAR-ATH (public sanity window) and IDSSE cross-validation, so a number in our
experiments log means exactly what it means on the leaderboard.
"""

import os

import numpy as np

from . import config, organisers
from .canonical import X

# The headline numbers we log for every candidate, in this order.
SUMMARY_KEYS = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells",
                "mean_spell_duration_s", "gt_mean_spell_duration_s", "alive_dead_f1",
                "team_acc_both_alive")


def evaluate(gt_codes, pred_codes, **kwargs):
    """The organisers' full metric dict. Codes are H=0, A=1, D=2, X=3."""
    pm = organisers.pm()
    return pm.evaluate(np.asarray(gt_codes, dtype=np.int8), np.asarray(pred_codes, dtype=np.int8),
                       **kwargs)


def summary(result):
    """The headline subset of an evaluate() result."""
    return {key: result[key] for key in SUMMARY_KEYS}


def format_summary(s):
    return ("spellF1 {spell_f1_mean:.4f}  macroF1 {macro_f1:.4f}  possessions {n_pred_spells} "
            "(gt {n_gt_spells})  mean {mean_spell_duration_s:.1f}s (gt {gt_mean_spell_duration_s:.1f}s)"
            "  aliveF1 {alive_dead_f1:.3f}  teamAcc {team_acc_both_alive:.3f}").format(**s)


def join_runs(arrays):
    """Concatenate per-half code arrays with one X frame between them.

    The scorer splits on GT X (valid_runs), so the separator guarantees no possession or transition
    is ever matched across halves or matches. Use the same call on GT and on predictions; pooled
    scoring over many IDSSE halves is then one evaluate() call.
    """
    pieces = []
    for i, codes in enumerate(arrays):
        if i:
            pieces.append(np.array([X], dtype=np.int8))
        pieces.append(np.asarray(codes, dtype=np.int8))
    return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int8)


def public_gt_path():
    return os.path.join(config.MTAH_REPO_DIR, "evaluation", "gt",
                        "BAR-ATH_possession_gt_public.csv")


def public_gt_codes():
    """Public 5-minute ground truth (frames 11185-18684). Sanity checks only -- never tuning."""
    pm = organisers.pm()
    return pm.label_codes(pm.read_label_csv(public_gt_path()))


def read_submission_codes(path, n_expected=None):
    """Read and validate a frame,label CSV with the organisers' own reader and validator."""
    pm = organisers.pm()
    table = pm.read_label_csv(path)
    if n_expected is None:
        n_expected = organisers.games().source_frames("tactical")
    reason = pm.validate_submission(table, n_expected)
    if reason:
        raise ValueError("{}: {}".format(path, reason))
    return pm.label_codes(table)
