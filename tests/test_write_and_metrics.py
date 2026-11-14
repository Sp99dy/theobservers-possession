"""Writer format and the metric wrapper, on synthetic sequences."""

import numpy as np

from possession import metrics, write
from possession.canonical import A, D, H, X


def test_render_exact_bytes():
    assert write.render(np.array([H, A, D, X])) == b"frame,label\n0,H\n1,A\n2,D\n3,X\n"
    assert write.render(np.array(["H", "D"])) == b"frame,label\n0,H\n1,D\n"


def test_write_returns_sha_and_lf_only(tmp_path):
    path = tmp_path / "s.csv"
    digest = write.write_submission(str(path), np.array([H, H, D]))
    data = path.read_bytes()
    assert b"\r" not in data
    assert digest == write.sha256_file(str(path))


def test_written_file_passes_organisers_validator(tmp_path):
    path = tmp_path / "s.csv"
    labels = np.array([X, H, H, A, D, X])
    write.write_submission(str(path), labels)
    codes = metrics.read_submission_codes(str(path), n_expected=len(labels))
    assert codes.tolist() == labels.tolist()


def _seq(*runs):
    return np.concatenate([np.full(n, code, dtype=np.int8) for code, n in runs])


def test_perfect_prediction_scores_one():
    gt = _seq((X, 5), (H, 50), (D, 10), (A, 40), (H, 30), (X, 5))
    s = metrics.summary(metrics.evaluate(gt, gt))
    assert s["spell_f1_mean"] == 1.0 and s["macro_f1"] == 1.0
    assert s["n_pred_spells"] == s["n_gt_spells"] == 3


def test_spurious_flip_splits_a_possession():
    gt = _seq((H, 100), (A, 100))
    pred = gt.copy()
    pred[45:55] = A                       # a 10-frame flicker inside the H possession
    s = metrics.summary(metrics.evaluate(gt, pred))
    assert s["n_pred_spells"] == 4
    assert s["spell_f1_mean"] < 1.0


def test_constant_guess_scores_zero_spell_f1():
    gt = _seq((H, 50), (A, 50), (H, 50), (A, 50))
    s = metrics.summary(metrics.evaluate(gt, np.full(len(gt), D, dtype=np.int8)))
    assert s["spell_f1_mean"] == 0.0


def test_join_runs_never_bridges_halves():
    # The same team "has the ball" at the end of half 1 and the start of half 2: two possessions.
    gt = metrics.join_runs([_seq((H, 30)), _seq((H, 30))])
    assert len(gt) == 61 and gt[30] == X
    s = metrics.summary(metrics.evaluate(gt, gt))
    assert s["n_gt_spells"] == 2
