"""The label bridge is a parameter; the default (2 s) keeps v4's labels and caches unchanged."""

import numpy as np

from possession import config, model
from possession.canonical import A, H
from possession.io_idsse import relabel_controlled


def _seq(*runs):
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in runs])


def test_default_bridge_is_two_seconds():
    assert config.DEFAULT_CONTROL_MIN_S == 2.0


def test_shorter_bridge_keeps_a_one_second_possession():
    lab = _seq((H, 100), (A, 30), (H, 100))            # A keeps it 1.2 s
    blocks = [(1, 0, len(lab))]
    assert (relabel_controlled(lab, blocks, 50) == H).all()
    assert relabel_controlled(lab, blocks, 25).tolist() == lab.tolist()


def test_default_bridge_returns_cached_labels_untouched(monkeypatch):
    monkeypatch.setattr(config, "CONTROL_MIN_S", config.DEFAULT_CONTROL_MIN_S)
    y = _seq((H, 10), (A, 10))
    assert model._bridged("J03WMX", False, y) is y


def test_wrong_team_swap_moves_track_to_other_team_spare():
    from possession import degrade
    T = 200
    players = np.zeros((T, 2, 2))
    players[:, 1] = 1.0                                  # opponents 1.4 m apart: in contact
    team = np.array([0, 1], dtype=np.int8)
    sp = {"hazard": 1.0, "runs": [50], "contact": True}
    out, t2 = degrade.wrong_team_swaps(np.random.default_rng(0), players, team, sp, 25)
    assert out.shape[1] == 2 + 2 * degrade.SWAP_SPARES and len(t2) == out.shape[1]
    present = ~np.isnan(out[..., 0])
    # slot 0 (home) is empty exactly where an away spare carries its positions
    away_spares = np.flatnonzero((t2 == 1) & (np.arange(len(t2)) >= 2))
    carried = present[:, away_spares].any(axis=1)
    assert carried.any() and not (carried & present[:, 0]).any()
    assert np.allclose(out[carried][:, away_spares][~np.isnan(out[carried][:, away_spares, 0])], 0.0)

