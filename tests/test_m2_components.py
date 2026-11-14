"""IDSSE label conversion, degrader mechanics and team-swap augmentation, on synthetic data."""

import numpy as np

from possession import degrade, metrics
from possession.canonical import A, AWAY, D, H, HOME, X, Match, measure_attack_sign
from possession.io_idsse import relabel_controlled


def _seq(*runs):
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in runs])


def test_short_touch_does_not_change_possession():
    lab = _seq((H, 100), (A, 20), (H, 100))           # a 0.8 s touch by A
    out = relabel_controlled(lab, [(1, 0, len(lab))], 50)
    assert (out == H).all()


def test_controlled_spell_changes_possession():
    lab = _seq((H, 100), (A, 60), (H, 100))           # A keeps it 2.4 s
    out = relabel_controlled(lab, [(1, 0, len(lab))], 50)
    assert out.tolist() == lab.tolist()


def test_restart_possessor_is_kept_even_if_brief():
    lab = _seq((D, 50), (A, 20), (H, 100))            # A takes the throw-in, loses it at once
    out = relabel_controlled(lab, [(1, 0, len(lab))], 50)
    assert out.tolist() == lab.tolist()


def test_loose_ball_stays_with_last_controller():
    lab = _seq((H, 100), (A, 10), (H, 10), (A, 10), (H, 100))
    out = relabel_controlled(lab, [(1, 0, len(lab))], 50)
    assert (out == H).all()


def test_drop_runs_share_and_no_overlap():
    rng = np.random.default_rng(0)
    T, run, q = 200_000, 250, 0.4
    h = q / ((1 - q) * run)
    drop = degrade._drop_runs(rng, np.ones(T, dtype=bool), h, np.array([run]))
    assert abs(drop.mean() - q) < 0.05
    starts, stops = degrade._runs(drop)
    assert ((stops - starts) % run == 0).all()       # never merged into odd lengths


def _match():
    T = 20
    team = np.array([HOME] * 6 + [AWAY] * 6, dtype=np.int8)
    players = np.zeros((T, 12, 2))
    players[:, team == HOME, 0] = -20.0
    players[:, team == AWAY, 0] = 15.0
    labels = _seq((H, 10), (A, 5), (D, 5))
    return Match(name="s", fps=25, pitch=(105.0, 68.0), players_xy=players,
                 players_speed=np.zeros((T, 12)), team_of_slot=team,
                 ball_xy=np.tile([[5.0, 1.0]], (T, 1)), ball_speed=np.zeros(T),
                 half=np.ones(T, dtype=np.int8), labels=labels)


def test_swap_teams_is_a_consistent_mirror():
    m = _match()
    s = degrade.swap_teams(m)
    assert s.labels.tolist() == _seq((A, 10), (H, 5), (D, 5)).tolist()
    assert measure_attack_sign(s) == {1: 1}          # new home still attacks +x
    np.testing.assert_allclose(s.ball_xy[0], [-5.0, -1.0])
    s2 = degrade.swap_teams(s)
    np.testing.assert_allclose(s2.players_xy, m.players_xy)
    assert s2.labels.tolist() == m.labels.tolist()


def test_smooth_positions_keeps_gaps():
    xy = np.arange(20, dtype=float).reshape(10, 1, 2)
    xy[4] = np.nan
    sm = degrade.smooth_positions(xy, 3)
    assert np.isnan(sm[4]).all() and not np.isnan(sm[3]).any()
