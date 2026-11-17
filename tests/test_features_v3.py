from types import SimpleNamespace

import numpy as np

from possession import features_v3 as F3


def _match(seed=0, T=400, swap=False):
    rng = np.random.default_rng(seed)
    P = np.cumsum(rng.normal(0, 0.2, (T, 22, 2)), 0) + rng.uniform(-30, 30, (1, 22, 2))
    ball = np.cumsum(rng.normal(0, 0.6, (T, 2)), 0)
    P[50:60, 3] = np.nan
    ball[100:120] = np.nan
    team = np.r_[np.zeros(11), np.ones(11)].astype(np.int8)
    if swap:                                  # teams swapped and pitch mirrored
        team = 1 - team
        P = P * np.array([-1.0, -1.0])
        ball = ball * np.array([-1.0, -1.0])
    return SimpleNamespace(fps=25, players_xy=P, ball_xy=ball, team_of_slot=team)


def test_symmetric_under_swap_and_mirror():
    a = F3.extra_columns(_match())
    b = F3.extra_columns(_match(swap=True))
    for k in a:
        np.testing.assert_allclose(a[k], b[k], equal_nan=True, atol=1e-9, err_msg=k)


def test_nan_where_ball_missing_and_ranges():
    c = F3.extra_columns(_match())
    for k, v in c.items():
        assert v.shape == (400,)
        assert np.all(np.isnan(v[100:120])), k
    assert np.nanmax(c["p3_opp_in5"]) <= 11
    assert np.all((np.nan_to_num(c["p3_cos_angle"]) >= -1) & (np.nan_to_num(c["p3_cos_angle"]) <= 1))
    assert np.nanmax(c["p3_dwell_s"]) <= F3.DWELL_CAP_S


def test_dwell_counts_same_nearest_player():
    m = _match()
    m.players_xy[:] = 50.0
    m.players_xy[:, 0] = 0.0
    m.ball_xy[:] = 0.1
    c = F3.extra_columns(m)
    assert c["p3_dwell_s"][0] == 0 and abs(c["p3_dwell_s"][24] - 24 / 25) < 1e-9
    assert c["p3_opp_in3"][10] == 0
