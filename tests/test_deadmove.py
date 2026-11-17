import numpy as np

from possession import degrade


def _toy(T=25 * 60, fps=25, seed=0):
    rng = np.random.default_rng(seed)
    S = 22
    team = np.repeat([0, 1], 11).astype(np.int8)
    base = rng.uniform([-40, -25], [40, 25], size=(S, 2))
    players = base[None] + np.cumsum(rng.normal(0, 0.02, size=(T, S, 2)), axis=0)
    ball = players[:, 3] + 0.5
    labels = np.zeros(T, dtype=np.int8)
    labels[300:700] = 2                      # 16 s dead, home restart
    labels[700:] = 0
    labels[1000:1200] = 2
    labels[1200:] = 1                        # away restart
    ball[300:700] = ball[299]                # DFL frozen dead ball
    ball[1000:1200] = ball[999]
    half = np.ones(T, dtype=np.int8)
    return ball, players, team, labels, half, fps


def test_dead_ball_moves_only_inside_dead_time_and_no_teleports():
    ball, players, team, labels, half, fps = _toy()
    sp = {"p_stretch": 1.0, "kick_every_s": 5.0}
    out = degrade.dead_ball_moves(np.random.default_rng(1), ball, players, team, labels, half,
                                  sp, fps)
    live = labels != 2
    assert np.array_equal(out[live], ball[live])
    dead = labels == 2
    step = np.linalg.norm(out[1:] - out[:-1], axis=1) * fps
    assert np.nanmax(step[dead[1:] & dead[:-1]]) <= 30.0
    v = step[dead[1:] & dead[:-1]]
    assert np.nanmean(v > 2.0) > 0.05        # the ball does move during dead time
    # ends at the restart spot, stationary
    assert np.allclose(out[699], ball[700]) and np.allclose(out[1199], ball[1200])


def test_dead_ball_moves_deterministic():
    ball, players, team, labels, half, fps = _toy()
    sp = {"p_stretch": 1.0, "kick_every_s": 10.0}
    a = degrade.dead_ball_moves(np.random.default_rng(3), ball, players, team, labels, half, sp, fps)
    b = degrade.dead_ball_moves(np.random.default_rng(3), ball, players, team, labels, half, sp, fps)
    assert np.array_equal(np.nan_to_num(a, nan=-1), np.nan_to_num(b, nan=-1))
