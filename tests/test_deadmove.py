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


def test_law9_mask_needs_five_frames_and_override_forces_d():
    from possession import rules
    out = np.zeros(30)
    out[2:6] = 1          # 4 frames: too short
    out[10:15] = 1        # 5 frames: out of play
    m = rules.law9_mask(out)
    assert not m[2:6].any() and m[10:15].all() and m.sum() == 5
    ll = np.log(np.full((30, 3), 1 / 3))
    ll2 = rules.law9_override(ll, m)
    assert np.all(np.argmax(ll2[m], 1) == 2) and np.array_equal(ll2[~m], ll[~m])


def test_ballfill_fills_only_short_agreeing_smooth_gaps():
    from possession import ballfill
    T = 400
    tac = np.zeros((T, 2))
    tac[:, 0] = np.linspace(0, 40, T)
    bro = tac + 0.5
    valid = np.ones(T, dtype=bool)
    graded = np.ones(T, dtype=bool)
    tac[50:60] = np.nan                       # short, edges agree, smooth -> filled
    tac[100:200] = np.nan                     # 4 s: too long
    tac[250:260] = np.nan
    bro[245] += 5.0                           # edge disagrees -> not filled
    tac[300:310] = np.nan
    bro[305] += 4.0                           # jump inside the gap -> not filled
    f = ballfill.gap_fill(tac, bro, valid, graded, 25)
    assert f[50:60].all() and f.sum() == 10


def test_replay_mask_uses_lagged_invalid_homography(monkeypatch):
    from types import SimpleNamespace
    from possession import ballfill, replay
    T = 50
    valid = np.ones(100, dtype=bool)
    valid[10:20] = False
    monkeypatch.setattr(ballfill, "broadcast_ball", lambda pitch, data_dir=None: (None, valid))
    monkeypatch.setattr(ballfill, "LAG", {1: 5})
    half = np.ones(T, dtype=np.int8)
    half[:3] = 0
    m = SimpleNamespace(pitch=(105, 68), n_frames=T, graded=half > 0,
                        half_blocks=lambda: [(1, 3, T)])
    r = replay.replay_mask(m)
    assert np.flatnonzero(r).tolist() == list(range(15, 25))
