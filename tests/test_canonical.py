"""Coordinate conversion and orientation, on synthetic data only."""

import numpy as np
import pytest

from possession.canonical import AWAY, HOME, Match, measure_attack_sign, normalised_to_metres, orient

PITCH = (105.3, 68.0)


def test_corners_and_centre():
    # Metrica/EPTS: origin top-left, y down. Canonical: centred, y up.
    x, y = normalised_to_metres([0.0, 1.0, 0.5, 0.0, 1.0], [0.0, 1.0, 0.5, 1.0, 0.0], PITCH)
    np.testing.assert_allclose(x, [-52.65, 52.65, 0.0, -52.65, 52.65])
    np.testing.assert_allclose(y, [34.0, -34.0, 0.0, -34.0, 34.0])


def test_distances_are_isotropic_in_metres():
    # 0.1 of the length and 0.1 of the width are different distances in metres.
    x, y = normalised_to_metres([0.5, 0.6, 0.5], [0.5, 0.5, 0.6], PITCH)
    assert abs((x[1] - x[0]) - 10.53) < 1e-9
    assert abs((y[0] - y[2]) - 6.8) < 1e-9


def test_nan_stays_nan():
    x, y = normalised_to_metres([np.nan], [np.nan], PITCH)
    assert np.isnan(x[0]) and np.isnan(y[0])


def _two_half_match(home_x_half1=-20.0, away_x_half1=15.0):
    """10 frames: 0-1 pre-match, 2-5 half 1, 6 halftime, 7-9 half 2 (teams have swapped ends).
    Six players a side (the direction check ignores frames with fewer than five)."""
    T, S = 10, 12
    team = np.array([HOME] * 6 + [AWAY] * 6, dtype=np.int8)
    half = np.array([0, 0, 1, 1, 1, 1, 0, 2, 2, 2], dtype=np.int8)
    players = np.zeros((T, S, 2))
    for t in range(T):
        sign = -1 if half[t] == 2 else 1
        players[t, team == HOME, 0] = sign * home_x_half1
        players[t, team == AWAY, 0] = sign * away_x_half1
        players[t, :, 1] = sign * np.array([5.0, -5.0, 3.0, -3.0, 1.0, -1.0] * 2)
    ball = np.tile([[10.0, 2.0]], (T, 1))
    return Match(name="synthetic", fps=25, pitch=PITCH, players_xy=players,
                 players_speed=np.zeros((T, S)), team_of_slot=team, ball_xy=ball,
                 ball_speed=np.zeros(T), half=half)


def test_attack_sign_per_half():
    m = _two_half_match()
    # Half 1: home sits at negative x, so it defends the -x goal and attacks +x.
    assert measure_attack_sign(m) == {1: 1, 2: -1}


def test_orient_rotates_only_the_flipped_half():
    m = _two_half_match()
    o = orient(m)
    assert o.oriented and o.attack_sign == {1: 1, 2: -1}
    # Half 2 rotated 180 degrees: home back on the -x side, y mirrored too.
    np.testing.assert_allclose(o.home_xy[8, :, 0], [-20.0] * 6)
    np.testing.assert_allclose(o.players_xy[8], m.players_xy[3])   # same picture as half 1
    np.testing.assert_allclose(o.ball_xy[8], [-10.0, -2.0])
    # Half 1 and non-half frames untouched.
    np.testing.assert_allclose(o.players_xy[3], m.players_xy[3])
    np.testing.assert_allclose(o.players_xy[0], m.players_xy[0])
    # After orienting, home attacks +x everywhere.
    assert measure_attack_sign(o) == {1: 1, 2: 1}
    # Idempotent.
    assert orient(o) is o


def test_half_blocks_must_be_contiguous():
    m = _two_half_match()
    assert m.half_blocks() == [(1, 2, 6), (2, 7, 10)]
    m.half[4] = 0
    with pytest.raises(ValueError):
        m.half_blocks()
