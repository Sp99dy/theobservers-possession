"""Decoder, features, kickoff and tag-prior logic on synthetic inputs."""

import numpy as np

from possession import decode, features, kickoff, tagging_prior
from possession.canonical import AWAY, HOME, X
from possession.io_tagging import Tag

H, A, D = 0, 1, 2
COST = ((0, 4.5, 5), (4.5, 0, 5), (3, 3, 0))


def _ll(seq, strength=1.5):
    """Step log-likelihoods favouring the given label sequence."""
    ll = np.zeros((len(seq), 3))
    ll[np.arange(len(seq)), seq] = strength
    return ll


def test_viterbi_follows_strong_long_segments():
    seq = np.array([H] * 30 + [A] * 30 + [D] * 30 + [H] * 30)
    assert decode.viterbi(_ll(seq), (5, 5, 10), COST).tolist() == seq.tolist()


def test_viterbi_removes_short_blip():
    seq = np.array([H] * 30 + [A] * 2 + [H] * 30)
    assert set(decode.viterbi(_ll(seq), (5, 5, 10), COST).tolist()) == {H}


def test_viterbi_enforces_min_duration_inside_sequence():
    # 4 steps of A evidence, strong enough to pay the switches, but A must last >= 5 steps.
    seq = np.array([H] * 30 + [A] * 4 + [H] * 30)
    path = decode.viterbi(_ll(seq, strength=10.0), (5, 5, 10), COST)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[-1, path, -1]) != 0))
    inner = runs[1:-1]
    assert (inner >= 5).all()


def test_viterbi_no_evidence_no_switch():
    path = decode.viterbi(np.zeros((50, 3)), (5, 5, 10), COST)
    assert len(set(path.tolist())) == 1


def test_decode_blocks_fills_outside_and_upsamples():
    T = 100
    ll = np.zeros((T, 3))
    ll[10:40, H] = 2.0
    ll[60:95, A] = 2.0
    out = decode.decode_blocks(ll, [(10, 40), (60, 95)], 5, 1.0, (1, 1, 1), COST, fill=X)
    assert (out[:10] == X).all() and (out[40:60] == X).all() and (out[95:] == X).all()
    assert (out[10:40] == H).all() and (out[60:95] == A).all()


def test_rolling_mean_ignores_nan():
    x = np.array([1.0, np.nan, 3.0, np.nan, np.nan, np.nan, np.nan])
    r = features.rolling_mean(x, 1)
    assert r[1] == 2.0 and r[0] == 1.0 and np.isnan(r[5])


def _kickoff_features(T, lineup):
    f = {k: np.zeros(T) for k in ("n_home", "n_away", "home_own_half", "away_own_half")}
    f["n_home"][:] = f["n_away"][:] = 10
    f["home_own_half"][:] = f["away_own_half"][:] = 0.5
    a, b = lineup
    f["home_own_half"][a:b] = f["away_own_half"][a:b] = 1.0
    f["player_speed"] = np.where(np.arange(T) < b, 0.5, 2.5)
    f["d_centre_home"] = np.full(T, 10.0)
    f["d_centre_away"] = np.full(T, 1.0)          # away player on the centre spot
    return f


def test_kickoff_detected_with_kicking_team():
    T = 2000
    f = _kickoff_features(T, (200, 400))
    half = np.ones(T, dtype=np.int8)
    kos = kickoff.detect(f, half)
    assert len(kos) == 1
    k = kos[0]
    assert abs(k.start - 200) <= 13 and abs(k.end - 400) <= 13 and k.kicking_team == AWAY
    assert kickoff.pre_kickoff_spans(kos, [(1, 0, T)]) == [(0, k.end)]


def test_short_separation_is_not_a_kickoff():
    f = _kickoff_features(2000, (200, 250))      # 2 s only
    assert kickoff.detect(f, np.ones(2000, dtype=np.int8)) == []


def test_tag_prior_signs():
    tags = [Tag("build up", 0, 10, HOME, ""), Tag("defending in middle third", 20, 30, HOME, ""),
            Tag("progression", 40, 50, None, "")]
    dz, dy = tagging_prior.evidence(60, tags, [])
    assert (dy[0:10] > 0).all() and (dy[20:30] < 0).all() and (dy[40:50] == 0).all()
    assert (dz == 0).all()
