"""Touch-aware decoder: exact against brute force on small random problems (no match data)."""

import itertools

import numpy as np
import pytest

from possession import decode


def path_score(path, ll, m, cost, bonus, fast_ok, fast_steps):
    """Score of a label path, or None if it breaks a minimum duration."""
    N = len(path)
    segs = []
    s = 0
    for t in range(1, N + 1):
        if t == N or path[t] != path[s]:
            segs.append((s, t, path[s]))
            s = t
    total = sum(ll[t, path[t]] for t in range(N))
    for i, (a, b, k) in enumerate(segs):
        if i > 0:
            j = segs[i - 1][2]
            c = cost[j][k]
            if {j, k} == {0, 1} and bonus is not None:
                c = max(c - bonus[a], 0.0)
            total -= c
        if i == 0 or i == len(segs) - 1:
            continue                                  # cut off by the start or end: no minimum
        need = m[k]
        if (fast_ok is not None and k in (0, 1) and segs[i - 1][2] == 1 - k and fast_ok[a]
                and fast_steps is not None and fast_steps < m[k]):
            need = fast_steps
        if b - a < need:
            return None
    return total


@pytest.mark.parametrize("seed", range(40))
def test_viterbi_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    N, K = 8, 3
    ll = rng.normal(0, 1.5, (N, K))
    m = [int(x) for x in rng.integers(1, 5, K)]
    cost = rng.uniform(0, 2, (K, K))
    bonus = rng.uniform(0, 2, N) * (rng.random(N) < 0.5) if seed % 2 else None
    fast_ok = rng.random(N) < 0.4 if seed % 4 >= 2 else None
    fast_steps = int(rng.integers(1, 4)) if fast_ok is not None else None
    best = max(sc for p in itertools.product(range(K), repeat=N)
               if (sc := path_score(p, ll, m, cost, bonus, fast_ok, fast_steps)) is not None)
    path = decode.viterbi(ll, m, cost, bonus, fast_ok, fast_steps)
    got = path_score(tuple(int(x) for x in path), ll, m, cost, bonus, fast_ok, fast_steps)
    assert got is not None and got == pytest.approx(best)


def test_no_change_input_is_unchanged():
    rng = np.random.default_rng(1)
    ll = rng.normal(0, 1, (400, 3))
    cost = ((0, 2, 3), (2, 0, 3), (1, 1, 0))
    a = decode.decode_blocks(ll, [(0, 400)], 5, 1.0, (4, 4, 3), cost, fill=3)
    b = decode.decode_blocks(ll, [(0, 400)], 5, 1.0, (4, 4, 3), cost, fill=3,
                             change=np.full(400, -5.0), chg_w=0.0, fast_thr=99.0, fast_steps=2)
    assert np.array_equal(a, b)
