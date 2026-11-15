"""Segmental decoder: Viterbi over {H, A, D} with switch costs and minimum durations.

Econometrically this is a hidden Markov (regime-switching) model whose regimes must last a minimum
time. Minimum durations are implemented exactly by expanding each label into a chain of m
sub-states (a semi-Markov model with a shifted-geometric duration). Every label change pays a
cost in nats, which is what stops flicker: a flip must be backed by more evidence than it costs.

The per-frame log-likelihoods are first averaged over a grid of STEP frames (0.2 s). Neighbouring
frames are strongly dependent, so summing per-frame scores would count the same observation many
times. Treating each 0.2 s step as one observation with weight STEP_WEIGHT is the honest discount.
"""

import numpy as np

H, A, D = 0, 1, 2
NEG = -1e18


def to_steps(ll, step):
    """(T, K) frame log-likelihoods -> (ceil(T/step), K) step means."""
    T, K = ll.shape
    n = -(-T // step)
    pad = n * step - T
    padded = np.vstack([ll, np.repeat(ll[-1:], pad, axis=0)]) if pad else ll
    return padded.reshape(n, step, K).mean(axis=1)


def viterbi(step_ll, min_steps, switch_cost, team_bonus=None, fast_ok=None, fast_steps=None):
    """Most likely label path under minimum durations and switch costs.

    step_ll: (N, K) log-likelihoods. min_steps: (K,) ints >= 1. switch_cost: (K, K) nats, the
    diagonal is ignored (staying is free). The start is free in every label, and no minimum
    applies to a segment cut off by the start or end of the sequence.

    Touch-aware option (labels 0 and 1 are the two teams): team_bonus (N,) nats are taken off the
    H <-> A switch cost at step t (floored at 0), and where fast_ok[t] a switch from the other
    team may start a possession that needs only fast_steps steps instead of min_steps.
    """
    N, K = step_ll.shape
    m = [int(x) for x in min_steps]
    delta = [np.full(mk, NEG) for mk in m]
    for k in range(K):
        delta[k][-1] = step_ll[0, k]
    enter_from = np.zeros((N, K), dtype=np.int8)
    stayed = np.zeros((N, K), dtype=bool)
    fast = np.zeros((N, K), dtype=bool)
    cost = np.asarray(switch_cost, dtype=np.float64)
    fpos = [None] * K
    if fast_ok is not None and fast_steps is not None:
        for k in (0, 1):
            if 1 <= int(fast_steps) < m[k]:
                fpos[k] = m[k] - int(fast_steps)

    for t in range(1, N):
        ends = np.array([d[-1] for d in delta])
        ct = cost
        if team_bonus is not None and team_bonus[t] > 0:
            ct = cost.copy()
            ct[0, 1] = max(cost[0, 1] - team_bonus[t], 0.0)
            ct[1, 0] = max(cost[1, 0] - team_bonus[t], 0.0)
        new = []
        for k in range(K):
            cand = ends - ct[:, k]
            cand[k] = NEG
            j = int(np.argmax(cand))
            enter = cand[j]
            enter_from[t, k] = j
            e = step_ll[t, k]
            d = delta[k]
            nd = np.empty_like(d)
            if m[k] == 1:
                stay = d[0]
                stayed[t, k] = stay >= enter
                nd[0] = max(stay, enter) + e
            else:
                nd[0] = enter + e
                nd[1:-1] = d[:-2] + e
                stay, advance = d[-1], d[-2]
                stayed[t, k] = stay >= advance
                nd[-1] = max(stay, advance) + e
                p = fpos[k]
                if p is not None and fast_ok[t]:
                    fe = cand[1 - k]                    # a fast start only from the other team
                    prev = max(stay, advance) if p == m[k] - 1 else d[p - 1]
                    if fe > prev:
                        fast[t, k] = True
                        nd[p] = fe + e
            new.append(nd)
        delta = new

    # Backtrack.
    k = int(np.argmax([d.max() for d in delta]))
    pos = int(np.argmax(delta[k]))
    path = np.empty(N, dtype=np.int8)
    for t in range(N - 1, -1, -1):
        path[t] = k
        if t == 0:
            break
        last = m[k] - 1
        if fast[t, k] and pos == fpos[k]:
            k = 1 - k                                 # fast start from the other team
            pos = m[k] - 1
            continue
        if pos == last and stayed[t, k]:
            continue                                  # stayed in the final sub-state
        if pos == 0:
            k = int(enter_from[t, k])
            pos = m[k] - 1
        else:
            pos -= 1
    return path


def steps_max(x, step):
    """(T,) frame values -> (ceil(T/step),) step maxima."""
    T = len(x)
    n = -(-T // step)
    pad = n * step - T
    padded = np.r_[x, np.repeat(x[-1:], pad)] if pad else x
    return padded.reshape(n, step).max(axis=1)


def decode_blocks(frame_ll, blocks, step, step_weight, min_steps, switch_cost, fill,
                  change=None, chg_w=0.0, chg_off=0.0, fast_thr=None, fast_steps=None):
    """Decode each [start, stop) block separately; frames outside every block get `fill`.

    change: optional (T,) log-odds that the ball changes team near each frame (the change head).
    Per step, the H <-> A switch cost is lowered by chg_w * max(g - chg_off, 0), g being the
    step's highest log-odds; where g >= fast_thr a switch may start a fast_steps-long possession.
    """
    T = frame_ll.shape[0]
    out = np.full(T, fill, dtype=np.int8)
    for start, stop in blocks:
        sl = step_weight * to_steps(frame_ll[start:stop], step)
        bonus = fast_ok = None
        if change is not None:
            g = steps_max(np.asarray(change[start:stop], dtype=np.float64), step)
            if chg_w > 0:
                bonus = chg_w * np.maximum(g - chg_off, 0.0)
            if fast_thr is not None and fast_steps is not None:
                fast_ok = g >= fast_thr
        path = viterbi(sl, min_steps, switch_cost, bonus, fast_ok, fast_steps)
        out[start:stop] = np.repeat(path, step)[:stop - start]
    return out
