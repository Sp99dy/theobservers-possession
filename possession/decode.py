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


def viterbi(step_ll, min_steps, switch_cost):
    """Most likely label path under minimum durations and switch costs.

    step_ll: (N, K) log-likelihoods. min_steps: (K,) ints >= 1. switch_cost: (K, K) nats, the
    diagonal is ignored (staying is free). The start is free in every label, and no minimum
    applies to a segment cut off by the start or end of the sequence.
    """
    N, K = step_ll.shape
    m = [int(x) for x in min_steps]
    delta = [np.full(mk, NEG) for mk in m]
    for k in range(K):
        delta[k][-1] = step_ll[0, k]
    enter_from = np.zeros((N, K), dtype=np.int8)
    stayed = np.zeros((N, K), dtype=bool)
    cost = np.asarray(switch_cost, dtype=np.float64)

    for t in range(1, N):
        ends = np.array([d[-1] for d in delta])
        new = []
        for k in range(K):
            cand = ends - cost[:, k]
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
        if pos == last and stayed[t, k]:
            continue                                  # stayed in the final sub-state
        if pos == 0:
            k = int(enter_from[t, k])
            pos = m[k] - 1
        else:
            pos -= 1
    return path


def decode_blocks(frame_ll, blocks, step, step_weight, min_steps, switch_cost, fill):
    """Decode each [start, stop) block separately; frames outside every block get `fill`."""
    T = frame_ll.shape[0]
    out = np.full(T, fill, dtype=np.int8)
    for start, stop in blocks:
        sl = step_weight * to_steps(frame_ll[start:stop], step)
        path = viterbi(sl, min_steps, switch_cost)
        out[start:stop] = np.repeat(path, step)[:stop - start]
    return out
