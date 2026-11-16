"""Make clean IDSSE tracking look like the BAR-ATH automatic tracking (ATD).

Every corruption parameter is fitted to statistics of the BAR-ATH *tracking* (never its labels):

1. Smoothing. ATD positions are much smoother than DFL's 1 cm-quantised raw positions (median
   |second difference| 0.0014 m vs 0.010 m for players). A centred moving average whose width is
   chosen to match that statistic is applied to players and, separately, to the ball.
2. Dead-ball placeholder. DFL freezes the ball while it is dead (exactly zero displacement on
   100% of dead frames, ~0.5% of live ones), so IDSSE has no real ball position then. Frozen runs
   are treated as "the ball's whereabouts are unknown" before any further corruption.
3. Ball gaps. The ATD loses the ball in long blocks (median 10.7 s). Missing runs are drawn from the
   ATD's own empirical run lengths, started by a per-frame hazard that depends on
   (a) the true state, live or dead, with two base hazards h_live and h_dead calibrated by
       simulation (grid search) so that degraded IDSSE reproduces the ATD's observable curve
       "share of frames with the ball missing, by mean-player-speed decile" (player speed is the
       label-free quantity both datasets share), and
   (b) the ball's distance to the nearest line, with the ATD's measured relative hazard profile.
   A dead ball that is not dropped keeps the frozen placeholder: a ball standing still at the
   restart spot, which is what a visible dead ball mostly looks like.
4. Out-of-pitch thinning. Visible balls more than 1 m outside the pitch are dropped (as whole runs)
   with the probability that brings their share of visible frames to the ATD's 4.2%.
5. Players. Independent on/off dropout runs per player and short duplicate tracklets, with rates
   matching the ATD's per-team counts (mean ~10.25 of 11, more than 11 on ~6% of frames).
6. Wrong-team tracklets (degrader version 2, only when params has "swap"). A player's track is
   shown in a spare slot of the OTHER team for a run drawn from the ATD's own "one team 12+, the
   other <= 10" run lengths; his own slot is empty meanwhile. Runs start at a per-frame hazard,
   optionally only while he is within SWAP_CONTACT_M of an opponent (a duel). The hazard is
   calibrated by simulation so that the degraded IDSSE reproduces the ATD's 12-vs-10 frame share.

Seeds are fixed per (match, copy), so the training set is reproducible bit for bit.
"""

import json
import os
from dataclasses import replace

import numpy as np

from . import config, features
from .canonical import D, X, orient

SPEED_BINS = 10
EDGE_EDGES = (-1e9, 0.0, 3.0, 10.0, 1e9)       # [M] bins used for the ATD hazard profile
PLAYER_OFF_RUN_S = 2.0                          # [F] a lost player track typically recovers in ~2 s
DUP_RUN_S = 1.0                                 # [F] duplicate tracklets are short-lived
DUP_OFFSET_M = 0.7                              # [F] a duplicate sits on/next to the real player
SWAP_CONTACT_M = 2.0                            # [F] "within ~1-2 m of an opponent"
SWAP_SPARES = 3                                 # [D] spare wrong-team slots per team


def _runs(mask):
    e = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def _second_diff_p50(xy, graded):
    d2 = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=-1)
    g = graded[2:]
    if d2.ndim == 2:
        g = np.broadcast_to(g[:, None], d2.shape)
    ok = g & ~np.isnan(d2)
    return float(np.median(d2[ok]))


def _second_diff_p90(xy, graded):
    d2 = np.linalg.norm(xy[2:] - 2 * xy[1:-1] + xy[:-2], axis=-1)
    ok = graded[2:] & ~np.isnan(d2)
    return float(np.percentile(d2[ok], 90))


def smooth_positions(xy, width):
    """Centred moving average of width `width` (odd) along time, NaN-aware, per slot."""
    if width <= 1:
        return xy
    h = width // 2
    flat = xy.reshape(len(xy), -1)
    out = np.empty_like(flat)
    for j in range(flat.shape[1]):
        col = flat[:, j]
        sm = features.rolling_mean(col, h)
        out[:, j] = np.where(np.isnan(col), np.nan, sm)
    return out.reshape(xy.shape)


# -- fitting to ATD statistics ----------------------------------------------------------------------

def atd_statistics(atd):
    """Label-free statistics of the BAR-ATH ATD that the degrader reproduces."""
    m = orient(atd)
    g = m.graded
    f = features.compute(m)
    present = m.ball_present
    blocks = m.half_blocks()
    miss_runs = []
    for _, s, e in blocks:
        a, b = _runs(~present[s:e])
        miss_runs.extend((b - a).tolist())
    speed = f["player_speed"]
    cuts = np.nanquantile(speed[g], np.linspace(0, 1, SPEED_BINS + 1)[1:-1])
    sbin = np.digitize(speed, cuts)
    miss_by_speed = [float(np.mean(~present[g & (sbin == k)])) for k in range(SPEED_BINS)]

    L, W = m.pitch
    b = m.ball_xy
    edge = np.fmin(L / 2 - np.abs(b[:, 0]), W / 2 - np.abs(b[:, 1]))
    nxt_missing = np.r_[~present[1:], False]
    ebin = np.digitize(edge, EDGE_EDGES[1:-1])
    hazard = []
    for k in range(len(EDGE_EDGES) - 1):
        s = g & present & (ebin == k)
        hazard.append(float(nxt_missing[s].mean()))
    hazard = np.array(hazard)
    out_share = float(np.mean(f["ball_out"][g & present] > 0))

    counts = {}
    for team, name in ((0, "home"), (1, "away")):
        n = (~np.isnan(m.players_xy[:, m.team_of_slot == team, 0])).sum(axis=1)[g]
        counts[name] = {"mean": float(n.mean()), "over11": float(np.mean(n > 11))}
    return {
        "missing_share": float(np.mean(~present[g])),
        "missing_runs": sorted(int(r) for r in miss_runs),
        "speed_cuts": cuts.tolist(),
        "missing_by_speed_bin": miss_by_speed,
        "edge_hazard_rel": (hazard / hazard[-1]).tolist(),
        "out_share_visible": out_share,
        "player_d2_p50": _second_diff_p50(m.players_xy, g),
        "ball_d2_p90": _second_diff_p90(m.ball_xy, g),
        "counts": counts,
    }


def fit_smoothing(target, clean_matches, which):
    """Smallest odd moving-average width whose second-difference statistic reaches the ATD's."""
    for width in range(1, 41, 2):
        vals = []
        for m in clean_matches:
            g = m.graded
            if which == "players":
                vals.append(_second_diff_p50(smooth_positions(m.players_xy, width), g))
            else:
                live = g & (m.labels != D)
                vals.append(_second_diff_p90(smooth_positions(m.ball_xy, width), live))
        if np.mean(vals) <= target:
            return width
    return 41


def _prepare(match, params):
    """Per-match arrays the ball-gap simulation needs (computed once)."""
    L, W = match.pitch
    dead = match.labels == D
    ball = match.ball_xy.copy()
    live_ball = smooth_positions(np.where(dead[:, None], np.nan, ball), params["ball_smooth"])
    ball = np.where(dead[:, None], ball, live_ball)
    players = smooth_positions(match.players_xy, params["player_smooth"])
    mo = orient(replace(match, players_xy=players, ball_xy=ball))
    f = features.compute(mo)
    return {"ball": ball, "dead": dead, "pitch": (L, W), "graded": match.graded,
            "speed": f["player_speed"]}


def _ball_missing(rng, prep, h_live, h_dead, stats, beta=0.0):
    """Missing-ball mask: renewal gaps (state-, edge- and speed-dependent hazard) + out-of-pitch
    thinning. beta (per m/s) makes the tracker lose the ball more in slow phases."""
    L, W = prep["pitch"]
    ball, dead = prep["ball"], prep["dead"]
    T = len(dead)
    edge = np.fmin(L / 2 - np.abs(ball[:, 0]), W / 2 - np.abs(ball[:, 1]))
    rel = np.array(stats["edge_hazard_rel"])[np.digitize(edge, EDGE_EDGES[1:-1])]
    rel = np.where(np.isnan(edge), 1.0, rel)
    speed = np.nan_to_num(prep["speed"], nan=SPEED_REF)
    h = np.where(dead, h_dead, h_live) * rel * np.exp(beta * (SPEED_REF - np.clip(speed, 0, 6)))
    missing = _drop_runs(rng, np.ones(T, dtype=bool), np.clip(h, 0, 1),
                         np.array(stats["missing_runs"]))
    over = np.fmax(np.abs(ball[:, 0]) - L / 2, np.abs(ball[:, 1]) - W / 2)
    out = (over > config.BALL_OUT_MARGIN_M) & ~missing
    visible = (~missing).sum()
    target = stats["out_share_visible"]
    if out.sum() > target * visible and out.sum() > 0:
        p_drop = 1 - target * visible / out.sum()
        a, b = _runs(out)
        for s, e in zip(a, b):
            if rng.random() < p_drop:
                missing[s:e] = True
    return missing


HAZARD_GRID = np.logspace(-5.0, -1.0, 17)   # [D] per-frame start hazards searched
BETA_GRID = (0.0, 0.5, 1.0, 1.5, 2.0)        # [D] slow-phase hazard slopes searched (per m/s)
SPEED_REF = 1.5                              # [M] ~median 1 s mean player speed on BAR-ATH


def fit_ball_hazards(stats, preps, seed=12345):
    """(h_live, h_dead) minimising the weighted squared gap between the degraded IDSSE and the ATD
    curves of "share of frames with the ball missing, by mean-player-speed decile"."""
    cuts = np.array(stats["speed_cuts"])
    target = np.array(stats["missing_by_speed_bin"])
    bins = [np.digitize(p["speed"], cuts) for p in preps]
    best = None
    grid = [(a, b, c) for c in BETA_GRID for a in HAZARD_GRID for b in HAZARD_GRID]
    for h_live, h_dead, beta in grid:
        if True:
            miss = np.zeros(SPEED_BINS)
            n = np.zeros(SPEED_BINS)
            for i, p in enumerate(preps):
                rng = np.random.default_rng(seed + i)
                m = _ball_missing(rng, p, h_live, h_dead, stats, beta)
                g = p["graded"]
                for k in range(SPEED_BINS):
                    s = g & (bins[i] == k)
                    miss[k] += m[s].sum()
                    n[k] += s.sum()
            curve = miss / np.maximum(n, 1)
            err = float(np.sum(n * (curve - target) ** 2) / n.sum())
            if best is None or err < best[0]:
                best = (err, float(h_live), float(h_dead), float(beta), curve.tolist())
    return best


def fit(atd, clean_matches):
    stats = atd_statistics(atd)
    params = {
        "stats": stats,
        "player_smooth": fit_smoothing(stats["player_d2_p50"], clean_matches, "players"),
        "ball_smooth": fit_smoothing(stats["ball_d2_p90"], clean_matches, "ball"),
    }
    preps = [_prepare(m, params) for m in clean_matches]
    err, h_live, h_dead, beta, curve = fit_ball_hazards(stats, preps)
    params.update({"h_live": h_live, "h_dead": h_dead, "beta": beta, "fit_error": err,
                   "fit_curve": curve})
    return params


def save(params, path):
    with open(path, "w") as handle:
        json.dump(params, handle, indent=1, sort_keys=True)


# -- applying the corruption -----------------------------------------------------------------------

def _drop_runs(rng, eligible, hazard, run_lengths):
    """Boolean mask from an alternating renewal process: while not dropping, a run starts at each
    eligible frame with the given per-frame hazard; a run lasts a length drawn from run_lengths.
    Runs never overlap, so the long-run dropped share is h*mu / (1 + h*mu) for constant h."""
    T = len(eligible)
    u = rng.random(T)
    candidates = np.flatnonzero((u < hazard) & eligible)
    lengths = rng.choice(run_lengths, size=len(candidates))
    drop = np.zeros(T, dtype=bool)
    free_at = 0
    for c, n in zip(candidates.tolist(), lengths.tolist()):
        if c < free_at:
            continue
        drop[c:c + n] = True
        free_at = c + n
    return drop


def degrade(match, params, seed):
    """A corrupted copy of a clean labelled match."""
    rng = np.random.default_rng(seed)
    st = params["stats"]
    fps = match.fps
    T = match.n_frames

    # 1-4. smoothing, ball gaps, out-of-pitch thinning
    prep = _prepare(match, params)
    players = smooth_positions(match.players_xy, params["player_smooth"])
    ball = prep["ball"].copy()
    missing = _ball_missing(rng, prep, params["h_live"], params["h_dead"], st, params["beta"])
    ball[missing] = np.nan

    # 5. players: dropouts and duplicates
    S = players.shape[1]
    off_frames = PLAYER_OFF_RUN_S * fps
    extra = []
    extra_team = []
    for team in (0, 1):
        name = "home" if team == 0 else "away"
        c = st["counts"][name]
        slots = np.flatnonzero(match.team_of_slot == team)
        p_dup = c["over11"]
        p_off = max(0.0, (11 - c["mean"] + p_dup) / 11)
        h_off = p_off / ((1 - p_off) * off_frames)
        for j in slots:
            drop = _drop_runs(rng, ~np.isnan(players[:, j, 0]), h_off,
                              np.array([int(off_frames)]))
            players[drop, j] = np.nan
        # one extra duplicate slot per team, active on a share p_dup of frames
        dup_frames = int(DUP_RUN_S * fps)
        h_dup = p_dup / ((1 - p_dup) * dup_frames) if p_dup < 1 else 1.0
        active = _drop_runs(rng, np.ones(T, dtype=bool), h_dup, np.array([dup_frames]))
        src = rng.choice(slots, size=T)
        dup = players[np.arange(T), src] + rng.normal(0, DUP_OFFSET_M, size=(T, 2))
        dup[~active] = np.nan
        extra.append(dup[:, None, :])
        extra_team.append(team)
    team_of_slot = match.team_of_slot
    if "swap" in params:
        players, team_of_slot = wrong_team_swaps(rng, players, team_of_slot, params["swap"], fps)
    players = np.concatenate([players] + extra, axis=1)
    team_of_slot = np.concatenate([team_of_slot, np.array(extra_team, dtype=np.int8)])

    return replace(match, players_xy=players, ball_xy=ball, team_of_slot=team_of_slot,
                   players_speed=np.full(players.shape[:2], np.nan), oriented=False,
                   attack_sign={}, name="{}#deg{}".format(match.name, seed))


def contact_mask(players, team_of_slot, dist=SWAP_CONTACT_M):
    """(T, S) True where the slot's player is within dist of any opponent."""
    T, S, _ = players.shape
    out = np.zeros((T, S), dtype=bool)
    for team in (0, 1):
        mine, opp = np.flatnonzero(team_of_slot == team), np.flatnonzero(team_of_slot != team)
        d = np.linalg.norm(players[:, mine, None, :] - players[:, None, opp, :], axis=-1)
        out[:, mine] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=2) < dist
    return out


def wrong_team_swaps(rng, players, team_of_slot, sp, fps):
    """Step 6: move runs of a player's track into spare slots of the other team.

    sp: {"hazard": per-frame start hazard, "runs": run lengths in frames, "contact": bool}."""
    T, S, _ = players.shape
    present = ~np.isnan(players[..., 0])
    eligible = present & (contact_mask(players, team_of_slot) if sp["contact"] else True)
    runs = np.array(sp["runs"], dtype=np.int64)
    spares = np.full((T, 2 * SWAP_SPARES, 2), np.nan)
    spare_team = np.repeat(np.array([1, 0], dtype=np.int8), SWAP_SPARES)   # spares of the OTHER team
    busy = np.zeros((T, 2 * SWAP_SPARES), dtype=bool)
    out = players.copy()
    for j in range(S):
        mask = _drop_runs(rng, eligible[:, j], sp["hazard"], runs)
        a, b = _runs(mask)
        cols = np.flatnonzero(spare_team != team_of_slot[j])
        for s0, e0 in zip(a.tolist(), b.tolist()):
            free = [c for c in cols if not busy[s0:e0, c].any()]
            if not free:
                continue
            c = free[0]
            busy[s0:e0, c] = True
            spares[s0:e0, c] = out[s0:e0, j]
            out[s0:e0, j] = np.nan
    return (np.concatenate([out, spares], axis=1),
            np.concatenate([team_of_slot, spare_team]).astype(np.int8))


def swap_teams(match):
    """The same match with home and away exchanged: rotate 180 degrees, swap team ids and H/A.

    Used as augmentation so the classifier cannot learn "home is better"."""
    from .canonical import A, H
    lab = match.labels
    if lab is not None:
        lab = np.where(lab == H, A, np.where(lab == A, H, lab)).astype(np.int8)
    return replace(match, players_xy=-match.players_xy, ball_xy=-match.ball_xy,
                   team_of_slot=(1 - match.team_of_slot).astype(np.int8), labels=lab,
                   oriented=False, attack_sign={}, name=match.name + "#swap")
