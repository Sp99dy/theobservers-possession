"""PFF FC World Cup 2022 open data -> canonical Match with event-derived H/A/D labels (+cache).

EXTRA TRAINING DATA ONLY: PFF matches are never a test fold and never used for any selection.
Source: PFF FC "World Cup 2022" open data release (registration at pff.com, Google Drive), files
kept outside the repo in PFF_DIR (default ~/pff-wc2022), never committed or uploaded. Labelled by
PFF FC's analysts (event data); the H/A/D frame labels below are DERIVED by us from those events.

Tracking: <game>.jsonl.bz2, one JSON object per broadcast frame (~29.97 fps), positions in metres,
pitch-centred. PFF tracking is broadcast-derived: off-camera players are estimated and the raw
ball is missing when not seen. We use the *Smoothed* player/ball streams (PFF's interpolation,
the closest thing to IDSSE's complete optical tracking) and resample to 25 fps by nearest
timestamp, so the BAR-ATH-fitted degrader can then impose BAR-ATH's own gaps on top, exactly as
for IDSSE.

Ball gaps: PFF's ball is missing on ~half of the broadcast frames, mostly in long runs (replays,
close-ups; median ~13 s). Frames inside a ball gap longer than LONG_GAP_S get label X (no loss) in
load_labelled, AFTER the bridge, so the network is never taught "no ball for 10 s => dead".
Shorter gaps are interpolated by scripts/pff_build.py before degrading.

Event rule (documented in TECHNICAL.md), from Event Data/<game>.json (event times are video
seconds, mapped to each period's clock through the tracking frames' videoTimeMs):
  * team: an on-the-ball event (OTB) gives the ball to its team (home_ball) from its frame; a loose
    ball stays with the last team (challenge definition).
  * dead: from a stoppage until the next restart. A restart is an OTB event with a set-piece type
    other than open play (kick-off, throw-in, corner, goal kick, free kick, penalty). The stoppage
    is the first OUT event since the previous OTB event if there is one, else the end of the
    previous OTB event (a foul or a goal is logged on the event before the restart). A dead
    stretch longer than MAX_DEAD_S is labelled X (ungraded for training) instead of D.
  * graded: each period from its first to its last game event.
Then the same 0.5 s controlled-possession bridge as for IDSSE (io_idsse.relabel_controlled).
"""

import bz2
import glob
import json
import os

import numpy as np

from . import config
from .canonical import A, D, H, X, Match
from .io_idsse import GAP, relabel_controlled

PFF_DIR = os.environ.get("PFF_DIR") or os.path.expanduser("~/pff-wc2022")
CACHE_DIR = os.path.join(config.REPO_DIR, "cache_pff")
PARSER_VERSION = 2
FPS = 25
MAX_DEAD_S = 120.0                 # [D] longer "dead" stretches are logging gaps -> X
LONG_GAP_S = 2.0                   # [D] ball gaps longer than this are masked (X) for training
RESTARTS = ("K", "T", "F", "G", "C", "P", "D")   # kick-off, throw-in, free kick, goal kick, corner, penalty, drop ball
OPEN_PLAY = ("O", None, "")
DEFAULT_PITCH = (105.0, 68.0)


def _find(game_id, kind):
    pats = {"tracking": "**/{}.jsonl.bz2", "tracking_plain": "**/{}.jsonl",
            "meta": "**/Metadata/{}.json", "events": "**/Event*/{}.json",
            "roster": "**/Roster*/{}.json"}
    hits = sorted(glob.glob(os.path.join(PFF_DIR, pats[kind].format(game_id)), recursive=True))
    return hits[0] if hits else None


def game_ids():
    """Games with all four files (tracking .jsonl.bz2, metadata, roster, event data)."""
    ids = set()
    for p in glob.glob(os.path.join(PFF_DIR, "**", "*.jsonl.bz2"), recursive=True):
        gid = os.path.basename(p).split(".")[0]
        if all(_find(gid, k) for k in ("meta", "events", "roster")):
            ids.add(gid)
    return sorted(ids, key=lambda s: int(s) if s.isdigit() else s)


def _open(path):
    return bz2.open(path, "rt") if path.endswith(".bz2") else open(path)


def _pitch(game_id):
    p = _find(game_id, "meta")
    if not p:
        return DEFAULT_PITCH
    with open(p) as h:
        meta = json.load(h)
    if isinstance(meta, list):
        meta = meta[0]
    try:
        pitch = meta["stadium"]["pitches"][0]
        return (float(pitch["length"]), float(pitch["width"]))
    except (KeyError, IndexError, TypeError):
        return DEFAULT_PITCH


def _xy(objs, key="jerseyNum"):
    out = {}
    for o in objs or ():
        if o is None or o.get("x") is None:
            continue
        out[str(o.get(key))] = (float(o["x"]), float(o["y"]))
    return out


def parse_tracking(path):
    """Per period: list of frames (t, video_s, home{jersey:xy}, away{..}, ball xy|None, raw ball
    seen, visible players home, away)."""
    periods = {}
    vis = lambda ps: sum(1 for p in ps or () if p and p.get("visibility") == "VISIBLE")
    with _open(path) as h:
        for line in h:
            if not line.strip():
                continue
            f = json.loads(line)
            per = f.get("period")
            if per not in (1, 2) or f.get("periodElapsedTime") is None:
                continue
            hp = _xy(f.get("homePlayersSmoothed") or f.get("homePlayers"))
            ap = _xy(f.get("awayPlayersSmoothed") or f.get("awayPlayers"))
            bs = f.get("ballsSmoothed")
            if isinstance(bs, list):
                bs = bs[0] if bs else None
            ball = (float(bs["x"]), float(bs["y"])) if bs and bs.get("x") is not None else None
            raw = f.get("balls") or []
            raw_seen = any(b and b.get("x") is not None for b in raw)
            periods.setdefault(per, []).append(
                (float(f["periodElapsedTime"]), float(f.get("videoTimeMs") or np.nan) / 1000.0,
                 hp, ap, ball, raw_seen, vis(f.get("homePlayers")), vis(f.get("awayPlayers"))))
    return periods


def parse_events(game_id, periods):
    """{period: [(t, type, home_ball, setpiece, end_t)]} with t on the period clock."""
    path = _find(game_id, "events")
    with open(path) as h:
        evs = json.load(h)
    clock = {}
    for per, frames in periods.items():
        v = np.array([f[1] for f in frames])
        e = np.array([f[0] for f in frames])
        ok = np.isfinite(v)
        o = np.argsort(v[ok])
        clock[per] = (v[ok][o], e[ok][o])
    out = {}
    for ev in evs:
        ge = ev.get("gameEvents") or {}
        per = ge.get("period")
        if per not in clock or ev.get("startTime") is None:
            continue
        v, e = clock[per]
        t = float(np.interp(float(ev["startTime"]), v, e))
        et = ev.get("endTime")
        tend = None if et is None else float(np.interp(float(et), v, e))
        typ = ge.get("gameEventType")
        if typ in ("FIRSTKICKOFF", "SECONDKICKOFF"):
            typ = "OTB"
        out.setdefault(per, []).append((t, typ, ge.get("homeTeam"), ge.get("setpieceType"), tend))
    return out


def event_labels(times, evs, fps=FPS):
    """(T,) labels on the period's 25 fps grid (times = grid seconds) from its events."""
    T = len(times)
    lab = np.full(T, X, dtype=np.int8)
    graded = np.zeros(T, dtype=bool)
    if not evs:
        return lab, graded
    evs = sorted(evs, key=lambda e: e[0])
    fr = lambda t: int(np.clip(np.searchsorted(times, t), 0, T))
    s, e = fr(evs[0][0]), fr(evs[-1][0]) + 1
    graded[s:e] = True
    otb = [ev for ev in evs if ev[1] == "OTB" and ev[2] is not None]
    owner = None
    for ev in otb:
        if owner is not None:
            lab[owner[1]:fr(ev[0])] = owner[0]
        owner = (H if ev[2] else A, fr(ev[0]))
    if owner is not None:
        lab[owner[1]:e] = owner[0]
    if otb:
        lab[s:fr(otb[0][0])] = H if otb[0][2] else A
    # dead stretches
    prev_otb = None
    out_since = None
    for ev in evs:
        if ev[1] == "OUT" and out_since is None:
            out_since = ev[0]
        if ev[1] != "OTB":
            continue
        if ev[3] in RESTARTS and prev_otb is not None:
            stop = out_since if out_since is not None else (
                prev_otb[4] if prev_otb[4] is not None else prev_otb[0] + 1.0)
            a, b = fr(stop), fr(ev[0])
            if b > a:
                lab[a:b] = D if (ev[0] - stop) <= MAX_DEAD_S else X
        prev_otb = ev
        out_since = None
    lab[~graded] = X
    return lab, graded


def build_match(game_id, periods, events, pitch):
    jerseys = {0: set(), 1: set()}
    for per in periods.values():
        for fr in per:
            jerseys[0].update(fr[2])
            jerseys[1].update(fr[3])
    order = [(0, j) for j in sorted(jerseys[0])] + [(1, j) for j in sorted(jerseys[1])]
    slot = {k: i for i, k in enumerate(order)}
    team_of_slot = np.array([t for t, _ in order], dtype=np.int8)
    S = len(order)
    parts = {k: [] for k in ("pxy", "bxy", "lab", "half", "raw", "vis")}
    for i, per in enumerate(sorted(periods)):
        frames = sorted(periods[per], key=lambda f: f[0])
        ts = np.array([f[0] for f in frames])
        grid = np.arange(ts[0], ts[-1], 1.0 / FPS)
        j = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
        j = np.where(np.abs(ts[j - 1] - grid) <= np.abs(ts[j] - grid), j - 1, j)
        T = len(grid)
        pxy = np.full((T, S, 2), np.nan)
        bxy = np.full((T, 2), np.nan)
        raw = np.zeros(T, dtype=bool)
        vis = np.zeros((T, 2), dtype=np.int16)
        for k, src in enumerate(j):
            _, _, hp, ap, ball, rs, vh, va = frames[src]
            for jer, xy in hp.items():
                pxy[k, slot[(0, jer)]] = xy
            for jer, xy in ap.items():
                pxy[k, slot[(1, jer)]] = xy
            if ball is not None:
                bxy[k] = ball
            raw[k] = rs
            vis[k] = (vh, va)
        lab, graded = event_labels(grid, events.get(per, []))
        if i:
            parts["pxy"].append(np.full((GAP, S, 2), np.nan))
            parts["bxy"].append(np.full((GAP, 2), np.nan))
            parts["lab"].append(np.full(GAP, X, dtype=np.int8))
            parts["half"].append(np.zeros(GAP, dtype=np.int8))
            parts["raw"].append(np.zeros(GAP, dtype=bool))
            parts["vis"].append(np.zeros((GAP, 2), dtype=np.int16))
        parts["pxy"].append(pxy)
        parts["bxy"].append(bxy)
        parts["lab"].append(lab)
        parts["half"].append(np.where(graded, per, 0).astype(np.int8))
        parts["raw"].append(raw)
        parts["vis"].append(vis)
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    m = Match(name="pff_{}".format(game_id), fps=FPS, pitch=pitch, players_xy=cat["pxy"],
              players_speed=np.full(cat["pxy"].shape[:2], np.nan), team_of_slot=team_of_slot,
              ball_xy=cat["bxy"], ball_speed=np.full(len(cat["bxy"]), np.nan), half=cat["half"],
              labels=cat["lab"])
    return m, cat["raw"], cat["vis"]


def _cache_path(game_id):
    return os.path.join(CACHE_DIR, "pff_{}_v{}.npz".format(game_id, PARSER_VERSION))


def load_match(game_id, use_cache=True, extras=False):
    """Clean Match (event labels, no bridge). extras=True also returns (raw_ball_seen, visible
    players per team) per frame for the visibility report."""
    cache = _cache_path(game_id)
    if use_cache and os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as z:
            m = Match(name="pff_{}".format(game_id), fps=FPS, pitch=tuple(z["pitch"].tolist()),
                      players_xy=z["pxy"], players_speed=np.full(z["pxy"].shape[:2], np.nan),
                      team_of_slot=z["team"], ball_xy=z["bxy"],
                      ball_speed=np.full(len(z["bxy"]), np.nan), half=z["half"], labels=z["lab"])
            return (m, z["raw"], z["vis"]) if extras else m
    path = _find(game_id, "tracking") or _find(game_id, "tracking_plain")
    periods = parse_tracking(path)
    events = parse_events(game_id, periods)
    m, raw, vis = build_match(game_id, periods, events, _pitch(game_id))
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = cache + ".tmp.npz"
        np.savez_compressed(tmp, pitch=np.array(m.pitch), pxy=m.players_xy.astype(np.float32), team=m.team_of_slot,
                 bxy=m.ball_xy, half=m.half, lab=m.labels, raw=raw, vis=vis)
        os.replace(tmp, cache)
    return (m, raw, vis) if extras else m


def load_labelled(game_id, control_min_s=None):
    m = load_match(game_id)
    bridge = config.CONTROL_MIN_S if control_min_s is None else control_min_s
    from dataclasses import replace
    lab = relabel_controlled(m.labels, m.half_blocks(), int(round(bridge * m.fps)))
    lab[long_ball_gaps(m)] = X
    return replace(m, labels=lab)


def long_ball_gaps(m, max_s=LONG_GAP_S):
    """(T,) bool: frames inside a run of missing ball longer than max_s."""
    miss = ~m.ball_present
    out = np.zeros_like(miss)
    d = np.diff(np.r_[0, miss.astype(np.int8), 0])
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if b - a > max_s * m.fps:
            out[a:b] = True
    return out
