"""IDSSE / DFL open data -> canonical Match with H/A/D labels (+cache).

DFL "positions raw observed" XML: one <FrameSet> per person (or the ball) per half, each holding
<Frame N=.. X=.. Y=..> at 25 fps. Coordinates are already metres, pitch-centred. The ball's frames
carry BallStatus (1 alive / 0 dead) and BallPossession (1 home / 2 away): DFL's official
corrected tracking flags, the same kind of source the BAR-ATH ground truth was derived from.

Labels: D where BallStatus == 0, else H / A from BallPossession. Referees are dropped.
Halves are concatenated with a GAP of unlabelled frames between them (half = 0, label X), so no
rolling feature or decoder segment can bridge the half-time break.
"""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from . import config
from .canonical import A, AWAY, D, H, HOME, X, Match

MATCH_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
COMPETITION = {"J03WMX": "DFL-COM-000001", "J03WN1": "DFL-COM-000001"}
FILENAME = {
    "meta": "DFL_02_01_matchinformation_{competition}_DFL-MAT-{match_id}.xml",
    "event": "DFL_03_02_events_raw_{competition}_DFL-MAT-{match_id}.xml",
    "tracking": "DFL_04_03_positions_raw_observed_{competition}_DFL-MAT-{match_id}.xml",
}
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/{filename}"
PARSER_VERSION = 1
GAP = 250          # 10 s of nothing between the halves
SECTIONS = ("firstHalf", "secondHalf")

_FRAMESET = re.compile(rb'<FrameSet ([^>]*)>')
_ATTR = re.compile(rb'(\w+)="([^"]*)"')
_FRAME = re.compile(rb'<Frame N="(\d+)"[^>]*? X="([-0-9.]+)" Y="([-0-9.]+)"')
_BALL = re.compile(rb'<Frame N="(\d+)"[^>]*? X="([-0-9.]+)" Y="([-0-9.]+)"[^>]*? S="([-0-9.]+)"'
                   rb'[^>]*? BallPossession="(\d)" BallStatus="(\d)"')


def _filename(match_id, kind):
    return FILENAME[kind].format(competition=COMPETITION.get(match_id, "DFL-COM-000002"),
                                 match_id=match_id)


def idsse_url(match_id, kind):
    return BASE_URL.format(filename=_filename(match_id, kind))


def local_path(match_id, kind):
    return os.path.join(config.IDSSE_DIR, _filename(match_id, kind))


def parse_meta(path):
    root = ET.parse(path).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    return {"home_id": general.get("HomeTeamId"), "away_id": general.get("GuestTeamId"),
            "pitch": (float(env.get("PitchX")), float(env.get("PitchY")))}


def parse_tracking(path, home_id, away_id):
    """{section: {"ball": (N, x, y, s, poss, status) arrays, "players": [(person, team, N, x, y)]}}"""
    with open(path, "rb") as handle:
        data = handle.read()
    heads = list(_FRAMESET.finditer(data))
    out = {s: {"ball": None, "players": []} for s in SECTIONS}
    for i, head in enumerate(heads):
        attrs = {k.decode(): v.decode() for k, v in _ATTR.findall(head.group(1))}
        body = data[head.end(): heads[i + 1].start() if i + 1 < len(heads) else len(data)]
        section = attrs.get("GameSection")
        if section not in out:
            continue
        team = attrs.get("TeamId")
        if team == "BALL":
            rows = np.array(_BALL.findall(body), dtype=np.float64)
            out[section]["ball"] = rows
        elif team in (home_id, away_id):
            rows = np.array(_FRAME.findall(body), dtype=np.float64)
            out[section]["players"].append((attrs["PersonId"], HOME if team == home_id else AWAY,
                                            rows))
    return out


def build_match(match_id, parsed, meta, fps=25):
    persons = {}
    for s in SECTIONS:
        for person, team, _ in parsed[s]["players"]:
            persons.setdefault(person, team)
    order = sorted(persons, key=lambda p: (persons[p], p))
    slot = {p: i for i, p in enumerate(order)}
    team_of_slot = np.array([persons[p] for p in order], dtype=np.int8)

    pieces = []
    for h, s in enumerate(SECTIONS, start=1):
        ball = parsed[s]["ball"]
        n0, n1 = int(ball[:, 0].min()), int(ball[:, 0].max())
        T = n1 - n0 + 1
        pxy = np.full((T, len(order), 2), np.nan)
        for person, _, rows in parsed[s]["players"]:
            idx = rows[:, 0].astype(np.int64) - n0
            ok = (idx >= 0) & (idx < T)
            pxy[idx[ok], slot[person]] = rows[ok, 1:3]
        bxy = np.full((T, 2), np.nan)
        bs = np.full(T, np.nan)
        lab = np.full(T, X, dtype=np.int8)
        idx = ball[:, 0].astype(np.int64) - n0
        bxy[idx] = ball[:, 1:3]
        bs[idx] = ball[:, 3]
        poss, status = ball[:, 4].astype(int), ball[:, 5].astype(int)
        lab[idx] = np.where(status == 0, D, np.where(poss == 1, H, np.where(poss == 2, A, X)))
        pieces.append((h, pxy, bxy, bs, lab))

    S = len(order)
    parts = {"pxy": [], "bxy": [], "bs": [], "lab": [], "half": []}
    for i, (h, pxy, bxy, bs, lab) in enumerate(pieces):
        if i:
            parts["pxy"].append(np.full((GAP, S, 2), np.nan))
            parts["bxy"].append(np.full((GAP, 2), np.nan))
            parts["bs"].append(np.full(GAP, np.nan))
            parts["lab"].append(np.full(GAP, X, dtype=np.int8))
            parts["half"].append(np.zeros(GAP, dtype=np.int8))
        parts["pxy"].append(pxy)
        parts["bxy"].append(bxy)
        parts["bs"].append(bs)
        parts["lab"].append(lab)
        parts["half"].append(np.full(len(lab), h, dtype=np.int8))
    cat = {k: np.concatenate(v) for k, v in parts.items()}
    return Match(name=match_id, fps=fps, pitch=meta["pitch"], players_xy=cat["pxy"],
                 players_speed=np.full(cat["pxy"].shape[:2], np.nan), team_of_slot=team_of_slot,
                 ball_xy=cat["bxy"], ball_speed=cat["bs"], half=cat["half"], labels=cat["lab"])


def relabel_controlled(labels, half_blocks, min_frames):
    """Apply the challenge's "controlled possession" rule to DFL's touch-level possession flag.

    CHALLENGE.md: a change of possession needs *controlled* possession by the other team; a touch,
    a deflection or a won aerial duel without control does not count, and while the ball is loose
    possession stays with the last team that had it. Every restart has a clear possessor.

    So inside each live stretch (between dead balls) the first spell keeps its team, and a later
    spell only changes the team if it lasts >= min_frames; shorter spells are given to the team
    that had control. Dead-ball frames are untouched.
    """
    out = labels.copy()
    for _, s, e in half_blocks:
        c = out[s:e]
        b = np.flatnonzero(np.diff(c)) + 1
        starts, stops = np.r_[0, b], np.r_[b, len(c)]
        current = None
        for a, z in zip(starts, stops):
            lab = c[a]
            if lab == D or lab == X:
                current = None
                continue
            if current is None or z - a >= min_frames:
                current = lab
            else:
                c[a:z] = current
        out[s:e] = c
    return out


def load_labelled(match_id, control_min_s=None):
    """Clean match with labels converted to the challenge's convention (bridge control_min_s,
    default config.CONTROL_MIN_S)."""
    m = load_match(match_id)
    bridge = config.CONTROL_MIN_S if control_min_s is None else control_min_s
    lab = relabel_controlled(m.labels, m.half_blocks(), int(round(bridge * m.fps)))
    from dataclasses import replace
    return replace(m, labels=lab)


def _cache_path(match_id):
    return os.path.join(config.CACHE_DIR, "idsse_{}_v{}.npz".format(match_id, PARSER_VERSION))


def load_match(match_id, use_cache=True):
    """Canonical labelled Match for one IDSSE game (clean, i.e. not degraded)."""
    cache = _cache_path(match_id)
    if use_cache and os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as z:
            return Match(name=match_id, fps=25, pitch=tuple(z["pitch"].tolist()),
                         players_xy=z["pxy"], players_speed=np.full(z["pxy"].shape[:2], np.nan),
                         team_of_slot=z["team"], ball_xy=z["bxy"], ball_speed=z["bs"],
                         half=z["half"], labels=z["lab"])
    meta = parse_meta(local_path(match_id, "meta"))
    parsed = parse_tracking(local_path(match_id, "tracking"), meta["home_id"], meta["away_id"])
    m = build_match(match_id, parsed, meta)
    if use_cache:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        tmp = cache + ".tmp.npz"
        np.savez(tmp, pitch=np.array(m.pitch), pxy=m.players_xy, team=m.team_of_slot,
                 bxy=m.ball_xy, bs=m.ball_speed, half=m.half, lab=m.labels)
        os.replace(tmp, cache)
    return m
