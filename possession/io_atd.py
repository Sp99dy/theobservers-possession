"""BAR-ATH tactical ATD (Metrica FIFA/EPTS export) -> canonical arrays, with an on-disk cache.

Raw line format: "<frame_id>:<slot>;<slot>;...;<slot>:<ball>" where every group is "x,y,speed",
coordinates normalised 0..1 (origin top-left, y down), NaN for a missing observation.
video_frame = frame_id - 1 (verified by the organisers, starter/load_atd.py).

The slot order and channel order are read from the metadata's DataFormatSpecification and checked,
not assumed. The parser reads the 195 MB file once; the cache makes later loads take ~1 s.
The cache lives in config.CACHE_DIR, which is gitignored: match data never enters the repository.
"""

import io
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from . import config, organisers
from .canonical import AWAY, HOME, Match, normalised_to_metres

# Bump when the parser or the cached layout changes, so stale caches are rebuilt.
PARSER_VERSION = 1
CHANNELS = ("x", "y", "s")


def _tag(element):
    return element.tag.split("}")[-1]


def parse_metadata(meta_path):
    """Team names, frame rate and the team of each slot, in the slot order of the raw file."""
    root = ET.parse(meta_path).getroot()
    teams = [t for t in root.iter() if _tag(t) == "Team"]
    if len(teams) != 2:
        raise ValueError("expected 2 teams, found {}".format(len(teams)))
    team_ids = [t.get("id") for t in teams]
    team_names = [t.findtext("{*}Name") or t.findtext("Name") for t in teams]

    team_of_player = {p.get("id"): p.get("teamId") for p in root.iter() if _tag(p) == "Player"}
    channel = {c.get("id"): (c.get("playerId"), c.get("channelId"))
               for c in root.iter() if _tag(c) == "PlayerChannel"}

    specs = [e for e in root.iter() if _tag(e) == "DataFormatSpecification"]
    if len(specs) != 1:
        raise ValueError("expected 1 DataFormatSpecification, found {}".format(len(specs)))
    slot_players = []
    for register in specs[0].iter():
        refs = [r for r in register if _tag(r) == "PlayerChannelRef"]
        if not refs:
            continue
        owners = {channel[r.get("playerChannelId")][0] for r in refs}
        channels = tuple(channel[r.get("playerChannelId")][1] for r in refs)
        if len(owners) != 1 or channels != CHANNELS:
            raise ValueError("unexpected slot layout: {} owners, channels {}".format(
                len(owners), channels))
        slot_players.append(owners.pop())

    ball = tuple(b.get("channelId") for b in root.iter() if _tag(b) == "BallChannelRef")
    if ball != CHANNELS:
        raise ValueError("unexpected ball channels {}".format(ball))

    # FIFA/EPTS convention (and kloppy's): the first team listed is the home side.
    team_of_slot = np.array([HOME if team_of_player[p] == team_ids[0] else AWAY
                             for p in slot_players], dtype=np.int8)
    rate = [e.text for e in root.iter() if _tag(e) == "FrameRate"]
    return {
        "team_names": team_names,
        "team_of_slot": team_of_slot,
        "n_slots": len(slot_players),
        "frame_rate": float(rate[0]) if rate else None,
    }


def parse_raw(raw_path, n_slots):
    """(frame_id[N], players[N, S, 3], ball[N, 3]) as float64, channels x, y, speed.

    Separator totals are checked before parsing, so a malformed line cannot be silently padded
    with NaN. pandas 3's default float parser gives bit-identical values to Python's float() on
    this file (checked on all 28.3M values against the old round_trip parser, 2026-11-14), which
    keeps us bit-compatible with the organisers' pure-Python baseline; a unit test guards it.
    """
    with open(raw_path, "rb") as handle:
        data = handle.read()
    data = data.replace(b"\r\n", b"\n")
    if not data.endswith(b"\n"):
        data += b"\n"
    n = data.count(b"\n")
    expected = {b":": 2 * n, b";": (n_slots - 1) * n, b",": 2 * (n_slots + 1) * n}
    for sep, count in expected.items():
        actual = data.count(sep)
        if actual != count:
            raise ValueError("{}: {!r} appears {} times, expected {} for {} lines".format(
                raw_path, sep, actual, count, n))
    data = data.replace(b":", b",").replace(b";", b",")
    n_cols = 1 + 3 * (n_slots + 1)
    table = pd.read_csv(io.BytesIO(data), header=None, dtype=np.float64, engine="c")
    if table.shape != (n, n_cols):
        raise ValueError("parsed shape {} != ({}, {})".format(table.shape, n, n_cols))
    values = table.to_numpy()
    frame_id = values[:, 0]
    if np.isnan(frame_id).any() or (frame_id != np.round(frame_id)).any():
        raise ValueError("non-integer frame ids")
    players = values[:, 1:1 + 3 * n_slots].reshape(n, n_slots, 3)
    ball = values[:, 1 + 3 * n_slots:]
    return frame_id.astype(np.int64), np.ascontiguousarray(players), np.ascontiguousarray(ball)


def _cache_path(raw_path):
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    return os.path.join(config.CACHE_DIR, "{}_v{}.npz".format(stem, PARSER_VERSION))


def _fingerprint(path):
    stat = os.stat(path)
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns),
            "parser_version": PARSER_VERSION}


def load_raw(meta_path, raw_path, use_cache=True):
    """Normalised arrays straight from the file, cached as .npz. Returns a dict."""
    meta = parse_metadata(meta_path)
    cache = _cache_path(raw_path)
    stamp = _fingerprint(raw_path)
    if use_cache and os.path.isfile(cache):
        with np.load(cache, allow_pickle=False) as z:
            if json.loads(str(z["stamp"])) == stamp:
                return dict(meta, frame_id=z["frame_id"], players=z["players"], ball=z["ball"])

    frame_id, players, ball = parse_raw(raw_path, meta["n_slots"])
    if use_cache:
        os.makedirs(config.CACHE_DIR, exist_ok=True)
        tmp = cache + ".tmp.npz"
        np.savez(tmp, frame_id=frame_id, players=players, ball=ball,
                 stamp=np.array(json.dumps(stamp)))
        os.replace(tmp, cache)
    return dict(meta, frame_id=frame_id, players=players, ball=ball)


def raw_to_video_index(raw, n_video_frames):
    """Re-index the parsed records by video frame (video_frame = frame_id - 1), NaN-padded."""
    video = raw["frame_id"] - 1
    if video.min() < 0 or video.max() >= n_video_frames or len(np.unique(video)) != len(video):
        raise ValueError("frame ids do not map one-to-one into 0..{}".format(n_video_frames - 1))
    n_slots = raw["players"].shape[1]
    players = np.full((n_video_frames, n_slots, 3), np.nan)
    ball = np.full((n_video_frames, 3), np.nan)
    players[video] = raw["players"]
    ball[video] = raw["ball"]
    return players, ball


def half_array(game, n_video_frames):
    """(T,) int8 half id per video frame from the organisers' published half boundaries."""
    half = np.zeros(n_video_frames, dtype=np.int8)
    for h, first, last in organisers.games().half_video_blocks(game):
        half[first:last + 1] = h
    return half


def load_match(game=None, data_dir=None, use_cache=True):
    """The BAR-ATH tactical ATD as a canonical Match (metres, not yet orientation-normalised)."""
    g = organisers.games()
    game = game or g.GAME
    meta_path, raw_path = g.require(["atd_meta", "atd_raw"], game, data_dir or config.MTAH_DATA_DIR)
    raw = load_raw(meta_path, raw_path, use_cache)
    n_video = g.source_frames("tactical", game)
    pitch = tuple(g.GAMES[game]["pitch"])
    players, ball = raw_to_video_index(raw, n_video)

    px, py = normalised_to_metres(players[..., 0], players[..., 1], pitch)
    bx, by = normalised_to_metres(ball[:, 0], ball[:, 1], pitch)
    return Match(
        name=game,
        fps=int(g.GAMES[game]["fps"]),
        pitch=pitch,
        players_xy=np.stack([px, py], axis=-1),
        players_speed=players[..., 2],
        team_of_slot=raw["team_of_slot"],
        ball_xy=np.stack([bx, by], axis=-1),
        ball_speed=ball[:, 2],
        half=half_array(game, n_video),
    )
