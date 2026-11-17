"""Simulated Smart Tagging on IDSSE: set-piece tags derived from the DFL events (review idea 7).

    python scripts/tag_simulate.py            # writes out/tag_events.json, prints alignment stats

Each DFL set-piece event (ThrowIn, CornerKick, GoalKick, FreeKick, KickOff, Penalty) is placed on
the match frame axis from its EventTime (via the first tracking frame of each half). A copy's tags
are then drawn per copy (seeded): each event is kept with the BAR-ATH per-type rate (BAR-ATH tags
per match / mean IDSSE events per match of that type, capped at 1; penalties as free kicks), and its
start is moved by an offset drawn from the BAR-ATH "tag start - restart" offsets
(out/tag_barath_stats.json, label-free). Features per frame: seconds since the last and to the next
tag start, clipped at CLIP_S. Labels are only used to print the event-vs-restart alignment check.
"""

import datetime as dt
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import io_idsse  # noqa: E402

OUT = os.path.join(REPO, "out")
TYPES = {"ThrowIn": "throw-in", "CornerKick": "corner kick", "GoalKick": "goal kick",
         "FreeKick": "free kick", "Penalty": "free kick", "KickOff": "kick off"}
CLIP_S = 60.0
FPS = 25
_EV = re.compile(rb'<Event [^>]*EventTime="([^"]+)"[^>]*>\s*<(' +
                 b"|".join(k.encode() for k in TYPES) + rb')[ >/]')
_BALLSET = re.compile(rb'<FrameSet GameSection="(firstHalf|secondHalf)"[^>]*TeamId="BALL"[^>]*>'
                      rb'\s*<Frame N="(\d+)" T="([^"]+)"')


def _t(s):
    return dt.datetime.fromisoformat(s.decode() if isinstance(s, bytes) else s).timestamp()


def match_events(mid, half):
    with open(io_idsse.local_path(mid, "tracking"), "rb") as h:
        head = h.read()
    starts = {}
    for m in _BALLSET.finditer(head):
        hn = 1 if m.group(1) == b"firstHalf" else 2
        starts[hn] = (_t(m.group(3)), int(np.flatnonzero(half == hn)[0]))
    del head
    with open(io_idsse.local_path(mid, "event"), "rb") as h:
        ev = h.read()
    out = []
    for m in _EV.finditer(ev):
        t = _t(m.group(1))
        hn = 2 if t >= starts[2][0] - 60 else 1
        f = starts[hn][1] + int(round((t - starts[hn][0]) * FPS))
        if 0 <= f < len(half) and half[f] == hn:
            out.append((f, TYPES[m.group(2).decode()]))
    return sorted(out)


def keep_rates(events, bar):
    per = {}
    for evs in events.values():
        for _, ty in evs:
            per[ty] = per.get(ty, 0) + 1
    n = len(events)
    bt = {}
    for k, v in bar["types"].items():
        key = "kick off" if k.startswith("kick off") else k
        bt[key] = bt.get(key, 0) + v
    return {ty: min(1.0, bt.get(ty, 0) / (c / n)) for ty, c in per.items()}


def copy_tags(events, rates, offsets, seed):
    rng = np.random.default_rng(seed)
    out = []
    for f, ty in events:
        if rng.random() < rates.get(ty, 0.0):
            out.append(int(f + round(rng.choice(offsets) * FPS)))
    return np.array(sorted(out), dtype=np.int64)


def tag_features(tag_starts, n):
    """(n, 2) float32: seconds since the last tag start and to the next one, clipped."""
    t = np.arange(n)
    since = np.full(n, CLIP_S, np.float32)
    to = np.full(n, CLIP_S, np.float32)
    if len(tag_starts):
        k = np.searchsorted(tag_starts, t, side="right")
        has = k > 0
        since[has] = np.minimum((t[has] - tag_starts[k[has] - 1]) / FPS, CLIP_S)
        nx = k < len(tag_starts)
        to[nx] = np.minimum((tag_starts[k[nx]] - t[nx]) / FPS, CLIP_S)
    return np.stack([since, to], 1)


_CACHE = {}


def load_sim():
    if not _CACHE:
        with open(os.path.join(OUT, "tag_events.json")) as h:
            _CACHE.update(json.load(h))
    return _CACHE


def features_for_copy(mid, offset, swapped, n):
    sim = load_sim()
    idx = io_idsse.MATCH_IDS.index(mid)
    seed = 777_000 + 100 * idx + 10 * (offset if offset is not None else 99) + int(swapped)
    tags = copy_tags([tuple(e) for e in sim["events"][mid]], sim["rates"], sim["offsets_s"], seed)
    return tag_features(tags, n)


def main():
    with open(os.path.join(OUT, "tag_barath_stats.json")) as h:
        bar = json.load(h)
    events, align = {}, []
    for mid in io_idsse.MATCH_IDS:
        m = io_idsse.load_labelled(mid, 0.5)
        evs = match_events(mid, m.half)
        events[mid] = evs
        lab = m.labels
        cut = np.flatnonzero(np.diff(lab)) + 1
        rs = np.array([c for c in cut if lab[c - 1] == 2 and lab[c] in (0, 1)])
        for f, _ in evs:
            if len(rs):
                align.append((f - rs[np.argmin(np.abs(rs - f))]) / FPS)
    rates = keep_rates(events, bar)
    a = np.array(align)
    print("events per match by type:", {ty: round(sum(e[1] == ty for v in events.values()
                                                       for e in v) / 7, 1) for ty in rates})
    print("keep rates (BAR-ATH tags / IDSSE events):", {k: round(v, 2) for k, v in rates.items()})
    print("event frame - nearest DFL restart (s): p10/50/90 {}  |within 1 s| {:.0%}".format(
        np.round(np.percentile(a, [10, 50, 90]), 2).tolist(), np.mean(np.abs(a) <= 1)))
    with open(os.path.join(OUT, "tag_events.json"), "w") as h:
        json.dump({"events": events, "rates": rates, "offsets_s": bar["offsets_s"]}, h)


if __name__ == "__main__":
    main()
