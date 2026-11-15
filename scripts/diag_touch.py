"""Diagnostic: is "team of the last touch" the signal behind the possessions v2 misses?

    python scripts/diag_touch.py

Uses DFL event data (every pass, ball action, claim, shot, restart, with team and time) as a
perfect touch detector, on IDSSE open data. Upper bound only: real touches would have to be
detected from the (degraded) ball trajectory.
"""
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from possession import config, cv, features, io_idsse, model  # noqa: E402
import diag_deadball as dd  # noqa: E402
import diag_liveplay as dl  # noqa: E402

TOUCH = {"Play": "Team", "OtherBallAction": "Team", "BallClaiming": "Team", "ShotAtGoal": "Team",
         "FreeKick": "Team", "ThrowIn": "Team", "CornerKick": "Team", "GoalKick": "Team"}


def touches(mid, home_id):
    root = ET.parse(io_idsse.local_path(mid, "event")).getroot()
    out = []
    for ev in root.iter("Event"):
        t = dd.ts(ev.get("EventTime"))
        for child in ev.iter():
            if child.tag in TOUCH and child.get(TOUCH[child.tag]):
                out.append((t, 0 if child.get(TOUCH[child.tag]) == home_id else 1))
                break
    return out


def main():
    with open(os.path.join(REPO, "models", "decoder_v2.json")) as fh:
        dp = json.load(fh)["decoder"]
    with open(os.path.join(config.CACHE_DIR, "degrader_params.json")) as fh:
        params = json.load(fh)
    oof = np.load(os.path.join(config.CACHE_DIR, "oof_v2.npz"))
    agg = {k: [0, 0] for k in ("all live frames", "matched", "missed", "missed, first 1 s")}
    nxt = {k: [0, 0] for k in ("missed",)}
    lag = []
    for mid in io_idsse.MATCH_IDS:
        _, y, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, features.FEATURE_VERSION)
        blocks = dl.blocks_from(g)
        ll = oof[mid].astype(np.float64)
        pred = cv.decode_ll(ll, blocks, dp, fill=3)
        meta = io_idsse.parse_meta(io_idsse.local_path(mid, "meta"))
        clock = dd.half_clock(mid)
        starts = [s for s, _ in blocks]
        last = np.full(len(y), -1)
        nextt = np.full(len(y), -1)
        idx = []
        for t, team in touches(mid, meta["home_id"]):
            i, _ = dd.to_index(t, clock, starts, len(y))
            if i is not None:
                idx.append((i, team))
        idx.sort()
        for i, team in idx:
            last[i] = team
        for s, e in blocks:            # forward-fill last touch, back-fill next touch, per half
            cur = -1
            for k in range(s, e):
                if last[k] >= 0:
                    cur = last[k]
                last[k] = cur
        for i, team in idx:
            nextt[i] = team
        for s, e in blocks:
            cur = -1
            for k in range(e - 1, s - 1, -1):
                if nextt[k] >= 0:
                    cur = nextt[k]
                nextt[k] = cur
        live = g & (y < 2)
        agg["all live frames"][0] += int(np.sum(last[live] == y[live]))
        agg["all live frames"][1] += int(live.sum())
        gs, ps = dl.spells(y, blocks), dl.spells(pred, blocks)
        mg, _ = dl.greedy(gs, ps)
        c = dl.classify(gs, ps, mg)
        tpos = np.array([i for i, _ in idx])
        for i, sp in enumerate(gs):
            k = c[i][0]
            sl = slice(sp[0], sp[1])
            if k in ("matched", "missed"):
                agg[k][0] += int(np.sum(last[sl] == sp[2]))
                agg[k][1] += sp[1] - sp[0]
            if k == "missed":
                f = slice(sp[0], min(sp[1], sp[0] + 25))
                agg["missed, first 1 s"][0] += int(np.sum(last[f] == sp[2]))
                agg["missed, first 1 s"][1] += f.stop - f.start
                nxt["missed"][0] += int(np.sum(nextt[sl] == sp[2]))
                nxt["missed"][1] += sp[1] - sp[0]
                near = tpos[np.abs(tpos - sp[0]) <= 75]
                if len(near):
                    lag.append((sp[0] - near[np.argmin(np.abs(near - sp[0]))]) / 25)
    print("Share of frames where 'team of the last touch' (DFL events) = true possession team")
    for k, (a, n) in agg.items():
        print("   {:22s} {:5.1%}  ({} frames)".format(k, a / n, n))
    a, n = nxt["missed"]
    print("   {:22s} {:5.1%}  (team of the NEXT touch, missed possessions)".format("", a / n))
    lag = np.array(lag)
    print("Missed possessions: label start minus nearest touch event: median {:+.2f}s, |.|<0.5 s for {:.0%}".format(
        np.median(lag), np.mean(np.abs(lag) < 0.5)))


if __name__ == "__main__":
    main()
