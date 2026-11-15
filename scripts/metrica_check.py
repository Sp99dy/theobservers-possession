"""Secondary validation (SPEC 6) on Metrica sample games 1 and 2, with event-derived labels.

    python scripts/metrica_check.py [--decoder PATH ...]

Scores the organisers' baseline rule, v1 (rules, no tags) and v2 (committed model, no tags) with
the organisers' metric. Never used to tune anything: it only checks that the ranking chosen on
IDSSE CV holds on Metrica's own tracking, whose ball is missing on ~40% of frames like BAR-ATH's.
Extra --decoder files (JSON with a "decoder" key) score v2 with other decoder settings.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import baseline, cv, io_metrica, metrics, model, v1, v2  # noqa: E402
from possession.canonical import X  # noqa: E402


def baseline50(m):
    players = np.concatenate([m.players_xy, np.zeros(m.players_xy.shape[:2] + (1,))], axis=2)
    ball = np.concatenate([m.ball_xy, np.zeros((m.n_frames, 1))], axis=1)
    codes = baseline.nearest_player_codes(players, ball, m.team_of_slot, (1.0, 1.0))
    letters = baseline.baseline_labels(codes, 50)
    out = np.select([letters == "H", letters == "A"], [0, 1], 2).astype(np.int8)
    out[~m.graded] = X
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", nargs="*", default=[])
    args = ap.parse_args()
    mdl = model.load(v2.MODEL_PATH)
    methods = {
        "baseline50": baseline50,
        "v1 (no tags)": lambda m: v1.predict(m, None, use_tags=False),
        "v2 (no tags)": lambda m: v2.predict(m, None, use_tags=False, mdl=mdl),
    }
    for path in args.decoder:
        with open(path) as handle:
            dec = json.load(handle)["decoder"]
        methods["v2 + " + os.path.basename(path)] = (
            lambda m, dec=dec: v2.predict(m, None, use_tags=False, mdl=mdl, decoder=dec))

    matches = [io_metrica.load_match(g) for g in io_metrica.GAMES]
    for name, fn in methods.items():
        items, preds = [], []
        for m in matches:
            pred = fn(m)
            blocks = [(s, e) for _, s, e in m.half_blocks()]
            items.append((None, m.labels, blocks))
            preds.append(pred)
            s = cv.pooled_score([items[-1]], preds=[pred])
            print("  {:<28} {:<10} spellF1 {:.4f}  macroF1 {:.4f}".format(
                name, m.name, s["spell_f1_mean"], s["macro_f1"]))
        s = cv.pooled_score(items, preds=preds)
        print("{:<30} pooled  {}".format(name, metrics.format_summary(s)), flush=True)


if __name__ == "__main__":
    main()
