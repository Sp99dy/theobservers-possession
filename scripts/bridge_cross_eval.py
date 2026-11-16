"""Train bridge x eval bridge: nested IDSSE CV of v4-style models trained at different bridges,
each scored against the labels of every bridge.

    python scripts/bridge_cross_eval.py [--train f2_all:2.0 b10_f2:1.0 b05_f2:0.5] [--jobs 9]

Needs cache/oof_<tag>.npz and cache/cv_<tag>_decoder.json from
`POSSESSION_CONTROL_MIN_S=<b> scripts/cv_idsse.py --tag <tag> --feature-version 2 --nested`.

For every (train tag, eval bridge) cell, exactly as cv_idsse.py: the decoder is tuned on pooled
out-of-fold spellF1 against the eval-bridge labels (starting from that run's decoder), then re-tuned
leaving each match out (NESTED). Per-match wins compare each cell with the 2 s model (v4's design)
in the same eval column, both nested. Writes cache/bridge_cross_eval.json.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, cv, io_idsse, model  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH, blocks_of  # noqa: E402


def _where(tag):
    """'dir/tag' -> (dir, tag); a bare tag lives in config.CACHE_DIR."""
    d, t = os.path.split(tag)
    return (d or config.CACHE_DIR), t


def eval_items(tag, bridge):
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    d, t = _where(tag)
    z = np.load(os.path.join(d, "oof_{}.npz".format(t)))
    items = []
    for mid in io_idsse.MATCH_IDS:
        _, _, g, _ = model.copy_features(mid, params, *model.EVAL_COPY, 2)
        y = model.copy_labels(mid, model.EVAL_COPY[1], bridge)
        items.append((z[mid].astype(np.float64), y, blocks_of(g)))
    return items


def cell(args):
    tag, bridge = args
    t0 = time.time()
    items = eval_items(tag, bridge)
    d, t = _where(tag)
    with open(os.path.join(d, "cv_{}_decoder.json".format(t))) as handle:
        start = json.load(handle)["decoder"]
    dp, tuned = cv.tune_decoder(items, start=start)
    preds = []
    for i in range(len(items)):
        rest = [it for j, it in enumerate(items) if j != i]
        dpi, _ = cv.tune_decoder(rest, start=dp, sweeps=1)
        preds.append(cv.decode_ll(items[i][0], items[i][2], dpi, fill=3))
    nested = cv.pooled_score(items, preds=preds)
    per = [cv.pooled_score([it], preds=[p])["spell_f1_mean"] for it, p in zip(items, preds)]
    return {"train": tag, "eval_bridge": bridge, "decoder": dp, "tuned": tuned, "nested": nested,
            "per_match": per, "seconds": time.time() - t0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", default=["f2_all:2.0", "b10_f2:1.0", "b05_f2:0.5"])
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--base", help="tag the per-match wins compare with (default: the 2 s model)")
    ap.add_argument("--out", default="bridge_cross_eval.json")
    args = ap.parse_args()
    train = [(t.split(":")[0], float(t.split(":")[1])) for t in args.train]
    bridges = sorted({b for _, b in train} | {2.0, 1.0, 0.5}, reverse=True)
    tasks = [(tag, e) for tag, _ in train for e in bridges]
    with ProcessPoolExecutor(args.jobs) as ex:
        res = list(ex.map(cell, tasks))
    by = {(r["train"], r["eval_bridge"]): r for r in res}
    base = args.base or train[[b for _, b in train].index(config.DEFAULT_CONTROL_MIN_S)][0]

    print("NESTED spellF1  (rows: train bridge, columns: eval bridge)")
    print("{:<16}".format("") + "".join("eval {:<9}".format("{:.1f}s".format(e)) for e in bridges))
    for tag, b in train:
        print("{:<16}".format("{} {:.1f}s".format(tag[-24:], b)) +
              "".join("{:<14.4f}".format(by[(tag, e)]["nested"]["spell_f1_mean"]) for e in bridges))
    print("\nper-match nested diff vs {} on the same eval labels; wins /7".format(base))
    for tag, b in train:
        if tag == base:
            continue
        for e in bridges:
            d = np.array(by[(tag, e)]["per_match"]) - np.array(by[(base, e)]["per_match"])
            print("{} train {:.1f}s eval {:.1f}s: {:+.4f}  [{}]  wins {}/7".format(
                tag, b, e, by[(tag, e)]["nested"]["spell_f1_mean"] - by[(base, e)]["nested"]["spell_f1_mean"],
                " ".join("{:+.3f}".format(x) for x in d), int((d > 0).sum())))
    print("\npossessions (nested pred / gt), mean length pred / gt")
    for tag, b in train:
        print("{} train {:.1f}s: ".format(tag, b) + "  ".join(
            "eval {:.1f}s {}/{} {:.1f}/{:.1f}s".format(
                e, by[(tag, e)]["nested"]["n_pred_spells"], by[(tag, e)]["nested"]["n_gt_spells"],
                by[(tag, e)]["nested"]["mean_spell_duration_s"],
                by[(tag, e)]["nested"]["gt_mean_spell_duration_s"]) for e in bridges))
    with open(os.path.join(config.CACHE_DIR, args.out), "w") as handle:
        json.dump(res, handle, indent=1)


if __name__ == "__main__":
    main()
