"""Simulate cross-platform float noise: perturb the ensemble emission by uniform +-eps (several
seeds) and count frames that change in the decoded output, with and without quantization.

    python scripts/check_float_robustness.py --model-dir models/v51 --eps 1e-6

Aggregates only (BAR-ATH frame counts, no labels).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import io_atd, io_tagging, v5  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models/v51")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    cfg = json.load(open(os.path.join(args.model_dir, "v5.json")))
    match = io_atd.load_match()
    tags = io_tagging.load()
    base_fn = v5.emission_fn(cfg["members"], args.model_dir)
    cache = {}

    def emission(X):
        if "ll" not in cache:
            cache["ll"] = base_fn(X)
        return cache["ll"] + cache.get("noise", 0.0)

    orig = v5.emission_fn
    v5.emission_fn = lambda members, model_dir=None: emission
    for q in (None, cfg.get("quantize")):
        cfg_q = dict(cfg, quantize=q)
        tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "robust_cfg")
        os.makedirs(tmp, exist_ok=True)
        json.dump(cfg_q, open(os.path.join(tmp, "v5.json"), "w"))
        cache.pop("noise", None)
        ref = v5.predict(match, tags, model_dir=tmp)
        for s in range(args.seeds):
            rng = np.random.default_rng(s)
            cache["noise"] = rng.uniform(-args.eps, args.eps, size=cache["ll"].shape)
            out = v5.predict(match, tags, model_dir=tmp)
            print("quantize {} eps {:g} seed {}: frames changed {}".format(
                q, args.eps, s, int((out != ref).sum())), flush=True)
    v5.emission_fn = orig


if __name__ == "__main__":
    main()
