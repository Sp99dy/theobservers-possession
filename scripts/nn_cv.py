"""Leave-one-match-out CV of the neural sequence emission model (HANDOFF §4.1).

    POSSESSION_CONTROL_MIN_S=0.5 python scripts/nn_cv.py --name gru05 --arch gru

Trains on offsets 0-8 (each with its mirror) of the six other matches, predicts the held-out
match's offset-9 copy at 25 fps, and writes cache/oof_nn_<name>.npz in the layout
ensemble_experiments.py reads. Each fold is saved to out/nn_<name>/<match>.npy as it finishes, so
an interrupted run resumes. Labels follow POSSESSION_CONTROL_MIN_S, as in cv_idsse.py.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, io_idsse, model, nn as pnn  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="gru", choices=sorted(pnn.ARCHS))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--windows", type=int, default=3000, help="training windows per epoch")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--offsets", default="0-8")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", nargs="*", default=None, help="subset of match ids (debug)")
    ap.add_argument("--train-cache", default=None,
                    help="read the TRAINING copies and their degrader params from this cache dir")
    ap.add_argument("--eval-caches", nargs="*", default=None,
                    help="score the held-out offset-9 copy from each of these cache dirs and write "
                         "oof_nn_<name>.npz into each (default: POSSESSION_CACHE_DIR only)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    lo, hi = (int(v) for v in args.offsets.split("-"))
    offsets = [o for o in range(lo, hi + 1) if o != model.EVAL_COPY[0]]
    log = lambda s: print(s, flush=True)
    t0 = time.time()
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    eval_dirs = [os.path.abspath(d) for d in (args.eval_caches or [config.CACHE_DIR])]
    home_cache = config.CACHE_DIR
    train_dir = os.path.abspath(args.train_cache or config.CACHE_DIR)
    with open(os.path.join(train_dir, "degrader_params.json")) as handle:
        tparams = json.load(handle)
    eparams = {}
    for d in eval_dirs:
        with open(os.path.join(d, "degrader_params.json")) as handle:
            eparams[d] = json.load(handle)
    fold_dir = os.path.join(config.REPO_DIR, "out", "nn_" + args.name)
    os.makedirs(fold_dir, exist_ok=True)
    log("bridge {} s, arch {}, offsets {}, train {}, eval {}".format(
        config.CONTROL_MIN_S, args.arch, offsets, train_dir, eval_dirs))

    copies = {}
    config.CACHE_DIR = train_dir
    for mid in io_idsse.MATCH_IDS:
        copies[mid] = [pnn.load_copy(mid, tparams, o, s)[:3] for o in offsets for s in (False, True)]
    config.CACHE_DIR = home_cache
    log("loaded {} copies, {:.0f}s".format(sum(map(len, copies.values())), time.time() - t0))

    multi = args.eval_caches is not None
    oof = {d: {} for d in eval_dirs}
    for mid in args.folds or io_idsse.MATCH_IDS:
        paths = {d: os.path.join(fold_dir, mid + ("__" + os.path.basename(d) if multi else "")
                                 + ".npy") for d in eval_dirs}
        if all(os.path.isfile(p) for p in paths.values()):
            for d, p in paths.items():
                oof[d][mid] = np.load(p)
            log("fold {}: cached".format(mid))
            continue
        train = [c for m in io_idsse.MATCH_IDS if m != mid for c in copies[m]]
        mu, sd = pnn.fit_scaler([X for X, _, _ in train])
        seqs = [(pnn.prepare(X, mu, sd), y, g) for X, y, g in train]
        net = pnn.train(seqs, arch=args.arch, epochs=args.epochs, windows_per_epoch=args.windows,
                        seed=args.seed, log=log, hidden=args.hidden)
        torch.save({"arch": args.arch, "hidden": args.hidden, "n_in": seqs[0][0].shape[1],
                    "mu": mu, "sd": sd, "state": net.state_dict()},
                   os.path.join(fold_dir, mid + ".pt"))
        for d in eval_dirs:
            config.CACHE_DIR = d
            X, y, g, _ = model.copy_features(mid, eparams[d], *model.EVAL_COPY, 2)
            config.CACHE_DIR = home_cache
            ll = pnn.predict_frames(net, X, mu, sd)
            ok = g & (y < 3)
            acc = float(np.mean(np.argmax(ll[ok], 1) == y[ok]))
            np.save(paths[d], ll)
            oof[d][mid] = ll
            log("fold {} [{}]: frame acc {:.3f}, {:.0f}s".format(
                mid, os.path.basename(d), acc, time.time() - t0))
    if args.folds is None:
        for d in eval_dirs:
            np.savez(os.path.join(d, "oof_nn_{}.npz".format(args.name)), **oof[d])
            log("wrote {}/oof_nn_{}.npz, total {:.0f}s".format(d, args.name, time.time() - t0))


if __name__ == "__main__":
    main()
