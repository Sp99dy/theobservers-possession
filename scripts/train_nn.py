"""Train the v5 neural members on all 7 IDSSE matches and write the v5 config.

    POSSESSION_CONTROL_MIN_S=0.5 python scripts/train_nn.py \
        --member gru:0:nn_gru05 --member gru:1:nn_gru05s1 --member tcn:0:nn_tcn05 \
        [--hgb emission_b05_f2.joblib] --decoder-from cache/cv_ens_<...>_e2_decoder.json

Same settings as scripts/nn_cv.py (offsets 0-8 + mirrors, 12 epochs x 3000 windows), CPU, fixed
seeds: the weights are byte-identical on rerun on the same platform. Writes <dir>/<name>.pt and
<dir>/v5.json (members + decoder), dir = --out-dir or POSSESSION_V5_DIR or out/models_v5. The
training copies come from POSSESSION_CACHE_DIR (v6: the dead-time ball cache).
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, io_idsse, nn as pnn, v5  # noqa: E402
from scripts.cv_idsse import PARAMS_PATH  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", action="append", required=True, help="arch:seed:name")
    ap.add_argument("--hgb", default=None, help="HGB joblib in the cache to add as a member")
    ap.add_argument("--decoder-from", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out-dir", default=None, help="model dir (default: v5.MODEL_DIR)")
    args = ap.parse_args()
    out_dir = args.out_dir or v5.MODEL_DIR
    assert config.CONTROL_MIN_S == 0.5, "train at the 0.5 s bridge (POSSESSION_CONTROL_MIN_S=0.5)"
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    log = lambda s: print(s, flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(PARAMS_PATH) as handle:
        params = json.load(handle)
    t0 = time.time()
    train = [pnn.load_copy(mid, params, o, s)[:3] for mid in io_idsse.MATCH_IDS
             for o in pnn.TRAIN_OFFSETS for s in (False, True)]
    mu, sd = pnn.fit_scaler([X for X, _, _ in train])
    seqs = [(pnn.prepare(X, mu, sd), y, g) for X, y, g in train]
    log("{} copies, {:.0f}s".format(len(seqs), time.time() - t0))
    members = []
    for spec in args.member:
        arch, seed, name = spec.split(":")
        net = pnn.train(seqs, arch=arch, seed=int(seed), log=log)
        path = os.path.join(out_dir, name + ".pt")
        torch.save({"arch": arch, "hidden": 64, "n_in": seqs[0][0].shape[1], "mu": mu, "sd": sd,
                    "state": net.state_dict()}, path)
        members.append(name + ".pt")
        log("{} saved, {:.0f}s".format(name, time.time() - t0))
    if args.hgb:
        shutil.copyfile(os.path.join(config.CACHE_DIR, args.hgb),
                        os.path.join(out_dir, args.hgb))
        members.append(args.hgb)
    with open(args.decoder_from) as handle:
        decoder = json.load(handle)["decoder"]
    config_path = os.path.join(out_dir, "v5.json")
    with open(config_path, "w") as handle:
        json.dump({"members": members, "decoder": decoder,
                   "decoder_from": os.path.basename(args.decoder_from)}, handle, indent=1)
    log("wrote {}".format(config_path))


if __name__ == "__main__":
    main()
