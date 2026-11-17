"""nn_cv.py with PFF WC2022 matches as extra TRAINING copies in every fold.

    POSSESSION_CONTROL_MIN_S=0.5 .venv/bin/python -W ignore scripts/pff_nn_cv.py \
        --name pff_w10_gru05 --arch gru --seed 0 --threads 3 --pff-weight 1.0

Every fold trains on the six other IDSSE matches (offsets 0-8 + mirrors, as v5) plus all PFF
copies in cache_pff/ (scripts/pff_build.py); the held-out IDSSE offset-9 copy is the only test.
PFF is never a test fold. --pff-weight scales how often PFF windows are drawn relative to their
length (1.0 = like an IDSSE match of the same length; the total is spread over the PFF copies,
so 12 matches x 6 copies weigh 12 matches, not 72). The scaler and class weights are fitted on
IDSSE + PFF together. Writes only out/nn_<name>/ (runs nn_cv with --folds = all seven).
"""

import glob
import os
import sys

import numpy as np
import torch
from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import io_idsse, nn as pnn  # noqa: E402
from scripts import nn_cv  # noqa: E402

WEIGHT = 1.0
if "--pff-weight" in sys.argv:
    k = sys.argv.index("--pff-weight")
    WEIGHT = float(sys.argv[k + 1])
    del sys.argv[k:k + 2]
IDSSE_COPIES_PER_MATCH = 18     # offsets 0-8 x mirror


def _pff_copies(step):
    out, games = [], set()
    dirs = os.environ.get("PFF_FEAT_DIRS", "cache_pff").split(",")
    paths = sorted(p for d in dirs for p in glob.glob(os.path.join(REPO, d, "pff_feat_*_v2.npz")))
    for path in paths:
        _, _, game, tag, _ = os.path.basename(path).split("_")
        offset, swapped = int(tag.rstrip("s")), tag.endswith("s")
        ph = pnn.phase_of(offset, swapped, step)
        with np.load(path) as z:
            out.append((z["X"][ph::step].astype(np.float32), z["y"][ph::step].astype(np.int64),
                        z["g"][ph::step]))
        games.add(game)
    assert out, "no PFF copies in cache_pff/"
    return out, len(games)


STATE = {}
_fit_scaler = pnn.fit_scaler


def fit_scaler(Xs):
    if "copies" not in STATE:
        STATE["copies"], STATE["games"] = _pff_copies(STATE.get("step", pnn.STEP))
    mu, sd = _fit_scaler(list(Xs) + [X for X, _, _ in STATE["copies"]])
    STATE["mu"], STATE["sd"] = mu, sd
    return mu, sd


def train(seqs, arch="gru", epochs=12, windows_per_epoch=3000, batch=32, lr=2e-3, seed=0,
          log=print, win=pnn.WIN, **kw):
    """pnn.train with the PFF copies appended; each PFF match weighs WEIGHT x an IDSSE match of
    the same length (copies per match differ: IDSSE 18, PFF len(copies)/games)."""
    pff = [(pnn.prepare(X, STATE["mu"], STATE["sd"]), y, g) for X, y, g in STATE["copies"]]
    allseq = list(seqs) + pff
    per_game = len(pff) / STATE["games"]
    wts = np.r_[np.ones(len(seqs)), np.full(len(pff), WEIGHT * IDSSE_COPIES_PER_MATCH / per_game)]
    log("  + {} PFF copies of {} games, weight {}".format(len(pff), STATE["games"], WEIGHT))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ys = np.concatenate([y[g & (y < 3)] for _, y, g in allseq])
    cnt = np.bincount(ys, minlength=3)[:3].astype(np.float64)
    w = torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32)
    net = pnn.ARCHS[arch](allseq[0][0].shape[1], **kw)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * (windows_per_epoch // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total)
    lens = np.array([len(s[0]) for s in allseq], dtype=np.float64) * wts
    pick_p = lens / lens.sum()
    lossf = nn.NLLLoss(weight=w, ignore_index=-1)
    for ep in range(epochs):
        net.train()
        tot, nb = 0.0, 0
        for _ in range(windows_per_epoch // batch):
            xb, yb = [], []
            for k in rng.choice(len(allseq), size=batch, p=pick_p):
                Z, y, g = allseq[k]
                s = int(rng.integers(0, max(1, len(Z) - win)))
                xb.append(Z[s:s + win])
                yy = y[s:s + win].copy()
                yy[(~g[s:s + win]) | (yy > 2)] = -1
                xb[-1], yy = pnn._pad(xb[-1], yy, win)
                yb.append(yy)
            out = net(torch.from_numpy(np.stack(xb)))
            loss = lossf(out.reshape(-1, 3), torch.from_numpy(np.stack(yb)).reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += float(loss)
            nb += 1
        log("  epoch {:2d} loss {:.4f}".format(ep, tot / nb))
    net.eval()
    return net


pnn.fit_scaler = fit_scaler
pnn.train = train

if __name__ == "__main__":
    if "--step" in sys.argv:
        STATE["step"] = int(sys.argv[sys.argv.index("--step") + 1])
    if "--folds" not in sys.argv:
        sys.argv += ["--folds", *io_idsse.MATCH_IDS]
    nn_cv.main()
