"""Neural sequence emission model (HANDOFF §4.1): a 2-layer BiGRU or a dilated temporal CNN over
the feature-v2 matrix at 5 fps, trained on degraded IDSSE copies (offsets 0-8 and their mirrors).

Input per step: each of the 122 columns standardised (training-fold mean / std), NaN set to 0, plus
one isnan indicator per column (244 channels). Output: log-softmax over H/A/D per step. Loss:
class-balanced cross-entropy on graded steps only. Windows of WIN steps (60 s); inference uses
windows with HOP overlap and keeps each window's centre, and runs all 5 subsampling phases so every
25 fps frame gets its own prediction. Everything runs on CPU with fixed seeds, so inference is
deterministic.
"""

import numpy as np
import torch
from torch import nn

from . import model

WIN = 300           # [D] 60 s at 5 fps
HOP = 150           # [D] 30 s overlap
STEP = model.SUBSAMPLE
TRAIN_OFFSETS = tuple(range(9))     # never offset 9 (the held-out evaluation copy)
CLIP = 8.0


def phase_of(offset, swapped, step=STEP):
    """Which subsampling phase a training copy uses (deterministic, spreads phases)."""
    return (offset + 2 * int(swapped)) % step


def win_of(step, win_s=60.0):
    """Window and hop in steps for a subsampling step and a window length (hop = half a window)."""
    w = int(round(win_s / 60.0 * WIN * STEP / step))
    return w, w // 2


def load_copy(match_id, params, offset, swapped, step=STEP):
    X, y, g, names = model.copy_features(match_id, params, offset, swapped, 2)
    ph = phase_of(offset, swapped, step)
    return X[ph::step].astype(np.float32), y[ph::step].astype(np.int64), g[ph::step], names


def fit_scaler(Xs):
    cat = np.concatenate(Xs)
    mu = np.nanmean(cat, axis=0)
    sd = np.nanstd(cat, axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def prepare(X, mu, sd):
    nan = ~np.isfinite(X)
    Z = np.clip((X - mu) / sd, -CLIP, CLIP)
    Z[nan] = 0.0
    return np.concatenate([Z, nan.astype(np.float32)], axis=1).astype(np.float32)


class BiGRU(nn.Module):
    def __init__(self, n_in, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.out = nn.Linear(2 * hidden, 3)

    def forward(self, x):                       # (B, T, C) -> (B, T, 3) log-probs
        h, _ = self.gru(self.inp(x))
        return torch.log_softmax(self.out(h), dim=-1)


class TCN(nn.Module):
    """Non-causal dilated residual 1D CNN; receptive field about 2 * sum(dilations) * 2 steps."""

    def __init__(self, n_in, hidden=64, dilations=(1, 2, 4, 8, 16, 32), dropout=0.2):
        super().__init__()
        self.inp = nn.Conv1d(n_in, hidden, 1)
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, 3, padding=d, dilation=d), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, 1)))
        self.out = nn.Conv1d(hidden, 3, 1)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(h)
        return torch.log_softmax(self.out(h), dim=1).transpose(1, 2)


ARCHS = {"gru": BiGRU, "tcn": TCN}


def train(seqs, arch="gru", epochs=12, windows_per_epoch=3000, batch=32, lr=2e-3, seed=0,
          log=print, win=WIN, **kw):
    """seqs: list of (Z (T, C) float32 prepared, y (T,) int, g (T,) bool), win steps per window."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ys = np.concatenate([y[g & (y < 3)] for _, y, g in seqs])
    cnt = np.bincount(ys, minlength=3)[:3].astype(np.float64)
    w = torch.tensor(cnt.sum() / (3 * np.maximum(cnt, 1)), dtype=torch.float32)
    net = ARCHS[arch](seqs[0][0].shape[1], **kw)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    total = epochs * (windows_per_epoch // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total)
    lens = np.array([len(s[0]) for s in seqs], dtype=np.float64)
    pick_p = lens / lens.sum()
    lossf = nn.NLLLoss(weight=w, ignore_index=-1)
    for ep in range(epochs):
        net.train()
        tot, nb = 0.0, 0
        for _ in range(windows_per_epoch // batch):
            xb, yb = [], []
            for k in rng.choice(len(seqs), size=batch, p=pick_p):
                Z, y, g = seqs[k]
                s = int(rng.integers(0, max(1, len(Z) - win)))
                xb.append(Z[s:s + win])
                yy = y[s:s + win].copy()
                yy[(~g[s:s + win]) | (yy > 2)] = -1
                xb[-1], yy = _pad(xb[-1], yy, win)
                yb.append(yy)
            xb = torch.from_numpy(np.stack(xb))
            yb = torch.from_numpy(np.stack(yb))
            out = net(xb)
            loss = lossf(out.reshape(-1, 3), yb.reshape(-1))
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


def _pad(x, y, win=WIN):
    if len(x) >= win:
        return x, y
    n = win - len(x)
    return (np.concatenate([x, np.zeros((n, x.shape[1]), np.float32)]),
            np.concatenate([y, np.full(n, -1, y.dtype)]))


@torch.no_grad()
def predict_steps(net, Z, win=WIN, hop=HOP):
    """Log-probs (T, 3) for one subsampled sequence: overlapping windows, each keeps its centre."""
    T = len(Z)
    out = np.zeros((T, 3), np.float32)
    if T <= win:
        return net(torch.from_numpy(Z)[None])[0].numpy()
    starts = list(range(0, T - win, hop)) + [T - win]
    q = (win - hop) // 2
    filled = 0
    for s in starts:
        o = net(torch.from_numpy(Z[s:s + win])[None])[0].numpy()
        lo = filled
        hi = T if s == starts[-1] else s + win - q
        out[lo:hi] = o[lo - s:hi - s]
        filled = hi
    return out


def predict_frames(net, X, mu, sd, step=STEP, win_s=60.0):
    """Frame-resolution (25 fps) log-probs: one pass per subsampling phase, interleaved."""
    out = np.zeros((len(X), 3), np.float32)
    win, hop = win_of(step, win_s)
    for ph in range(step):
        out[ph::step] = predict_steps(net, prepare(X[ph::step].astype(np.float32), mu, sd),
                                      win, hop)
    return out
