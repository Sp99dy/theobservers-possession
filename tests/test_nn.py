import numpy as np
import pytest

torch = pytest.importorskip("torch")

from possession import model, nn as pnn  # noqa: E402


def test_never_trains_on_eval_copy():
    assert model.EVAL_COPY[0] not in pnn.TRAIN_OFFSETS


def test_prepare_adds_nan_channels():
    X = np.array([[1.0, np.nan], [3.0, 2.0]], np.float32)
    mu, sd = pnn.fit_scaler([X])
    Z = pnn.prepare(X, mu, sd)
    assert Z.shape == (2, 4)
    assert np.isfinite(Z).all()
    assert Z[0, 1] == 0.0 and Z[0, 3] == 1.0 and Z[1, 3] == 0.0


class _StepIndex(torch.nn.Module):
    """Outputs the position inside the window, to check which window part each step comes from."""

    def forward(self, x):
        t = torch.arange(x.shape[1], dtype=torch.float32)
        return torch.stack([t, t, t], -1)[None].expand(x.shape[0], -1, -1)


@pytest.mark.parametrize("T", [50, 300, 301, 449, 450, 1000, 1234])
def test_windows_cover_every_step_from_centres(T):
    out = pnn.predict_steps(_StepIndex(), np.zeros((T, 2), np.float32))
    assert out.shape == (T, 3)
    if T > pnn.WIN:
        q = (pnn.WIN - pnn.HOP) // 2
        inner = out[q:T - q, 0]          # away from the sequence ends: never a window's edge
        assert inner.min() >= q - 1 and inner.max() <= pnn.WIN - q


def test_predict_frames_deterministic():
    torch.manual_seed(0)
    net = pnn.BiGRU(8, hidden=8).eval()
    X = np.random.default_rng(0).normal(size=(997, 4)).astype(np.float32)
    X[::7, 1] = np.nan
    mu, sd = pnn.fit_scaler([X])
    a = pnn.predict_frames(net, X, mu, sd)
    b = pnn.predict_frames(net, X, mu, sd)
    assert a.shape == (997, 3)
    assert np.array_equal(a, b)
