"""The label bridge is a parameter; the default (2 s) keeps v4's labels and caches unchanged."""

import numpy as np

from possession import config, model
from possession.canonical import A, H
from possession.io_idsse import relabel_controlled


def _seq(*runs):
    return np.concatenate([np.full(n, c, dtype=np.int8) for c, n in runs])


def test_default_bridge_is_two_seconds():
    assert config.DEFAULT_CONTROL_MIN_S == 2.0


def test_shorter_bridge_keeps_a_one_second_possession():
    lab = _seq((H, 100), (A, 30), (H, 100))            # A keeps it 1.2 s
    blocks = [(1, 0, len(lab))]
    assert (relabel_controlled(lab, blocks, 50) == H).all()
    assert relabel_controlled(lab, blocks, 25).tolist() == lab.tolist()


def test_default_bridge_returns_cached_labels_untouched(monkeypatch):
    monkeypatch.setattr(config, "CONTROL_MIN_S", config.DEFAULT_CONTROL_MIN_S)
    y = _seq((H, 10), (A, 10))
    assert model._bridged("J03WMX", False, y) is y
