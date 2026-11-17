"""v5 candidate: v4's pipeline with neural sequence emissions (HANDOFF §4.1).

The emission is the equal-weight mean of the members' log-probs, exactly as in
scripts/ensemble_experiments.py: neural members (possession/nn.py, trained on all 7 IDSSE matches
at the 0.5 s label bridge by scripts/train_nn.py) and optionally the HGB trained at the same bridge.
Each member's log-probs are rounded to float32 first, as in the out-of-fold files the decoder was
tuned on. Kickoff and Smart Tagging layers and the segmental decoder are v4's code; the decoder
settings are the pooled-tuned ones from the chosen ensemble run (IDSSE CV, 2 s labels). Nothing
here sees BAR-ATH labels.

Weights for v5.1+ are committed under models/v51 ... models/v55 (IDSSE-trained, a few MB) and loaded
via model_dir; the original v5 set lives in POSSESSION_V5_DIR (default out/models_v5). All are
regenerated deterministically by scripts/train_nn.py.
"""

import json
import os

import numpy as np
import torch

from . import config as C
from . import model, nn as pnn, v2

FEATURE_VERSION = 2
MODEL_DIR = os.environ.get("POSSESSION_V5_DIR") or os.path.join(C.OUT_DIR, "models_v5")
CONFIG_PATH = os.path.join(MODEL_DIR, "v5.json")


def load_member(name, model_dir=None):
    model_dir = model_dir or MODEL_DIR
    if name.endswith(".joblib"):
        mdl = model.load(os.path.join(model_dir, name))
        return lambda X: model.emission_ll(mdl, X).astype(np.float32)
    ck = torch.load(os.path.join(model_dir, name), weights_only=False)
    net = pnn.ARCHS[ck["arch"]](ck["n_in"], hidden=ck["hidden"], **ck.get("arch_kw", {}))
    net.load_state_dict(ck["state"])
    net.eval()
    return lambda X: pnn.predict_frames(net, X, ck["mu"], ck["sd"], ck.get("step", pnn.STEP),
                                            ck.get("win_s", 60.0))


def emission_fn(members, model_dir=None):
    fns = [load_member(m, model_dir) for m in members]

    def emission(X):
        return np.mean([f(X).astype(np.float64) for f in fns], axis=0)
    return emission


def replay_mask_cached(match):
    """v5.4: replay frames as 0/1 floats (possession/replay.py)."""
    from . import replay
    return replay.replay_mask(match).astype(np.float64)


def predict(match, tags=None, use_tags=True, return_debug=False, model_dir=None):
    """model_dir: another model set with the same layout (v6 uses out/models_v6)."""
    torch.use_deterministic_algorithms(True)
    model_dir = model_dir or MODEL_DIR
    with open(os.path.join(model_dir, "v5.json")) as handle:
        cfg = json.load(handle)
    if cfg.get("ballfill"):
        from . import ballfill
        match = ballfill.fill(match)[0]
    d_bonus = None
    if cfg.get("replay_llr"):
        d_bonus = cfg["replay_llr"] * replay_mask_cached(match)
    if cfg.get("inference_threads"):
        torch.set_num_threads(int(cfg["inference_threads"]))
    return v2.predict(match, tags, use_tags=use_tags, return_debug=return_debug,
                      decoder=cfg["decoder"], feature_version=FEATURE_VERSION,
                      emission=emission_fn(cfg["members"], model_dir),
                      law9=cfg.get("law9", False), quantize=cfg.get("quantize"),
                      d_bonus=d_bonus)
