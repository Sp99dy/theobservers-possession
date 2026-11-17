"""Metrica-augmented members through mean_decoder (Law 9, 1e-3, mean metric, nested vs v5.2).

    python scripts/metrica_mix_mean.py --arm w10 --v5-weight 0.0 ... (mean_decoder args)

Metrica members: out/nn_metrica_<arm>_{gru05,gru05s1,tcn05}/<match>.npy (out-of-fold, same folds).
ll = v * mean(v5.2 members) + (1 - v) * mean(Metrica members); v = 0 is the Metrica ensemble alone,
v = 0.5 the 6-member equal mix.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import mean_decoder as MD  # noqa: E402

V5 = ["nn_gru05", "nn_gru05s1", "nn_tcn05"]
MEM = ("gru05", "gru05s1", "tcn05")


def _arg(name, cast):
    k = sys.argv.index(name)
    v = cast(sys.argv[k + 1])
    del sys.argv[k:k + 2]
    return v


ARM = _arg("--arm", str)
V = _arg("--v5-weight", float)
_load = MD.load_items


def load_items(members, law9, quantize, swap_tta=None):
    params = MD.json.load(open(MD.PARAMS_PATH))
    base = _load(V5, False, None)
    out = []
    for (la, labs, bl), mid in zip(base, MD.io_idsse.MATCH_IDS):
        lm = np.mean([np.load(os.path.join(MD.config.REPO_DIR, "out", "nn_metrica_{}_{}".format(ARM, m),
                                           mid + ".npy")).astype(np.float64) for m in MEM], 0)
        assert lm.shape == la.shape
        ll = V * la + (1 - V) * lm
        if law9:
            ll = MD.rules.law9_override(ll, MD.law9_eval_mask(mid, params))
        if quantize:
            ll = np.round(ll / quantize) * quantize
        out.append((ll, labs, bl))
    return out


MD.load_items = load_items
if __name__ == "__main__":
    sys.argv += ["--members", "metrica_" + ARM]
    MD.main()
