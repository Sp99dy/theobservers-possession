"""v6 dead-move members, alone or mixed with v5.2's members, through mean_decoder (Law 9, 1e-3).

    python scripts/v6mix_mean.py --v6-weight 0.5 --tag v6mix50     (6-member equal mix)
    python scripts/v6mix_mean.py --v6-weight 0.33 --tag v6mix33

ll = (1 - w) * mean(v5 members) + w * mean(v6 members), then Law 9 and rounding as in load_items.
All other behaviour (tuning, nesting, outputs cache/cv_<tag>_mean_decoder.json and
cache/nested_preds_<tag>_mean_e*.npz) is mean_decoder.main's.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import mean_decoder as MD  # noqa: E402

V5 = ["nn_gru05", "nn_gru05s1", "nn_tcn05"]
V6 = ["nn_gru05dm", "nn_gru05dms1", "nn_tcn05dm"]
k = sys.argv.index("--v6-weight")
W = float(sys.argv[k + 1])
del sys.argv[k:k + 2]
if "--group" in sys.argv:
    k = sys.argv.index("--group")
    V6 = sys.argv[k + 1].split(",")
    del sys.argv[k:k + 2]
_load = MD.load_items


def load_items(members, law9, quantize, swap_tta=None):
    a = _load(V5, False, None)
    b = _load(V6, False, None)
    out = []
    for (la, labs, bl), (lb, _, _), mid in zip(a, b, MD.io_idsse.MATCH_IDS):
        ll = (1 - W) * la + W * lb
        if law9:
            ll = MD.rules.law9_override(ll, MD.law9_eval_mask(mid, MD.json.load(open(MD.PARAMS_PATH))))
        if quantize:
            ll = np.round(ll / quantize) * quantize
        out.append((ll, labs, bl))
    return out


MD.load_items = load_items
if __name__ == "__main__":
    sys.argv += ["--members"] + V5 + V6
    MD.main()
