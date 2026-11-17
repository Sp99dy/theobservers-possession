"""Build the feature-v3 cache (cache_v3/): every IDSSE copy (offsets 0-9, mirrors) = the cached v2
matrix + features_v3.extra_columns on the same degraded copy. Files keep the _v2 name so nn_cv
reads them with --train-cache cache_v3 --eval-caches cache_v3 (cache_v3 is a separate directory).

    python scripts/v3_build.py --part 0 --parts 2
"""

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config, features_v3, io_idsse, model  # noqa: E402

DST = os.path.join(REPO, "cache_v3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    a = ap.parse_args()
    params = json.load(open(os.path.join(config.CACHE_DIR, "degrader_params.json")))
    jobs = [(m, o, s) for m in io_idsse.MATCH_IDS for o in range(10) for s in (False, True)]
    for k, (mid, o, s) in enumerate(jobs):
        if k % a.parts != a.part:
            continue
        name = os.path.basename(model._feature_cache(mid, o, s, 2))
        out = os.path.join(DST, name)
        if os.path.isfile(out):
            continue
        t0 = time.time()
        with np.load(os.path.join(config.CACHE_DIR, name), allow_pickle=False) as z:
            X2, y, g, names = z["X"], z["y"], z["g"], z["names"].tolist()
        m = model.make_copy(mid, params, o, s)
        assert m.n_frames == len(X2) and np.array_equal(m.graded, g), name
        ex = features_v3.extra_columns(m)
        X = np.concatenate([X2, np.column_stack(list(ex.values())).astype(np.float32)], 1)
        tmp = out + ".tmp.npz"
        np.savez(tmp, X=X, y=y, g=g, names=np.array(names + list(ex)))
        os.replace(tmp, out)
        print("{} {:.0f}s".format(name, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
