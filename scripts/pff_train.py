"""Final training of the PFF variant: v5.1 members trained on all 7 IDSSE + all PFF copies in
cache_pff/ (weight 1.0), same settings as scripts/train_nn.py. Shipping config like models/v52.

    POSSESSION_CONTROL_MIN_S=0.5 .venv/bin/python -W ignore scripts/pff_train.py \
        --member gru:0:nn_gru05 --member gru:1:nn_gru05s1 --out-dir models/v5x_pff \
        --decoder-from cache/cv_pff_w10_mean_decoder.json --law9 --threads 3
    .venv/bin/python scripts/pff_train.py --write-config models/v5x_pff \
        --decoder-from cache/cv_pff_w10_mean_decoder.json

Members can be trained in separate processes into the same --out-dir; --write-config then writes
the v5.json listing all three (law9, quantize 1e-3, 1 inference thread, as models/v52).
With --replay-from models/v54/v5.json the frozen v5.4 replay evidence is added.
Inference: POSSESSION_V54_DIR=models/v5x_pff python run.py --method v54
"""

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

MEMBERS = ("nn_gru05.pt", "nn_gru05s1.pt", "nn_tcn05.pt")

if __name__ == "__main__":
    if "--write-config" in sys.argv:
        d = sys.argv[sys.argv.index("--write-config") + 1]
        src = sys.argv[sys.argv.index("--decoder-from") + 1]
        missing = [m for m in MEMBERS if not os.path.isfile(os.path.join(d, m))]
        assert not missing, "missing members {}".format(missing)
        with open(src) as h:
            dec = json.load(h)["decoder"]
        cfg = {"members": list(MEMBERS), "decoder": dec,
               "decoder_from": "{} (scripts/mean_decoder.py on PFF-augmented oof: mean spellF1 "
                               "over 2/1/0.5 s, law9, quantize 1e-3)".format(os.path.basename(src)),
               "law9": True, "quantize": 0.001, "inference_threads": 1,
               "training_data": "7 IDSSE matches + PFF FC WC2022 games {} (3 degraded copies each, "
                                "weight 1.0)".format(sorted({os.path.basename(p).split("_")[2]
                                    for p in __import__("glob").glob(os.path.join(
                                        REPO, "cache_pff", "pff_feat_*_v2.npz"))}))}
        if "--replay-from" in sys.argv:          # v5.4: copy the frozen replay evidence keys
            with open(sys.argv[sys.argv.index("--replay-from") + 1]) as h:
                base = json.load(h)
            for k in ("replay_llr", "replay_note"):
                if k in base:
                    cfg[k] = base[k]
        with open(os.path.join(d, "v5.json"), "w") as h:
            json.dump(cfg, h, indent=1)
        print("wrote", os.path.join(d, "v5.json"))
        sys.exit(0)
    from scripts import pff_nn_cv  # noqa: F401  (patches pnn.fit_scaler / pnn.train with PFF)
    from scripts import train_nn
    train_nn.main()
