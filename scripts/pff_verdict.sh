#!/bin/zsh
# w1.0 only: tcn05 folds split over 2 processes after the two GRU runs, then nested verdict vs v5.2.
cd $HOME/theobservers-nn
export POSSESSION_CONTROL_MIN_S=0.5 OMP_NUM_THREADS=3 MTAH_REPO_DIR=$HOME/hack2027/vendor/official MTAH_DATA_DIR=$HOME/mtah-data
NN=".venv/bin/python -W ignore scripts/pff_nn_cv.py --name pff_w10_tcn05 --arch tcn --seed 0 --threads 3 --pff-weight 1.0"
lane() { while kill -0 $1 2>/dev/null; do sleep 20; done
  echo "$(date +%H:%M:%S) start w10_tcn05 folds $2" >> out/pff_progress.log
  ${=NN} --folds ${=2} > out/nn_pff_w10_tcn05_$3.log 2>&1
  echo "$(date +%H:%M:%S) done w10_tcn05 $3 rc=$?" >> out/pff_progress.log; }
lane 12840 "J03WMX J03WN1 J03WOH J03WOY" a &
lane 12971 "J03WPY J03WQQ J03WR9" b &
wait
.venv/bin/python - <<'PY' >> out/pff_progress.log 2>&1
import numpy as np, os
from possession import io_idsse
for m in ("gru05", "gru05s1", "tcn05"):
    d = "out/nn_pff_w10_" + m
    np.savez("cache/oof_nn_pff_w10_{}.npz".format(m), **{k: np.load(os.path.join(d, k + ".npy")) for k in io_idsse.MATCH_IDS})
    print("oof", m, "ok")
PY
echo "$(date +%H:%M:%S) verdict: mean_decoder start" >> out/pff_progress.log
.venv/bin/python -W ignore scripts/mean_decoder.py --members nn_pff_w10_gru05 nn_pff_w10_gru05s1 nn_pff_w10_tcn05 \
  --law9 --quantize 0.001 --start cache/cv_ens_nn_gru05+nn_gru05s1+nn_tcn05_e1_decoder.json \
  --base v52_mean --tag pff_w10 > out/pff_verdict.log 2>&1
echo "$(date +%H:%M:%S) verdict rc=$?" >> out/pff_progress.log
tail -3 out/pff_verdict.log >> out/pff_progress.log
