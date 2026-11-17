#!/bin/zsh
# PFF nested CV runs: 2 lanes x 3 threads. Usage: scripts/pff_run.sh <lane 1|2>
cd $HOME/theobservers-nn
export POSSESSION_CONTROL_MIN_S=0.5 OMP_NUM_THREADS=3 MTAH_REPO_DIR=$HOME/hack2027/vendor/official MTAH_DATA_DIR=$HOME/mtah-data
run() { echo "$(date +%H:%M:%S) start $1" >> out/pff_progress.log
  .venv/bin/python -W ignore scripts/pff_nn_cv.py --name pff_$1 --arch $2 --seed $3 --threads 3 --pff-weight $4 > out/nn_pff_$1.log 2>&1
  echo "$(date +%H:%M:%S) done $1 rc=$?" >> out/pff_progress.log; }
if [[ $1 == 1 ]]; then
  run w10_gru05 gru 0 1.0; run w10_tcn05 tcn 0 1.0; run w05_gru05s1 gru 1 0.5
else
  echo "$(date +%H:%M:%S) start v5.1 baseline scoring" >> out/pff_progress.log
  .venv/bin/python -W ignore -c "import sys; sys.path.insert(0, '.'); from scripts import pff_score; pff_score.run('v5.1')" > out/pff_score_v51.log 2>&1
  echo "$(date +%H:%M:%S) done v5.1 baseline rc=$?" >> out/pff_progress.log
  run w10_gru05s1 gru 1 1.0; run w05_gru05 gru 0 0.5; run w05_tcn05 tcn 0 0.5
fi
