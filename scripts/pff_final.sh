#!/bin/zsh
# Final PFF variant (run only on a positive nested verdict). 2 processes x 3 threads, ~25 min.
cd $HOME/theobservers-nn
export POSSESSION_CONTROL_MIN_S=0.5 OMP_NUM_THREADS=3 MTAH_REPO_DIR=$HOME/hack2027/vendor/official MTAH_DATA_DIR=$HOME/mtah-data
D=models/v5x_pff; DEC=cache/cv_pff_w10_mean_decoder.json; P=".venv/bin/python -W ignore scripts/pff_train.py"
echo "$(date +%H:%M:%S) final training start" >> out/pff_progress.log
${=P} --member gru:0:nn_gru05 --member gru:1:nn_gru05s1 --out-dir $D --decoder-from $DEC --law9 --threads 3 > out/pff_final_a.log 2>&1 &
${=P} --member tcn:0:nn_tcn05 --out-dir $D --decoder-from $DEC --law9 --threads 3 > out/pff_final_b.log 2>&1 &
wait
${=P} --write-config $D --decoder-from $DEC --replay-from models/v54/v5.json >> out/pff_progress.log 2>&1 || exit 1
echo "$(date +%H:%M:%S) final training done; inference" >> out/pff_progress.log
POSSESSION_V54_DIR=$D .venv/bin/python -W ignore run.py --method v54 --out out/TheObservers_pff.csv > out/pff_final_run.log 2>&1
echo "$(date +%H:%M:%S) run.py rc=$?" >> out/pff_progress.log
tail -5 out/pff_final_run.log >> out/pff_progress.log
