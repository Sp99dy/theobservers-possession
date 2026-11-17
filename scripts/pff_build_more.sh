#!/bin/zsh
# Speculative build of PFF games 12..N (default 64) into cache_pff_more/, one game at a time, kept
# apart from cache_pff so the running 11-match verdict never sees them. Before each game waits
# until swap free >= 1.5 GB (Q&A rule); 1 job; disk floor 10 GB.
cd $HOME/theobservers-nn
export POSSESSION_CONTROL_MIN_S=0.5 OMP_NUM_THREADS=1 PFF_FEAT_DIR=cache_pff_more PFF_MIN_FREE_GB=10
N=${1:-64}
for i in $(seq 11 $((N - 1))); do
  until .venv/bin/python -c "import sys;sys.path.insert(0,'.');from scripts.pff_build import swap_free_gb as f;sys.exit(f()<1.5)"; do sleep 30; done
  .venv/bin/python -W ignore scripts/pff_build.py --start $i --games $((i + 1)) --jobs 1 >> out/pff_build_more.log 2>&1
  echo "$(date +%H:%M:%S) build_more game $i rc=$? ($(ls cache_pff_more 2>/dev/null | wc -l | tr -d ' ') copies)" >> out/pff_progress.log
done
echo "$(date +%H:%M:%S) build_more done" >> out/pff_progress.log
