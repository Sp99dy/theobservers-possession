# TheObservers: frame-level possession for More Than a Hack 2027

This is Team TheObservers' Phase 1 submission. For every frame of the BAR-ATH tactical video it labels
who has the ball: `H` (Barcelona, ball alive), `A` (Athletic, ball alive), `D` (dead ball), or `X`
(not graded).

> **Status: M0 scaffold.** The loader, metric wrapper and writer work. The pipeline reproduces the
> organisers' four baselines byte for byte. The real model comes in M1 and M2 (rules + HMM, then
> emissions learned on IDSSE open data with a segmental decoder). See `experiments.md`.

## Layout

```
run.py                      data -> out/TheObservers.csv, then the organisers' validator
possession/
  config.py                 all paths and hyperparameters (each says where its value came from)
  organisers.py             imports games.py / possession_metrics.py from the organisers' repo
  canonical.py              Match: metres, pitch-centred, per-half orientation
  io_atd.py                 BAR-ATH ATD (FIFA/EPTS) -> Match, cached
  metrics.py                wrapper over the organisers' scorer (also used for IDSSE CV)
  write.py                  frame,label CSV writer (LF, byte-stable)
  baseline.py               the organisers' baseline through our pipeline (pipeline check)
scripts/
  reproduce_baseline.py     byte-identity check against the organisers' baseline_*.csv
  public_check.py           public 5-minute window, sanity only (never used for selection)
tests/                      pytest; synthetic data, plus count-only checks that run if the data is present
```

## Install

Python 3.14 (tested on 3.14.0, Windows 11):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt     # Windows; use .venv/bin/python elsewhere
```

## Data and the organisers' repository

The code imports the official scorer and the published half boundaries from the organisers'
repository instead of copying them, so we always score with exactly their code. Clone it next to this repo:

```
FC Barcelona/
  more-than-a-hack-2027/         https://github.com/metrica-sports/more-than-a-hack-2027
    data/                        the gated match bundle (tactical/, broadcast/)
  theobservers-possession/       this repository
```

Override the locations with `MTAH_REPO_DIR` and `MTAH_DATA_DIR` (same variable as the
organisers' `games.py`). Parsed arrays are cached in `cache/` (`POSSESSION_CACHE_DIR`), and outputs
go to `out/` (`POSSESSION_OUT_DIR`). Both folders are gitignored. **No match data is ever committed.**

## Regenerate the submission

```bash
python run.py
```

This writes `out/TheObservers.csv`, prints its sha256 and runs the organisers'
`validate_submission.py` on it. It never retrains. Its output is deterministic.

## Checks

```bash
python -m pytest -q                     # unit tests (+ data checks when the bundle is present)
python scripts/reproduce_baseline.py    # our pipeline vs the organisers' baselines, byte for byte
```

## Rules we follow

No hand labels on BAR-ATH. No tuning on the public ground truth window. No human-produced data
about BAR-ATH. No frame constants except the half boundaries in `games.py`. See `TECHNICAL.md`.
