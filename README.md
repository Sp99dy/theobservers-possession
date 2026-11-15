# TheObservers: frame-level possession for More Than a Hack 2027

This is Team TheObservers' Phase 1 submission. For every frame of the BAR-ATH tactical video it labels
who has the ball: `H` (Barcelona, ball alive), `A` (Athletic, ball alive), `D` (dead ball), or `X`
(not graded).

> **Status: v4 is the default (`python run.py`).** A gradient-boosting classifier trained on IDSSE
> open data, which is corrupted to look like the BAR-ATH tracking, gives per-frame evidence. Its
> features include ball-touch detection and the collective slowdown of players at stoppages.
> Kickoff detection and a Smart Tagging prior are added, and a minimum-duration Viterbi decoder over
> {H, A, D} turns the evidence into labels. IDSSE leave-one-match-out CV (nested) spellF1 is 0.684,
> against 0.643 for v2 (M2, the last version submitted) and 0.389 for the v1 rules. See
> `experiments.md` and `TECHNICAL.md`.
>
> **Joining the project? Start with [`HANDOFF.md`](HANDOFF.md)**: setup, the rules, what has been
> tried and what to try next.

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
  features.py               per-frame identity-free features (one code path for every source)
  rules.py                  v1 evidence: dead-ball score z and home-vs-away score y
  kickoff.py                algorithmic kickoff detection (law 8 line-up)
  io_tagging.py             Smart Tagging -> frame intervals
  tagging_prior.py          Smart Tagging as soft evidence for the decoder
  decode.py                 minimum-duration Viterbi over {H, A, D}
  v1.py                     the v1 pipeline (rules; M1 baseline)
  io_idsse.py               IDSSE / DFL open data -> Match + controlled-possession labels
  degrade.py                corrupts IDSSE to match BAR-ATH tracking statistics (fitted on tracking only)
  model.py                  training copies, HGB emission classifier, save/load
  cv.py                     leave-one-match-out CV and decoder tuning (all model selection)
  v2.py                     the v2 pipeline (M2)
  v3.py                     v2 + larger classifier + v1-rule blend (M3; fallback)
  v4.py                     v2 pipeline on feature version 2: touch detection + slowdown (M3b, default)
  io_metrica.py             Metrica sample games 1+2 -> Match (secondary check only)
  stack.py                  second-stage context features (experiment, not adopted)
  video.py, video_detect.py M4 video ball detector -- experiment only, dropped, not used by run.py
models/                     emission_v*.joblib + decoder_v*.json (committed; run.py never retrains)
scripts/
  reproduce_baseline.py     byte-identity check against the organisers' baseline_*.csv
  public_check.py           public 5-minute window, sanity only (never used for selection)
  download_idsse.py         fetch the IDSSE open data (CC-BY 4.0)
  idsse_conventions.py      label-convention comparison (DFL touch-level vs controlled possession)
  build_idsse_features.py   fit the degrader, build degraded IDSSE feature caches
  cv_idsse.py               LOMO CV, ablations, nested decoder tuning
  train_idsse.py            final model on all 7 IDSSE matches -> models/ (--version v2|v3|v4)
  decoder_experiments.py    M3 decoder / ensemble variants on cached out-of-fold emissions
  ensemble_experiments.py   scores any cached out-of-fold emissions (or a mean) nested, vs v4 per match
  change_head.py, change_decoder_experiments.py  touch-aware decoder (not adopted)
  stack_experiments.py      second-stage model (not adopted)
  postfilter_experiments.py short-possession post-filter (not adopted); caches v4's nested predictions
  metrica_check.py          secondary check on Metrica sample games (never used for selection)
  diag_*.py                 diagnostics on IDSSE OOF emissions (dead balls, live-play losses, touches)
  viewer_export.py          local match viewer data -> out/viewer/ (LOCAL ONLY: holds match extracts)
  viewer_serve.py           serves viewer/index.html + out/viewer on 127.0.0.1 only
  video_*.py                M4 experiments (requirements-video.txt); dropped
viewer/index.html           local viewer: video, tracking, model internals vs the public truth
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
