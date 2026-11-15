# Handoff: where the model stands and what to try next

**In one paragraph.** We label every frame of BAR-ATH as home ball (H), away ball (A) or dead ball
(D). The ranking metric is spellF1, which scores whole possessions, not frames. The current model is
**v4**, the default of `python run.py`. A gradient-boosting classifier gives per-frame P(H, A, D),
and a minimum-duration Viterbi decoder turns those probabilities into possessions. The classifier is
trained on 7 public Bundesliga matches (IDSSE / DFL open data) whose tracking is corrupted to look
like BAR-ATH's. On held-out IDSSE matches, v4 scores **spellF1 0.6835** (v2: 0.643). Tonight's
attempts at a v5 (bigger trees, a touch-aware decoder, a second-stage model, ensembles) all landed
within +0.008 of v4, which is inside the noise. **The next gain has to come from something
structurally new.** The two best bets are a neural sequence model and more realistic training
noise. Both run much faster on a strong machine.

---

## 0. Rules (non-negotiable)

- **The BAR-ATH match data never goes to a cloud service or a cloud LLM.** That covers raw files,
  extracts, frames, video and screenshots of it. LLMs are fine for code. Only paste aggregates
  (scores, counts, shapes).
- **The repo is public.** Never commit `data/`, `cache/`, `out/`, `*.csv`, `*.npz`, video or
  weights. The `.gitignore` already blocks these: don't force-add.
- **No tuning on the public 5-minute ground-truth window.** Every choice is made on IDSSE
  leave-one-match-out CV. Run the public window at most once, as a sanity check.
- Work on **your own branch**. Tobias decides what goes to `main` and resubmits.

## 1. Setup

```
FC Barcelona/
  more-than-a-hack-2027/        https://github.com/metrica-sports/more-than-a-hack-2027 (+ data/ bundle)
  idsse-data/                   created by download_idsse.py
  theobservers-possession/      this repo
```

Python 3.14 and the pinned `requirements.txt`. Paths can be overridden with `MTAH_REPO_DIR`,
`MTAH_DATA_DIR`, `IDSSE_DIR` and `POSSESSION_CACHE_DIR`.

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt   # .venv/bin/python on Linux/Mac
python run.py                     # must print sha256 ce5399e8bb93b6022e667726963f02b368efc25a4532cdc218be64d4854b92be
python -m pytest -q               # 84+ tests
```

Then build the IDSSE pipeline (one-off, cached in `cache/`):

```bash
python scripts/download_idsse.py                                  # 7 matches, CC-BY 4.0
python scripts/build_idsse_features.py --feature-version 2 --jobs 16   # degrader fit + corrupted feature caches
python scripts/cv_idsse.py --tag f2_all --feature-version 2 --nested --skip-v1   # v4's CV: 0.6835 nested
python scripts/postfilter_experiments.py --tag f2_all             # also caches v4's nested predictions (the baseline)
```

The degrader fit reads the BAR-ATH tracking (statistics only, never labels), so the data bundle
must be present. On Tobias's laptop a 7-fold CV takes about 30–60 min of training plus about
30 min of decoder tuning. A strong machine should be much faster: raise `--jobs` and run variants
in parallel.

## 2. How we decide (the bar)

`cv_idsse.py` trains on 6 matches, predicts the 7th (a corruption never trained on), and repeats
7 times. It then tunes the decoder **nested**, re-tuning without the held-out match. **Only the
NESTED spellF1 counts**; the "tuned" number is optimistic.

A candidate **beats v4** only if both hold:

1. nested spellF1 ≥ **0.6935** (+0.01 over 0.6835);
2. it wins on **at least 5 of the 7 matches**, as printed by `scripts/ensemble_experiments.py`.

We set this bar because per-match differences swing ±0.02–0.03 by chance; tonight's ensembles
looked like +0.011 tuned and shrank to +0.004–0.008 nested.

**Plugging in any new emission model.** Write `cache/oof_<tag>.npz`: one array per IDSSE match id,
of shape (T, 3) float32 holding log P(H), log P(A), log P(D) for every frame of that match's
held-out corruption (`model.copy_features(mid, params, *model.EVAL_COPY, 2)`). It must come from
a model that never saw that match. Then:

```bash
python scripts/ensemble_experiments.py --members <tag>             # standalone, nested, vs v4 per match
python scripts/ensemble_experiments.py --members f2_all <tag>      # averaged with v4
```

## 3. Already tried (don't repeat)

All on IDSSE nested CV; details in `experiments.md`.

| idea | nested spellF1 | verdict |
|---|---|---|
| v2: HGB on feature v1 | 0.643 | old baseline |
| bigger HGB (600 / 1000 trees) on feature v1 | 0.654 / 0.661 | helped then, superseded |
| v1-rule blend (v3) | 0.666 | fallback |
| **v4: + touch detection + collective slowdown features** | **0.6835** | **current** |
| v4 + 1000 trees, 127 leaves | 0.689 | +0.006, noise |
| split dead-ball states, 8 corruption copies, no minimum durations | ≤ +0.004 | noise |
| post-filter: short possession between two stoppages → dead | −0.003 to −0.06 | worse |
| touch-aware decoder (change head lowers switch cost / allows short starts) | 0.6855 | +0.002, noise |
| second-stage (stacked) model on stage-1 outputs over ±8 s | 0.6892 | +0.006, 4/7 wins |
| ensembles of the above | 0.688–0.691 | +0.004–0.008, noise |
| M4: YOLO ball detector on the video | – | dropped (40% precision) |

**What the diagnostics say about the remaining error** (`scripts/diag_liveplay.py`):

- **Missed short possessions are the biggest loss.** 21% of true possessions are missed entirely
  (median 3.4 s), and possessions under 4 s are 29% of the truth with only about 30% caught. On
  those, the classifier itself prefers the wrong team: often a loose ball after an interception or
  clearance, with no player close.
- **Not the decoder.** Tonight's touch-aware decoder showed that even free switching can't recover
  possessions the evidence doesn't support. The evidence has to improve.

## 4. What to try (ranked)

### 1. Neural sequence model (best use of a strong machine)

- **Why.** Every model so far is per-frame trees plus hand-made rolling windows. A model that reads
  the whole feature sequence can learn the temporal patterns we can't hand-design: touch, ball
  speed-up, the receiving team's shape changing.
- **Input.** The feature-v2 matrix `X` (122 columns) from `model.copy_features(mid, params, offset,
  swapped, 2)`, per frame, plus the labels `y` and the graded mask `g`. Subsample to 5 fps
  (`model.SUBSAMPLE` = 5). Standardise each column, then add `isnan` indicator channels and set
  NaN to 0.
- **Architecture to start with.** A 1D temporal CNN with dilations, or a 2-layer BiGRU, over
  windows of about 60 s (300 steps) with about 30 s overlap. Output log-softmax over H/A/D at each
  step, with a class-balanced cross-entropy on graded steps only.
- **Data augmentation (cheap here).** The degrader makes unlimited fresh corruptions: offsets 0–8
  are valid (`make_copy` / `copy_features` build and cache any new offset). Every copy also has its
  team-swapped mirror (`swapped=True`). **Never train on offset 9**: it's the held-out evaluation
  copy.
- **Protocol.** 7 folds, leave one match out. Predict the held-out match's offset-9 copy, repeat
  each step's output over its 5 frames back to frame resolution, and write
  `cache/oof_nn_<name>.npz`. Score with `ensemble_experiments.py`, standalone and averaged with
  `f2_all`. The ensemble with v4 is the most likely winner even if the network alone is only level.
- **If it beats the bar.** Tobias or Claude wires it into a `v5.py` like `v4.py`: train on all 7
  matches, save the weights to `models/`, and make sure regeneration is deterministic (fixed seeds,
  CPU inference).

### 2. More realistic training noise (degrader)

- **Why.** BAR-ATH's tracking has errors that `possession/degrade.py` does not simulate, so the
  model never learned to ignore them. Measured on BAR-ATH (aggregates):
  - during play a team shows **12+ players on the pitch** on 7.7% (home) / 4.6% (away) of frames.
    That's duplicates plus wrong-team tracks; the degrader only simulates duplicates;
  - **about 4.9 identity jumps per minute**: a track moving faster than 20 m/s, meaning the tracker
    swapped two players;
  - the ball is present on 57% of frames, in long gaps (median 9.2 s).
- **What to add.**
  - Wrong-team tracklets: a player's positions for a stretch of seconds appear in a slot of the
    other team.
  - Identity swaps: two players' tracks exchange from some frame on.
  - Fit the rates so the corrupted IDSSE reproduces the two numbers above. Measure the current
    degrader's 12+ rate first; the gap is what to add.
- **Careful.** The feature caches are keyed by match / offset / feature version, **not** by
  degrader version. Use a separate `POSSESSION_CACHE_DIR` for the new degrader, or stale caches
  will silently mix in.
- **Evaluate.** Run the full CV with the new degrader for training *and* the held-out copy, then
  compare with v4 under the same new evaluation corruption: score v4's design under both
  corruptions. The question is whether training with the noise helps when the noise is there.

### 3. Cheap but now affordable

- **Fully nested stacking.** It's 42 stage-1 fits: for each held-out match, the inner 6-fold
  out-of-fold predictions. Tonight's stacking (`scripts/stack_experiments.py`) used the usual
  shortcut and got +0.006. A clean version would say whether that's real.
- **HGB hyperparameter sweep on feature v2.** Learning rate, leaves, `min_samples_leaf`, l2, more
  copies (`--more-copies`). Use `cv_idsse.py --hgb '{"max_iter": 800, ...}' --feature-version 2
  --nested --skip-v1 --tag <name>`. Bigger trees gave +0.006, so don't expect much, but it's free
  on a big machine.

### 4. Not recommended

- **Rules on top of the decoder output.** The post-filter lost score.
- **Anything tuned on the public window.** It's 17 possessions; one possession is about 0.03.

## 5. Handing results back

- Add one row per experiment to `experiments.md`: date, idea, nested spellF1, per-match wins and a
  one-line verdict. Commit only code and docs.
- If something beats the bar, open a PR. Tobias retrains on all 7 matches, regenerates, checks the
  sha on a fresh clone, and resubmits.

## 6. Useful files

| file | what |
|---|---|
| `possession/features.py` | all features; `design_matrix(..., version=2)` is v4's input |
| `possession/degrade.py` | the IDSSE → BAR-ATH-like corruption |
| `possession/model.py` | training copies, HGB, `copy_features` (cached) |
| `possession/cv.py`, `possession/decode.py` | decoder and its tuning |
| `scripts/cv_idsse.py` | LOMO CV for an HGB variant |
| `scripts/ensemble_experiments.py` | scores any `oof_<tag>.npz` (or a mean of several) nested, vs v4 per match |
| `scripts/diag_liveplay.py` | where the remaining spellF1 is lost |
| `viewer/` + `scripts/viewer_export.py` / `viewer_serve.py` | local-only match viewer (127.0.0.1); never share its output |
| `TECHNICAL.md` | the technical document (method, results, reproduction) |
