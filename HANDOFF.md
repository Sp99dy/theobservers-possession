# Handoff: where the model stands and what to try next

## v5.3 broadcast ball fill (2026-11-17): not adopted, v5.2 stays

`possession/ballfill.py` fills tactical ball gaps from the broadcast ATD (constant per-half lag
4014 / 26920 frames, measured label-free by `scripts/measure_broadcast_lag.py`: median ball
distance 0.80 / 0.85 m). The broadcast ball is noisy (70–74% within 2 m where both exist), so the
reviewed rule is conservative and all-or-nothing per gap: gap ≤ 3 s, the broadcast ball within 2 m
of the tactical one on the 5 visible frames at both edges, valid homography and no step > 3 m.
**No frame qualifies**: tactical gaps are long (150 gaps, 64.6k frames, ~17 s mean); the ≤ 3 s gaps
hold 75 frames and none passes the edge check. `run.py --method v53` therefore reproduces v5.2's
sha 5a9f6e23. A looser first version (any length, runs joining the tactical ball within 5 m) filled
3 879 frames (sha 45787031, commit f5659a5) but was superseded: unverifiable, and outside the
reviewed rule.

## v5.2 (2026-11-17): v5.1 + a decoder tuned for every label convention

BAR-ATH's label convention is unknown, so the selection metric is now the **mean nested spellF1
over the 2 s, 1 s and 0.5 s conventions** (one decoder for all three). `scripts/mean_decoder.py`
tunes the decoder on that mean (grid + min_team 10, start = the 1 s decoder), with Law 9 and 1e-3
rounding active, nested over the 7 IDSSE matches:

| | mean | 2 s | 1 s | 0.5 s | wins (per-match mean) |
|---|---|---|---|---|---|
| v5.1 (2 s-tuned decoder) | 0.6650 | 0.7264 | 0.6585 | 0.6101 | |
| **v5.2 (mean-tuned)** | **0.6801** | 0.7201 | 0.6825 | 0.6377 | 6/7 (+0.0151) |

Decoder: step weight 2, team cost 0, enter D 10, leave D 5, min possession 10 steps (2 s), min D 1,
D bias 0.5. Weights are v5.1's (`models/v52/v5.json` points at `models/v51/*.pt`).
`python run.py --method v52` → sha256 `5a9f6e23f91ca68382e94d0e6b352bea55a3893183a8c52b024c7bb003b508f6`
(two runs identical). BAR-ATH: 282 possessions (12.8 s mean, ≈ 4.7 per live minute), D share 0.433;
2120 frames differ from v5.1.

## v5.1 (2026-11-17, branch `fadel/nn`): v5 + Law 9, hardened for byte-identical rebuilds

**What changed vs v5.** (1) **Law 9 override**: where the tracked ball is more than 1 m beyond a
touchline or goal line for at least 5 frames (0.2 s), H and A get log-likelihoods 30 nats below D
before decoding, so the decoder's minimum durations still apply (`rules.law9_mask` /
`law9_override`). The constants come from the law and the existing 1 m margin; nothing was swept.
The decoder was re-tuned with the rule active and came back with v5's settings. (2) **Committed
weights**: the three IDSSE-trained members and the config live in `models/v51/` (1.6 MB, like
`models/emission_v4.joblib`), so `run.py --method v51` loads them and never retrains. (3)
**Cross-platform hardening**: the final log-likelihoods are rounded to 1e-3 nats before Viterbi
(`v2.quantize_ll`, config `quantize`), and inference runs on 1 torch thread with deterministic
algorithms (config `inference_threads`).

| eval labels | v5 nested | **v5.1 nested** | v5.1 + rounding |
|---|---|---|---|
| 2 s | 0.7279 | 0.7264 (−0.0015) | 0.7264 (identical nested predictions) |
| 1 s | 0.6824 | 0.6821 (−0.0003) | 0.6821 (identical) |
| 0.5 s | 0.6411 | 0.6404 (−0.0007) | 0.6404 (identical) |

Law 9 cannot gain on IDSSE: 99.8% of the 12 214 masked eval-copy frames are truly dead, and the NN
already calls them D. On BAR-ATH (label-free), H/A frames on an out-of-play stretch go from 422 to 2
and H/A spells touching one from 32 to 2. Adopted under its brief: non-inferior (≥ v5 − 0.002) at
every bridge, plus that aggregate.

**Reproduce**: `python run.py --method v51`. Expected sha256
`b8239637e98e0aa68e58f4426131890ad0f7bcf9fce82502b2beaa70a96b674b`. Rounding and 1 thread change 0
frames vs the unrounded 4-thread run, so the sha is unchanged. Two runs are identical (82 s on 1
thread). `scripts/check_float_robustness.py` perturbs the ensemble emission by uniform ±1e-6 and
±1e-4 (a stand-in for arm64 vs x86 / BLAS differences): 0 frames change, with or without rounding.
**Still unchecked: an actual run on Windows x86.** To rebuild the weights instead of using the
committed ones: the v5 commands below, then `train_nn.py ... --law9 --out-dir models/v51`
(byte-identical on this Mac), and add `"quantize": 0.001, "inference_threads": 1` to its
`v5.json`.


## v5 candidate (2026-11-16, branch `fadel/nn`): for Tobias to decide

**What changed vs v4.** (1) Training labels use the **0.5 s label bridge** (`POSSESSION_CONTROL_MIN_S=0.5`)
instead of v4's 2 s. (2) The HGB emission is replaced by the equal-weight mean of **three neural
sequence models** (HANDOFF §4.1): two 2-layer BiGRUs (seeds 0 and 1) and one dilated TCN, over
feature v2 at 5 fps. They are trained on degrader offsets 0–8 plus mirrors, never offset 9.
Kickoff, Smart Tagging and the segmental decoder are v4's code; the decoder settings come from the
ensemble's own pooled IDSSE CV (`possession/v5.py`, `possession/nn.py`).

**Nested IDSSE CV spellF1** (7-fold LOMO, decoder re-tuned without the held-out match, per eval-label
bridge; wins = matches beaten vs v4-b05 at the same eval bridge):

| eval labels | v4 (2 s train) | v4-b05 (0.5 s train) | **v5 (NN ensemble b)** | wins vs v4-b05 |
|---|---|---|---|---|
| 2 s | 0.6835 | 0.6962 | **0.7279** | 7/7 |
| 1 s | 0.5907 | 0.6241 | **0.6824** | 7/7 |
| 0.5 s | 0.5496 | 0.5838 | **0.6411** | 7/7 |

Every NN variant cleared the §2 bar on its own (single GRU 0.7154–0.7232, TCN 0.7215). Details and
alternatives (a)/(c) are in `experiments.md`.

**Reproduce** (Python 3.14, `pip install -r requirements.txt`, which now pins `torch==2.15.0`; data
paths as in §1):

```bash
POSSESSION_CONTROL_MIN_S=0.5 python scripts/cv_idsse.py --tag b05_f2 --feature-version 2 --nested --skip-v1   # v4-b05: decoder start + base
POSSESSION_CONTROL_MIN_S=0.5 python scripts/nn_cv.py --name gru05 --arch gru
POSSESSION_CONTROL_MIN_S=0.5 python scripts/nn_cv.py --name gru05s1 --arch gru --seed 1
POSSESSION_CONTROL_MIN_S=0.5 python scripts/nn_cv.py --name tcn05 --arch tcn
python scripts/ensemble_experiments.py --members b05_f2 --base b05_f2 --eval-bridge 2.0 --save-nested
cp cache/nested_preds_ens_b05_f2_e2.npz cache/nested_preds_b05_f2_e2.npz                  # base for per-match wins
python scripts/ensemble_experiments.py --members nn_gru05 nn_gru05s1 nn_tcn05 --base b05_f2 --eval-bridge 2.0   # writes the decoder json
POSSESSION_CONTROL_MIN_S=0.5 python scripts/train_nn.py --member gru:0:nn_gru05 --member gru:1:nn_gru05s1 \
    --member tcn:0:nn_tcn05 --decoder-from "cache/cv_ens_nn_gru05+nn_gru05s1+nn_tcn05_e2_decoder.json"
python run.py --method v5          # expected sha256 b4f23924e7216db4b9477ba202f3b9f71b420a1687d49029bb4156d4f6627f86
```

`train_nn.py` trains on all 7 matches (about 8 min on 4 CPU threads). The feature caches for offsets
2–8 build on demand (about 10 s per copy). The v5 decoder is tuned on the NN out-of-fold files and
starts from `cv_b05_f2_decoder.json`, so a fresh clone runs the CV steps first. All of them are deterministic.

**Weights are not committed** (public repo). They live in `out/models_v5/` (`nn_gru05.pt`,
`nn_gru05s1.pt`, `nn_tcn05.pt`, `v5.json`), or in `POSSESSION_V5_DIR`, and are regenerated by `train_nn.py`.

**Determinism.** On macOS arm64 (CPU, torch 2.15.0), a full retrain into a fresh directory gave
byte-identical weights and the same CSV sha. Two inference runs also matched, and so did a fold
trained at 2 vs 4 threads. v4's sha (`ce5399e8…`) is unchanged.
**Open caveat: byte-identical regeneration on Windows x86 is unchecked.** CPU float kernels can
differ between platforms, so check the sha there before resubmitting.

**Also on `fadel/bridge` (3cb4618), neither adopted.** Degrader v2 (wrong-team swaps fitted to
BAR-ATH's 12v10 rate) and fully nested stacking.

---

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
