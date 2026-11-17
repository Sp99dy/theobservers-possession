# TheObservers — technical document (draft)

*Draft, filled in as the milestones land (outline from SPEC §10). Current state: v4 (§5.0b) is the default of `run.py`; v2 (M2) was the last version submitted.*

## 1. Approach in one paragraph

The ranking metric scores whole possessions, so a flicker inside a possession costs as much as a
wrong possession. We therefore split the problem in two. First, per-frame **evidence** for H, A and D
is computed from the tracking geometry. Second, a **segmental state-space decoder** turns that
evidence into a few long, stable segments. The decoder is a hidden Markov / regime-switching model
over {H, A, D} with switching costs and minimum durations, decoded by Viterbi. The evidence comes
from a gradient-boosting classifier trained on **external** labelled tracking data (IDSSE / DFL open
data). That data is first corrupted to look like the BAR-ATH automatic tracking, so the model learns
from inputs as noisy as the ones it will see. The decoder's costs and durations are chosen by
leave-one-match-out cross-validation on that external data. On held-out IDSSE matches (nested CV)
this scores spellF1 0.643, against 0.389 for the hand-set rules of our v1. v4 adds features that
detect ball touches from the ball's trajectory and the collective slowdown of the players at
stoppages, and reaches 0.684 nested.

## 2. Assets used / dropped

- **Tactical ATD**: core input (players, ball).
- **Smart Tagging**: a weak prior added to the decoder's log-likelihoods (§5.3). It is never copied
  across as labels.
- **Homography + tactical video**: not used. Detecting the ball in the video to fill the 41% ball
  gaps is a stretch goal.
- **IDSSE / DFL open data** (external): training data for the evidence model and the decoder (§3).
- **Broadcast video / ATD**: dropped. It is on its own unsynchronised timeline.

## 3. Training data

**IDSSE / DFL open data.** Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12, 195,
doi:10.1038/s41597-025-04505-y, CC-BY 4.0. The set has 7 Bundesliga and 2. Bundesliga matches,
about 11 hours of 25 fps tracking for all 22 players and the ball. The labels are DFL's official
per-frame `BallStatus` (alive/dead) and `BallPossession` (home/away) flags. They were produced by
DFL's data provider, not by us. **Nothing was labelled on BAR-ATH**, and no BAR-ATH labels were
used for anything. The public window was run once per version as a bug check (§6).

**Label convention bridge.** DFL's possession flag follows *touches*: 4.6 possessions per minute,
median 3.7 s, 37% shorter than 2 s. The challenge counts *controlled* possession: "a simple touch
of the ball without control" is not a change, and a loose ball stays with the last team. We apply
the written definition as a rule. A team change counts only once the new team keeps the ball for
≥ 2 s ("a few seconds", lower end), and the first possessor after every restart is kept. This
gives 3.0 possessions per minute and a mean of 11.0 s. That is the same regime as the challenge's
published reference (3.4 per minute, 14.2 s), which was used only for this qualitative comparison.
The 2 s value was not tuned.

**IDSSE's dead-ball ball is a placeholder.** On 100% of dead-ball frames DFL's ball has exactly
zero displacement: a frozen position, usually on or just past a line. During live play this happens
on only ~0.5% of frames. The degrader below accounts for it.

## 4. Handling missing and noisy tracking

Measured on the BAR-ATH tracking (aggregates only, no labels):
- The ball is missing on 40.7% of graded frames, always in blocks longer than 2 s (median 10.7 s).
- The provided ball `speed` channel is unusable: its maximum is 7 m/s. Speeds computed from ball
  positions are realistic (median 5 m/s, p95 17 m/s), so the positions are not smoothed. We compute
  all speeds from positions.
- Slots are tracklets (46 slots, 22 players), so every feature is a team aggregate: nearest distance
  to the ball per team, mean speed, counts, centroid.
- The attacking direction is measured per half from the teams' median positions. Teams swap ends
  at half-time, and coordinates are rotated so that home always attacks +x.

**The degrader** (`possession/degrade.py`) makes clean IDSSE look like the BAR-ATH ATD. Every
parameter is fitted to statistics of the BAR-ATH *tracking*, never to its labels:

| statistic | BAR-ATH ATD | clean IDSSE | degraded IDSSE |
|---|---|---|---|
| ball missing, share of graded frames | 0.407 | 0.000 | 0.457 |
| missing-run length, median / p90 (frames) | 268 / 845 | – | 268 / 989 |
| ball missing, slowest / fastest player-speed decile | 0.90 / 0.10 | 0 / 0 | 0.87 / 0.26 |
| visible ball > 1 m outside the pitch | 2.7% | ~6% | 1.9% |
| players per team (mean) | 10.2–10.3 | 11.0 | 10.2 |
| player jitter, median second difference (m) | 0.0014 | 0.0100 | 0.0013 |

- **Ball gaps** come from an alternating renewal process, with run lengths drawn from the ATD's
  own gap lengths.
- **Gap starts** have a per-frame hazard with three factors:
  - the true state (live/dead), the factor that lets the gaps carry information;
  - the ball's distance to the nearest line, using the ATD's measured relative hazard;
  - slow play, exp(β·(1.5 − speed)).
- **Calibration.** The live and dead rates and β were set by simulation, so that degraded IDSSE
  reproduces the ATD's *observable* curve of missing-ball share by player-speed decile. Player
  speed is a quantity both datasets share without labels.
- **Out-of-pitch balls** are thinned to the ATD's share. Players get dropouts and short duplicate
  tracklets, and both players and ball are smoothed to the ATD's jitter.
- **Team-swap augmentation.** Each corrupted copy is also mirrored: rotated 180°, with home and
  away exchanged. This keeps the classifier team-agnostic.

## 5. Model and possession rules

### 5.0 v2 (submitted): learned evidence

- **Features** (`features.design_matrix`, 89 columns). The same code path builds them for IDSSE and
  for BAR-ATH. They do not depend on player identity and are orientation-normalised:
  - the ball: presence, position, distance to the lines, speed computed from positions, out of
    pitch;
  - the nearest home and nearest away distance to the ball, and their margin;
  - across ball gaps: which team had close control when the ball was last seen and when it is next
    seen, and how far away in time that was;
  - team shape: centroids, spreads, defensive and forward lines, own-half shares;
  - motion: speeds and forward velocities per team;
  - centred rolling means over ±0.5 s, ±2 s and ±5 s. This is offline labelling, so using future
    frames is allowed.
- **Classifier**: scikit-learn 1.9.1 `HistGradientBoostingClassifier` over H, A and D.
  - 300 iterations, 31 leaves, balanced class weights, a fixed `random_state` and no early stopping.
  - Training uses every 5th frame of 4 copies of each of the 7 IDSSE matches: 2 corruption seeds,
    each also mirrored.
  - Retraining reproduces the committed model file byte for byte.
- **Emissions** are the classifier's log-probabilities. Because training used balanced classes,
  they act as scaled likelihoods under a flat prior.
- **BAR-ATH-only layers**: the kickoff detector (5.1) and the Smart Tagging prior (5.3) are added on
  top, converted into the classifier's units.
  - The conversion factor is measured without labels. On clear close-control frames the classifier
    gives 3.8 times the home-vs-away evidence that the v1 rules give (4.6 nats against 1.2).
  - These terms are multiplied by 3.8, so a tag keeps the same weight relative to a clear ball
    observation as it had in v1.
- **Decoder settings**, chosen on IDSSE CV (coordinate descent on pooled out-of-fold spellF1):
  - evidence weight 3.0 per 0.2 s step;
  - team switch 2 nats;
  - entering a stoppage 10 nats, leaving one 5 nats;
  - minimum possession 2.4 s, minimum stoppage 4 s;
  - dead-ball bias +0.5 nats.

### 5.0b v4 (M3b, the default): touch and slowdown features

v4 is the v2 pipeline with feature version 2 (`features.design_matrix(..., version=2)`, 122
columns). It came out of the live-play loss breakdown (`scripts/diag_liveplay.py`): most lost
possessions are short ones after interceptions or clearances, where the nearest player is a poor
guide to who has the ball.
- **Touch detection** (`features.detect_touches`): a ball contact is a change in ball velocity of
  at least 3 m/s with a player within 3 m; the team of the nearest player is the toucher. Features:
  team of the last and next touch, time since and until them, touch balance over the surrounding
  seconds. On BAR-ATH this finds about 14 touches per minute (degraded IDSSE: 8–12).
- **Slowdown** (`features.slowdown_features`): mean speed over the previous and next 3, 6 and 12 s of
  all players and of the 4 players nearest the ball, and the share of players under 2 m/s. Players
  slow down together when play stops (fouls, offsides, the ball going out).
- **Classifier**: same HGB settings as v2. A larger model (1000 trees, 127 leaves) gave +0.006
  nested, within noise, so it was not adopted.
- **Decoder** (re-tuned on IDSSE CV for this classifier): evidence weight 4.5, team switch 4.5
  nats, entering a stoppage 7, leaving one 5, minimum possession 2.4 s, minimum stoppage 3 s,
  dead-ball bias +0.5.
- The kickoff and Smart Tagging layers are unchanged; their conversion factor is re-measured for
  the new classifier (3.9).

The v1 rules below remain the documented baseline (`run.py --method v1`). v2 reuses their kickoff
detector and Smart Tagging prior.

## 5b. v1 rules (M1 baseline)

### 5.1 Evidence (nats per frame)

- **Dead vs live, z**:
  - Mean player speed (1 s window), linear around the 37.5% quantile of the match's own speed
    distribution. 37.5% is the share of time a football match is typically dead.
  - Ball beyond a line by more than 1 m: strong dead.
  - Ball missing: weakly dead.
  - Ball moving faster than 4 m/s: live.
  - Ball standing still: weakly dead.
- **Home vs away, y**: the gap between the nearest home and nearest away player to the ball, weighted
  by how close the nearest player is (close control is strong evidence, a ball in flight weak). Plus
  a very weak territory term.
- **Kickoffs** (law 8: every player in his own half). We detect stretches of at least 3 s where
  ≥85% of each team is in its own half while players stand still. These are strong dead evidence, and
  the team with a player nearest the centre spot gets the ball afterwards. A half whose graded span
  begins with such a line-up is dead until the kickoff. Six kickoffs were found: the second-half
  start and the four restarts after goals. No open-play false positives. No frame constants are used.

### 5.2 Decoder

- Evidence is averaged over 0.2 s steps, each treated as one observation. This discounts the strong
  dependence between neighbouring frames.
- Minimum durations: possession ≥ 1 s, stoppage ≥ 2 s. They are implemented exactly by chain-state
  expansion (a semi-Markov model).
- Switch costs:
  - Team change: 4.5 nats, about 0.75 s of typical close control. The definition asks for
    *controlled* possession, and a touch or deflection lasts less than that.
  - Entering a stoppage: 5 nats. Leaving one: 3 nats.
- Each half is decoded separately, with a flat initial distribution.

### 5.3 Smart Tagging prior

- Attacking phases vote for the tagged team. "Defending…" and "Defensive transition" vote for the
  other team. Each vote is 0.25 nats/frame, a sixth of a clear nearest-player call.
- Set pieces add mild dead-ball evidence just before the tagged start, and favour the tagged team
  just after it.
- Goals make the ball dead until the next detected kickoff.
- One overall scalar weight (1.0). The terms were sized by reasoning, not by score.

Every value in `possession/config.py` carries its source: football reasoning, a measurement of
BAR-ATH *tracking*, or a decoder design argument.

## 6. Results

**IDSSE leave-one-match-out CV** is the only basis for model selection.
- Each match is scored on a held-out corruption that training never saw, using the organisers' own
  `possession_metrics`.
- Scores are pooled over all 14 halves, which contain 2017 true possessions.

| version | spellF1 | macroF1 | possessions (GT 2017) | mean duration (GT 11.0 s) |
|---|---|---|---|---|
| v1 rules (no tags: IDSSE has none) | 0.389 | 0.788 | 1148 | 19.5 s |
| v2 evidence, v1 decoder settings | 0.633 | 0.896 | 1595 | 14.0 s |
| v2, decoder tuned on all folds (in-sample for the 7 decoder values) | 0.652 | 0.897 | 1716 | 12.8 s |
| **v2, nested** (decoder re-tuned without the held-out match) | **0.643** | **0.895** | 1701 | 13.0 s |

- Per-match v2 spellF1 ranges from 0.55 to 0.70. The mean over matches is 0.654, with a 90%
  bootstrap CI of 0.621–0.682.
- The classifier alone, taking the most likely label per frame, is 90.7% accurate.
- Almost all of the gain over v1 comes from the learned evidence. The decoder search surface is
  flat: different settings move spellF1 by about ±0.01.
- **Ablations and refinements** (same CV; "nested" re-tunes the decoder without the held-out
  match). Each row changes one thing relative to v2:

  | variant | spellF1 | reading |
  |---|---|---|
  | ball features only (tuned) | 0.491 | |
  | ball + team shape, no motion (tuned) | 0.555 | team shape +0.06, motion +0.10 |
  | trained on clean IDSSE, no degrader (tuned) | 0.566 | the degrader is worth +0.09 |
  | plain HMM, no minimum durations (nested) | 0.637 | minimum durations +0.006 |
  | split dead-ball states D_after_H / D_after_A (nested) | 0.646 | noise. The restart goes to the team that had the ball only 48% of the time |
  | 8 corrupted copies per match instead of 4 (nested) | 0.645 | no gain |
  | kickoff layer evaluated on IDSSE (nested) | 0.644 | neutral |
  | larger classifier: 600 trees, 63 leaves (nested) | 0.654 | capacity helps |
  | larger classifier: 1000 trees, 127 leaves (nested) | 0.661 | capacity helps further |
  | 600/63 + v1 rule evidence at weight 0.35 (nested) | 0.658 | ensemble adds a little |
  | 1000/127 + v1 rule evidence at weight 0.35 (nested) — **v3**, fallback | 0.666 | +0.004 over capacity alone, noise |
  | **feature version 2: touch + slowdown (nested) — v4** | **0.684** | per-match 0.688, CI 0.668–0.707 |
  | feature v2 without touch features (nested) | 0.676 | slowdown alone +0.034 |
  | feature v2 without slowdown features (nested) | 0.677 | touch alone +0.034; the two overlap |
  | feature v2 + 1000 trees, 127 leaves (nested) | 0.689 | +0.006, noise; not adopted |

- **What v2 misses is short possessions.** Recovered (same team, IoU ≥ 0.5) by true duration:

  | true duration | under 2.5 s | 2.5–5 s | 5–10 s | 10–20 s | 20–40 s | over 40 s |
  |---|---|---|---|---|---|---|
  | recovered | 20% | 41% | 63% | 86% | 91% | 99% |

  Possessions under 5 s are 38% of all true possessions.

**Secondary check: Metrica sample games 1 and 2** (public, `scripts/metrica_check.py`). These are
Metrica's own tracking and events.
- **Labels.** There is no per-frame possession, so H/A/D labels are derived from the events
  (`possession/io_metrica.py`):
  - the team of the last possession event owns the ball;
  - the ball is dead from a stoppage (ball out, a shot out or a goal, a foul, offside) until the
    next set piece or goal kick;
  - then the same 2 s control rule as for IDSSE.

  This gives 2.6–2.8 possessions per minute (mean 16–17 s).
- **Why it is useful.** The tracking is not degraded, but its ball is already missing on about 40%
  of frames, as in BAR-ATH.
- **Use.** Never used to tune anything. It only checks whether the ranking chosen on IDSSE holds on
  another provider's tracking.

| version | spellF1 | macroF1 | possessions (GT 509) |
|---|---|---|---|
| organisers' baseline rule (nearest player, smooth 50) | 0.425 | 0.737 | 943 |
| v1 rules (no tags) | 0.603 | 0.760 | 380 |
| **v2** (no tags) | **0.702** | **0.779** | 441 |

The ranking holds: baseline, then v1, then v2.

**Public 5-minute window — sanity check only, not used for any selection.** Each version was run
once.

| version | spellF1 | macroF1 | possessions (GT 17) | mean duration (GT 14.2 s) |
|---|---|---|---|---|
| organisers' baseline_smooth50 | 0.326 | 0.621 | 26 | 10.8 s |
| v1 without Smart Tagging | 0.400 | 0.683 | 11 | 25.6 s |
| v1 | 0.443 | 0.718 | 11 | 25.2 s |
| v2 without Smart Tagging | 0.462 | 0.763 | – | – |
| **v2 (submitted)** | **0.514** | **0.768** | 11 | 25.1 s |
| v3 (fallback) | 0.465 | 0.793 | 14 | 19.4 s |
| **v4 (default)** | 0.493 | 0.810 | 13 | 20.8 s |

On the full match, v2 predicts 226 possessions (mean 15.8 s) and 129 dead-ball stretches. Dead ball
takes 44% of graded frames. v1 predicted 193 possessions.

With 17 true possessions, one matched possession moves spellF1 by about 0.03, so v2 and v4 are not
separable on this window. v4 labels the offside stoppage at 1:07 of the window (5.5 s) as dead
ball on 109 of its 138 frames; v2 labels none of it dead. On the full match v4 predicts 222
possessions (mean 15.9 s).

## 7. Limitations and future work

- **Domain shift.** The degrader reproduces the ATD's main statistics, but not all of them. In the
  fastest decile of play, degraded IDSSE still loses the ball on 26% of frames, against 10% in the
  ATD. Bundesliga tracking also differs from the tracking used for BAR-ATH in ways we did not model.
- **Possessions get merged.** v2 leans towards fewer, longer possessions:
  - on IDSSE it finds 85% of the true number;
  - on BAR-ATH it predicts 226, about 2.1 per graded minute, against 3.4 per minute in the public
    reference window.

  For spellF1 this is the safer error, but it is still an error.
- **Label convention.** The 2 s controlled-possession bridge is our reading of the written
  definition. The challenge's reference labels may treat contested balls differently.
- **Video (tried, dropped).** We tried to fill the ATD's ball gaps from the tactical video:
  - **Method.** A small public detector (Ultralytics YOLO11n) was fine-tuned on full-resolution
    tiles. The labels were automatic: the ATD's own ball position, projected through the provided
    homography. No human labelled anything.
  - **Labels are aligned, ball is hard to see.** Measured by local contrast, the labels sit on
    the ball, but the ball is a visible blob at only about 30% of ATD positions: it is 4–9 px wide
    and often blurred. Detectors trained on single frames, then on frame-difference ("motion")
    inputs, did not learn.
  - **Best attempt.** Trained only on examples where the ball is visible. On held-out minutes its
    top detection is on the ATD ball only 40–56% of the time, even when confident. Gaps are
    harder still, so filling them would add mostly wrong ball positions. We dropped it; the
    scripts remain (`scripts/video_*.py`, `requirements-video.txt`).
  - **What could work with more time.** A dedicated tiny-ball model with longer temporal context
    (a heatmap network over several frames).

## 8. Reproduction

```bash
python run.py        # -> out/TheObservers.csv
```

**v5.1** (NN ensemble + Law 9, the candidate submission):

```bash
python run.py --method v51 --out out/TheObservers_v51.csv
```

It loads the committed IDSSE-trained weights and config in `models/v51/` (two BiGRUs and one TCN,
1.6 MB; `torch==2.15.0` from requirements.txt) and never retrains. Inference runs on one CPU thread
with torch's deterministic algorithms. The final log-likelihoods are rounded to 1e-3 nats before the
Viterbi decoder, so last-bit float differences between CPUs or BLAS builds cannot change the path.
Perturbing the emissions by ±1e-4 changes 0 frames. v5.1 sha256:
`b8239637e98e0aa68e58f4426131890ad0f7bcf9fce82502b2beaa70a96b674b`.

`run.py` loads the committed v4 model (`models/emission_v4.joblib`) and decoder settings
(`models/decoder_v4.json`). It never retrains. v4 sha256:
`ce5399e8bb93b6022e667726963f02b368efc25a4532cdc218be64d4854b92be`.

The previous submission, v2 (`models/emission_v2.joblib`, `models/decoder_v2.json`):

```bash
python run.py --method v2 --out out/TheObservers_v2.csv
```

v2 sha256: `f258bdea093f03c52f6fb24a258dbc23e6d17125a27738a7045f3ad576b041e6`.

To rebuild the model from scratch (this gives a byte-identical model file):

```bash
python scripts/download_idsse.py              # IDSSE open data
python scripts/build_idsse_features.py        # degrader fit (BAR-ATH tracking stats only) + IDSSE features
python scripts/cv_idsse.py --tag v2 --nested  # LOMO CV and decoder tuning
python scripts/train_idsse.py                 # final model on all 7 matches
# v4: feature version 2
python scripts/cv_idsse.py --tag f2_all --feature-version 2 --nested
python scripts/train_idsse.py --version v4 --tag f2_all
```

Pipeline validation: our ATD loader, slot-to-team mapping, frame alignment
(`video_frame = frame_id - 1`) and CSV writer reproduce all four of the organisers' baselines byte
for byte (`scripts/reproduce_baseline.py`).
