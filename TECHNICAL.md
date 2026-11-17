# TheObservers: technical document

More Than a Hack 2027, phase 1 (BAR-ATH possession). Submitted model: **v5.5**.
Code: https://github.com/Sp99dy/theobservers-possession (repository main branch).

---

## 1. Approach

The ranking metric (spellF1) scores whole possessions, so a short flicker inside a possession costs
about as much as a wrong possession. We therefore split the task in two:

1. **Evidence.** Neural sequence models read 60 s of tracking-derived features and give per-frame
   log-probabilities for H, A and D.
2. **Decoding.** A segmental HMM (switching costs plus minimum durations, exact Viterbi) turns the
   evidence into a small number of stable segments.

The evidence model never sees a BAR-ATH label. It learns from **external** labelled tracking: 7
IDSSE / DFL matches whose tracking we corrupt to look like the BAR-ATH automatic tracking. Every
model choice is made by **nested leave-one-match-out cross-validation on IDSSE** (the decoder is
re-tuned without the held-out match).

**v5.5, in one list:**

- **Emission model:** the mean log-probability of five networks, all trained on the 7 IDSSE
  matches with a **0.5 s control bridge** on the labels (§3):
  two 2-layer BiGRUs (hidden 64, seeds 0 and 1) and a dilated TCN (6 residual blocks, dilations
  1–32), at 5 fps in 60 s windows with 30 s overlap; a **switch-weighted BiGRU** (3× loss on steps
  within 1 s of a label change); and a **12.5 fps BiGRU** (60 s windows). Input: 122 tracking
  features ("feature version 2") plus one missing-value channel per column.
- **Law 9 override:** where the tracked ball is more than 1 m beyond a touchline or goal line for at
  least 5 frames (0.2 s), H and A are set 30 nats below D, before decoding.
- **Rounding:** the final log-likelihoods are rounded to 1e-3 nats, so last-bit float differences
  between CPUs cannot change the Viterbi path.
- **Broadcast replay evidence:** on frames where the broadcast homography is invalid (a replay,
  close-up or crowd shot), D gets +2.195 nats (§2, §5). The weight was estimated label-free by EM
  on v5.2's decoding and is reused frozen.
- **Kickoff detector and Smart Tagging prior** (unchanged since v1/v4): weak, BAR-ATH-only terms
  added to the log-likelihoods.
- **Decoder:** tuned nested on the **mean spellF1 over three possession conventions** (labels
  bridged at 2 s, 1 s and 0.5 s), because the convention of the BAR-ATH ground truth is unknown.

**Why this and not something else.** Hand-written rules (our v1) scored 0.389 nested on IDSSE;
learned evidence scored far higher (§6). Per-frame gradient boosting saturated at about 0.68–0.69
whatever we added; a sequence model reads the build-up before a turnover and the restart after a
stoppage, and gained +0.03 at every convention. The segmental decoder enforces what the metric
rewards: few, stable possessions.

**How we got here.** v1 (14 Nov) was a rule-based evidence model with the same decoder: 0.389
nested. v2 replaced the rules by a gradient-boosting classifier trained on degraded IDSSE: 0.643.
v3 (a larger classifier plus the v1 rules, 0.666) was kept as a fallback only. v4 added
ball-touch detection and collective-slowdown features: 0.6835. v5 (16 Nov) replaced the
classifier by a three-network ensemble and moved the label bridge from 2 s to 0.5 s: 0.7279 at the
2 s convention. v5.1 added the Law 9 override and byte-identical reproduction; v5.2 re-tuned the
decoder on the mean over conventions (0.6801); v5.3 (broadcast ball fill) was not adopted; v5.4
added the replay evidence; v5.5 (17 Nov) adds two members aimed at short possessions and
re-tunes the decoder (0.6836).

---

## 2. Assets: used and dropped

| asset | use |
|---|---|
| Tactical ATD (players, ball) | Core input. All features are team aggregates (46 slots are tracklets, not players), orientation-normalised per half. |
| Smart Tagging | Weak prior in the decoder: attacking phases vote for the tagged side; set pieces add mild dead-ball evidence just before the tag and favour the tagged side after it; goals keep the ball dead until the next detected kickoff. Never copied across as labels. |
| Broadcast homography (+ broadcast ATD timeline) | **Used since v5.4** as replay evidence: homography invalid at broadcast frame t − 4014 (1st half) / t − 26920 (2nd half) → +2.195 nats on D. |
| Broadcast ATD ball | **Tried, dropped** (v5.3): a conservative fill of tactical ball gaps. 0 frames qualify (§6). |
| Tactical video | **Tried, dropped:** a YOLO11n ball detector fine-tuned on ATD-projected positions (no human labels). The ball is a visible blob at only about 30% of ATD positions; the best attempt had 40–56% top-1 precision. Too unreliable to fill gaps. |
| Tactical audio | **Checked, not used:** a tonal whistle detector finds 29–117 events depending on threshold; 52–64% fall in stretches v5 already calls dead. IDSSE has no audio, so it could not be validated, and throw-ins, corners and goal kicks have no whistle. |
| Referee | Not in the ATD (all 46 slots belong to the home or away side). |
| Public 5-minute window | Pipeline check only (§4). |

**Credit.** The constant broadcast-to-tactical lag and the idea of using broadcast cut-aways as
stoppage evidence come from public sources; we re-measured the lag and estimated the evidence
weight ourselves, label-free. (`scripts/measure_broadcast_lag.py`: median tactical–broadcast ball
distance 0.80 / 0.85 m at those lags; a blind ±80k-frame search lands at 4000 / 26900. The weight
comes from EM, §5, not from the public ground truth.)

---

## 3. Training data

| dataset | source / licence | labelled by | amount | used for |
|---|---|---|---|---|
| **IDSSE / DFL open data** | Bassek, Rein, Weber & Memmert (2025), *Scientific Data* 12, 195, doi:10.1038/s41597-025-04505-y; CC-BY 4.0 | DFL's official data provider: per-frame `BallStatus` (alive/dead) and `BallPossession` (home/away) flags | 7 Bundesliga / 2. Bundesliga matches, about 11 h of 25 fps tracking; 2017 possessions at the 2 s convention | **Training of v5.5 (all 7)** and nested CV (leave one match out) |
| PFF FC World Cup 2022 | PFF FC; registration; licence checked and confirmed by us | Per-frame H/A/D derived by our rule from PFF FC's event data; tracking is broadcast-derived | 64 matches available; 11 tested as extra training data | Tested, **not adopted** (§6): −0.017 nested |
| Metrica sample games 1 and 2 | Metrica Sports public sample data | Events by Metrica; per-frame H/A/D derived by our rule from those events + 0.5 s bridge | 2 matches (339 / 308 possessions) | Tested, **not adopted** (§6): −0.025 alone, −0.007 mixed |

**No label comes from BAR-ATH.** Nobody labelled any BAR-ATH frame, clip or transition, and no
constant was fitted to anything seen on BAR-ATH video. The only BAR-ATH-derived numbers in the
model are label-free: the degrader's tracking statistics and the replay weight (estimated by EM from
our own model's decoding, a pseudo-label method RULES.md allows).

**Label bridge.** DFL's possession flag follows touches. The challenge counts controlled
possession, so a team change in our training labels counts only once the new side keeps the ball
for ≥ 0.5 s, and the first possessor after every restart is kept. This is a deterministic transform
of DFL's flags. 0.5 s was chosen on nested IDSSE CV, where training at 0.5 s beat training at 2 s
under every evaluation convention (2 s / 1 s / 0.5 s: 0.6962 / 0.6241 / 0.5838 vs 0.6835 / 0.5907 /
0.5496, gradient-boosting model); for the BiGRU, training at 2 s lost 0.008 (0/7, §6). The bridge
removes short spells but never delays a switch.

**Degrader.** `possession/degrade.py` corrupts clean IDSSE so that it looks like the BAR-ATH ATD.
Every parameter is fitted to statistics of the BAR-ATH *tracking*, never to its labels:

| statistic | BAR-ATH ATD | clean IDSSE | degraded IDSSE |
|---|---|---|---|
| ball missing, share of graded frames | 0.407 | 0.000 | 0.457 |
| missing-run length, median / p90 (frames) | 268 / 845 | – | 268 / 989 |
| players per side (mean) | 10.2–10.3 | 11.0 | 10.2 |
| player jitter, median second difference (m) | 0.0014 | 0.0100 | 0.0013 |

Ball gaps follow a renewal process with the ATD's own run lengths and a hazard that depends on the
true state, the distance to the nearest line and slow play. Players get dropouts, short duplicate
tracklets and smoothing. Each copy is also mirrored (rotated 180°, H and A exchanged). The networks
train on 9 independent corruptions per match (offsets 0–8) plus mirrors; offset 9 is the held-out
evaluation copy. Ablation (v2 era): the degrader is worth +0.09 spellF1.

---

## 4. What we did not do

- **No hand labels on BAR-ATH**, in any amount or for any purpose.
- **No tuning or selection on the public 5-minute window.** v5.4 was scored on it once, after it
  had been chosen, as a pipeline check (frame alignment, file format): spellF1 0.619, 90% bootstrap
  interval 0.41–0.81, 17 possessions. That number was never used for any choice; v5.5 was chosen on
  nested IDSSE CV alone, on a bar declared before the run.
- **BAR-ATH never went to a cloud service or a cloud LLM**: no raw files, extracts, frames, video or
  screenshots. Only aggregates (scores, counts) left the machines.
- **The football expert's viewer review informed feature and rule ideas only.** Our teammate watched
  BAR-ATH clips in a local viewer (allowed by RULES.md for understanding the data) and described
  failure *types* in words. Nothing from it became a label, a fitted constant, or a selection
  criterion; every idea was then tested on IDSSE.

---

## 5. Possession rules: where we drew the line

The model is learned, so most rules are not `if` statements. Each enters in one of three places:
**labels** (how the 0.5 s bridge builds the training targets from DFL's flags), **features** (what
the networks can see), or the **decoder** (switch costs, minimum durations, explicit overrides).

| situation | our line | where it lives |
|---|---|---|
| **Touch vs control** | A team change needs the new side to keep the ball. One touch, a block or a deflection that does not lead to control is not a change. | Labels: 0.5 s bridge. Features: touch detector (ball velocity change ≥ 3 m/s with a player within 3 m; side of last / next touch). |
| **Proximity is not possession** | Being near the ball is never enough; a pass running past an opponent stays with the passer. | Features: touches and ball motion, not only nearest-player distance. |
| **When the switch happens** | At the winning side's touch, not back-dated to a pass in flight and not delayed. | Labels (DFL flips at the touch; the bridge never delays). If the winning run starts with a deflection, the switch is at the deflection. A constant switch delay was tested on IDSSE and lost (§6). The BAR-ATH reference appears to credit some interceptions slightly later (≈0.6 s, after the first controlled action); we did not fit that, since the only evidence is the public window. |
| **Loose balls and 50/50s** | The ball stays with the last side until one side comes out with control; the change is placed at the first touch of the winner's run. | Labels (bridge removes back-and-forth spells shorter than 0.5 s); decoder minimum possession 1.6 s. |
| **Aerial duels** | Winning the header is a possession only if the side keeps the ball afterwards. | As touch vs control. |
| **Tackle followed by a foul** | The fouled side keeps the ball; the ball is dead from the foul. | Labels (DFL dead at the whistle); slowdown features (all players slow together). |
| **Advantage** | Ordinary live play; possession changes only if the opponent gains control. | No special handling. |
| **Keeper** | A catch is a possession; a parry or punch is a touch like any other. | Learned from the DFL labels like any other player. |
| **Ball out (Law 9)** | Dead from the moment the ball is out, whatever the players do next. | Labels (DFL `BallStatus`); features (ball beyond a line, last sighting heading out); **override**: tracked ball > 1 m out for ≥ 5 frames forces D. Constants come from the law and the existing 1 m margin; nothing was swept. |
| **Whistle / offside** | Dead from the decision, even if players play on for a few seconds. | Labels; collective-slowdown features. |
| **Replays and close-ups on the broadcast** | Broadcasters cut away mostly while play is stopped, so a cut-away is evidence (not proof) of a dead ball. | +2.195 nats on D where the broadcast homography is invalid at the measured lag. Added after rounding, then re-rounded; the decoder's minimum durations still apply. |
| **Slow build-up** | Never dead because it is slow. | Slowdown features use all players and the ball's own movement, not speed alone. |
| **Restarts** | The restart side owns the ball from the restart. For throw-ins, corners and goal kicks that is the side that did not touch the ball last. | Labels (first possessor after a restart is kept); Smart Tagging set pieces; kickoff detector (≥ 85% of each side in its own half, players still, for ≥ 3 s). |
| **When a restart goes live** | At the restart kick or throw. | Labels: IDSSE goes live at the restart kick (measured on IDSSE by our label builder). The BAR-ATH reference sometimes goes live later, after the first pass; we accept the reference as the grader but did not fit a delay to the public window. See §7. |
| **Ball not tracked** | The last side keeps the ball unless the evidence on both sides of the gap (control when last and next seen, team shape) says otherwise. | Features across ball gaps; the degrader trains on BAR-ATH-like gaps. |

**Decoder (v5.5).** Evidence is averaged over 0.2 s steps. Settings, tuned nested on the mean over
the three conventions: step weight 1, team-switch cost 0, entering D 7 nats, leaving D 5, minimum
possession 8 steps (1.6 s), minimum stoppage 15 steps (3 s), D bias +0.5. Minimum durations are
exact (chain-state expansion). Each half is decoded separately.

**Replay weight.** 2.195 = log P(replay | D) / P(replay | live), estimated by hard EM on BAR-ATH from
v5.2's own decoding (iterations +2.020 / +2.179 / +2.195; P(r | D) 0.659, P(r | live) 0.073). It is
frozen in `models/v55/v5.json` (`replay_llr`) and was not re-estimated for v5.5, so reproduction
never runs EM. The bonus is added to the frame log-likelihoods before the decoder's step scaling,
so its weight relative to the network evidence is the same as under v5.4. No labels are involved.

---

## 6. Results

All scores are **nested** leave-one-match-out spellF1 on the 7 IDSSE matches, each scored on a
held-out corruption, with the organisers' metric. "Mean" is the mean over the 2 s, 1 s and 0.5 s
label conventions; since 17 Nov it is our selection metric, with a bar of ≥ +0.003 mean and ≥ 5/7
matches won.

| model | 2 s | 1 s | 0.5 s | mean | note |
|---|---|---|---|---|---|
| v1 rules | 0.389 | – | – | – | |
| v2 gradient boosting | 0.643 | – | – | – | |
| v4 (touch + slowdown features) | 0.6835 | 0.5907 | 0.5496 | – | decoder re-tuned per convention |
| v4 trained at 0.5 s bridge | 0.6962 | 0.6241 | 0.5838 | – | decoder re-tuned per convention |
| v5 NN ensemble | 0.7279 | 0.6824 | 0.6411 | – | decoder re-tuned per convention; 7/7 wins vs the row above at each convention |
| v5.1 (+ Law 9, rounding; one 2 s-tuned decoder) | 0.7264 | 0.6585 | 0.6101 | 0.6650 | |
| v5.2 (one decoder tuned on the mean) | 0.7201 | 0.6825 | 0.6377 | 0.6801 | +0.0151 vs v5.1, wins 6/7 |
| v5.4 = v5.2 + replay evidence | = v5.2 | = v5.2 | = v5.2 | = v5.2 | replay is unscorable on IDSSE (no broadcast); fallback |
| **v5.5 = 5 members, re-tuned decoder** | 0.7147 | 0.6890 | 0.6470 | **0.6836** | +0.0035 vs v5.2/v5.4, wins 5/7; the only combination tested, declared in advance |

v5.5 gives up 0.005 at the 2 s convention for +0.007 at 1 s and +0.009 at 0.5 s. (v5.2 had already
traded 0.006 at 2 s for +0.024 at 1 s and +0.028 at 0.5 s against a 2 s-tuned decoder.) Each new
member alone did not clear the bar (below), the two together did.

**On BAR-ATH (label-free aggregates only).** v5.5 predicts 289 possessions (mean 12.2 s), D share
0.444; 2365 frames differ from v5.4 (277 possessions, D share 0.442). Under v5.5 the replay evidence
changes 1430 frames (1500 under v5.4) and raises the D share by 0.009 in both. Law 9 (v5.1 vs v5):
H/A frames on an out-of-play stretch 422 → 2.

### Tried and not adopted

All on nested IDSSE CV, mean metric, against v5.2 (0.6801) unless stated.

| experiment | result | why not |
|---|---|---|
| **PFF WC2022 (11 matches) as extra training data** | 0.6635 (−0.017, 1/7); mixed with v5.2 at weight 0.5 / 0.33: −0.0023 / −0.0008 (3/7 each) | worse. Likely causes: event-derived labels give fewer, longer possessions (2.75–3.74 per min vs 3.5–4.9 on IDSSE), and about 60% of player positions are estimated |
| **Metrica games 1+2 as extra training data** | members trained with Metrica alone 0.6552 (−0.025, 0/7); mixed with v5.2 at 0.5: 0.6729 (−0.007, 2/7) | worse; event-derived labels are coarser (fewer, longer possessions, half the dead share) |
| v6: synthetic ball movement during dead time | members alone 0.6722 (−0.0079, 3/7); mixed equally with v5.2 0.6802 (+0.0001, 3/7); mixed at 0.33 0.6787 (−0.0014, 2/7) | no gain (see §7) |
| Feature v3: pressure / control / relative-speed features | vs v5.5: −0.0031 mean, +0.0065 at 2 s, wins 2/7 | worse on the mean |
| BiGRU trained at the 2 s bridge | −0.008 (0/7); in v5.5 −0.010 (0/7) | 0.5 s bridge better |
| 12.5 fps BiGRU alone (swapped for one GRU) | +0.0002 (5/7) | a tie; kept only as a v5.5 member |
| Switch-weighted BiGRU alone | +0.0018 (4/7) | below the bar; kept only as a v5.5 member |
| 120 s window BiGRU | +0.0005 (4/7) | 60 s context suffices |
| 24 epochs instead of 12 | +0.0001 (5/7) | no gain |
| Decoder grid with large team-switch costs (up to 30) and long minimum stoppage | +0.0000: the same decoder is chosen in every fold | no effect |
| Boundary refinement with our own touch evidence | −0.0003 (2/7) | no gain |
| Team-swap test-time augmentation | +0.0010 (4/7) | below the bar; members already train on mirrors |
| 7-member ensemble (4 GRU + 3 TCN seeds) | −0.0005 (4/7) | more seeds alone do not help |
| Conservative broadcast ball fill (v5.3) | 0 frames qualify (150 gaps; 17 gaps ≤ 3 s hold 75 frames, none passes the edge check) | nothing to fill |
| Consensus / minimum-Bayes-risk over 26 decodings | −0.0020 (2/7) | worse |
| Emission-gated short spells | +0.0000: the mean-tuned decoder never opens the gate (7/7 folds) | no effect |
| Constant switch delay 0.2 / 0.4 / 0.6 s | −0.0048 / −0.0144 / −0.0314; nested picks 0 s | labels flip at the touch |
| Snap stoppage start to ball exit | ±0.0000; nested picks off | almost never fires |
| Restart-side dead states | +0.0008 (4/7) | below the bar |
| Stacking (second-stage model on stage-1 outputs; v4 era, 2 s) | context-only +0.0057 (4/7, one match carries it); with raw features −0.013; fully nested −0.008 | noise or worse |
| Wrong-side swap degrader (v5 era, 2 s) | −0.003 / +0.001 | no gain |
| Identity-swap degrader (fitted to BAR-ATH's 4.9 jumps/min) | not evaluated: feature build stalled at the hard stop | no result in time |

Earlier (v2–v4 era), also within noise or worse: a touch-aware decoder (+0.002), HGB ensembles
(≤ +0.008), split dead-ball states (+0.004), a short-possession post-filter (−0.003).

### Where the remaining error is

An oracle decomposition on v5.2's nested predictions (mean 0.6801) replaces one part of the
prediction by the truth at a time. A perfect alive/dead mask adds only +0.004. The true side during
live play adds **+0.259** (this includes cutting our spells at the true changes). Snapping our
boundaries to true ones within ±1 s adds +0.029, but refining boundaries with our own evidence gave
−0.0003, so that gain is not reachable with the current evidence. Even the truth with every
possession under 2 s merged away scores only 0.861 on the mean (0.979 at 2 s, 0.768 at 0.5 s): the
ceiling for any decoder with a 2 s minimum. The error is therefore almost all *which side has the
ball in live play*, not stoppages.

---

## 7. Limitations

- **Missed short possessions.** On IDSSE (v5, 2 s convention) the model misses 51% of true
  possessions of ≤ 3 s (184 of 364). In 61% of those misses the emission already shows the right side
  for ≥ 0.3 s, and the decoder's minimum duration suppresses it. Typical cases: aerial balls, spells
  inside ball gaps, a ball lost straight back, the first touch after a restart. The mean-tuned
  decoders admit more short spells (v5.2: short-spell recall at 2 s 0.495 → 0.657, precision 0.672
  → 0.512), and v5.5's minimum possession is 1.6 s, but most short possessions remain hard.
- **Ball moved during dead time.** On BAR-ATH the ball is often moved while play is stopped (set-up
  passes to the restart taker, kickabouts during celebrations, VAR, substitutions): inside v5-predicted
  stoppages the ball is tracked on 17% of frames and moving on 40% of those, with a hand-over in 7%
  of stretches. IDSSE freezes the dead ball (moving on 0.4% of dead frames, no hand-overs), so the
  model never learned this and sometimes calls it a 2–4 s possession. Our synthetic fix (v6) cut such
  spells on BAR-ATH (short H/A between two D: 21 → 6) but did not gain on IDSSE, alone or mixed, and
  our standard evaluation cannot reward it. The replay evidence and v5.5's 3 s minimum stoppage
  address part of it.
- **Restart go-live convention.** Our labels go live at the restart kick. The BAR-ATH reference may go
  live later; we did not fit a delay, because the only evidence is a few public-window restarts. A
  start 0.5 s late on a 10 s possession still has IoU 0.95, so the spellF1 cost is small.
- **Tracking errors not in the training noise.** The away goalkeeper's track sits in the home box
  next to the home keeper for 6 350 (1st half) + 9 595 (2nd half) ≈ 16k frames, a median 6 m apart;
  which track is the real keeper cannot be told label-free, so we did not repair it. Wrong-side
  tracklets near duels: a side has 12+ players on 7.7% / 4.6% of frames per half; identity jumps
  (> 20 m/s) occur 4.9 times per minute, 0.142 per minute onto an opponent. Our degrader does not
  simulate wrong-side tracklets; the identity-swap degrader was not evaluated in time.
- **Convention uncertainty.** We do not know BAR-ATH's possession convention. The decoder is tuned on
  the mean over three conventions. If the truth is closer to 2 s, v5.5 gives up 0.005 against v5.2
  (0.7147 vs 0.7201) and 0.012 against a 2 s-tuned decoder (0.7264); if it is closer to 0.5 s, a 2 s-
  tuned decoder would have lost 0.037.
- **Replay evidence is unscorable.** IDSSE has no broadcast feed, so the replay term was adopted on
  label-free evidence only (EM-estimated weight, aggregates above), not on a CV score. Its weight was
  estimated on v5.2's decoding and reused for v5.5 without re-estimation.
- **Degrader mismatch in live play.** On our IDSSE evaluation copies the ball is missing on 28% of
  live frames; on BAR-ATH (proxy: v5.2-predicted live frames) about 9%. Ball gaps are therefore a
  weaker dead-ball cue in training than on the target, and the CV score may understate or misplace
  the model's live-play errors.
- **Seven matches, small margins.** v5.5's gain (+0.0035, 5/7) is near the noise of 7 matches (seed
  spread about 0.008). External data did not help: PFF and Metrica, with event-derived labels, made
  the score worse (§6), so more data of a different labelling style is not a free gain.

**Future work.** The oracle decomposition (§6) places the remaining error in which side has the
ball during live play. Our roadmap follows from that:

1. **Co-train the pressure / control / relative-speed features (feature v3) with a decoder tuned for
   them.** On their own they already raise frame accuracy on all 7 folds (by 0.001–0.002) and gain
   +0.0065 at the 2 s convention.
2. **Training data with ball movement during stoppages:** labelled tracking where the ball is not
   frozen when dead, so the stoppage evidence that BAR-ATH shows can be learned and scored.
3. **Validation with several seeds and several held-out corruptions per match,** so that gains of
   about 0.003 become measurable.

---

## 8. Reproduction

Python 3.14, `pip install -r requirements.txt` (pins `torch==2.15.0`, CPU). Data bundle paths via
`MTAH_REPO_DIR` / `MTAH_DATA_DIR` (see README). The model also reads the broadcast `_atd.txt/.xml`
and `_homography.csv` from the bundle.

```bash
python run.py
```

- Writes `out/TheObservers.csv` (the default method is `v55`).
- Expected sha256: `d7e4bcfe2fa4470903fc3fef28e2ce8040d1dd3559c296209769231fbfaf2a92`. Two runs
  identical; about 2 min on 1 thread.
- Loads the committed weights (`models/v51/*.pt`, three networks, and `models/v55/*.pt`, two
  networks; 2.8 MB in all) and `models/v55/v5.json` (members, decoder, Law 9, rounding, 1 inference
  thread, frozen replay weight). It never retrains and never runs EM.
- Determinism: inference on 1 torch thread with deterministic algorithms; log-likelihoods rounded to
  1e-3 before Viterbi. Perturbing the emissions by ±1e-4 changes 0 frames. Retraining the members
  from scratch (IDSSE download, feature caches, `scripts/train_nn.py`; commands in `HANDOFF.md`) gives
  byte-identical weights on macOS arm64.
- Cross-platform: a teammate rebuilt **v5.2** (`--method v52`, sha256 `5a9f6e23f91c…08f6`) on
  Windows x86 byte for byte (about 170 s), and the submitted v5.5 file was regenerated on the same
  Windows machine and checked against the sha above before upload.
- **Fallback:** `python run.py --method v54` gives v5.4, sha256
  `443667c05bfedc30f3c8125265fc5af62fde16f8a69f8ad485eae7a6e1b4a154`.
- Pipeline check: our loader, frame alignment and CSV writer reproduce the organisers' four baselines
  byte for byte (`scripts/reproduce_baseline.py`). Tests: `python -m pytest -q`, 109 pass.
