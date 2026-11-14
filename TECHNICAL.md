# TheObservers — technical document (draft)

*Draft, filled in as the milestones land (outline from SPEC §10). Current state: v1 (M1).*

## 1. Approach in one paragraph

The ranking metric scores whole possessions, so a flicker inside a possession costs as much as a
wrong possession. We therefore split the problem in two. First, per-frame **evidence** for H, A and D
is computed from the tracking geometry. Second, a **segmental state-space decoder** turns that
evidence into a few long, stable segments. The decoder is a hidden Markov / regime-switching model
over {H, A, D} with switching costs and minimum durations, decoded by Viterbi. In v1 the evidence is
hand-set rules argued from football. In v2 it will be a classifier trained on external, labelled
tracking data (IDSSE / DFL open data).

## 2. Assets used / dropped

- **Tactical ATD**: core input (players, ball).
- **Smart Tagging**: a weak prior added to the decoder's log-likelihoods (§5.3). It is never copied
  across as labels.
- **Homography + tactical video**: not used in v1 (stretch goal M4).
- **Broadcast video / ATD**: dropped. It is on its own unsynchronised timeline.

## 3. Training data

v1 has no training. No labels on BAR-ATH were used for anything, and the public window was run
once per version as a bug check (§6).

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

## 5. Possession rules implemented (v1)

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

*IDSSE leave-one-match-out CV: M2.*

**Public 5-minute window — sanity check only, not used for any selection:**

| version | spellF1 | macroF1 | possessions (GT 17) | mean duration (GT 14.2 s) |
|---|---|---|---|---|
| organisers' baseline_smooth50 | 0.326 | 0.621 | 26 | 10.8 s |
| v1 without Smart Tagging | 0.400 | 0.683 | 11 | 25.6 s |
| **v1** | **0.443** | **0.718** | 11 | 25.2 s |

On the full match v1 predicts 193 possessions (mean 19.8 s) and 101 dead-ball stretches, with a
dead-ball share of 0.40. Both the 5-minute window and the GT-free diagnostics suggest v1 merges
possessions; learning the switch costs on IDSSE (M2) is the principled fix.

## 7. Limitations and future work

*To be completed.*

## 8. Reproduction

```bash
python run.py        # -> out/TheObservers.csv
```

The run takes about 3 s with a warm cache and about 5 s cold (Windows 11, i5-13600K). v1 sha256:
`80a5f2352acd500c3297f8eaa0ef5cdc6ee9388682ee49eef995c7e3e1790f09`.

Pipeline validation: our ATD loader, slot-to-team mapping, frame alignment
(`video_frame = frame_id - 1`) and CSV writer reproduce all four of the organisers' baselines byte
for byte (`scripts/reproduce_baseline.py`).
