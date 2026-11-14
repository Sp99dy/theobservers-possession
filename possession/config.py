"""Every path and hyperparameter in one place.

Rule for this file: each tunable value carries a comment saying where it came from. Allowed sources
are (a) the organisers' published constants (games.py), (b) football / physical reasoning,
(c) measured statistics of the BAR-ATH tracking itself (never of its ground truth), and
(d) cross-validation on the external IDSSE data. Never the public ground-truth window.
"""

import os

# -- paths ---------------------------------------------------------------------------------------

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The organisers' repository (read-only). We import games.py and evaluation/possession_metrics.py
# from it rather than copying them, so we always score with exactly their code.
MTAH_REPO_DIR = os.environ.get("MTAH_REPO_DIR") or os.path.join(
    os.path.dirname(REPO_DIR), "more-than-a-hack-2027")

# The gated match bundle. Same variable and same default as the organisers' games.py.
MTAH_DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(MTAH_REPO_DIR, "data")

# Parsed arrays are cached here. Gitignored: cached match data must never be committed.
CACHE_DIR = os.environ.get("POSSESSION_CACHE_DIR") or os.path.join(REPO_DIR, "cache")

# Generated submissions go here. Gitignored for the same reason.
OUT_DIR = os.environ.get("POSSESSION_OUT_DIR") or os.path.join(REPO_DIR, "out")

SUBMISSION_NAME = "TheObservers.csv"

# -- canonical coordinates -----------------------------------------------------------------------

# Pitch size in metres, from games.py ("measured from the corrected tracking"). The ATD metadata
# says 105 x 68; the organisers' value is the one the ground truth was built on.
# Read from games.py at runtime (see organisers.pitch()); this is only the fallback for tests.
PITCH_FALLBACK = (105.3, 68.0)
FPS = 25

# ================================================================================================
# v1 (M1) hyperparameters. NONE of these was chosen by looking at a score on any ground truth.
# Sources: [F] football / physical reasoning, [M] measured on BAR-ATH *tracking* (never its labels),
# [D] a design choice about the decoder, argued in the comment.
# ================================================================================================

# -- features ------------------------------------------------------------------------------------
BALL_OUT_MARGIN_M = 1.0      # [F/M] tracking noise near the lines is ~0.5 m; 1 m beyond the line
                             # is unambiguously out (whole ball over the line = out, law 9).
BALL_SPEED_CAP = 40.0        # [F] faster than any kick; caps detector jumps.
PLAYER_SPEED_CAP = 12.0      # [F] above the human sprint maximum (~10-11 m/s): tracklet switches.

# -- dead-ball evidence (z, nats per frame) --------------------------------------------------------
DEAD_SHARE_PRIOR = 0.375     # [F] the ball is out of play for ~35-40% of a professional match
                             # (effective playing time is typically 55-65 of ~95 minutes).
                             # The speed threshold is the DEAD_SHARE_PRIOR quantile of the match's
                             # own 1 s mean player speed [M], so the speed cue splits the match
                             # in about the proportion football says it should.
SPEED_SLOPE = 1.0            # [F] nats per m/s: a walking team (~1 m/s) vs a playing one (~2.5 m/s)
SPEED_CLIP = 1.5             #     differs by ~1.5 nats/frame, never more (slow live phases exist).
BALL_OUT_LLR = 3.0           # [F] ball beyond the line: about as decisive as a cue gets.
BALL_MISSING_LLR = 0.4       # [F] trackers lose the ball more when it is out/held, but a missing
                             # ball is mostly occlusion in live play: weak.
BALL_MOVING_SPEED = 4.0      # [F] m/s, a pass or a dribble; a ball carried to a restart is slower.
BALL_MOVING_LLR = 0.7
BALL_STILL_SPEED = 0.6       # [M] the frame-to-frame jitter floor of the ATD ball is ~0.5 m/s.
BALL_STILL_LLR = 0.5         # [F] a ball standing still is usually being placed for a restart.

# -- team evidence (y, nats per frame) -----------------------------------------------------------
CONTROL_DISTANCE_M = 2.0     # [F] within ~2 m a player can play the ball; evidence halves at 2 m
                             # and fades as 1/d^2 beyond (ball in flight).
TEAM_MARGIN_SCALE_M = 2.0    # [F] a 2 m gap between the nearest home and nearest away player is a
TEAM_MARGIN_CLIP = 1.5       #     clear call; beyond 3 m more distance adds nothing.
TERRITORY_SCALE_M = 20.0     # [F] centroid 20 m into a half = sustained territorial pressure.
TERRITORY_LLR = 0.1          # [F] weak: territory and possession correlate only loosely.

# -- kickoffs --------------------------------------------------------------------------------------
KICKOFF_HALFWAY_TOL_M = 1.5  # [M] player position noise around the halfway line.
KICKOFF_OWN_HALF_FRAC = 0.85 # [F/M] law 8 says all players; 0.85 tolerates one mis-teamed tracklet.
KICKOFF_MIN_PLAYERS = 6      # [M] tracking often sees only part of a team during line-ups.
KICKOFF_MIN_DURATION_S = 3.0 # [F] a kickoff line-up lasts many seconds; open play never shows
                             # both teams separated by the halfway line for 3 s while still.
KICKOFF_MAX_SPEED = 1.5      # [F] m/s: players stand still for a kickoff.
PRE_KICKOFF_WINDOW_S = 180.0 # [F] a half that is graded before its kickoff has the line-up within
                             # its first minutes (teams take the field, then kick off).
KICKOFF_DEAD_LLR = 4.0       # [F] the line-up is a dead ball with near certainty.
KICKOFF_LIVE_LLR = 1.0       # [F] the 3 s after the line-up breaks are live play...
KICKOFF_TEAM_WINDOW_S = 3.0
KICKOFF_TEAM_LLR = 1.5       # ...and the kicking team has the ball.

# -- Smart Tagging (BAR-ATH only) -------------------------------------------------------------------
TAG_WEIGHT = 1.0             # [D] single scalar on all tag terms; the terms below are already set
                             # so that tags stay below a confident ball observation.
TAG_TEAM_LLR = 0.25          # [D] a phase tag is a weak team vote: a sixth of a clear nearest-
                             # player call (1.5). Enough to steer a ball-less gap, too weak to
                             # override the ball when it is seen.
SET_PIECE_DEAD_BEFORE_S = 4.0  # [F] a restart is preceded by a stoppage; tag starts are loose,
SET_PIECE_DEAD_AFTER_S = 1.0   #     so the window is a few seconds around the tagged start.
SET_PIECE_DEAD_LLR = 0.8
SET_PIECE_TEAM_S = 5.0       # [F] the tagged team takes the set piece, so has the ball after it.
GOAL_DEAD_LLR = 1.5          # [F] after a goal the ball is dead until the kickoff.
GOAL_TO_KICKOFF_MAX_S = 180.0  # [F] celebrations + VAR rarely exceed 3 minutes.

# -- decoder ---------------------------------------------------------------------------------------
STEP = 5                     # [D] 0.2 s grid; boundaries move by <= 4 frames, which the metric
                             # barely notices (3 frames late still scores 0.98).
STEP_WEIGHT = 1.0            # [D] one 0.2 s step = one observation's worth of evidence.
MIN_STEPS = (5, 5, 10)       # [F] H, A >= 1 s (controlled possession takes time; a touch is not a
                             # change); D >= 2 s (no stoppage restarts faster than that).
# switch_cost[from][to] in nats.
# Team change [F+M]: "controlled" possession = the new team keeps the ball, which we take as
# ~0.75 s of close control (a touch or deflection lasts ~0.3-0.5 s). Typical close-control evidence
# measured on BAR-ATH tracking (ball within 1.5 m of the nearest player, no labels) is a median
# |y| of 1.2 nats/frame, so 0.75 s = 0.75 x 5 steps x 1.2 = 4.5 nats.
# (A first draft used 6.0 on an assumed 1.5 nats/frame; the measured value showed that needed a
# full second of unbroken control, and the GT-free check found 34 >= 1 s close-control spells by
# the other team that the decoder ignored. Corrected before any ground truth was consulted.)
# Stoppages [D]: entering costs 5, leaving 3, so a whole D stretch costs 8 nats and needs ~2-3 s
# of clear dead-ball evidence.
TEAM_SWITCH_COST = 4.5
SWITCH_COST = ((0.0, TEAM_SWITCH_COST, 5.0),
               (TEAM_SWITCH_COST, 0.0, 5.0),
               (3.0, 3.0, 0.0))
