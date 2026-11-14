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
