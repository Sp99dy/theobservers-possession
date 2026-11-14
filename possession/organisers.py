"""Import shim for the organisers' repository (games.py and evaluation/possession_metrics.py).

We import their code instead of vendoring it, so the metric we optimise is byte-for-byte the one
that ranks us. The repository location comes from config.MTAH_REPO_DIR (env var MTAH_REPO_DIR).
"""

import importlib
import os
import sys

from . import config


def _ensure_path():
    repo = config.MTAH_REPO_DIR
    if not os.path.isfile(os.path.join(repo, "games.py")):
        raise ImportError(
            "organisers' repository not found at {!r}. Clone "
            "https://github.com/metrica-sports/more-than-a-hack-2027 next to this repo, or set "
            "MTAH_REPO_DIR.".format(repo))
    for path in (repo, os.path.join(repo, "evaluation")):
        if path not in sys.path:
            sys.path.insert(0, path)


def games():
    """The organisers' games.py module."""
    _ensure_path()
    return importlib.import_module("games")


def pm():
    """The organisers' possession_metrics.py module (the official scorer)."""
    _ensure_path()
    return importlib.import_module("possession_metrics")


def baseline():
    """The organisers' starter/baseline.py module (for its smoothing rule)."""
    _ensure_path()
    starter = os.path.join(config.MTAH_REPO_DIR, "starter")
    if starter not in sys.path:
        sys.path.insert(0, starter)
    return importlib.import_module("baseline")
