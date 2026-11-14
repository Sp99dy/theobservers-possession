import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from possession import config  # noqa: E402


def _have_match_data():
    tactical = os.path.join(config.MTAH_DATA_DIR, "tactical")
    return os.path.isfile(os.path.join(tactical, "BAR-ATH_tactical_atd.txt"))


# Tests marked requires_data read the gated match bundle locally. They assert shapes and counts
# only, never print data, and are skipped in a fresh clone without the bundle.
requires_data = pytest.mark.skipif(not _have_match_data(),
                                   reason="gated match data not available")
