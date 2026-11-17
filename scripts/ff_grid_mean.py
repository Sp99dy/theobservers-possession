"""mean_decoder with a high team-switch-cost grid extension: team_cost adds {12, 20, 30} (6 already
in the grid), min_d already holds {5, 10} steps. Same protocol as v5.2 (Law 9, 1e-3, nested).

    python scripts/ff_grid_mean.py --members nn_gru05 nn_gru05s1 nn_tcn05 --law9 --quantize 0.001 \
        --start <v5 e1 decoder> --base v52_mean --tag ffgrid
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import mean_decoder as MD  # noqa: E402

MD.EXTRA = {"min_team": (10,), "team_cost": (6.0, 12.0, 20.0, 30.0), "min_d": (5, 10)}

if __name__ == "__main__":
    MD.main()
