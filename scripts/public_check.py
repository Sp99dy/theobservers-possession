"""Score a submission on the public 5-minute window. SANITY CHECK ONLY.

    python scripts/public_check.py out/TheObservers.csv [more.csv ...]

Use it to catch pipeline bugs (alignment, format, label meaning). Never use it to choose between
versions or tune anything: 17 possessions cannot rank models (EVALUATION.md), and the rules forbid
it. Model selection happens on IDSSE cross-validation.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import metrics  # noqa: E402


def main(paths):
    gt = metrics.public_gt_codes()
    print("public window -- sanity only, not for selection")
    for path in paths:
        pred = metrics.read_submission_codes(path, n_expected=len(gt))
        s = metrics.summary(metrics.evaluate(gt, pred))
        print("{:<28} {}".format(os.path.basename(path), metrics.format_summary(s)))
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))
