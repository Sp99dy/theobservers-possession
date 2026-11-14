"""End to end: match data -> out/TheObservers.csv, then the organisers' format validation.

    python run.py                       # default method
    python run.py --method baseline50   # the organisers' published floor, recomputed by us

Never retrains anything. Deterministic: the same data bundle always gives the same bytes, and the
sha256 printed at the end is the one to quote in the technical document.
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from possession import baseline, config, write  # noqa: E402


def method_baseline50():
    """M0: the organisers' baseline_smooth50, recomputed through our loader (pipeline check)."""
    return baseline.baseline_labels(baseline.baseline_codes(), 50)


METHODS = {
    "baseline50": method_baseline50,
}
DEFAULT_METHOD = "baseline50"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", choices=sorted(METHODS), default=DEFAULT_METHOD)
    parser.add_argument("--out", default=os.path.join(config.OUT_DIR, config.SUBMISSION_NAME))
    args = parser.parse_args()

    labels = METHODS[args.method]()
    digest = write.write_submission(args.out, labels)
    print("method {}  ->  {}".format(args.method, args.out))
    print("sha256 {}\n".format(digest), flush=True)

    validator = os.path.join(config.MTAH_REPO_DIR, "evaluation", "validate_submission.py")
    return subprocess.run([sys.executable, validator, args.out]).returncode


if __name__ == "__main__":
    sys.exit(main())
