"""M0 pipeline check: rebuild the organisers' four baselines through our loader and writer and
confirm each is byte-identical to the committed file in their repository.

    python scripts/reproduce_baseline.py

The organisers' files are compared as committed in git (LF line endings). A Windows checkout with
core.autocrlf=true shows them with CRLF on disk, so the on-disk copy is compared after CRLF->LF.
Prints hashes and label counts only.
"""

import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import baseline, config, write  # noqa: E402

VARIANTS = (("baseline_raw", 0), ("baseline_smooth25", 25),
            ("baseline_smooth50", 50), ("baseline_smooth125", 125))


def reference_sha(name):
    """sha256 of the organisers' committed file: git blob if available, else CRLF-normalised disk copy."""
    rel = "evaluation/results/{}.csv".format(name)
    try:
        blob = subprocess.run(["git", "-C", config.MTAH_REPO_DIR, "show", "HEAD:" + rel],
                              capture_output=True, check=True).stdout
        import hashlib
        return hashlib.sha256(blob).hexdigest(), "git HEAD"
    except (OSError, subprocess.CalledProcessError):
        path = os.path.join(config.MTAH_REPO_DIR, rel)
        return write.sha256_file(path, normalise_crlf=True), "disk (CRLF->LF)"


def main():
    start = time.time()
    codes = baseline.baseline_codes()
    print("nearest-player codes built in {:.1f}s: home {}  away {}  none {}".format(
        time.time() - start, int(np.sum(codes == 0)), int(np.sum(codes == 1)),
        int(np.sum(codes == baseline.NOBODY))))

    ok = True
    for name, window in VARIANTS:
        labels = baseline.baseline_labels(codes, window)
        path = os.path.join(config.OUT_DIR, "{}.csv".format(name))
        ours = write.write_submission(path, labels)
        theirs, source = reference_sha(name)
        same = ours == theirs
        ok &= same
        print("{:<20} {}  ours {}...  theirs {}... ({})".format(
            name, "IDENTICAL" if same else "DIFFERENT", ours[:12], theirs[:12], source))
    print("\nALL IDENTICAL" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
