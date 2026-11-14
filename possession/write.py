"""Submission writer: one "frame,label" row per video frame, LF line endings, no trailing spaces.

Line endings are pinned to LF explicitly. Python's text mode would write CRLF on Windows and LF
elsewhere, and the file must be byte-identical wherever it is regenerated.
"""

import hashlib
import os

import numpy as np

from .canonical import LETTERS


def to_letters(labels):
    """Accept H/A/D/X codes (ints) or letters; return a numpy array of single letters."""
    labels = np.asarray(labels)
    if labels.dtype.kind in "iu":
        if labels.min(initial=0) < 0 or labels.max(initial=0) > 3:
            raise ValueError("label codes must be 0..3")
        return LETTERS[labels]
    return labels.astype("<U1")


def render(labels):
    """The exact bytes of a submission file."""
    letters = to_letters(labels)
    body = "".join("{},{}\n".format(i, l) for i, l in enumerate(letters.tolist()))
    return ("frame,label\n" + body).encode("ascii")


def write_submission(path, labels):
    """Write the file and return its sha256 hex digest."""
    data = render(labels)
    folder = os.path.dirname(os.path.abspath(path))
    os.makedirs(folder, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, normalise_crlf=False):
    with open(path, "rb") as handle:
        data = handle.read()
    if normalise_crlf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()
