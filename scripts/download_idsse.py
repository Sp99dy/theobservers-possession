"""Download the IDSSE / DFL open data (7 matches, CC-BY 4.0) to config.IDSSE_DIR.

    python scripts/download_idsse.py

Bassek, Rein, Weber & Memmert (2025), "An integrated dataset of spatiotemporal and event data in
elite soccer", Scientific Data 12, 195, doi:10.1038/s41597-025-04505-y. Files are fetched from the
URLs kloppy uses (huggingface.co/datasets/pysport/idsse-data). Existing files are skipped.
"""

import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config  # noqa: E402
from possession.io_idsse import MATCH_IDS, idsse_url, local_path  # noqa: E402


def fetch(url, path):
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return "cached"
    tmp = path + ".part"
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in r.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    os.replace(tmp, path)
    return "downloaded"


def main():
    os.makedirs(config.IDSSE_DIR, exist_ok=True)
    for match_id in MATCH_IDS:
        for kind in ("meta", "tracking", "event"):
            path = local_path(match_id, kind)
            status = fetch(idsse_url(match_id, kind), path)
            print("{} {:<8} {:>8.1f} MB  {}".format(match_id, kind,
                                                   os.path.getsize(path) / 1e6, status), flush=True)


if __name__ == "__main__":
    main()
