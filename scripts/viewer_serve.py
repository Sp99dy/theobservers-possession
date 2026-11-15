"""Serve the match viewer on this machine only (127.0.0.1).

    python scripts/viewer_serve.py            # then open http://127.0.0.1:8765/
    python scripts/viewer_serve.py --port 9000

Routes
    /                 viewer/index.html
    /data/<name>.json out/viewer/<name>.json  (written by scripts/viewer_export.py)
    /video            the tactical broadcast mp4, with HTTP Range so the browser can seek

Binds to localhost only: the match data and video are gated and must not leave this PC.
"""

import argparse
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from possession import config as C  # noqa: E402

INDEX = os.path.join(REPO, "viewer", "index.html")
DATA_DIR = os.path.join(C.OUT_DIR, "viewer")
VIDEO = os.path.join(C.MTAH_DATA_DIR, "tactical", "BAR-ATH_tactical.mp4")
CHUNK = 1 << 20


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _file(self, path, ctype, allow_range=False):
        if not os.path.isfile(path):
            self.send_error(404, "not found: " + os.path.basename(path))
            return
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = self.headers.get("Range") if allow_range else None
        if rng:
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                    end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    start = max(0, size - int(m.group(2)))
                end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", "bytes {}-{}/{}".format(start, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as fh:
            fh.seek(start)
            left = end - start + 1
            try:
                while left > 0:
                    buf = fh.read(min(CHUNK, left))
                    if not buf:
                        break
                    self.wfile.write(buf)
                    left -= len(buf)
            except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
                pass  # the browser cancels range requests whenever it seeks

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/index.html"):
            return self._file(INDEX, "text/html; charset=utf-8")
        if path == "/video":
            return self._file(VIDEO, "video/mp4", allow_range=True)
        m = re.fullmatch(r"/data/([A-Za-z0-9_\-]+)\.json", path)
        if m:
            return self._file(os.path.join(DATA_DIR, m.group(1) + ".json"), "application/json")
        if path == "/data/":
            names = sorted(f[:-5] for f in os.listdir(DATA_DIR) if f.endswith(".json")) \
                if os.path.isdir(DATA_DIR) else []
            body = ("[" + ",".join('"{}"'.format(n) for n in names) + "]").encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    do_HEAD = do_GET


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("match viewer on http://127.0.0.1:{}/   (Ctrl+C to stop)".format(args.port), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
