"""M4 feasibility probe for video ball detection. Prints aggregates only (no frames, no rows).

    python scripts/video_probe.py

* video container: resolution, frame count, codec, decode speed on a short stretch;
* ball size in pixels where the ATD sees the ball (projected diameter, via the homography);
* where the ATD loses the ball: the last/next seen ball position projected into the image, to
  see how often a gap starts inside the picture (so a detector has something to find).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from possession import config, io_atd, organisers  # noqa: E402

BALL_DIAMETER_M = 0.22


def homographies():
    g = organisers.games()
    sys.path.insert(0, config.MTAH_REPO_DIR)
    from starter import load_homography
    H, err, valid = load_homography.load(data_dir=config.MTAH_DATA_DIR)
    return H, err, valid, load_homography.project


def main():
    import av
    g = organisers.games()
    video = g.asset_paths(data_dir=config.MTAH_DATA_DIR)["video"]
    with av.open(video) as c:
        s = c.streams.video[0]
        print("video: {}x{}  frames {}  fps {}  codec {}  pix {}".format(
            s.codec_context.width, s.codec_context.height, s.frames, s.average_rate,
            s.codec_context.name, s.codec_context.pix_fmt))
        s.thread_type = "AUTO"
        t0, n = time.time(), 0
        for frame in c.decode(s):
            frame.to_ndarray(format="rgb24")
            n += 1
            if n == 500:
                break
        print("decode + rgb convert: {:.0f} frames/s".format(n / (time.time() - t0)))

    H, err, valid, project = homographies()
    meta_path, raw_path = g.require(["atd_meta", "atd_raw"], g.GAME, config.MTAH_DATA_DIR)
    raw = io_atd.load_raw(meta_path, raw_path)
    n_video = g.source_frames("tactical", g.GAME)
    _, ball = io_atd.raw_to_video_index(raw, n_video)
    length, width = g.GAMES[g.GAME]["pitch"]
    graded = io_atd.half_array(g.GAME, n_video) > 0

    seen = graded & ~np.isnan(ball[:, 0])
    idx = np.flatnonzero(seen)[::25]
    px, py = [], []
    d = []
    for f in idx:
        x0, y0 = project(H[f], ball[f, 0], ball[f, 1])
        x1, y1 = project(H[f], ball[f, 0] + BALL_DIAMETER_M / length, ball[f, 1])
        px.append(x0)
        py.append(y0)
        d.append(np.hypot(x1 - x0, y1 - y0))
    px, py, d = map(np.asarray, (px, py, d))
    inside = (px >= 0) & (px < 1920) & (py >= 0) & (py < 1080)
    print("ATD-seen ball, sampled 1/s: n {}  inside image {:.3f}".format(len(idx), inside.mean()))
    print("ball diameter px: p5 {:.1f}  p25 {:.1f}  median {:.1f}  p75 {:.1f}  p95 {:.1f}".format(
        *np.percentile(d[inside], [5, 25, 50, 75, 95])))
    print("ball pixel y: p5 {:.0f}  median {:.0f}  p95 {:.0f};  x: p5 {:.0f}  median {:.0f}  "
          "p95 {:.0f}".format(*np.percentile(py[inside], [5, 50, 95]),
                              *np.percentile(px[inside], [5, 50, 95])))

    # Gap edges: where was the ball just before the ATD lost it?
    miss = graded & np.isnan(ball[:, 0])
    starts = np.flatnonzero(miss[1:] & ~miss[:-1]) + 1
    edge_in = []
    for f in starts:
        x, y = project(H[f - 1], ball[f - 1, 0], ball[f - 1, 1])
        edge_in.append((0 <= x < 1920) and (0 <= y < 1080) and 0 <= ball[f - 1, 0] <= 1
                       and 0 <= ball[f - 1, 1] <= 1)
    print("gap starts: {}  last-seen ball inside image and pitch: {:.3f}".format(
        len(starts), np.mean(edge_in)))
    print("homography reprojection error on graded frames: median {:.2f}  p95 {:.2f}".format(
        *np.percentile(err[graded], [50, 95])))


if __name__ == "__main__":
    main()
