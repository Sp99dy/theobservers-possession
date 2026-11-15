"""Tactical video helpers for M4 (ball detection to fill the ATD's ball gaps).

Nothing here is ever shown to a person: frames are decoded, cut into tiles and fed to code. The
training labels are AUTOMATIC: the ATD's own ball position projected into the image through the
provided homography (both are algorithm outputs, not human annotation).

Geometry. The homography maps normalised pitch coordinates (0-1, the ATD's own convention) to
pixels. The ball is 4-9 px wide (median 6), so detection runs on full-resolution 640x640 tiles.
"""

import os
import sys

import numpy as np

from . import config, io_atd, organisers

W, HGT = 1920, 1080
# The ball is 4-9 px wide and often blurred, so the detector sees MOTION and a 2x ZOOM:
#   * input = 3 channels: grey frame t, |t - (t-GAP)| and |(t+GAP) - t| (amplified), so a moving
#     ball lights up even where a single frame shows no clear blob;
#   * 320x320 full-resolution tiles, scaled to 640 by the detector.
TILE = 320
GAP = 2
MOTION_GAIN = 4.0
# Tile origins: x covers the full width, y the band where the pitch is (ATD ball p5-p95 y is
# 225-883 px), with 40 px overlap.
TILE_X = (0, 280, 560, 840, 1120, 1400, 1600)
TILE_Y = (120, 380, 640)
BALL_DIAMETER_M = 0.22


def video_path():
    return organisers.games().asset_paths(data_dir=config.MTAH_DATA_DIR)["video"]


def load_homographies():
    sys.path.insert(0, config.MTAH_REPO_DIR)
    from starter import load_homography
    H, err, valid = load_homography.load(data_dir=config.MTAH_DATA_DIR)
    return H, err, valid


def project(Hm, x, y):
    """Normalised pitch -> pixel, for one 3x3 matrix and scalar or array x, y."""
    p = Hm @ np.vstack([np.atleast_1d(x), np.atleast_1d(y), np.ones(np.size(x))])
    return p[0] / p[2], p[1] / p[2]


def unproject(Hm, u, v):
    """Pixel -> normalised pitch (ground plane)."""
    return project(np.linalg.inv(Hm), u, v)


def atd_ball_video():
    """(T, 3) ATD ball (normalised x, y, speed) indexed by video frame, and the graded mask."""
    g = organisers.games()
    meta_path, raw_path = g.require(["atd_meta", "atd_raw"], g.GAME, config.MTAH_DATA_DIR)
    raw = io_atd.load_raw(meta_path, raw_path)
    n = g.source_frames("tactical", g.GAME)
    _, ball = io_atd.raw_to_video_index(raw, n)
    return ball, io_atd.half_array(g.GAME, n) > 0


def motion_image(prev, cur, nxt):
    """Three grey frames (uint8) -> HxWx3 uint8 detector input (grey, back diff, forward diff)."""
    c = cur.astype(np.float32)
    back = np.clip(np.abs(c - prev) * MOTION_GAIN, 0, 255)
    fwd = np.clip(np.abs(nxt - c) * MOTION_GAIN, 0, 255)
    return np.dstack([cur, back.astype(np.uint8), fwd.astype(np.uint8)])


def iter_motion(wanted):
    """Yield (frame_index, HxWx3 uint8 motion image) for the sorted frame indices in `wanted`.
    Frames within GAP of either end of the video are skipped."""
    import av
    import cv2
    wanted = np.asarray(sorted(set(int(f) for f in wanted)))
    if not len(wanted):
        return
    need = set()
    for f in wanted:
        need.update((f - GAP, f, f + GAP))
    buf = {}
    k = 0
    with av.open(video_path()) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(s)):
            if i in need:
                buf[i] = cv2.cvtColor(frame.to_ndarray(format="bgr24"), cv2.COLOR_BGR2GRAY)
            while k < len(wanted) and wanted[k] + GAP <= i:
                f = wanted[k]
                if f - GAP in buf and f in buf and f + GAP in buf:
                    yield int(f), motion_image(buf[f - GAP], buf[f], buf[f + GAP])
                k += 1
                for old in [j for j in buf if j < wanted[k] - GAP] if k < len(wanted) else list(buf):
                    del buf[old]
            if k == len(wanted):
                return


def iter_frames(wanted):
    """Yield (frame_index, HxWx3 uint8 BGR) for the sorted frame indices in `wanted`."""
    import av
    wanted = np.asarray(sorted(set(int(f) for f in wanted)))
    if not len(wanted):
        return
    k = 0
    with av.open(video_path()) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(s)):
            if i == wanted[k]:
                yield i, frame.to_ndarray(format="bgr24")
                k += 1
                if k == len(wanted):
                    return


def tiles():
    return [(x, y) for y in TILE_Y for x in TILE_X]
