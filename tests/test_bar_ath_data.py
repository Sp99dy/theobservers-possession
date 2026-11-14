"""Checks against the real BAR-ATH bundle. Shapes and counts only; skipped without the data."""

import hashlib
import subprocess

import numpy as np

from conftest import requires_data
from possession import baseline, config, io_atd, write
from possession.canonical import AWAY, HOME, measure_attack_sign


@requires_data
def test_atd_shape_and_team_split():
    m = io_atd.load_match()
    assert m.n_frames == 199319
    assert m.players_xy.shape == (199319, 46, 2)
    assert int(np.sum(m.team_of_slot == HOME)) == 24
    assert int(np.sum(m.team_of_slot == AWAY)) == 22
    # 24 home slots come first in the file.
    assert m.team_of_slot[:24].tolist() == [HOME] * 24


@requires_data
def test_half_masks_match_organisers():
    m = io_atd.load_match()
    assert m.half_blocks() == [(1, 11185, 90652), (2, 110808, 190198)]
    assert int(m.graded.sum()) == 158857


@requires_data
def test_coordinates_in_metres_and_plausible():
    m = io_atd.load_match()
    g = m.graded
    x = m.players_xy[g, :, 0]
    # Players stay within a few metres of the pitch.
    assert np.nanmin(x) > -60 and np.nanmax(x) < 60
    assert 0.55 < m.ball_present[g].mean() < 0.65          # 59.3% measured


@requires_data
def test_teams_swap_ends_at_halftime():
    signs = measure_attack_sign(io_atd.load_match())
    assert signs[1] == -signs[2]


@requires_data
def test_baseline_smooth50_byte_identical():
    codes = baseline.baseline_codes()
    ours = hashlib.sha256(write.render(baseline.baseline_labels(codes, 50))).hexdigest()
    rel = "evaluation/results/baseline_smooth50.csv"
    try:
        blob = subprocess.run(["git", "-C", config.MTAH_REPO_DIR, "show", "HEAD:" + rel],
                              capture_output=True, check=True).stdout
        theirs = hashlib.sha256(blob).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        theirs = write.sha256_file(config.MTAH_REPO_DIR + "/" + rel, normalise_crlf=True)
    assert ours == theirs


@requires_data
def test_v1_is_deterministic_and_graded_only():
    from possession import io_tagging, v1
    from possession.canonical import X
    m = io_atd.load_match()
    tags = io_tagging.load()
    first = v1.predict(m, tags)
    second = v1.predict(io_atd.load_match(), io_tagging.load())
    assert hashlib.sha256(write.render(first)).digest() == hashlib.sha256(write.render(second)).digest()
    assert (first[~m.graded] == X).all() and (first[m.graded] != X).all()


@requires_data
def test_v2_is_deterministic_and_graded_only():
    from possession import io_tagging, v2
    from possession.canonical import X
    m = io_atd.load_match()
    first = v2.predict(m, io_tagging.load())
    second = v2.predict(io_atd.load_match(), io_tagging.load())
    assert write.render(first) == write.render(second)
    assert (first[~m.graded] == X).all() and (first[m.graded] != X).all()
