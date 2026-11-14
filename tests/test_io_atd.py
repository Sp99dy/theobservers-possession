"""ATD parsing on a tiny synthetic EPTS pair written to a temp folder."""

import numpy as np
import pytest

from possession import io_atd
from possession.baseline import NOBODY, nearest_player_codes

META = """<?xml version="1.0" encoding="utf-8"?>
<main>
  <Metadata>
    <GlobalConfig><FrameRate>25.0</FrameRate></GlobalConfig>
    <Teams>
      <Team id="T_HOME"><Name>Home FC</Name></Team>
      <Team id="T_AWAY"><Name>Away FC</Name></Team>
    </Teams>
    <Players>
      <Player id="P1" teamId="T_HOME"/>
      <Player id="P2" teamId="T_HOME"/>
      <Player id="P3" teamId="T_AWAY"/>
    </Players>
    <Devices><Device><Sensors><Sensor><Channels>
      <Channel id="x"/><Channel id="y"/><Channel id="s"/>
    </Channels></Sensor></Sensors></Device></Devices>
    <PlayerChannels>
      {channels}
    </PlayerChannels>
  </Metadata>
  <DataFormatSpecifications>
    <DataFormatSpecification>
      <StringRegister name="frameCount"/>
      <SplitRegister separator=";">
        {slots}
      </SplitRegister>
      <SplitRegister separator=",">
        <BallChannelRef channelId="x"/><BallChannelRef channelId="y"/><BallChannelRef channelId="s"/>
      </SplitRegister>
    </DataFormatSpecification>
  </DataFormatSpecifications>
</main>
"""


def write_meta(path, slot_order=("P1", "P2", "P3")):
    channels = "\n".join('<PlayerChannel id="{p}_{c}" playerId="{p}" channelId="{c}"/>'.format(
        p=p, c=c) for p in ("P1", "P2", "P3") for c in ("x", "y", "s"))
    slots = "\n".join('<SplitRegister separator=",">{}</SplitRegister>'.format("".join(
        '<PlayerChannelRef playerChannelId="{}_{}"/>'.format(p, c) for c in ("x", "y", "s")))
        for p in slot_order)
    path.write_text(META.format(channels=channels, slots=slots))


RAW = ("1:0.1,0.2,1.5;0.9,0.8,2.0;0.5,0.5,0.0:0.11,0.21,3.0\n"
       "2:NaN,NaN,NaN;0.9,0.8,2.0;0.5,0.5,0.0:NaN,NaN,NaN\n"
       "3:0.1,0.2,1.5;NaN,NaN,NaN;0.52,0.5,0.0:0.5,0.5,0.1\n")


def test_metadata_slot_order_and_teams(tmp_path):
    meta = tmp_path / "m.xml"
    write_meta(meta, slot_order=("P3", "P1", "P2"))   # spec order, not Player order, decides
    info = io_atd.parse_metadata(str(meta))
    assert info["n_slots"] == 3
    assert info["team_of_slot"].tolist() == [1, 0, 0]
    assert info["team_names"] == ["Home FC", "Away FC"]
    assert info["frame_rate"] == 25.0


def test_parse_raw_shapes_and_values(tmp_path):
    raw = tmp_path / "r.txt"
    raw.write_bytes(RAW.replace("\n", "\r\n").encode())   # CRLF tolerated
    frame_id, players, ball = io_atd.parse_raw(str(raw), 3)
    assert frame_id.tolist() == [1, 2, 3]
    assert players.shape == (3, 3, 3) and ball.shape == (3, 3)
    assert players[0, 0].tolist() == [0.1, 0.2, 1.5]
    assert np.isnan(players[1, 0]).all() and np.isnan(ball[1]).all()
    assert ball[2].tolist() == [0.5, 0.5, 0.1]
    # Same values Python's float() gives (correctly rounded parse).
    assert players[0, 0, 0] == float("0.1")


def test_parse_raw_rejects_short_line(tmp_path):
    raw = tmp_path / "r.txt"
    raw.write_text(RAW + "4:0.1,0.2,1.5;0.9,0.8,2.0:0.1,0.1,0.1\n")
    with pytest.raises(ValueError):
        io_atd.parse_raw(str(raw), 3)


def test_video_index_is_frame_id_minus_one(tmp_path):
    raw = tmp_path / "r.txt"
    raw.write_text(RAW)
    frame_id, players, ball = io_atd.parse_raw(str(raw), 3)
    p, b = io_atd.raw_to_video_index({"frame_id": frame_id, "players": players, "ball": ball}, 5)
    assert p.shape == (5, 3, 3)
    assert p[0, 0, 0] == 0.1                      # frame_id 1 -> video frame 0
    assert np.isnan(p[3]).all() and np.isnan(p[4]).all()


def test_nearest_player_codes():
    pitch = (100.0, 50.0)
    players = np.array([
        [[0.10, 0.20, 0], [0.90, 0.80, 0], [0.50, 0.50, 0]],
        [[np.nan] * 3, [0.90, 0.80, 0], [0.50, 0.50, 0]],
        [[0.10, 0.20, 0], [np.nan] * 3, [0.52, 0.50, 0]],
        [[np.nan] * 3, [np.nan] * 3, [np.nan] * 3],
    ])
    ball = np.array([[0.11, 0.21, 0], [np.nan] * 3, [0.5, 0.5, 0], [0.5, 0.5, 0]])
    team = np.array([0, 0, 1], dtype=np.int8)
    codes = nearest_player_codes(players, ball, team, pitch)
    assert codes.tolist() == [0, NOBODY, 1, NOBODY]


def test_nearest_player_tie_goes_to_first_slot():
    players = np.array([[[0.4, 0.5, 0], [0.6, 0.5, 0]]])   # equidistant from the ball
    ball = np.array([[0.5, 0.5, 0]])
    assert nearest_player_codes(players, ball, np.array([1, 0], dtype=np.int8),
                                (100.0, 50.0)).tolist() == [1]
