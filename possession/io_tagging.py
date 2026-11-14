"""Smart Tagging (SportsCode XML) -> frame intervals with a team side.

Time base (verified by the organisers): seconds from the first frame of the tactical video, so
frame = round(seconds * 25). Codes and tag values are matched case-insensitively.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional

from . import config, organisers
from .canonical import AWAY, HOME

# Phases where the tagged team has the ball, and phases where the tagged team is defending
# (so the other team has it). Every non-set-piece code in DATA.md falls in exactly one group.
ATTACK_CODES = {"build up", "progression", "ball in final third", "ball in the box",
                "players in the box", "creating chances", "long ball", "attacking transition"}
DEFEND_CODES = {"defending in defensive third", "defending in middle third",
                "defending in attacking third", "defensive transition"}
SET_PIECE_CODE = "set pieces"
GOAL_CODE = "goals"


@dataclass
class Tag:
    code: str                  # lower case
    start: int                 # video frame
    end: int                   # video frame, exclusive
    team: Optional[int]        # HOME / AWAY / None (N/A)
    type: str                  # lower case "Type" tag, "" if absent


def load(game=None, data_dir=None, fps=25):
    g = organisers.games()
    game = game or g.GAME
    path = g.require(["smart_tagging"], game, data_dir or config.MTAH_DATA_DIR)[0]
    # The home side is the first team in the ATD metadata (Barcelona); the tagging spells the away
    # club "Atletic Club". Map by name, case-insensitively.
    home_names = {"barcelona", "fc barcelona"}
    away_names = {"atletic club", "athletic club", "athletic bilbao"}
    tags = []
    for inst in ET.parse(path).iter("instance"):
        labels = {(l.findtext("group") or "").strip().lower(): (l.findtext("text") or "").strip()
                  for l in inst.findall("label")}
        team_name = labels.get("team", "").lower()
        team = HOME if team_name in home_names else AWAY if team_name in away_names else None
        tags.append(Tag(code=(inst.findtext("code") or "").strip().lower(),
                        start=int(round(float(inst.findtext("start")) * fps)),
                        end=int(round(float(inst.findtext("end")) * fps)),
                        team=team, type=labels.get("type", "").lower()))
    return tags
