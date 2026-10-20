import math
from types import SimpleNamespace

import pytest

from body.poses import Mood
from perception.face_reader import FaceReader

ENGAGE_YAW = 0.4
DISENGAGE_YAW = 0.6
ENGAGE_DWELL = 0.5
DISENGAGE_DWELL = 1.0
DT = 0.125


@pytest.fixture
def reader() -> FaceReader:
    return FaceReader(
        engage_yaw=ENGAGE_YAW,
        disengage_yaw=DISENGAGE_YAW,
        engage_dwell=ENGAGE_DWELL,
        disengage_dwell=DISENGAGE_DWELL,
    )


def face(yaw: float = 0.0, **blendshapes: float) -> SimpleNamespace:
    s, c = math.sin(yaw), math.cos(yaw)
    rotation = [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    scores = [
        SimpleNamespace(category_name=name, score=score)
        for name, score in blendshapes.items()
    ]
    return SimpleNamespace(
        facial_transformation_matrixes=[rotation], face_blendshapes=[scores]
    )


def feed(reader: FaceReader, result, steps: int):
    reading = None
    for _ in range(steps):
        reading = reader.read(result, DT)
    return reading


def steps_for(seconds: float) -> int:
    return round(seconds / DT)


def test_engages_only_after_dwell(reader):
    assert not feed(reader, face(), steps_for(ENGAGE_DWELL) - 1).engaged
    assert feed(reader, face(), 1).engaged


def test_glance_away_resets_engage_dwell(reader):
    feed(reader, face(), steps_for(ENGAGE_DWELL) - 1)
    feed(reader, None, 1)
    assert not feed(reader, face(), steps_for(ENGAGE_DWELL) - 1).engaged


def test_never_engages_when_turned_away(reader):
    assert not feed(reader, face(yaw=0.5), 100).engaged


def test_hysteresis_keeps_engagement_at_wider_yaw(reader):
    feed(reader, face(), steps_for(ENGAGE_DWELL))
    assert feed(reader, face(yaw=0.5), 100).engaged


def test_disengages_after_face_is_lost_for_dwell(reader):
    feed(reader, face(), steps_for(ENGAGE_DWELL))
    assert feed(reader, None, steps_for(DISENGAGE_DWELL) - 1).engaged
    assert not feed(reader, None, 1).engaged


@pytest.mark.parametrize(
    "result, mood",
    [
        (None, Mood.NEUTRAL),
        (face(), Mood.NEUTRAL),
        (face(mouthSmileLeft=0.9), Mood.HAPPY),
        (face(mouthSmileRight=0.9), Mood.HAPPY),
        (face(browInnerUp=0.9), Mood.SAD),
        (face(mouthSmileLeft=0.9, browInnerUp=0.9), Mood.HAPPY),
        (face(mouthSmileLeft=0.1, browInnerUp=0.1), Mood.NEUTRAL),
    ],
)
def test_mood_from_blendshapes(reader, result, mood):
    assert reader.read(result, DT).mood == mood
