from datetime import datetime

import pytest

import dialogue.response as response_module
from action.vla import PointResult
from body.poses import Mood
from config import MOOD_CHECK_COOLDOWN
from dialogue.phrases import (
    LOOK_EMPTY,
    LOOK_REPLIES,
    MOOD_OPENINGS,
    POINT_REPLIES,
    RECALL_NO,
    RECALL_YES,
    REPLIES,
    Category,
    PointOutcome,
)
from dialogue.response import ResponseManager
from perception.vision_brain import Memory


@pytest.fixture
def response() -> ResponseManager:
    return ResponseManager()


@pytest.fixture
def clock(monkeypatch) -> list[float]:
    now = [1000.0]
    monkeypatch.setattr(response_module.time, "monotonic", lambda: now[0])
    return now


def filled(templates: list[str], **values) -> set[str]:
    return {template.format(**values) for template in templates}


def test_opening_greets_once_per_day(response):
    assert response.opening(Mood.NEUTRAL).startswith("Hello!")
    assert response.opening(Mood.NEUTRAL) is None


def test_no_mood_checkin_when_neutral(response, clock):
    assert response.mood_checkin(Mood.NEUTRAL) is None


def test_mood_checkin_respects_cooldown(response, clock):
    assert response.mood_checkin(Mood.SAD) in MOOD_OPENINGS[Mood.SAD]
    clock[0] += MOOD_CHECK_COOLDOWN
    assert response.mood_checkin(Mood.SAD) is None
    clock[0] += 1
    assert response.mood_checkin(Mood.HAPPY) in MOOD_OPENINGS[Mood.HAPPY]


def test_look_with_nothing_seen(response):
    assert response.look(None) in LOOK_EMPTY


def test_look_speaks_label_with_article(response):
    assert response.look("cell_phone") in filled(
        LOOK_REPLIES, article="a", label="cell phone"
    )
    assert response.look("apple") in filled(LOOK_REPLIES, article="an", label="apple")


@pytest.mark.parametrize(
    "result, outcome",
    [
        (PointResult("cup", found=True, centered=True), PointOutcome.FOUND),
        (PointResult("cup", found=True, centered=False), PointOutcome.OFF),
        (PointResult("cup", found=False, centered=False), PointOutcome.MISSING),
    ],
)
def test_point_report_matches_outcome(response, result, outcome):
    expected = filled(POINT_REPLIES[outcome], article="a", label="cup")
    assert response.point_report(result) in expected


def test_recall_without_memory(response):
    assert response.recall(None) in RECALL_NO


def test_recall_says_what_and_when(response):
    memory = Memory(label="cup", at=datetime(2026, 1, 1, 15, 5), box=(0, 0, 1, 1))
    expected = filled(RECALL_YES, article="a", label="cup", when="3:05 pm")
    assert response.recall(memory) in expected


def test_respond_answers_the_date(response):
    assert response.respond("what day is it", Mood.NEUTRAL).startswith("Today is")


def test_respond_admits_unknown_questions(response):
    reply = response.respond("how tall are you?", Mood.NEUTRAL)
    assert reply == "I'm not sure how to answer that."


@pytest.mark.parametrize(
    "text, category",
    [
        ("thanks a lot", Category.GRATITUDE),
        ("pretty good", Category.AFFIRMATIVE),
        ("I'm exhausted", Category.NEGATIVE),
        ("the sky is blue", Category.UNKNOWN),
    ],
)
def test_respond_picks_reply_by_keywords(response, text, category):
    assert response.respond(text, Mood.NEUTRAL) in REPLIES[category]


@pytest.mark.parametrize(
    "mood, category",
    [(Mood.SAD, Category.NEGATIVE), (Mood.HAPPY, Category.AFFIRMATIVE)],
)
def test_respond_falls_back_to_mood(response, mood, category):
    assert response.respond("the sky is blue", mood) in REPLIES[category]
