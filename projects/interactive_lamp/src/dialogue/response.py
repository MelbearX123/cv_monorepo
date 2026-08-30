import random
import re
import time
from datetime import date
from typing import TYPE_CHECKING

from body.poses import Mood
from dialogue.phrases import (
    Category,
    PointOutcome,
    KEYWORDS,
    REPLIES,
    LOOK_KEYWORDS,
    LOOK_PHRASES,
    LOOK_REPLIES,
    LOOK_EMPTY,
    RECALL_PHRASES,
    RECALL_YES,
    RECALL_NO,
    POINT_PHRASES,
    POINT_REPLIES,
    MOOD_OPENINGS,
)
from config import MOOD_CHECK_COOLDOWN

if TYPE_CHECKING:
    from perception.vision_brain import Memory
    from action.vla import PointResult


class ResponseManager:
    def __init__(self) -> None:
        self._rng = random.Random()
        self._last_greeting_date = None
        self._last_mood_check = None

    def opening(self, mood: Mood) -> str | None:
        today = date.today()
        if self._last_greeting_date == today:
            return None
        self._last_greeting_date = today
        return f"Hello! Today is {today:%A, %B} {today.day}."

    def mood_checkin(self, mood: Mood) -> str | None:
        if mood == Mood.NEUTRAL:
            return None
        now = time.monotonic()
        if (
            self._last_mood_check is not None
            and now - self._last_mood_check <= MOOD_CHECK_COOLDOWN
        ):
            return None
        self._last_mood_check = now
        return self._pick(MOOD_OPENINGS.get(mood, []))

    def wants_look(self, text: str) -> bool:
        lowered = text.lower()
        if any(phrase in lowered for phrase in LOOK_PHRASES):
            return True
        words = set(re.findall(r"[a-z']+", lowered))
        return bool(words & LOOK_KEYWORDS)

    def look(self, label: str | None) -> str:
        if label is None:
            return self._pick(LOOK_EMPTY)
        spoken = label.replace("_", " ")
        return self._pick(LOOK_REPLIES).format(article=_article(spoken), label=spoken)

    def wants_recall(self, text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in RECALL_PHRASES)

    def recall(self, memory: "Memory | None") -> str:
        if memory is None:
            return self._pick(RECALL_NO)
        spoken = memory.label.replace("_", " ")
        when = memory.at.strftime("%I:%M %p").lstrip("0").lower()
        return self._pick(RECALL_YES).format(
            article=_article(spoken), label=spoken, when=when
        )

    def wants_point(self, text: str) -> bool:
        lowered = text.lower()
        return any(phrase in lowered for phrase in POINT_PHRASES)

    def point_report(self, result: "PointResult") -> str:
        if not result.found:
            outcome = PointOutcome.MISSING
        elif result.centered:
            outcome = PointOutcome.FOUND
        else:
            outcome = PointOutcome.OFF
        spoken = result.label.replace("_", " ")
        return self._pick(POINT_REPLIES[outcome]).format(
            article=_article(spoken), label=spoken
        )

    def respond(self, text: str, mood: Mood) -> str:
        lowered = text.lower()
        if "date" in lowered or "what day" in lowered:
            today = date.today()
            return f"Today is {today:%A, %B} {today.day}."
        if "?" in text:
            return "I'm not sure how to answer that."
        category = self._classify(text)
        if category == Category.UNKNOWN and mood != Mood.NEUTRAL:
            category = Category.NEGATIVE if mood == Mood.SAD else Category.AFFIRMATIVE
        return self._pick(REPLIES.get(category) or REPLIES[Category.UNKNOWN])

    def _classify(self, text: str) -> Category:
        words = set(re.findall(r"[a-z']+", text.lower()))
        for category, keywords in KEYWORDS.items():
            if words & keywords:
                return category
        return Category.UNKNOWN

    def _pick(self, options: list[str]) -> str:
        return self._rng.choice(options) if options else ""


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"
