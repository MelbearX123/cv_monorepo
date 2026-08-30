import random
import re

from poses import Mood
from datetime import date
from enum import StrEnum
import time

from config import MOOD_CHECK_COOLDOWN


class Category(StrEnum):
    GRATITUDE = "gratitude"
    AFFIRMATIVE = "affirmative"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class ResponseManager:
    def __init__(self) -> None:
        self._rng = random.Random()
        self._last_greeting_date = None
        self._last_mood_check = None

        self._keywords: dict[Category, set[str]] = {
            Category.GRATITUDE: {"thanks", "thank", "appreciate", "grateful"},
            Category.AFFIRMATIVE: {"yes", "yeah", "yep", "sure", "good", "great",
                                   "fine", "okay", "ok", "better", "happy"},
            Category.NEGATIVE: {"no", "nope", "not", "tired", "stressed", "rough",
                                "bad", "sad", "down", "awful", "terrible", "exhausted"},
        }
        self._replies: dict[Category, list[str]] = {
            Category.GRATITUDE: [
                "Anytime.",
                "Happy to help.",
                "Of course.",
            ],
            Category.AFFIRMATIVE: [
                "Glad to hear it.",
                "That's good to hear.",
                "Love that.",
            ],
            Category.NEGATIVE: [
                "I'm sorry to hear that.",
                "That sounds tough. I'm here.",
                "Rough one. Take it easy.",
            ],
            Category.UNKNOWN: [
                "I hear you.",
                "Got it.",
                "Thanks for telling me.",
            ],
        }
        self._mood_openings: dict[Mood, list[str]] = {
            Mood.HAPPY: [
                "You look cheerful today!",
                "Someone's in a good mood.",
            ],
            Mood.SAD: [
                "You seem a little down. Everything okay?",
                "Rough day? I'm here if you want to talk.",
            ],
        }

    def opening(self, mood: Mood, engaged_seconds: float) -> str | None:
        today = date.today()
        now = time.monotonic()
        greeting = None
        if self._last_greeting_date != today:
            greeting = f"Hello! Today is {today:%A, %B} {today.day}."
            self._last_greeting_date = today
        elif mood != Mood.NEUTRAL and (
            self._last_mood_check is None
            or now - self._last_mood_check > MOOD_CHECK_COOLDOWN
        ):
            greeting = self._pick(self._mood_openings.get(mood, []))
            self._last_mood_check = now
        return greeting

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
        options = self._replies.get(category) or self._replies[Category.UNKNOWN]
        return self._pick(options)

    def _classify(self, text: str) -> Category:
        words = set(re.findall(r"[a-z']+", text.lower()))
        for category, keywords in self._keywords.items():
            if words & keywords:
                return category
        return Category.UNKNOWN

    def _pick(self, options: list[str]) -> str:
        return self._rng.choice(options) if options else ""
