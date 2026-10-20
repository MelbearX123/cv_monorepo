import re
from enum import StrEnum

from dialogue.phrases import (
    LOOK_KEYWORDS,
    LOOK_PHRASES,
    MUSIC_PHRASES,
    POINT_PHRASES,
    RECALL_PHRASES,
    STOP_MUSIC_PHRASES,
)


class Intent(StrEnum):
    STOP_MUSIC = "stop_music"
    PLAY_MUSIC = "play_music"
    POINT = "point"
    RECALL = "recall"
    LOOK = "look"
    CONVERSE = "converse"


def classify(text: str) -> Intent:
    lowered = text.lower()
    if _contains_any(lowered, STOP_MUSIC_PHRASES):
        return Intent.STOP_MUSIC
    if _contains_any(lowered, MUSIC_PHRASES):
        return Intent.PLAY_MUSIC
    if _contains_any(lowered, POINT_PHRASES):
        return Intent.POINT
    if _contains_any(lowered, RECALL_PHRASES):
        return Intent.RECALL
    if _contains_any(lowered, LOOK_PHRASES) or _words(lowered) & LOOK_KEYWORDS:
        return Intent.LOOK
    return Intent.CONVERSE


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text))
