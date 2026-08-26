from dataclasses import dataclass
from poses import Mood
from enum import StrEnum

class Activity(StrEnum):
    IDLE = "idle"
    SPEAKING = "speaking"
    GESTURING = "gesturing"

@dataclass
class CharacterState:
    user_emotion: str | None = None
    mood: Mood = Mood.NEUTRAL
    memory: list = []
    is_engaged: bool = False
    activity: Activity = Activity.IDLE
