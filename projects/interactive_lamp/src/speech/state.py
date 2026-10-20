from enum import StrEnum


class SpeechState(StrEnum):
    IDLE = "idle"
    SPEAKING = "speaking"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
