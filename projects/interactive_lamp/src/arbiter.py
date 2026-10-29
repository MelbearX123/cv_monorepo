from dataclasses import dataclass
from enum import StrEnum

from speech.state import SpeechState


class BodyOwner(StrEnum):
    GESTURE = "gesture"
    POINT = "point"


class LightCue(StrEnum):
    BODY = "body"
    THINKING = "thinking"
    SPEAKING = "speaking"


@dataclass(frozen=True)
class Snapshot:
    engaged: bool
    speech: SpeechState
    pointing: bool
    looking: bool
    sfx_playing: bool


@dataclass(frozen=True)
class Decision:
    body: BodyOwner
    light: LightCue
    duck_music: bool
    accept_commands: bool
    start_listening: bool


def arbitrate(snapshot: Snapshot) -> Decision:
    return Decision(
        body=BodyOwner.POINT if snapshot.pointing else BodyOwner.GESTURE,
        light=_light_cue(snapshot),
        duck_music=snapshot.speech in (SpeechState.LISTENING, SpeechState.TRANSCRIBING),
        accept_commands=snapshot.engaged,
        start_listening=snapshot.engaged and not _is_busy(snapshot),
    )


def _light_cue(snapshot: Snapshot) -> LightCue:
    if not snapshot.engaged:
        return LightCue.BODY
    if snapshot.speech == SpeechState.TRANSCRIBING:
        return LightCue.THINKING
    if snapshot.speech == SpeechState.SPEAKING:
        return LightCue.SPEAKING
    return LightCue.BODY


def _is_busy(snapshot: Snapshot) -> bool:
    return (
        snapshot.pointing
        or snapshot.looking
        or snapshot.sfx_playing
        or snapshot.speech != SpeechState.IDLE
    )
