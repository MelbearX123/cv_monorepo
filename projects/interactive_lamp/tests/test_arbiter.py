import pytest

from arbiter import BodyOwner, LightCue, Snapshot, arbitrate
from speech.state import SpeechState


def snapshot(**overrides) -> Snapshot:
    defaults = dict(
        engaged=True,
        speech=SpeechState.IDLE,
        pointing=False,
        looking=False,
        sfx_playing=False,
    )
    return Snapshot(**(defaults | overrides))


def test_idle_and_engaged_starts_listening():
    decision = arbitrate(snapshot())
    assert decision.start_listening
    assert decision.accept_commands
    assert decision.body == BodyOwner.GESTURE
    assert decision.light == LightCue.BODY


def test_disengaged_ignores_commands_and_never_listens():
    decision = arbitrate(snapshot(engaged=False))
    assert not decision.accept_commands
    assert not decision.start_listening


def test_pointing_takes_the_body():
    assert arbitrate(snapshot(pointing=True)).body == BodyOwner.POINT


def test_pointing_keeps_the_body_after_disengage():
    assert arbitrate(snapshot(engaged=False, pointing=True)).body == BodyOwner.POINT


@pytest.mark.parametrize(
    "busy",
    [
        {"pointing": True},
        {"looking": True},
        {"sfx_playing": True},
        {"speech": SpeechState.SPEAKING},
        {"speech": SpeechState.LISTENING},
        {"speech": SpeechState.TRANSCRIBING},
    ],
)
def test_busy_lamp_does_not_start_listening(busy):
    assert not arbitrate(snapshot(**busy)).start_listening


def test_sound_effect_blocks_listening_so_lamp_does_not_hear_itself():
    assert not arbitrate(snapshot(sfx_playing=True)).start_listening


@pytest.mark.parametrize(
    "speech, cue",
    [
        (SpeechState.IDLE, LightCue.BODY),
        (SpeechState.LISTENING, LightCue.BODY),
        (SpeechState.TRANSCRIBING, LightCue.THINKING),
        (SpeechState.SPEAKING, LightCue.SPEAKING),
    ],
)
def test_light_follows_speech_while_engaged(speech, cue):
    assert arbitrate(snapshot(speech=speech)).light == cue


def test_disengaged_light_stays_with_the_body():
    decision = arbitrate(snapshot(engaged=False, speech=SpeechState.SPEAKING))
    assert decision.light == LightCue.BODY


@pytest.mark.parametrize(
    "speech, ducked",
    [
        (SpeechState.IDLE, False),
        (SpeechState.SPEAKING, False),
        (SpeechState.LISTENING, True),
        (SpeechState.TRANSCRIBING, True),
    ],
)
def test_music_ducks_only_while_the_user_is_heard(speech, ducked):
    assert arbitrate(snapshot(speech=speech)).duck_music == ducked
