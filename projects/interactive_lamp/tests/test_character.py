from datetime import datetime

import pytest

from body.poses import Gesture, Mood, poses
from character import Character
from config import LOOK_CONFIRM_FRAMES, THINK_COLOR
from dialogue.phrases import MOOD_OPENINGS, RECALL_NO
from fakes import FakeAudio, FakeFace, FakeSpeech, FakeVision
from perception.vision_brain import Detection, Memory
from speech.state import SpeechState

DT = 0.03
CUP = Detection("cup", 0.9, (300, 220, 40, 40))


class Rig:
    def __init__(self, lamp, frame) -> None:
        self.face = FakeFace()
        self.vision = FakeVision(
            detections=[CUP], main_objects=[CUP] * LOOK_CONFIRM_FRAMES
        )
        self.speech = FakeSpeech()
        self.audio = FakeAudio()
        self.frame = frame
        self.character = Character(
            lamp, self.face, self.vision, self.speech, self.audio
        )

    def tick(self, times: int = 1) -> None:
        for _ in range(times):
            self.character.tick(DT, self.frame)

    def engage(self) -> None:
        self.face.engaged = True
        self.tick()
        self.speech.finish()

    def disengage(self) -> None:
        self.face.engaged = False
        self.tick()

    def hear(self, transcript: str) -> None:
        self.speech.finish(transcript)
        self.tick()


@pytest.fixture
def rig(lamp, frame) -> Rig:
    return Rig(lamp, frame)


def test_stays_quiet_until_engaged(rig):
    rig.tick(10)
    assert rig.speech.said == []
    assert rig.speech.listen_count == 0
    assert rig.audio.events == []


def test_engaging_powers_on_and_greets(rig):
    rig.engage()
    assert rig.audio.events == ["power_on"]
    assert rig.speech.said[0].startswith("Hello!")


def test_greets_only_once_per_day(rig):
    rig.engage()
    rig.disengage()
    rig.engage()
    greetings = [line for line in rig.speech.said if line.startswith("Hello!")]
    assert len(greetings) == 1


def test_listens_when_idle(rig):
    rig.engage()
    rig.tick()
    assert rig.speech.listen_count == 1


def test_does_not_listen_while_sound_effect_plays(rig):
    rig.engage()
    rig.audio.sfx = True
    rig.tick(5)
    assert rig.speech.listen_count == 0


def test_happy_mood_checks_in_instead_of_listening(rig):
    rig.face.mood = Mood.HAPPY
    rig.engage()
    rig.tick()
    assert rig.speech.said[-1] in MOOD_OPENINGS[Mood.HAPPY]
    assert rig.speech.listen_count == 0


@pytest.mark.parametrize(
    "transcript, event",
    [("play some music", "play_music"), ("stop", "stop_music")],
)
def test_music_commands_reach_audio(rig, transcript, event):
    rig.engage()
    rig.hear(transcript)
    assert rig.audio.events[-1] == event


def test_point_command_starts_pointing(rig):
    rig.engage()
    rig.hear("point at the cup")
    assert rig.character._vla.active


def test_look_command_votes_then_names_the_object(rig):
    rig.engage()
    rig.hear("look at this")

    rig.tick(LOOK_CONFIRM_FRAMES)

    assert "cup" in rig.speech.said[-1]
    assert rig.vision.remembered == [CUP]


def test_recall_points_at_remembered_object(rig):
    rig.vision.memory = Memory("cup", datetime.now(), CUP.box)
    rig.engage()
    rig.hear("have you seen my cup")
    assert rig.character._vla.active
    assert "cup" in rig.speech.said[-1]


def test_recall_of_unknown_object_does_not_move(rig):
    rig.engage()
    rig.hear("have you seen my keys")
    assert not rig.character._vla.active
    assert rig.speech.said[-1] in RECALL_NO


def test_unrecognised_speech_gets_a_reply(rig):
    rig.engage()
    rig.hear("the sky is blue")
    assert len(rig.speech.said) == 2


def test_disengage_cancels_everything(rig):
    rig.engage()
    rig.hear("point at the cup")
    rig.speech.finish()
    rig.hear("look at this")

    rig.disengage()

    assert not rig.character._vla.active
    assert not rig.character._look.active
    assert rig.speech.silence_count == 1
    assert rig.audio.events[-1] == "power_off"


def test_transcript_arriving_after_disengage_is_ignored(rig):
    rig.engage()
    rig.tick()
    rig.disengage()

    rig.hear("play some music")

    assert "play_music" not in rig.audio.events


def test_thinking_light_pulses_in_think_color(rig, lamp):
    rig.engage()
    rig.speech.state = SpeechState.TRANSCRIBING
    rig.tick()
    ratios = [shown / base for shown, base in zip(lamp.get_light(), THINK_COLOR)]
    assert ratios == pytest.approx([ratios[0]] * 3)


def test_speaking_light_is_engage_light(rig, lamp):
    rig.engage()
    rig.speech.state = SpeechState.SPEAKING
    rig.tick()
    assert lamp.get_light() == pytest.approx(poses[Gesture.ENGAGE]["light"])


def test_music_ducks_while_listening(rig):
    rig.engage()
    rig.speech.state = SpeechState.LISTENING
    rig.tick()
    assert rig.audio.ducked
