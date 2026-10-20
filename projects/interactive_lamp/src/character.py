from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np

from action.vla import VLAManager
from arbiter import BodyOwner, Decision, LightCue, Snapshot, arbitrate
from body.lamp_body import LampBody
from body.lamp_gesture import LampGesture
from body.poses import Gesture, poses
from config import THINK_COLOR, THINK_PULSE_HZ
from dialogue.character_state import CharacterState
from dialogue.intent import Intent, classify
from dialogue.response import ResponseManager
from perception.look import LookBehavior

if TYPE_CHECKING:
    from audio.audio import AudioManager
    from perception.perception import FacePerception
    from perception.vision_brain import VisionBrain
    from speech.speech import SpeechManager

log = logging.getLogger(__name__)


class Character:
    def __init__(
        self,
        lamp: LampBody,
        face: FacePerception,
        vision: VisionBrain,
        speech: SpeechManager,
        audio: AudioManager,
    ) -> None:
        self._lamp = lamp
        self._face = face
        self._vision = vision
        self._speech = speech
        self._audio = audio

        self._state = CharacterState()
        self._response = ResponseManager()
        self._gesture = LampGesture(lamp, mood=self._state.mood)
        self._vla = VLAManager(lamp, vision, self._state.mood)
        self._look = LookBehavior(vision)
        self._clock = 0.0

        rest = poses[Gesture.DISENGAGE]
        lamp.set_joints(rest["joints"])
        lamp.set_light(rest["light"])

    def tick(self, dt: float, frame: np.ndarray | None) -> None:
        self._clock += dt
        self._sense(frame, dt)
        self._speech.step()
        self._look.step(frame)
        self._report_results()

        decision = arbitrate(self._snapshot())
        self._move(decision.body, dt, frame)
        self._show_light(decision.light)
        self._audio.duck(decision.duck_music)
        self._respond(decision, frame)

    def _sense(self, frame: np.ndarray | None, dt: float) -> None:
        reading = self._face.read(frame, dt)
        if reading is None:
            return
        was_engaged = self._state.is_engaged
        self._state.is_engaged = reading.engaged
        self._state.mood = reading.mood
        if reading.engaged and not was_engaged:
            self._on_engage()
        elif was_engaged and not reading.engaged:
            self._on_disengage()

    def _on_engage(self) -> None:
        self._play(Gesture.ENGAGE)
        self._audio.power_on()
        greeting = self._response.opening(self._state.mood)
        if greeting:
            self._speech.say(greeting)

    def _on_disengage(self) -> None:
        self._vla.cancel()
        self._look.cancel()
        self._speech.silence()
        self._speech.take_transcript()
        self._audio.power_off()
        self._play(Gesture.DISENGAGE)

    def _play(self, gesture: Gesture) -> None:
        self._gesture = LampGesture(self._lamp, mood=self._state.mood)
        self._gesture.play(gesture)

    def _report_results(self) -> None:
        look = self._look.take_result()
        if look is not None:
            self._speech.say(self._response.look(look.label))

        point = self._vla.take_result()
        if point is not None:
            self._speech.say(self._response.point_report(point))

    def _snapshot(self) -> Snapshot:
        return Snapshot(
            engaged=self._state.is_engaged,
            speech=self._speech.state,
            pointing=self._vla.active,
            looking=self._look.active,
            sfx_playing=self._audio.sfx_playing(),
        )

    def _move(self, owner: BodyOwner, dt: float, frame: np.ndarray | None) -> None:
        if owner == BodyOwner.POINT:
            self._vla.step(dt, frame)
        else:
            self._gesture.step(dt)

    def _show_light(self, cue: LightCue) -> None:
        if cue == LightCue.THINKING:
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * THINK_PULSE_HZ * self._clock)
            self._lamp.set_light([c * (0.35 + 0.65 * pulse) for c in THINK_COLOR])
        elif cue == LightCue.SPEAKING:
            self._lamp.set_light(poses[Gesture.ENGAGE]["light"])

    def _respond(self, decision: Decision, frame: np.ndarray | None) -> None:
        heard = self._speech.take_transcript()
        if not decision.accept_commands:
            return
        if heard:
            log.info("heard %r", heard)
            self._handle(classify(heard), heard, frame)
        elif decision.start_listening:
            self._prompt()

    def _handle(self, intent: Intent, heard: str, frame: np.ndarray | None) -> None:
        match intent:
            case Intent.STOP_MUSIC:
                self._audio.stop_music()
            case Intent.PLAY_MUSIC:
                self._audio.play_music()
            case Intent.POINT:
                if frame is not None:
                    self._vla.point_at_label(heard, frame)
            case Intent.RECALL:
                self._recall(heard)
            case Intent.LOOK:
                self._look.start()
            case Intent.CONVERSE:
                self._speech.say(self._response.respond(heard, self._state.mood))

    def _recall(self, heard: str) -> None:
        memory = self._vision.seen(heard)
        if memory is not None:
            self._vla.point_at_box(memory.box)
        self._speech.say(self._response.recall(memory))

    def _prompt(self) -> None:
        checkin = self._response.mood_checkin(self._state.mood)
        if checkin:
            self._speech.say(checkin)
        else:
            self._speech.listen()
            log.info("listening")
