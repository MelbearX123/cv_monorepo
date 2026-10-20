from body.poses import Mood
from perception.face_reader import FaceReading
from perception.vision_brain import Detection, Memory
from speech.state import SpeechState


class FakeFace:
    def __init__(self) -> None:
        self.engaged = False
        self.mood = Mood.NEUTRAL

    def read(self, frame, dt: float) -> FaceReading:
        return FaceReading(self.engaged, self.mood)


class FakeVision:
    def __init__(
        self,
        detections: list[Detection] | None = None,
        main_objects: list[Detection | None] | None = None,
        memory: Memory | None = None,
    ) -> None:
        self.detections = detections or []
        self._main_objects = iter(main_objects or [])
        self.memory = memory
        self.remembered: list[Detection] = []

    def detect(self, frame) -> list[Detection]:
        return list(self.detections)

    def main_object(self, frame) -> Detection | None:
        return next(self._main_objects, None)

    def remember(self, detection: Detection) -> None:
        self.remembered.append(detection)

    def seen(self, query: str) -> Memory | None:
        return self.memory


class FakeSpeech:
    def __init__(self) -> None:
        self.state = SpeechState.IDLE
        self.said: list[str] = []
        self.listen_count = 0
        self.silence_count = 0
        self.pending_transcript: str | None = None

    def say(self, text: str) -> None:
        if self.state == SpeechState.IDLE:
            self.said.append(text)
            self.state = SpeechState.SPEAKING

    def listen(self) -> None:
        if self.state == SpeechState.IDLE:
            self.listen_count += 1
            self.state = SpeechState.LISTENING

    def silence(self) -> None:
        self.silence_count += 1

    def step(self) -> None:
        pass

    def take_transcript(self) -> str | None:
        text, self.pending_transcript = self.pending_transcript, None
        return text

    def finish(self, transcript: str | None = None) -> None:
        self.state = SpeechState.IDLE
        self.pending_transcript = transcript


class FakeAudio:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.ducked = False
        self.sfx = False

    def power_on(self) -> None:
        self.events.append("power_on")

    def power_off(self) -> None:
        self.events.append("power_off")

    def play_music(self) -> None:
        self.events.append("play_music")

    def stop_music(self) -> None:
        self.events.append("stop_music")

    def duck(self, ducked: bool) -> None:
        self.ducked = ducked

    def sfx_playing(self) -> bool:
        return self.sfx
