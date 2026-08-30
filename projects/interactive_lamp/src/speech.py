from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from enum import StrEnum

import numpy as np
import sounddevice as sd

from silero_vad import load_silero_vad
from silero_vad import VADIterator
from faster_whisper import WhisperModel
from kokoro import KPipeline

from config import (
    STT_SAMPLE_RATE,
    TTS_SAMPLE_RATE,
    AUDIO_CHANNELS,
    VAD_FRAME,
    WHISPER_MODEL,
    WHISPER_COMPUTE,
    WHISPER_DEVICE,
    KOKORO_LANG,
    KOKORO_VOICE,
    KOKORO_SPEED,
    VAD_THRESHOLD,
    VAD_MIN_SILENCE,
    VAD_SPEECH_PAD,
    MAX_RECORD_SECONDS,
)


class SpeechState(StrEnum):
    IDLE = "idle"
    SPEAKING = "speaking"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"


class SpeechManager:
    """Owns the mic, speaker, and the three speech models. Non-blocking to the main loop."""

    def __init__(self) -> None:
        self._vad = load_silero_vad(onnx=True)
        self._whisper = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
        )
        self._tts = KPipeline(lang_code=KOKORO_LANG)

        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._state = SpeechState.IDLE
        self._transcript: str | None = None

    @property
    def state(self) -> SpeechState:
        return self._state

    @property
    def active(self) -> bool:
        return self._future is not None and not self._future.done()

    def say(self, text: str) -> None:
        if self.active:
            return
        self._state = SpeechState.SPEAKING
        self._future = self._pool.submit(self._speak, text)

    def listen(self) -> None:
        if self.active:
            return
        self._state = SpeechState.LISTENING
        self._future = self._pool.submit(self._listen)

    def step(self) -> None:
        if self._future is None or not self._future.done():
            return
        result = self._future.result()
        if isinstance(result, str):
            self._transcript = result
        self._future = None
        self._state = SpeechState.IDLE

    def take_transcript(self) -> str | None:
        text, self._transcript = self._transcript, None
        return text

    def _speak(self, text: str) -> None:
        chunks = [
            chunk
            for _, _, chunk in self._tts(text, voice=KOKORO_VOICE, speed=KOKORO_SPEED)
        ]
        if not chunks:
            return
        audio = np.concatenate(chunks).astype(np.float32)
        sd.play(audio, samplerate=TTS_SAMPLE_RATE)
        sd.wait()

    def _listen(self) -> str:
        audio = self._record_until_silence()
        self._state = SpeechState.TRANSCRIBING
        return self._transcribe(audio)

    def _record_until_silence(self) -> np.ndarray:
        vad = VADIterator(
            self._vad,
            threshold=VAD_THRESHOLD,
            sampling_rate=STT_SAMPLE_RATE,
            min_silence_duration_ms=int(VAD_MIN_SILENCE * 1000),
            speech_pad_ms=int(VAD_SPEECH_PAD * 1000),
        )
        frames: list[np.ndarray] = []
        speech_started = False
        max_frames = int(MAX_RECORD_SECONDS * STT_SAMPLE_RATE / VAD_FRAME)

        with sd.InputStream(
            samplerate=STT_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="float32",
            blocksize=VAD_FRAME,
        ) as stream:
            for _ in range(max_frames):
                block, _ = stream.read(VAD_FRAME)
                chunk = block[:, 0].copy()
                frames.append(chunk)
                event = vad(chunk)
                if event is None:
                    continue
                if "start" in event:
                    speech_started = True
                elif "end" in event:
                    break

        vad.reset_states()
        if not speech_started:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(frames).astype(np.float32)

    def _transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._whisper.transcribe(audio, language="en", beam_size=1)
        return " ".join(seg.text.strip() for seg in segments).strip()

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def __enter__(self) -> "SpeechManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
