from __future__ import annotations

from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

from config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_POWER_ON,
    AUDIO_POWER_OFF,
    AUDIO_MUSIC,
    AUDIO_MUSIC_DUCK,
    AUDIO_DUCK_STEP,
    AUDIO_POWER_OFF_GAIN,
)


def _load(path: Path, sample_rate: int) -> np.ndarray:
    if not path.exists():
        return np.zeros(0, dtype=np.float32)
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if sr != sample_rate:
        n = int(round(len(mono) * sample_rate / sr))
        src = np.arange(len(mono))
        dst = np.linspace(0, len(mono), n, endpoint=False)
        mono = np.interp(dst, src, mono).astype(np.float32)
    return mono


class AudioManager:
    def __init__(self) -> None:
        sr = AUDIO_SAMPLE_RATE
        self._power_on = _load(AUDIO_POWER_ON, sr)
        self._power_off = _load(AUDIO_POWER_OFF, sr) * AUDIO_POWER_OFF_GAIN
        self._music = _load(AUDIO_MUSIC, sr)

        self._sfx: np.ndarray | None = None
        self._sfx_pos = 0
        self._music_on = False
        self._music_pos = 0
        self._gain = 1.0
        self._target_gain = 1.0

        self._stream = sd.OutputStream(
            samplerate=sr, channels=1, dtype="float32",
            blocksize=1024, callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, time_info, status) -> None:
        out = np.zeros(frames, dtype=np.float32)

        sfx = self._sfx
        if sfx is not None:
            clip = sfx[self._sfx_pos : self._sfx_pos + frames]
            out[: len(clip)] += clip
            self._sfx_pos += frames
            if self._sfx_pos >= len(sfx):
                self._sfx = None

        if self._music_on and len(self._music) > 0:
            buf = np.empty(frames, dtype=np.float32)
            pos, i, need = self._music_pos, 0, frames
            while need > 0:
                take = min(need, len(self._music) - pos)
                buf[i : i + take] = self._music[pos : pos + take]
                pos = (pos + take) % len(self._music)
                i += take
                need -= take
            self._music_pos = pos
            start = self._gain
            self._gain += float(
                np.clip(self._target_gain - self._gain, -AUDIO_DUCK_STEP, AUDIO_DUCK_STEP)
            )
            out += buf * np.linspace(start, self._gain, frames, dtype=np.float32)

        outdata[:, 0] = np.clip(out, -1.0, 1.0)

    def sfx_playing(self) -> bool:
        return self._sfx is not None

    def _play_sfx(self, clip: np.ndarray) -> None:
        if len(clip) == 0:
            return
        self._sfx_pos = 0
        self._sfx = clip

    def power_on(self) -> None:
        self._play_sfx(self._power_on)

    def power_off(self) -> None:
        self.stop_music()
        self._play_sfx(self._power_off)

    def play_music(self) -> None:
        if len(self._music) == 0:
            return
        self._music_pos = 0
        self._music_on = True

    def stop_music(self) -> None:
        self._music_on = False

    def duck(self, speaking: bool) -> None:
        self._target_gain = AUDIO_MUSIC_DUCK if speaking else 1.0

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    def __enter__(self) -> "AudioManager":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
