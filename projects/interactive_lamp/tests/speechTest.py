"""Manual check for SpeechManager._record_until_silence().

Run from the src/ folder:   python speechTest.py
Speak after the prompt; it stops on its own when you go quiet.
Each take is saved to a .wav so you can play it back and confirm the audio is real.
"""

import sys
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import STT_SAMPLE_RATE
from speech.speech import SpeechManager


def describe(audio: np.ndarray) -> None:
    if audio.size == 0:
        print(
            "  -> NO SPEECH DETECTED (VAD never fired 'start'). Empty array returned."
        )
        return
    duration = audio.size / STT_SAMPLE_RATE
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    print(f"  samples     : {audio.size}")
    print(f"  duration    : {duration:.2f} s")
    print(f"  peak amp    : {peak:.3f}   (near 0 = silence, near 1 = loud/clipping)")
    print(f"  rms level   : {rms:.4f}")
    if peak < 0.01:
        print("  !! very quiet -- is the right mic selected / unmuted?")


def main() -> None:
    print("=== audio devices (input-capable have input_channels > 0) ===")
    print(sd.query_devices())
    print(f"\ndefault input device: {sd.default.device}")
    print(f"sample rate        : {STT_SAMPLE_RATE} Hz\n")

    print("Loading speech models (first run downloads weights)...")
    sm = SpeechManager()
    print("Models loaded.\n")

    take = 0
    try:
        while True:
            cmd = input("Press ENTER to record (or 'q' then ENTER to quit): ").strip()
            if cmd.lower() == "q":
                break

            print("Recording... speak now, then pause.")
            audio = sm._record_until_silence()
            print("Stopped.")
            describe(audio)

            if audio.size > 0:
                take += 1
                path = f"take_{take}.wav"
                sf.write(path, audio, STT_SAMPLE_RATE)
                print(f"  saved -> {path}  (play it to confirm it recorded you)")

                print("  transcribing...")
                text = sm._transcribe(audio)
                print(f'  heard: "{text}"')

                if text:
                    print("  reading it back...")
                    sm._speak(text)
                print()
            else:
                print()
    finally:
        sm.close()
        print("Closed. Bye.")


if __name__ == "__main__":
    main()
