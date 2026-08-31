# Setup & Run — Ubuntu 24.04 Target

Tested against the target described in [`starter_pack/CHALLENGE.md`](starter_pack/CHALLENGE.md):
Ubuntu 24.04, 4-core CPU, 8 GB RAM, **no GPU**, an integrated/USB camera, and a
microphone + speaker exposed through the standard Linux audio stack
(ALSA → PipeWire/PulseAudio, which is the desktop default on 24.04).

Everything below runs as a normal user in a graphical desktop session. Audio and
camera need that session — a headless/SSH/WSL2 shell has no sound server or
camera device, so the app will start but mic/speaker/webcam features stay inert
there (see [Troubleshooting](#troubleshooting)).

---

## 1. System packages (apt)

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.12 python3.12-venv python3-pip \
  libportaudio2 libsndfile1 \
  libgl1 libglib2.0-0 libegl1 libgles2
```

What these cover:

| Package | Needed by |
|---|---|
| `python3.12-venv` | creating the virtual environment |
| `libportaudio2` | `sounddevice` — audio playback/mixing + mic capture |
| `libsndfile1` (≥ 1.1) | `soundfile` — decoding the `.mp3` assets |
| `libgl1`, `libegl1`, `libgles2`, `libglib2.0-0` | MuJoCo rendering + OpenCV |

A working desktop session already runs PipeWire, so no audio server needs to be
installed or started.

---

## 2. Python environment

From the project directory (`interactive_lamp/`):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the **CPU** build of PyTorch via
`--extra-index-url https://download.pytorch.org/whl/cpu`, so no CUDA/GPU is pulled in.

---

## 3. Models

Two groups of model files are used.

**a) Ship with the repo** — already present in [`models/`](models/):

- `face_landmarker.task` — MediaPipe face landmarker (engagement)
- `efficientdet_lite2.tflite` — object detection (the "point at the X" moment)

If `models/` is empty (e.g. it was git-ignored on your copy), download them:

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
curl -L -o models/efficientdet_lite2.tflite \
  https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/int8/1/efficientdet_lite2.tflite
```

**b) Auto-download on first run** — the speech stack (`faster-whisper` base int8,
`kokoro` TTS, `silero-vad`) fetches weights from Hugging Face the **first time**
the app runs, then caches them under `~/.cache/`. This needs network access on
the first launch only. To pre-warm the cache before a demo, just run the app once
with network available.

---

## 4. Run

The app uses absolute package imports rooted at `src/`, so it **must be launched
from `src/`**:

```bash
cd src
../.venv/bin/python main.py
```

(Running it from anywhere else fails with `No module named 'body'` and similar.)

A MuJoCo window (960×720) opens showing the lamp. Sit in front of the camera to
engage it; speak to trigger the speech/response moments; try "play music",
"stop", and "point at the ⟨object⟩".

---

## Troubleshooting

**No audio / `PortAudioError: Device unavailable` / no default output device**
There is no running sound server or no audio device. Confirm you're in a desktop
session (`pactl info` should report a PipeWire/PulseAudio server). This is the
expected state under SSH-only, containers, and WSL2.

**No camera / opens but stays black**
Confirm a device exists (`ls /dev/video*`) and no other app holds it. `CAMERA_INDEX`
in [`src/config.py`](src/config.py) defaults to `0`; change it if the camera is
enumerated differently.

**First run is slow / stalls at startup**
That's the one-time Hugging Face download of the speech models (section 3b). Let
it finish; subsequent launches read from `~/.cache/`.

**MuJoCo window fails to create / GL errors**
Ensure the GL packages from section 1 are installed and you're in a session with
a display. For a purely headless render you'd need EGL offscreen mode — not
required for the intended desktop demo.
