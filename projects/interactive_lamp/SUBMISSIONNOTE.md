# Interactive Lamp — Technical Note

A live expressive character on a simulated 5-DOF MuJoCo lamp. The lamp perceives a
person through a webcam and microphone and responds with coordinated **motion,
light, voice, sound effects, and music**. Everything runs on-device, CPU-only, with
no LLM and no cloud calls.

## Architecture

```mermaid
flowchart LR
    subgraph IN["Sensors"]
        CAM["Webcam"]
        MIC["Mic"]
    end
    subgraph PERC["Perception"]
        FACE["FaceTracker/Reader<br/>MediaPipe → (engaged, mood)"]
        VB["VisionBrain<br/>EfficientDet-Lite2 + object memory"]
    end
    subgraph COG["Dialogue (no LLM)"]
        CS["CharacterState<br/>engaged, mood"]
        RESP["ResponseManager<br/>intent match + templated phrases"]
    end
    SM["SpeechManager<br/>silero-VAD → faster-whisper → kokoro TTS"]
    subgraph ACT["Action / Body"]
        VLA["VLAManager<br/>point state machine"]
        GES["LampGesture (easing)"]
        BODY["LampBody<br/>5-DOF joints + light"]
    end
    AUD["AudioManager<br/>mix: SFX + music + duck"]
    REN["MuJoCo Renderer"]
    MAIN(["main.py — per-frame loop, owns the clock"])

    CAM --> FACE --> CS --> RESP
    CAM --> VB -- box/memory --> VLA
    MIC --> SM -- transcript --> RESP
    RESP -- intent --> MAIN
    MAIN --> VLA --> BODY --> REN
    MAIN --> GES --> BODY
    RESP -- phrase --> SM --> AUD
    MAIN --> AUD
    CS -. mood .-> GES & VLA & RESP
```

Solid arrows are data (frames, transcript, detections, joint/light targets, audio).
Dotted `mood` edges are cross-cutting state: mood shapes **delivery** (motion timing,
phrasing, gesture energy), never spoken content.

## Data flow

One pass of the per-frame loop, following a spoken command from sensor to actuator:

```mermaid
flowchart LR
    F["Webcam frame<br/>+ mic audio"] --> P["read_face →<br/>(engaged, mood)"]
    P --> E{"engage /<br/>disengage<br/>edge?"}
    E -->|yes| T["fire pose + SFX<br/>+ greeting / cutoff"]
    P --> S["VAD + Whisper<br/>→ transcript"]
    S --> I{"match intent"}
    I -->|music / stop| AUD["AudioManager"]
    I -->|point / recall| VLA["VLAManager<br/>→ box → joint targets"]
    I -->|look| VB["VisionBrain<br/>→ detect + remember"]
    I -->|converse| R["ResponseManager<br/>→ templated phrase"]
    R --> TTS["kokoro TTS"]
    VLA --> BODY["LampBody<br/>joints + light"]
    E --> BODY
    BODY --> REN["MuJoCo render"]
    TTS --> SPK["Speaker"]
    AUD --> SPK
```

Every stage is non-blocking: perception, transcription, and audio run on their own
threads, and the loop advances each subsystem once per frame regardless of whether a
command is in flight. Interaction stages run only while engaged; a disengage edge
cuts speech, drops any pending transcript, and clears in-progress look/point state.

## Protocol

The whole system is one process with a single **orchestration loop** in `main.py`
that owns the clock. Every subsystem exposes a `step(dt)` method, and the loop calls
each one once per frame. Subsystems only hand results back up to the loop and never call each other.

Those results are plain typed values:

- perception → `(is_engaged, mood)`
- speech → a transcript `str`
- action → a `PointResult`
- vision → a list of `Detection`

Slow work is kept off the loop. Mic capture, Whisper transcription, and the audio
output stream all run on their own threads and streams; the loop simply **polls**
them each frame (`take_transcript()`, `take_result()`) and moves on. So a slow
transcription never stalls rendering or motion. The payoff is that ownership is
explicit and the loop stays deterministic and easy to reason about.

## Model-to-action

Model outputs are grounded into actuator space, never just printed:

- **Engagement:** MediaPipe face landmarks → head-yaw / gaze thresholds →
  `is_engaged`. A rising edge fires the ENGAGE pose, power-on SFX, and a greeting; a
  falling edge fires DISENGAGE, power-off SFX, and stops music. Mood (from facial
  expression) selects the delivery variant.
- **Point-at-object (VLA):** EfficientDet-Lite2 returns a box → center normalized to
  `[-1, 1]` → mapped to `NECK_YAW` (mirrored pan, anchored level) and `HEAD_PITCH`
  (anchored to the engage pose) → eased over time by `LampGesture` into MuJoCo joint
  targets. On arrival the lamp re-detects and corrects once if it has drifted, reports
  verbally, holds, then self-returns to the engage pose. A state machine
  (`IDLE→MOVING→HOLDING→RETURNING→IDLE`) guards concurrency.
- **Speech intents:** the transcript is matched against templated intents (music /
  stop / point / recall / look / converse), which is deterministic string/keyword matching.
  No generative model, so behavior is predictable and cheap.

## Simulation

The lamp is a 5-DOF MuJoCo model (`lamp.xml`) with an actuated light element. Motion
is produced by writing **joint targets** and easing toward them over a duration, not
by teleporting poses, so movement has believable acceleration and settle. The render
uses an orbit camera framing the lamp at 960×720. The webcam is treated as a
world-fixed sensor, not mounted on the lamp head. So "point at X" is a **snapshot**
aim from a single observation, not closed-loop visual tracking.

## Deployment

Target: Ubuntu 24.04, 4-core, 8 GB, **no GPU**. CPU-only throughout — `requirements.txt`
pins the CPU PyTorch wheel. System deps: `libportaudio2`, `libsndfile1` (audio),
`libgl1 libegl1 libgles2 libglib2.0-0` (MuJoCo/OpenCV). Two vision models ship in
`models/`; the speech models (Whisper base int8, kokoro, silero-VAD) auto-download
from Hugging Face on first run and cache under `~/.cache/`. Audio and camera use the
standard Linux stack (ALSA → PipeWire); on a headless/SSH session with no sound
server or camera the app still starts and degrades gracefully. Launch from `src/`
(`python main.py`); full steps in `SETUP.md`.

## Key design choices

- **One manager per subsystem.** The system is split into focused managers —
  `VisionBrain`, `SpeechManager`, `ResponseManager`, `VLAManager`, `AudioManager`,
  and the `LampBody`/`LampGesture` pair — each owning a single concern (perception,
  speech I/O, dialogue, pointing, sound, and motion, respectively). Each manager
  hides its own models, threads, and state behind a small interface, exposing a
  `step(dt)` method plus a few typed accessors. This keeps the boundaries clean, so
  a subsystem can be swapped or tested in isolation and `main.py` reads as a short
  list of coordinated managers rather than tangled logic.
- **One loop owns time.** A single per-frame loop drives every manager — simpler and
  more debuggable than an event/actor system; heavy work is pushed to threads and
  polled so the loop never blocks.
- **No LLM.** Templated, intent-driven dialogue keeps latency low, cost zero, and
  behavior deterministic on an 8 GB CPU box.
- **Coordinated modalities.** Engage/disengage transitions fire motion + light + SFX
  together; music ducks automatically while the user is speaking.

## Measurements

Engagement reliability and memory were measured directly; the latency, frame-rate,
and CPU figures are **engineering estimates** derived from the known cost of the
model stack (silero-VAD, faster-whisper base int8, kokoro, MediaPipe,
EfficientDet-Lite2) running CPU-only. Each row is marked accordingly. Memory was
read on the Windows development machine, so the target's RSS may differ; the other
estimates are intended to bound expected behavior. The methodology for capturing the
remaining exact figures is noted after the table.

| Metric | Value |
|---|---|
| Engagement — correct engage on approach | **20 / 20** (measured) |
| Engagement — spurious/incorrect fires (20 trials) | **0** (measured) |
| Response latency — end-of-speech → transcript (Whisper base int8) | ~0.6–1.5 s (est.) |
| Response latency — transcript → first TTS audio (kokoro) | ~0.2–0.6 s (est.) |
| Render loop rate (idle) | ~30 fps (est.) |
| CPU — steady state | ~60–120% of 400% (≈0.6–1.2 cores) (est.) |
| CPU — transient (Whisper transcription) | up to ~250–350% (est.) |
| Memory — resident set | **~2.5 GB** (measured, Windows dev box) |


## Known limitations

- **Open-mic picks up the speaker.** With no echo cancellation, loud music/TTS can be
  re-captured by the mic; mitigated by ducking music while listening.
- **Pointing is directional, not true 3D aim.** The webcam is world-fixed and not
  co-located with the lamp, so the system only maps the object's 2D image position
  to a pan/tilt direction. The
  lamp therefore points in the object's *general* direction rather than precisely at
  it, and the aim is a single snapshot (corrected once on arrival) rather than
  continuous tracking. On a real robot with a head-mounted camera, or a known
  camera-to-lamp transform, this becomes a genuine point at where the object was
  last seen.
- **Fixed detector vocabulary.** Object recall/pointing is limited to the COCO-80
  classes EfficientDet knows; out-of-vocabulary objects can't be named or pointed at.
- **Templated dialogue.** Responses are pattern-matched, so phrasing is bounded and
  won't handle open-ended conversation the way an LLM would. This is a deliberate trade for latency, determinism, and on-device cost.
- **No enforced actuator limits.** The sim eases joint targets but does not model
  torque/velocity limits a physical lamp would impose.
