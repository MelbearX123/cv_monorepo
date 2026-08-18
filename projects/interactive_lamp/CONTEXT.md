# Project Context — Human Computer Lab: Live Character Robot Challenge

## What this is
A take-home challenge for an ML/robotics software internship at Human Computer Lab
(makers of LeLamp, an open-source expressive robot lamp based on Apple's ELEGNT paper).
Completing it skips the first few application stages.

**Timebox: 6–8 hours. This is a hard constraint and part of the evaluation.**
A submission that obviously took 30+ hours signals inability to scope. If scope is
overrun, state it honestly rather than hiding it.

## The task
Build a live, expressive character around a supplied fictional 5-DOF lamp robot.
- Laptop **camera** = eyes, **microphone** = ears, **speaker** = voice/music/SFX
- Supplied URDF provides the simulated body (motion + light)
- Supplied files: `robot/dummy_lamp_6dof.urdf`, `robot/assets/lamp_shade.stl`, `SUBMISSION.md`

### Required demonstration — ONE continuous interaction containing:
1. **Engagement** — person looks toward the character, it recognizes the opportunity to
   interact; disengages appropriately when attention moves elsewhere
2. **Character response** — acknowledges the person via intentional combination of
   motion, light, voice, SFX, or music
3. **Spoken interaction** — listens via mic, responds via speaker
4. **Scene memory** — observes ≥1 object via camera, retains useful info, later answers
   a spoken question about it

Across the full demo, show purposeful use of **motion, light, voice, a sound effect,
and music**.

### Free choices (explicitly stated in the brief)
Languages, frameworks, repo structure, sim/visualization platform, comms between
character software and robot body, perception/speech/LLM/memory approaches, local vs
cloud vs hybrid, packaging/deployment.

Cloud APIs allowed — but must document what data is sent, where decisions are made,
and why that fits.

## THE ACTUAL RUBRIC
> "The result should not feel like a collection of unrelated AI demonstrations."

All four moments are table stakes. Most submissions will hit them and still fail because
they'll be four separate pipelines glued to a state machine. The differentiator is
**coherence**.

## Architectural decisions already made (do not relitigate without good reason)

### 1. One character state, many renderers
Maintain a single character state, e.g.:
```
{ attention_target, arousal, mood, memory[] }
```
Motion, light, voice tone, sound effects, and music are all **renderers of that state**,
not independent subsystems. Anti-pattern to avoid: a `speak()` that doesn't touch the
light, a `move()` that doesn't know the mood. If those exist, the architecture is wrong.

### 2. Two timescales — reflex vs deliberative (highest-leverage decision)
- **Reflex loop (local, <100ms):** face appears → perk up, light warms, small sound.
  Zero network calls.
- **Deliberative loop (cloud, 1–2s):** speech understanding, LLM response, TTS.

If everything blocks on an LLM round-trip, the character reads as broken, not thoughtful.
Instant physical reaction followed by speech reads as alive. This must be explicit in
the writeup.

## Scoping decisions (chosen to fit the timebox)

| Component | Approach | Rationale |
|---|---|---|
| Engagement | MediaPipe Face Landmarker on CPU (face + head pose, realtime, no training). Gate on head yaw/pitch toward camera. | Fast, local, no training time |
| **Hysteresis** | Dwell timer + hysteresis on the engage/disengage threshold — **required** | Naive thresholding flickers and destroys the illusion. Disengagement is explicitly graded. |
| Speech | Single realtime speech-to-speech API rather than assembling ASR + LLM + TTS | Collapses 3 components into 1; saves hours. Document tradeoffs: latency, cloud dependency, what audio leaves the machine. |
| Scene memory | **NO vector DB.** On trigger, send one frame to a VLM, store `{timestamp, caption, thumbnail}` in a plain Python list, stuff list into prompt at question time. | Over-engineering memory is the classic way to burn 6 of 8 hours. State in writeup that the trivial version was deliberate + what would change at scale. |
| Motion | Small library of hand-authored gestures with easing curves: perk-up, look-away, nod, idle breathing. **No IK, no RL.** | Expressiveness comes from timing/easing, not solvers |
| Coupling | Blend light + sound into the same gesture primitive so they're inseparable by construction | Directly serves the "not unrelated demos" rubric |

## Simulation / MuJoCo note
Platform choice is free. MuJoCo loads URDF directly, so it's a viable pick and makes the
sim a ~1 hour tool decision, not a project. Do not turn this into a physics-engine
learning exercise — the lamp needs to *move expressively*, not simulate contact dynamics.
Any lightweight visualizer that renders the URDF and a controllable light is acceptable.

## Details worth catching (free credibility)
- Brief says **5 DOF**, supplied file is named `dummy_lamp_6dof.urdf`. Verify which it
  actually is and **mention the discrepancy** in the writeup.
- The robot lineage is Apple's **ELEGNT** paper. Reference its principles — anticipation,
  follow-through, intention-expressive motion — to show understanding of the design
  philosophy rather than treating this as a generic robot arm.

## SUBMISSION.md — likely 30–50% of the grade
The brief asks for explanation of choices and tradeoffs **twice**, plus an explicit
statement of what was intentionally left out. **Budget ~1 hour for the writeup.**
Must cover:
- The single-character-state architecture and why
- Reflex vs deliberative split and why
- What data goes to cloud services, where decisions are made, why that fits
- What was completed vs deliberately omitted, and what would change with more time
- The 5-DOF / 6-DOF discrepancy

## Constraint on AI-assisted development
AI tools are explicitly allowed, BUT: "You remain responsible for the architecture,
behavior, measurements, code, and technical decisions... and should be able to explain
and modify the work as requested."

This implies a **live modification round**. Do not accept generated architecture that
can't be redrawn on a whiteboard from memory. Prefer fewer, simpler, well-understood
components over anything clever.

## Working principle
Prefer a smaller, coherent system with clear decisions over a wide, unfinished one.
This is stated in the brief and should govern every tradeoff.
