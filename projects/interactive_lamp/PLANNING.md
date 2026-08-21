# Planning

A live, expressive character built on the supplied 5-DOF lamp robot. 
Needs to be "one aware character, not a collection of unrelated AI demos."

---

### The demo — ONE continuous interaction with all 5 moments
1. **Engagement** — person looks at it and notices -> disengages when attention leaves.
2. **Character response** — acknowledges via intentional motion + light + voice/SFX/music.
3. **Spoken interaction** — listens on mic, replies on speaker.
4. **Scene memory** — sees ≥1 object, remembers it, later answers a spoken question about it.
5. **Goal-directed (VLA)** — spoken goal about the live scene -> vision + language pick a
   *sequence* of lamp actions → **re-observe the scene** → complete the goal.

Across the whole demo, purposefully use all five: **motion, light, voice, a sound effect, music**
(not necessarily at once).

### Deliverables
- **Source code** + dependency declarations.
- **Setup + run instructions** for the Ubuntu target (below).
- **≤2-page technical note**: one architecture/data-flow diagram; choices for protocol,
  model→action, sim, deployment; **measurements** (engagement reliability, response latency,
  CPU + memory); known limitations.

### Hard constraints
- **Timebox 6–8h.** Smaller + coherent beats wide + unfinished. State what was cut.
- **Runs on Ubuntu 24.04, 4 cores, 8 GB RAM, NO GPU/CUDA.**
- Cloud APIs allowed — must document what data leaves the machine, where decisions are made, why.

### Grading weights (design toward these)
Architecture/protocol 25% · Character integration 20% · Perception/action/memory 20% ·
Deployment 15% · Physical reasoning 10% · Evidence/communication 10%.

---

## Architectural spine (decided)
- **One character state** `{attention_target, arousal, mood, memory[]}`. Motion, light, voice,
  SFX, music are all **renderers of that state** — never independent subsystems.
- **Two timescales:** *reflex loop* (local, <100ms: face appears → perk up, light warms, small
  sound, zero network) vs *deliberative loop* (cloud, 1–2s: speech, LLM, VLM). Instant physical
  reaction first, speech after = reads as alive.

---

## Suggested build order

**0. Foundations** ✅ deps installed (mediapipe, mujoco), URDF confirmed rendering in MuJoCo.

**1. Body layer** — Wrap the URDF in MJCF adding a `<light>` + `<camera>` on the head frame.
Build a thin `LampBody` adapter: `set_joints()`, `set_light(color/intensity)`, `get_camera_frame()`.
This is the single boundary between "character brain" and "robot body."

**2. Character state + renderers** — Define the state object. Write a small gesture library
(perk-up, look-away, nod, idle-breathing) with easing curves that blend **motion + light + sound
together** so they're inseparable. No IK, no RL — expressiveness is timing/easing.

**3. Reflex loop (Moment 1 + 2)** — MediaPipe face + head-pose on CPU → engage/disengage gate
with **dwell timer + hysteresis** (naive thresholds flicker and break the illusion). Fire an
instant local acknowledgement gesture. No network calls here.

**4. Deliberative loop (Moment 3)** — Single realtime speech-to-speech API for listen→respond
(collapses ASR+LLM+TTS into one). Voice tone/light tied to mood.

**5. Scene memory (Moment 4)** — On trigger, send ONE frame to a VLM → store
`{timestamp, caption, thumbnail}` in a plain Python list → stuff list into the prompt at question
time. **No vector DB** (deliberate; note what would change at scale).

**6. Goal-directed / VLA (Moment 5)** — Spoken goal → VLM observes current scene → language picks
a **sequence** of gesture/light actions → execute → **re-observe** → confirm completion. Keep the
action representation tiny and explicit (a named-gesture list), and state where model decisions
end and body execution begins.

**7. Stitch into one continuous demo** — Chain the moments; make sure a **sound effect** and
**music** cue appear somewhere purposeful so all five modalities are covered.

**8. Evidence + writeup + deployment** — Measure engagement reliability, latency, CPU/RAM on a
CPU-only profile. Write the 2-page note (incl. the 5-DOF fact, cloud data-flow, what was cut).
Document the Ubuntu 24.04 / 4-core / 8 GB / no-GPU setup + run steps.

---

## Watch-outs
- **No CUDA in the shipped path** — the deploy target has no GPU. Keep perception CPU-friendly.
- **Disengagement is explicitly graded** — don't only build "engage."
- **Filename says 5-DOF** and inspection confirms 5 hinge joints — mention in the note.
- Budget ~1h for the writeup; it's ~35% of the grade (deployment + evidence + architecture prose).
