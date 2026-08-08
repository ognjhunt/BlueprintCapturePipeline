# ADP-009D lab-format episode evidence

Backlog binding: ADP-009D (deterministic variation, abstention, and full
rehearsal). This work makes each Task Evaluation Run episode legible to
robotics teams in the format they already consume, without changing what is
simulated, rendered, or scored.

## What was already lab-correct, and stays

| Property | Value | Why it is right |
|---|---|---|
| Control rate | 15 Hz (`sim.dt = 1/120`, `decimation = 8`) | DROID's published control contract; one policy action row per env step |
| Camera resolution | 320x180 | The exact DROID RLDS release resolution; reproduces both candidates' consumed pixels byte-identically (pi05 pads to 224x224 keeping 224x126, GR00T keeps 320x180) |
| View names | `observation/exterior_image_1_left`, `observation/wrist_image_left` | openpi DROID observation keys, verbatim |
| Action space | 7 joint velocities + absolute gripper, 15-row chunks, 8 executed | DROID convention; `droid_row_to_isaac_action` bounds each row |
| Review codec | H.264 `avc1`, decode round-trip proven | The mp4v QuickTime failure is already codified |

Rendering above 320x180 for policy-facing streams is not higher quality — it
is a 131x slowdown that the resize immediately discards (see the P2/P3
findings). Human "hero" renders are a separate diagnostic lane
(`BLUEPRINT_ADP009D_CAMERA_RESOLUTION=diagnostic`, 1280x720), never the
default.

## What changed (schema v3)

1. **Per-step trace retained** (`adp009d_episode_step_trace.py`). The loop
   already observed joints after every step, held every commanded row, and
   sampled object state per step — then discarded them. The receipt now
   carries one row per control step: pre/post joint state, executed DROID
   action, position target, Isaac action, query/chunk indices, phase, and the
   object sample. Fail-closed consistency: each commanded row's
   `observed_before_rad` must equal the joint trace, and policy rows must
   fill whole chunks.
2. **Motion quality derived, not guessed.** Finite-difference velocity,
   acceleration, jerk (max/RMS), chunk-boundary velocity discontinuity
   (commanded and observed), joint-limit minimum margin, DROID gripper
   transitions, and end-effector path length. "Was it smooth?" is now a
   number in the receipt instead of a guess from a 4 fps video.
3. **Review video plays at real rate.** Query-cadence frames are 8/15 s
   apart, so the composite review video is encoded at 1.875 fps with
   `playback_realtime_factor: 1.0` recorded. The old 4.0 fps default played
   motion 2.13x fast — wrong evidence for judging speed or smoothness.
4. **Dataset capture profile** (`adp009d_dataset_capture.py`,
   `BLUEPRINT_ADP009D_EVIDENCE_PROFILE=dataset`). One H.264 stream per DROID
   camera at the true 15 Hz, one frame per environment step plus terminal,
   incremental encode (no PNG flood, no frame buffering), per-frame raw RGB
   digests, decode round-trip proven. The query-cadence policy-input PNGs
   remain the authoritative record of what the policy consumed.
5. **LeRobot v2.1 export** (`adp009d_lerobot_export.py`, local tool).
   Receipts export to the tree openpi/GR00T tooling loads directly:
   `meta/info.json` + `tasks.jsonl` + `episodes.jsonl` +
   `episodes_stats.jsonl`, per-episode parquet
   (`observation.state` = 7 joints + gripper width, `action` = executed
   DROID row, timestamps on the 1/15 s grid), and per-camera videos when the
   capture profile ran. Receipts without step traces (v65 and earlier) are
   refused by name — their per-step record no longer exists and synthesizing
   it would be fabrication.

## Rendering cost: how 15 fps gets cheap

Measured baseline (v62/v65, 320x180, L40-class): ~172–195 ms per rendered
frame; 520 renders per episode; render is ~100% of the 89 s episode wall
clock; π0.5 inference is ~91 ms/query (~6%) and not worth optimizing.

The lanes, in order of leverage:

- **Tier 0 — retain, don't re-render (landed here).** The runtime already
  renders once per env step. The dataset profile records those frames;
  15 fps output costs encoding only (~5–15 s/episode), zero extra renders.
- **Tier 1 — render only what the policy consumes (sibling branch
  `claude/adp009d-p3-render-interval`).** `render_interval = decimation x
  open_loop_horizon = 64` renders once per policy query: projected
  89.3 s → 11.7 s/episode (7.6x). Mutually exclusive with 15 fps live
  retention; requires the stale-frame alignment assert before adoption.
- **Tier 2 — per-frame cost knobs (bounded canary, unproven).** GPU class is
  a measured 3x (L40/RTX 6000 Ada vs A6000 at 720p; keep the avoidlist
  current). Sweep `maxGaussiansToAccumulate` {48, 256, 1024} for *time*
  (48→1024 was already shown quality-neutral at <1e-4 frame delta). At
  320x180 fixed per-update overhead (RTX accumulation, denoiser, app update)
  likely dominates pixels; measure before believing any knob.
- **Tier 3 — replay-render lane (the architectural answer; next).** Run
  episodes at Tier-1 speed, then re-render offline by kinematically scrubbing
  the retained step trace (joints + gripper width + can pose per step —
  the trace's `replay_sufficiency` field states this contract) with physics
  and policy out of the loop. Live paid time drops 7.6x while 15 fps (or
  720p hero) video is produced by a renderer running flat out, batched
  across episodes on the same warm process. Honesty label: replay frames are
  derived renders of the same sealed states, not the frames the policy saw;
  the policy-input PNGs remain authoritative. Needs one paid canary to prove
  scrub-render parity before any claim.

An Arena-level `TiledCamera` experiment (both cameras in one render product)
is plausible but unproven against NuRec volumes; treat as a bounded
experiment, not a plan.

## Operating it

```bash
# Worker-side: control-rate per-camera streams next run
BLUEPRINT_ADP009D_EVIDENCE_PROFILE=dataset  # baked into the bundle entrypoint

# Local: export receipts to a LeRobot v2.1 dataset
python -c "
import json, pathlib
from blueprint_pipeline.adp009d_lerobot_export import export_lerobot_dataset
result = json.loads(pathlib.Path('adp009d_native_microcheck.json').read_text())
receipts = [e for b in result['policy_episode']['batches'] for e in b['episodes'] if e.get('status') == 'scored']
export_lerobot_dataset(episode_receipts=receipts, output_dir='lerobot_out', media_root='<runtime_output_dir>')
"
```

The batch receipt rows carry `step_trace`, `object_samples`,
`motion_quality`, `dataset_capture`, and `dataset_contract` per episode
(batch schema v3), so the persisted `adp009d_native_microcheck.json` is
sufficient input for the export.
