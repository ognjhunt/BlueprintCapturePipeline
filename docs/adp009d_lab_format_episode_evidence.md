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

The lanes, in order of leverage — Tiers 0, 1, and 3 are now code on this
branch; only the paid canaries remain:

- **Tier 0 — retain, don't re-render (landed).** The dataset profile records
  the frames the runtime already renders; 15 fps output costs encoding only
  (~5–15 s/episode), zero extra renders.
- **Tier 1 — render only what the policy consumes (landed, cadence-derived).**
  `resolve_render_interval` grants `decimation x open_loop_horizon = 64`
  (once per policy query; projected 89.3 s → 11.7 s/episode, 7.6x) only when
  every bound candidate declares `per_query` frame cadence in
  `CANDIDATE_OBSERVATION_FRAME_CADENCE`. Any per-step candidate (GR00T's
  t-minus-15 history), the dataset profile, an unknown candidate, or
  `BLUEPRINT_ADP009D_RENDER_PER_QUERY=0` forces per-step rendering — the
  always-safe cadence. The saving is guarded, not assumed: the adapter
  stamps each observation with its rendered sim time and the episode
  refuses any query frame whose stamp differs from the episode's own step
  clock (a merely-monotonic check would pass a constant misalignment).
  Receipts record `observation_sim_times` and intervals, so a claimed
  cadence is shown, not asserted.
- **Tier 2 — per-frame cost knobs (bounded canary, unproven).** GPU class is
  a measured 3x (L40/RTX 6000 Ada vs A6000 at 720p; keep the avoidlist
  current). Sweep `maxGaussiansToAccumulate` {48, 256, 1024} for *time*
  (48→1024 was already shown quality-neutral at <1e-4 frame delta). All the
  tuning variables now actually reach the worker — the entrypoint exports
  camera resolution, render cadence, episode count, gaussian cap, evidence
  profile, and replay flag at bundle-build time.
- **Tier 3 — replay-render lane (landed; parity canary pending).** Run
  episodes at Tier-1 speed, then `episode_replay_render` re-renders offline
  by kinematically scrubbing the retained step trace — the exact full DOF
  vector per step (arm plus every gripper joint, so no width-to-joint
  mapping to get wrong) plus the sealed object pose — with physics, policy,
  and server out of the loop. Provenance is structural: the recorder must be
  labeled `kinematic_replay` and digest-bound to the trace it derives from,
  and the LeRobot export carries `video_source` so derived frames can never
  masquerade as live capture. Enabled by `EVIDENCE_PROFILE=replay` or
  forced alongside any profile with `BLUEPRINT_ADP009D_REPLAY_RENDER=1`;
  running it with the dataset profile makes the runtime compare live and
  replay streams per episode (`replay_parity`, mean/max pixel deltas) —
  the parity canary is one receipt read. Until that canary passes a
  preregistered threshold, replay video is evidence of the lane, not a
  claimed substitute for live capture.

An Arena-level `TiledCamera` experiment (both cameras in one render product)
is plausible but unproven against NuRec volumes; treat as a bounded
experiment, not a plan.

## Operating it

```bash
# Worker-side profiles (baked into the bundle entrypoint at build time)
BLUEPRINT_ADP009D_EVIDENCE_PROFILE=eval     # default: fast, per-query renders when cadence allows
BLUEPRINT_ADP009D_EVIDENCE_PROFILE=dataset  # per-step renders + live 15 fps per-camera streams
BLUEPRINT_ADP009D_EVIDENCE_PROFILE=replay   # fast live leg + offline kinematic replay render
BLUEPRINT_ADP009D_REPLAY_RENDER=1           # force the replay pass alongside any profile (parity canary: use with dataset)

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
