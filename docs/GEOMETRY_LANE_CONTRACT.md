# Geometry Lane Contract

This document defines the new `raw/walkthrough.mov|mp4 -> pipeline/geometry/` lane for
Meta-glasses and other non-ARKit video captures that need world-model-friendly geometry
conditioning.

The target is **not** photorealistic rendering and **not** Gaussian splatting.

The target is a grounded geometry bundle that can be consumed by:

- downstream world-model services / APIs
- `scene_memory` conditioning
- pose-aware semantic extraction
- later TSDF / occupancy / site-world packaging

This lane is **derived** and **non-authoritative**. It does not rewrite qualification truth.

## Primary Decision

Use the geometry lane to produce:

- camera intrinsics
- camera trajectory / poses
- dense or semi-dense depth
- aligned confidence maps
- keyframe selection metadata
- run-level readiness and scale-confidence summaries

For the current Meta-glasses path, treat the result as **world-model conditioning** unless and
until scale validation / scaffolding is added.

## On-Disk Contract

The lane writes under:

```text
scenes/<scene_id>/captures/<capture_id>/pipeline/geometry/
```

Required files:

```text
pipeline/geometry/
  geometry_manifest.json
  geometry_summary.json
  geometry_run_status.json
  geometry_inputs.json
  IMPLEMENTATION_NOTES.md
  camera/
    intrinsics.json
    poses.jsonl
    trajectory_summary.json
  frames/
    keyframes.json
    frame_index.jsonl
  depth/
    depth_manifest.json
    *.npy | *.npz | *.png
  confidence/
    confidence_manifest.json
    *.npy | *.npz | *.png
  logs/
    provider_request.json
    provider_result.json
```

Optional files:

```text
pipeline/geometry/
  camera/
    poses.preview.json
  frames/
    preview_contact_sheet.jpg
  depth/
    preview_manifest.json
  confidence/
    preview_manifest.json
  logs/
    runner.stdout.log
    runner.stderr.log
```

## File Semantics

### `geometry_manifest.json`

Authoritative family index for the geometry lane.

Required fields:

- `schema_version`
- `generated_at`
- `manifest_type = "geometry_manifest"`
- `stage = "geometry"`
- `status`
- `capture_identity`
- `provider`
- `artifacts`
- `world_model_contract`

Status values:

- `contract_staged`
- `running`
- `failed`
- `completed`

### `geometry_summary.json`

Compact summary for qualification / scene-memory / downstream routing.

Required fields:

- `status`
- `ready_for_world_model`
- `source_video`
- `provider`
- `scale_assessment`
- `deliverables`

This is the file later stages should read first.

### `geometry_run_status.json`

Mutable heartbeat during execution.

Required fields:

- `status`
- `ready_for_world_model`
- `blocking_issues`
- `provider`
- `model`
- `execution_mode`

The runner updates this first when transitioning between `running`, `failed`, and `completed`.

### `geometry_inputs.json`

Frozen record of the exact raw inputs and provider configuration used for the run.

Required fields:

- `capture_identity`
- `source`
- `video_probe`
- `raw_manifest_hints`
- `descriptor_hints`
- `provider_config`

### `camera/intrinsics.json`

Single-camera intrinsics for the processed video stream.

Required fields:

- `camera_model`
- `image_width`
- `image_height`
- `fx`
- `fy`
- `cx`
- `cy`
- `distortion`
- `source`

If the provider outputs per-frame intrinsics instead of one stable camera model, keep the
canonical average / chosen intrinsics here and store the per-frame data in `frame_index.jsonl`.

### `camera/poses.jsonl`

One JSON object per sampled frame or keyframe.

Required fields per line:

- `frame_index`
- `timestamp_seconds`
- `image_path`
- `world_from_camera`
- `camera_from_world`
- `pose_confidence`
- `is_keyframe`

Matrix convention:

- 4x4 row-major arrays serialized as nested lists.
- `world_from_camera` is the preferred canonical transform.

### `camera/trajectory_summary.json`

Rollup over `poses.jsonl`.

Required fields:

- `pose_count`
- `keyframe_count`
- `track_length_m`
- `loop_closure_detected`
- `scale_status`
- `confidence_summary`

### `frames/keyframes.json`

Keyframe selection record.

Required fields:

- `sampling_strategy`
- `frames`

Required fields per `frames[]` item:

- `frame_index`
- `timestamp_seconds`
- `image_path`
- `blur_score`
- `overlap_hint`

### `frames/frame_index.jsonl`

Frame-level index that ties raw frames to geometry outputs.

Required fields per line:

- `frame_index`
- `timestamp_seconds`
- `image_path`
- `depth_path`
- `confidence_path`
- `pose_present`
- `intrinsics_present`

### `depth/depth_manifest.json`

Index for depth outputs.

Required fields:

- `representation = "per_frame_depth_map"`
- `unit = "meters"`
- `frame_count`
- `artifacts`

Required fields per `artifacts[]` item:

- `frame_index`
- `timestamp_seconds`
- `path`
- `format`
- `width`
- `height`
- `min_depth_m`
- `max_depth_m`

Recommended on-disk format:

- `float32 .npy` or `float16 .npz` for canonical processing
- optional `.png` previews for debugging

### `confidence/confidence_manifest.json`

Index for confidence outputs aligned to depth.

Required fields:

- `representation = "per_frame_confidence_map"`
- `frame_count`
- `artifacts`

Required fields per `artifacts[]` item:

- `frame_index`
- `timestamp_seconds`
- `path`
- `format`
- `width`
- `height`
- `value_range`

Recommended numeric convention:

- normalize to `[0, 1]`

### `logs/provider_request.json`

Frozen request payload sent to the model runner.

Purpose:

- reproducibility
- future reruns
- debugging model/provider drift

### `logs/provider_result.json`

Provider-native result summary.

Required fields:

- `status`
- `artifacts_written`
- `metrics`
- `errors`

## Runner Behavior

The minimal runner is:

```bash
python3 scripts/run_geometry_lane.py \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider da3 \
  --model DA3Nested-Giant-Large-1.1
```

Current behavior:

- resolves the staged capture root
- validates that a walkthrough video exists in `raw/`
- creates `pipeline/geometry/`
- writes placeholder manifests and status files

Future full behavior:

1. probe the video
2. sample keyframes or define streaming windows
3. run `DA3Nested` or `DA3-Streaming`
4. write poses, intrinsics, depth, and confidence outputs
5. compute readiness / scale / coverage summaries
6. update `geometry_run_status.json` and `geometry_manifest.json` to `completed`

## Model Selection

Current recommendation:

- default provider label: `da3`
- default model: `DA3Nested-Giant-Large-1.1`
- execution mode:
  - `standard` for shorter clips / pre-sampled windows
  - `streaming` for long clips or lower-memory GPUs

Interpretation:

- `DA3Nested` is the main reconstruction model
- `DA3-Streaming` is the execution strategy for long videos
- `MapAnything` is optional later as a unified evaluation / model-swap framework

## Integration Points

The next implementation agent should wire these outputs into:

### Qualification

Consume `pipeline/geometry/geometry_summary.json` for advisory signals only:

- pose coverage
- confidence coverage
- scale confidence
- world-model conditioning readiness

Do not let these fields override deterministic qualification truth.

### Scene Memory

Use:

- `camera/poses.jsonl`
- `camera/intrinsics.json`
- `depth/depth_manifest.json`
- `confidence/confidence_manifest.json`

### Privacy / Semantics

Run SAM3 / VIP / DeepPrivacy2 after or alongside geometry extraction, not before destroying
geometry cues from the source frames.

### Evaluation Prep / World-Model API

World-model-facing bundle should point to:

- original walkthrough video
- geometry manifest
- depth manifest
- confidence manifest
- poses
- intrinsics
- semantic/privacy masks if present

## Acceptance Criteria For Full Implementation

The full implementation is complete when:

1. `run_geometry_lane.py` produces `completed` status on a staged capture with a real video.
2. `camera/poses.jsonl` contains at least one pose entry.
3. `camera/intrinsics.json` contains numeric intrinsics.
4. `depth_manifest.json` and `confidence_manifest.json` index real files.
5. `geometry_summary.json` exposes machine-readable readiness / scale / coverage fields.
6. Qualification and downstream consumers can ingest `geometry_summary.json` without guessing.

## Non-Goals

This lane is not required to:

- perform Gaussian splatting
- become authoritative metric truth without scaffolding
- block qualification if geometry generation fails

Failure should be explicit and fail-closed in `geometry_run_status.json`.
