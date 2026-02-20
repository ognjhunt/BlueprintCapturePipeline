# BlueprintCapturePipeline

NuRec-first orchestration for converting BlueprintCapture descriptors into sim-ready scenes with swappable assets.

## Scope

This repo is intentionally thin. It orchestrates one path:

`capture_descriptor.json` -> NuRec reconstruction -> swap candidate policy -> SAM3D-first asset materialization -> interactive articulation validation/fallback -> simready + USD assembly.

It reuses existing jobs and helper logic from:

- `/Users/nijelhunt_1/workspace/BlueprintPipeline`

## Entry Points

Orchestrator:

```bash
python -m blueprint_pipeline.swap_orchestrator \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json
```

Or via installed script:

```bash
blueprint-capture-swap --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json
```

NuRec worker (consumes `nurec_job_spec.json`, writes `.nurec_complete`/`.nurec_failed`):

```bash
python -m blueprint_pipeline.nurec_worker \
  --job-spec /mnt/gcs/<bucket>/scenes/<scene_id>/captures/<capture_id>/pipeline/nurec_job_spec.json \
  --storage-root /mnt/gcs
```

## Required Descriptor Inputs

- `schema_version: v1`
- `scene_id`, `capture_id`
- `raw_prefix_uri`, `frames_index_uri`
- `qa_report_uri` (or sibling `qa_report.json`)

Supported aliases:

- `intended_space_type` -> `environment_type_hint`
- `capture_bundle.arkit_poses_uri` -> `arkit_poses_uri`
- `capture_bundle.arkit_intrinsics_uri` -> `arkit_intrinsics_uri`

## Contract Outputs

Written under:

`scenes/<scene_id>/captures/<capture_id>/pipeline/`

- `nurec_job_spec.json`
- `nurec_outputs.json`
- `swap_candidates.json`
- `swap_execution_report.json`
- `runtime_preflight_report.json`
- `advanced_quality_report.json`
- `swap_quality_report.json`
- `pipeline_summary.json`
- `.swap_pipeline_complete` or `.swap_pipeline_failed.json`

Scene artifacts written under:

- `scenes/<scene_id>/assets/scene_manifest.json`
- `scenes/<scene_id>/layout/scene_layout_scaled.json`
- `scenes/<scene_id>/seg/inventory.json`

NuRec artifact roles under `.../pipeline/nurec/`:

- `export_last.usdz`: NuRec volume visual for Isaac Sim / Omniverse rendering.
- `visual_mesh.glb`: generic-viewer visual mesh (textured when available; vertex-color fallback).
- `visual_pointcloud.ply`: colored dense point cloud debug artifact.
- `nvblox_mesh.ply`: collision/physics mesh (not intended to look photoreal).
- `capture_quality_report.json`: frame blur/brightness/motion stats + SfM registration ratio.
- `sam3_preflight_report.json`: SAM3 auth/import/cache preflight result and skip/fail reason.
- `mesh_manifest.json`: role manifest describing which artifact to use for visual vs collision.

## Environment

- `GCS_ROOT` (default: `/mnt/gcs`)
- `BLUEPRINTPIPELINE_ROOT` (default: `/Users/nijelhunt_1/workspace/BlueprintPipeline`)
- `BLUEPRINTPIPELINE_COMMIT_HASH` (optional pin)
- `FAIL_ON_BLUEPRINTPIPELINE_COMMIT_MISMATCH` (`true` by default)
- `RUNTIME_PREFLIGHT_ENABLED` (`true` by default)
- `PIPELINE_COMPLETION_MODE` (`best_effort` default, `full_required` for strict non-stub completion)
- `PIPELINE_STANDALONE_MODE` (`true` by default; must be `false` in `full_required`)

NuRec worker dispatch:

- `NUREC_WORKER_MODE` (`local_worker` default, `command`, `external_markers`)
- `NUREC_WORKER_COMMAND` (required when `NUREC_WORKER_MODE=command`)
- `NUREC_PIPELINE_COMMAND` (required for `local_worker` unless `NUREC_SKIP_PIPELINE_COMMAND=true`)
- `NUREC_SKIP_PIPELINE_COMMAND` (dev/testing only: skip command and validate pre-generated artifacts)

NuRec shim Fixer routing (when using `scripts/nurec_shim.py`):

- `FIXER_MODE` (`auto` default; `h100`, `local`)
- `FIXER_H100_SCRIPT` (default: `/app/scripts/fixer_h100_stage.sh`)
- `FIXER_H100_INSTANCE_ID` (optional existing Vast.ai instance)
- `FIXER_H100_KEEP_INSTANCE` (`true`/`false`)
- `FIXER_H100_MAX_HOURLY` (default: `2.50`)
- `FIXER_H100_DISK_GB` (default: `80`)
- `FIXER_H100_REMOTE_SETUP_CMD` (optional custom setup command for remote Fixer env)
- `NUREC_QUALITY_PROFILE` (`quality_first` default; `balanced`, `fast`)
- `MAX_FRAMES` (`450` default in `quality_first`; requested floor for adaptive frame budget)
- `EXTRACT_FPS` (`6` default in `quality_first`; requested FPS before duration-aware downsampling)
- `ADAPTIVE_MAX_FRAMES` (`true` default; scales frame budget up for longer captures)
- `ADAPTIVE_MAX_FRAMES_TARGET_FPS` (`3.0` default; target temporal density for long-capture budgeting)
- `ADAPTIVE_MAX_FRAMES_HARD_CAP` (`6000` default; maximum auto-expanded frame budget)
- `ADAPTIVE_EXTRACT_FPS` (`true` default; reduces extraction FPS so frame sampling spans full clip)
- `ADAPTIVE_EXTRACT_FPS_WARN_FLOOR` (`0.15` default; warning threshold for very sparse long-capture sampling)
- `COLMAP_SIFT_GPU` (`auto` default; `on`, `off`)
- `COLMAP_MAPPER_NUM_THREADS` (`0` default = auto/all visible CPU cores)
- `COLMAP_MATCHER_MODE` (`auto` default in quality-first profile; `auto`/`sequential`/`exhaustive`)
- `COLMAP_AUTO_EXHAUSTIVE_MAX_FRAMES` (`600` default; `auto` uses exhaustive below this frame count)
- `COLMAP_SEQUENTIAL_OVERLAP` (`30` default in quality-first profile)
- `COLMAP_CHUNKED_MODE` (`auto` default; `auto`/`off`/`on` for long-capture chunked SfM)
- `COLMAP_CHUNK_MIN_FRAMES` (`900` default; `auto` enables chunked SfM at/above this frame count)
- `COLMAP_CHUNK_SIZE_FRAMES` (`600` default; per-chunk SfM window size)
- `COLMAP_CHUNK_OVERLAP_FRAMES` (`120` default; shared frames between adjacent chunks)
- `COLMAP_CHUNK_MAX_CHUNKS` (`24` default; caps number of chunk windows)
- `COLMAP_CHUNK_MATCHER_MODE` (`sequential` default; matcher used inside each chunk)
- `COLMAP_MIN_REGISTERED_RATIO` (`0.80` default; retry threshold trigger)
- `COLMAP_RETRY_MIN_REGISTERED_RATIO` (`0.75` default; hard fail threshold after forced retry)
- `COLMAP_RETRY_MATCHER_MODE` (`auto` default; retry matcher `auto`/`sequential`/`exhaustive`)
- `COLMAP_RETRY_SEQUENTIAL_OVERLAP` (`60` default when retry matcher resolves to sequential)
- `BLUR_FILTER_KEEP_RATIO` (`1.0` default = disabled; e.g. `0.7` keeps sharpest 70% before SfM)
- `BLUR_FILTER_MIN_FRAMES` (`120` default; safety floor when blur filtering is enabled)
- `NUREC_RESUME` (`false` default in quality-first wrapper profile)
- `NUREC_PARALLEL_POST_STAGE6` (`true` default; runs Stage 7 visual mesh and Stage 9 SAM3 concurrently)
- `NUREC_DEPENDENCY_PREFLIGHT` (`true` default; fail fast before COLMAP if 3DGRUT deps are missing)
- `NUREC_PREFLIGHT_CHECK_FUSED_SSIM` (`true` default; checks fused_ssim import/torch ABI during preflight)
- `SAM3_PREFLIGHT_STRICT` (`false` default; if true, fail before reconstruction when SAM3 access is unavailable)
- `SAM3_N_FRAMES` (`0` default = auto-scaled by capture length)
- `SAM3_MIN_FRAME_DETECTIONS` (`0` default = env-aware auto)
- `SCENE_SEMANTICS_GEMINI_MODEL` (default: `gemini-3.0-pro`)
- `GOOGLE_GENAI_API_KEY` (optional; when missing, scene semantics falls back to local auto)
- `SAM3_TRACKING_MODE` (`auto` default in wrapper; resolves to `full_video` or `sampled`)
- `SAM3_FULL_VIDEO_MAX_FRAMES` (`600` default; `auto` uses sampled tracking above this)
- `DA3_MODEL_PATH` (default: `/opt/da3/weights/metric_large`, local path preferred)
- `DA3_MODEL_NAME` (default: `da3metric-large`)
- `HF_HOME` (default in VM guide: `/opt/hf`, shared HuggingFace cache)
- `NUREC_VISUAL_PRIMARY` (`usdz` default; `mesh`, `auto` control scene-shell visual prim routing)
- `OPEN_OMNIVERSE_PREVIEW` (`false` default; set `true` in `run_full_pipeline.sh` to print a preview plan)
- `ISAAC_WEBRTC_ENDPOINT` (required for client/stream launch hints, example: `https://10.0.0.10:3000`)
- `ISAAC_WEBRTC_REMOTE_TARGET` (optional `user@host` for asset upload with `scp`)
- `ISAAC_WEBRTC_REMOTE_PATH` (optional, default `/tmp/omniverse-preview`)
- `ISAAC_WEBRTC_REMOTE_PORT` (optional, default `22`)
- `ISAAC_WEBRTC_LOCAL_STAGE` (optional local staging dir for HTTP preview, default `/tmp/omniverse_webrtc_preview`)
- `ISAAC_WEBRTC_HTTP_PORT` (optional local HTTP port for temporary staging, default `8000`)
- `VISUAL_MESH_ENABLED` (`true` default; set `false` to skip viewer mesh export)
- `VISUAL_MESH_METHOD` (`textured_colmap` default; fallback chain `gaussian_tsdf` -> `quick_poisson`)
- `VISUAL_MESH_TARGET_FACES` (default: `500000`)
- `VISUAL_MESH_POISSON_DEPTH` (default: `12`; used for smaller clouds)
- `VISUAL_MESH_POISSON_DEPTH_LARGE` (default: `9`; used when cloud exceeds large-threshold)
- `VISUAL_MESH_POISSON_LARGE_THRESHOLD` (default: `500000` points)
- `VISUAL_MESH_TEXTURE_SIZE` (default: `4096`)
- `VISUAL_MESH_TEXTURE_MAX_ATLASES` (default: `2`)
- `OPEN3D_CPU_THREADS` (`0` default = Open3D/runtime default threading; set explicit thread count for CPU Poisson stages)
- `COLLISION_MAX_EDGE_M` (default: `5.0`; long-edge spike filter threshold)
- `COLLISION_SPIKE_MAX_RATIO` (default: `0.02`; collision spike gate threshold)

For production runtimes, pre-bake 3DGRUT build dependencies into the image (tiny-cuda-nn submodules and fused_ssim built against the image's torch) to avoid rebuild delays during retries.

## Omniverse Preview (Recommended)

Primary visual asset routing is now configured by `NUREC_VISUAL_PRIMARY`:
- `usdz` (default): prefer `export_last.usdz` for photoreal neural rendering in Omniverse/Isaac Sim.
- `mesh`: prefer `visual_mesh.glb` for broad app compatibility.
- `auto`: prefer textured mesh only when available, otherwise USDZ.

Use the helper script to choose the same artifact path orchestrator recorded:

```bash
bash scripts/launch_omniverse_preview.sh /Users/nijelhunt_1/Downloads/pipeline_output
# add --launch to attempt auto-open with local apps when available
bash scripts/launch_omniverse_preview.sh /Users/nijelhunt_1/Downloads/pipeline_output auto --launch
```

For WebRTC-only hosts (like this Mac), use the dedicated flow:

```bash
ISAAC_WEBRTC_ENDPOINT="https://<server-host>:<port>" \
ISAAC_WEBRTC_REMOTE_TARGET="<user>@<server-host>" \
ISAAC_WEBRTC_REMOTE_PATH="/tmp/omniverse-preview" \
bash scripts/preview_omniverse_webrtc.sh /Users/nijelhunt_1/Downloads/pipeline_output auto
```

`preview_omniverse_webrtc.sh` outputs:
1. resolved primary visual asset (USDZ preferred),
2. local staging + HTTP hosting command,
3. optional `scp` upload command,
4. the endpoint + client launch line.

To auto-run this plan after `run_full_pipeline.sh`, set:

```bash
export OPEN_OMNIVERSE_PREVIEW=true
export ISAAC_WEBRTC_ENDPOINT="https://<server-host>:<port>"
export ISAAC_WEBRTC_REMOTE_TARGET="<user>@<server-host>"
```

Asset generation/retrieval providers:

- `CROP_CLEANUP_PROVIDER` (`skip` default in `run_full_pipeline.sh`; options: `skip`, `together_qwen_image_edit`, `qwen_image_edit`, `nano_banana`, `gpt_image`)
- `TOGETHER_API_KEY` (required only when `CROP_CLEANUP_PROVIDER=together_qwen_image_edit`)
- `TOGETHER_QWEN_IMAGE_EDIT_MODEL` (optional override; tries known Together Qwen model IDs by default)
- `TOGETHER_IMAGE_EDIT_ENDPOINT` (optional override; default `https://api.together.xyz/v1/images/generations`)
- `TOGETHER_QWEN_IMAGE_EDIT_WIDTH` / `TOGETHER_QWEN_IMAGE_EDIT_HEIGHT` (optional; defaults `1024x1024`)
- `TOGETHER_QWEN_IMAGE_EDIT_STEPS` (optional; default `28`)
- `TOGETHER_QWEN_IMAGE_EDIT_TIMEOUT_SECONDS` (optional; default `90`)
- `TEXT_ASSET_GENERATION_PROVIDER_CHAIN` (default: `sam3d,hunyuan3d`)
- `TEXT_SAM3D_API_HOST` + `TEXT_SAM3D_API_KEY` (or `SAM3D_API_HOST` + `SAM3D_API_KEY`)
- `TEXT_HUNYUAN_API_HOST` + `TEXT_HUNYUAN_API_KEY` (or `HUNYUAN_API_HOST` + `HUNYUAN_API_KEY`)

Interactive backend env:

- `PARTICULATE_MODE` (`remote`/`local`/`mock`/`skip`)
- `PARTICULATE_ENDPOINT` (required for `PARTICULATE_MODE=remote`)
- `PARTICULATE_LOCAL_ENDPOINT` + `PARTICULATE_LOCAL_MODEL` (required for `PARTICULATE_MODE=local`)
- `ARTICULATION_BACKEND` (`auto` default)

Swap policy tuning:

- `SWAP_POLICY_CONFIG_PATH` (optional YAML path; default baked policy used if unset)
- `SWAP_INCLUDE_HEURISTIC_AS_EXPLICIT` (`false` recommended; `true` is flagged as unsafe in preflight)

Advanced quality gates:

- `ADVANCED_QUALITY_GATES_ENABLED` (`true` by default)
- `QUALITY_MAX_COLLISION_FACES`
- `QUALITY_DROP_MIN_PASS_RATE`
- `QUALITY_JITTER_MAX_DRIFT_M`
- `QUALITY_TUNNELING_MAX_PENETRATION_M`
- `QUALITY_PERF_MAX_STEP_MS`
- `QUALITY_THRESHOLD_PROFILE` (`default`, `delaunay_relaxed`, etc.; persisted in quality report)

Async trigger dispatch:

- `SWAP_TRIGGER_DISPATCH_MODE` (`pubsub` default, `cloud_tasks`, or `direct`)
- `SWAP_TRIGGER_PUBSUB_TOPIC` (required for `pubsub` mode)
- `SWAP_TRIGGER_TASK_QUEUE`, `SWAP_TRIGGER_TASK_LOCATION`, `SWAP_TRIGGER_TASK_URL` (required for `cloud_tasks` mode)
- `SWAP_TRIGGER_TASK_SERVICE_ACCOUNT` (optional OIDC SA for Cloud Tasks HTTP target)
- `SWAP_TRIGGER_ALLOW_DIRECT=true` (required to allow sync direct mode in local/dev)

Worker entrypoints in `functions/storage_trigger.py`:

- `on_storage_finalize`: enqueue-only storage trigger.
- `on_swap_dispatch`: Pub/Sub worker consumer.
- `on_swap_dispatch_http`: HTTP worker target for Cloud Tasks.

## Pipeline Stages

1. Runtime preflight checks mounted storage, BlueprintPipeline runtime, provider env, NuRec worker config, and quality-gate deps.
2. Intake validates descriptor + `qa_report.json` and loads raw manifest/object index.
3. NuRec stage writes `nurec_job_spec.json`, dispatches worker, and waits for completion marker.
4. Candidate selection applies policy signals (descriptor hints + object index + environment tuning).
5. SAM3D-first materialization generates swappable assets and scene shell assets.
6. Manifest/layout synthesis writes scene files for BlueprintPipeline Stage 2+ jobs.
7. Interactive articulation validation runs interactive-job.
8. Retrieval fallback resolves required articulation failures (hard fail if unresolved).
9. SimReady + USD assembly runs unchanged BlueprintPipeline jobs.
10. Advanced quality gates run drop/jitter/tunneling/perf and complexity budgets.
11. Completion writes `swap_quality_report.json`, `pipeline_summary.json`, and `.swap_pipeline_complete`.

## Tests

```bash
pytest -q
```
