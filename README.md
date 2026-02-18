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
- `visual_mesh.glb`: generic-viewer visual mesh (vertex-colored).
- `visual_pointcloud.ply`: colored dense point cloud debug artifact.
- `nvblox_mesh.ply`: collision/physics mesh (not intended to look photoreal).
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
- `COLMAP_SIFT_GPU` (`auto` default; `on`, `off`)
- `SAM3_N_FRAMES` (`0` default = auto-scaled by capture length)
- `SAM3_MIN_FRAME_DETECTIONS` (`0` default = env-aware auto)
- `SCENE_SEMANTICS_GEMINI_MODEL` (default: `gemini-3.0-pro`)
- `GOOGLE_GENAI_API_KEY` (optional; when missing, scene semantics falls back to local auto)
- `SAM3_TRACKING_MODE` (`full_video` recommended/default in wrapper)
- `DA3_MODEL_PATH` (default: `/opt/da3/weights/metric_large`, local path preferred)
- `DA3_MODEL_NAME` (default: `da3metric-large`)
- `HF_HOME` (default in VM guide: `/opt/hf`, shared HuggingFace cache)
- `VISUAL_MESH_ENABLED` (`true` default; set `false` to skip viewer mesh export)
- `VISUAL_MESH_METHOD` (`quick_poisson` default; `gaussian_tsdf` for robust path)
- `VISUAL_MESH_TARGET_FACES` (default: `500000`)
- `COLLISION_MAX_EDGE_M` (default: `5.0`; long-edge spike filter threshold)
- `COLLISION_SPIKE_MAX_RATIO` (default: `0.02`; collision spike gate threshold)

Asset generation/retrieval providers:

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
