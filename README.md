# BlueprintCapturePipeline

Qualification-first orchestration for converting BlueprintCapture raw uploads or descriptors into authoritative qualification artifacts. Metric-ready captures now flow by default into canonical scene-memory and preview-prep artifacts for downstream adapters, while video-only captures remain supported but are prevented from silently becoming decision-grade.

## Scope

This repo is intentionally thin. It now orchestrates three lanes:

- Default qualification lane:
  `capture_descriptor.json` -> QA/completeness -> scoping -> risk extraction -> qualification artifacts
- Default modern downstream lane:
  qualification artifacts + capture evidence -> scene-memory manifest -> backend adapter manifests -> preview simulation prep
- Explicit advanced geometry lane:
  `capture_descriptor.json` -> NuRec reconstruction -> swap candidate policy -> SAM3D-first asset materialization -> interactive articulation validation/fallback -> simready + USD assembly

Qualification artifacts remain authoritative. Scene memory is the canonical downstream substrate. World-model outputs, preview simulation prep, and any advanced geometry exports are derived artifacts only.

This repo emits qualification, scene-memory, and preview-prep handoff artifacts. It does not own the high-volume synthetic-data factory role; bounded preview generation and downstream adapters belong here, while large-scale synthetic generation belongs in downstream systems.

Raw-upload materialization path:

- `raw/capture_upload_complete.json` -> descriptor materialization -> modality-aware `qa_report.json` -> qualification orchestration

Local raw-capture contract for `blueprint-run-e2e`:

- required:
  - `raw/manifest.json`
  - `raw/intake_packet.json`
  - `raw/capture_context.json`
  - one video file under `raw/` or a `video_uri` in `manifest.json`
  - `raw/capture_upload_complete.json`
- optional:
  - ARKit pose/intrinsics/depth files
  - scaffolding calibration assets
  - splat / 3DGS artifacts

Splat / 3DGS artifacts are supplemental only. They can be attached to agent review, scene-memory conditioning, and advanced-geometry compatibility packaging, but they do not bypass intake, QA, or readiness gates.

It reuses existing jobs and helper logic from:

- `/Users/nijelhunt_1/workspace/BlueprintPipeline`

## Entry Points

Default lane orchestrator:

```bash
python -m blueprint_pipeline.capture_orchestrator \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json
```

Or via installed script:

```bash
blueprint-capture-pipeline --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json
```

Local raw-capture preflight:

```bash
blueprint-preflight-capture \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id>
```

Local agent review over qualification artifacts:

```bash
blueprint-agent-review \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider claude
```

One-command local report flow:

```bash
blueprint-run-e2e \
  --capture-root /path/to/<bucket>/scenes/<scene_id>/captures/<capture_id> \
  --provider openai
```

Advanced geometry orchestrator:

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

- Default qualification artifacts:
  - `site_intake.json`
  - `capture_package_manifest.json`
  - `capture_qa_scorecard.json`
  - `task_scope_record.json`
  - `qualification_record.json`
  - `qualification_brief.json`
  - `opportunity_handoff.json`
  - `task_targets.json`
  - `runtime_preflight_report.json`
  - `qualification_quality_report.json`
  - `human_actions_required.json`
  - `swap_quality_report.json` (compatibility alias)
  - `pipeline_summary.json`
  - `agent_review_bundle.json` (when `blueprint-agent-review` or `blueprint-run-e2e` is used)
  - `agent_readiness_memo.md` (when `blueprint-agent-review` or `blueprint-run-e2e` is used)
  - `.qualification_pipeline_complete` or `.qualification_pipeline_failed.json`
  - `.swap_pipeline_complete` or `.swap_pipeline_failed.json` (compatibility aliases)
- Scene-memory and preview-prep artifacts:
  - `scene_memory/scene_memory_manifest.json`
  - `scene_memory/scene_memory_readiness.json`
  - `scene_memory/conditioning_bundle.json`
  - `scene_memory/adapter_manifests/gen3c.json`
  - `scene_memory/adapter_manifests/neoverse.json`
  - `scene_memory/adapter_manifests/cosmos_transfer.json`
  - `preview_simulation/preview_simulation_manifest.json`
  - `evaluation_prep/qualified_opportunity_handoff.json`
  - `evaluation_prep/task_run_manifest.json`
  - `evaluation_prep/task_anchor_manifest.json`
  - `evaluation_prep/hosted_session_runtime_manifest.json`
  - `evaluation_prep/site_normalization_package.json`
  - `evaluation_prep/benchmark_suite_manifest.json`
  - `evaluation_prep/compatibility_matrix.json`
  - `evaluation_prep/recapture_diff.json`
  - `evaluation_prep/launchable_export_bundle.json`
  - `evaluation_prep/evaluation_prep_manifest.json`
- Advanced geometry artifacts when explicitly requested:
  These are compatibility outputs for downstream geometry-oriented flows. They are not the default end product.
  - `nurec_job_spec.json`
  - `nurec_outputs.json`
  - `swap_candidates.json`
  - `swap_execution_report.json`
  - `advanced_quality_report.json`
  - `advanced_geometry/advanced_geometry_bundle.json`
  - `advanced_geometry/labels.json`
  - `advanced_geometry/structure.json`
  - `advanced_geometry/task_targets.synthetic.json`
  - optional `advanced_geometry/3dgs_compressed.ply`

Scene artifacts written under:

- `scenes/<scene_id>/assets/scene_manifest.json`
- `scenes/<scene_id>/layout/scene_layout_scaled.json`
- `scenes/<scene_id>/seg/inventory.json`

NuRec artifact roles under `.../pipeline/nurec/`:

- `export_last.usdz`: NuRec volume visual for Isaac Sim / Omniverse rendering.
- `export_last_refined.usdz`: post-Stage-4 refined visual volume (when refinement gate passes).
- `export_last_refined.ply`: post-Stage-4 refined Gaussian point cloud (when refinement gate passes).
- `visual_mesh.glb`: generic-viewer visual mesh (textured when available; vertex-color fallback).
- `visual_pointcloud.ply`: colored dense point cloud debug artifact.
- `nvblox_mesh.ply`: collision/physics mesh (not intended to look photoreal).
- `capture_quality_report.json`: frame blur/brightness/motion stats + SfM registration ratio.
- `sam3_preflight_report.json`: SAM3 auth/import/cache preflight result and skip/fail reason.
- `mesh_manifest.json`: role manifest describing which artifact to use for visual vs collision.
- `.fixer_stage_complete.json`: Stage 5 Fixer completion marker (backend + refined image count) used for safe resume.
- `gap_analysis_report.json`: Stage 4.5 gap observability summary + pseudo-view candidate stats.
- `view_repair_report.json`: Stage 5A per-view repair metrics and acceptance decisions.
- `post_stage4_distill_report.json`: Stage 5B distillation outputs/metrics summary.
- `refinement_quality_gate.json`: Stage 5C rollback gate status and thresholds.

## Environment

- `PIPELINE_LANE` (optional; `qualification`, `scene_memory`, `advanced_geometry`, or `all`; default resolves from descriptor `requested_lanes` and falls back to `qualification`)
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

- `FIXER_MODE` (`local` default; `auto` aliases `local`; `h100` explicit opt-in)
- `FIXER_H100_SCRIPT` (default: `/app/scripts/fixer_h100_stage.sh`)
- `FIXER_H100_INSTANCE_ID` (optional existing Vast.ai instance)
- `FIXER_H100_KEEP_INSTANCE` (`true`/`false`)
- `FIXER_H100_MAX_HOURLY` (default: `2.50`)
- `FIXER_H100_DISK_GB` (default: `80`)
- `FIXER_H100_REMOTE_SETUP_CMD` (optional custom setup command for remote Fixer env)
- `FIXER_RERUN` (`false` default; set `true` with resume to force rerun Stage 5 Fixer)
- `FIXER_REQUIRED` (`false` default; set `true` to fail instead of falling back to unrefined renders)
- `POST_STAGE4_REFINE` (`auto` default; `off`/`auto`/`force` for Stage 4.5/5A/5B/5C flow)
- `POST_STAGE4_REFINE_MODEL` (`worldforge+gsfix3d` default; `fixer`, `fixer+gsfix3d`, `worldforge`, or `worldforge+gsfix3d`)
- `POST_STAGE4_WORLDFORGE_IMAGE_COMMAND` (optional command template override using placeholders `{input}`, `{mask}`, `{output}`)
- `POST_STAGE4_WORLDFORGE_ROOT` (default: `/opt/WorldForge`)
- `POST_STAGE4_WORLDFORGE_BACKEND` (`longcat` default; `longcat` or `wan`)
- `POST_STAGE4_WORLDFORGE_CHECKPOINT_DIR` (required for native `longcat` backend)
- `POST_STAGE4_WORLDFORGE_MODELS_DIR` (required for native `wan` backend)
- `POST_STAGE4_WORLDFORGE_PROMPT` (optional text prompt for native `longcat`)
- `POST_STAGE4_WORLDFORGE_SCENE` (optional scene key for native `wan`; default: `truck`)
- `POST_STAGE4_WORLDFORGE_RESOLUTION` (`480p` default; `480p` or `720p`)
- `POST_STAGE4_WORLDFORGE_NUM_FRAMES` (`17` default)
- `POST_STAGE4_WORLDFORGE_NUM_INFERENCE_STEPS` (`16` default for `longcat`, `50` for `wan`)
- `POST_STAGE4_WORLDFORGE_TIMEOUT_SECONDS` (`600` default)
- `POST_STAGE4_WORLDFORGE_STATIC` (`True` default; `True`/`False`)
- `POST_STAGE4_MAX_PSEUDOVIEWS` (`96` default; candidate pseudo-view cap)
- `POST_STAGE4_MAX_VIRTUAL_CANDIDATES` (`48` default; cap on generated virtual void-fill cameras)
- `POST_STAGE4_DISTILL_ITERS` (`1600` default; refinement distillation iterations)
- `POST_STAGE4_TIME_BUDGET_MIN` (`90` default; refinement distillation budget)
- `POST_STAGE4_MIN_PARALLAX_DEG` (`7.0` default; minimum pseudo-view parallax)
- `REFINEMENT_QUALITY_GATE_PROFILE` (`auto` default; `strict`/`hallucination`; `auto` resolves to `hallucination` when `PIPELINE_MODE=photoreal_hallucination`)
- `REFINEMENT_GATE_MIN_HOLE_IMPROVEMENT_RATIO` (optional gate override; default depends on profile)
- `REFINEMENT_GATE_MAX_SHARPNESS_DROP_RATIO` (optional gate override; default depends on profile)
- `REFINEMENT_GATE_MAX_PSNR_DROP_DB` (optional gate override; default depends on profile)
- `REFINEMENT_GATE_ENFORCE_PSNR` (optional gate override; default depends on profile)
- `VOID_FILL_ROUNDS` (`0` default; enables iterative virtual-render→repair→distill loop)
- `VOID_FILL_TARGET_HOLE_RATIO` (`0.05` default; stop threshold based on virtual probe p90 hole ratio)
- `VOID_FILL_DISTILL_ITERS` (`5000` default; distillation iterations per void-fill round)
- `VOID_FILL_MIN_HOLE_RATIO` (`0.03` default; lower bound for virtual renders selected for repair)
- `VOID_FILL_MAX_HOLE_RATIO` (`0.98` default; upper bound to reject near-fully-empty virtual renders)
- `VOID_FILL_MAX_REPAIR_PER_ROUND` (`24` default; top-N bounded ranking cap for repaired virtual renders)
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
- `BLUR_FILTER_KEEP_RATIO` (profile defaults: `0.85` for `quality_first`, `0.90` for `balanced`, `1.0` for `fast`; set `1.0` to disable)
- `BLUR_FILTER_MIN_FRAMES` (`120` default; safety floor when blur filtering is enabled)
- `NUREC_RESUME` (`false` default in quality-first wrapper profile)
- `NUREC_PARALLEL_POST_STAGE6` (`true` default; runs Stage 7 visual mesh and Stage 9 SAM3 concurrently)
- `NUREC_DEPENDENCY_PREFLIGHT` (`true` default; fail fast before COLMAP if 3DGRUT deps are missing)
- `NUREC_PREFLIGHT_CHECK_FUSED_SSIM` (`true` default; checks fused_ssim import/torch ABI during preflight)
- `SAM3_PREFLIGHT_STRICT` (`false` default; if true, fail before reconstruction when SAM3 access is unavailable)
- `SAM3_N_FRAMES` (`0` default = auto-scaled by capture length)
- `SAM3_MIN_FRAME_DETECTIONS` (`0` default = env-aware auto)
- `PIPELINE_MODE` (`full` default; `photorealistic_scene` for baseline-clear 3DGRUT-first output, `photoreal_hallucination` for clarity-first high-capacity baseline + forced synthetic repair)
- `SCENE_CLEANING_MODE` (`off` default; `auto` best-effort candidate-scoped cleaning, `force` hard-fails when prerequisites/cleaning fail)
- `SAM3_MASK_EXPORT_SPACE` (`undistorted` default when cleaning is enabled; `raw` or `undistorted`)
- `RECONSTRUCTION_BACKEND` (`nurec_3dgrut` default; supports `ttt_lrm`, `loger`, `neoverse`, `gen3c`)
- `RECONSTRUCTION_COMPARE_BACKENDS` (optional comma-separated candidate backends for A/B)
- `RECONSTRUCTION_COMPARE_WINNER` (`auto` or backend name; selects winner output when comparing)
- `RECONSTRUCTION_COMPARE_REPORT` (path to backend comparison JSON report)
- `TTT_LRM_CMD_TEMPLATE` (command template for experimental tttLRM; placeholders: `INPUT_VIDEO`, `OUTPUT_DIR`, `SCENE_ID`, `CAPTURE_ID`, `JOB_SPEC_PATH`)
- `TTT_LRM_EXECUTABLE` (fallback executable for tttLRM; receives `--input-video` and `--output-dir`)
- `WORLD_MODEL_SERVICE_URL` / `WORLD_MODEL_SERVICE_API_KEY` (shared fallback for remote Stage 1 world-model backends)
- `WORLD_MODEL_SERVICE_TIMEOUT_SECONDS` / `WORLD_MODEL_SERVICE_POLL_SECONDS` (remote NeoVerse / GEN3C orchestration tuning)
- `NEOVERSE_SERVICE_URL` / `NEOVERSE_SERVICE_API_KEY` (optional overrides for NeoVerse Stage 1 remote execution)
- `GEN3C_SERVICE_URL` / `GEN3C_SERVICE_API_KEY` (optional overrides for GEN3C Stage 1 remote execution)
- `RECONSTRUCTION_ARKIT_POSES_PATH`, `RECONSTRUCTION_ARKIT_INTRINSICS_PATH`, `RECONSTRUCTION_ARKIT_DEPTH_DIR` (required for `gen3c` unless advanced geometry is supplied)
- `RECONSTRUCTION_ARKIT_CONFIDENCE_DIR` (optional extra conditioning metadata)
- `RECONSTRUCTION_SCENE_MEMORY_BUNDLE_PATH` / `RECONSTRUCTION_ADVANCED_GEOMETRY_BUNDLE_PATH` (optional canonical bundle inputs for remote backends)
- `INPAINT360GS_DIR` (default: `/opt/Inpaint360GS`)
- `INPAINT360GS_PYTHON` (default: `python3.10`)
- `INPAINT360GS_RESOLUTION` (`2` default; `1`=full, `2`=half, `4`=quarter)
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
- `NUREC_RERUN_PROFILE` (`default` wrapper profile; `clear_over_faithful` applies high-capacity baseline-only settings, `photoreal_hallucination` applies baseline + aggressive synthetic repair settings)

`scripts/run_full_pipeline.sh` behavior note:
- In `best_effort` completion mode, if swap-orchestrator dependencies are missing (BlueprintPipeline runtime or required provider credentials), Phase 3 is skipped or soft-failed and the script still completes with NuRec outputs for visual QA (`orchestrator_run_report.json` records the fallback state).
- Stage 1 now routes reconstruction through `scripts/reconstruction_backend_router.py`.
- Use `--reconstruction-backend nurec_3dgrut|ttt_lrm|loger|neoverse|gen3c`, plus optional:
  - `--reconstruction-compare-backends nurec_3dgrut,neoverse`
  - `--reconstruction-compare-winner auto|ttt_lrm|nurec_3dgrut|neoverse|gen3c`
  - `--reconstruction-compare-report /tmp/compare_report.json`
- The wrapper now writes the canonical Stage 1 job spec through `scripts/write_reconstruction_job_spec.py`; remote backends consume that spec unchanged.
- `neoverse` is the preferred first remote Stage 1 world-model backend. It can operate from video-only input and should be rolled out behind compare mode first.
- `gen3c` is the stricter second backend. It requires ARKit poses + intrinsics + depth, or an advanced geometry bundle that the remote service accepts as equivalent.
- Winner metadata is written to `reconstruction_backend_meta.json` and comparison details to the configured report path.
- Cosmos remains Stage 5 `Fixer` infrastructure in this phase. It is not the Stage 1 world-model backend path here.
- The old execution gap was broader than the router alone: Stage 1 shell guardrails, runtime preflight, remote service runners, backend normalization adapters, and docs/tests now all participate in backend support.

### Run Operations Toolkit

- Each wrapper run now writes:
  - `full_pipeline/run_summary.json` + `full_pipeline/run_summary.md` (`inputs`, `commit`, `params`, `outputs`, `runtime`, `failures`)
  - `full_pipeline/log_summary.json` + `full_pipeline/log_summary.md` (parsed stage timings/errors from pipeline logs)
- Log parser can be run manually:

```bash
python3 scripts/summarize_pipeline_logs.py --pipeline-dir /path/to/full_pipeline
```

- Fast smoke wiring check (tiny synthetic fixture, no expensive NuRec/orchestrator execution):

```bash
bash scripts/run_pipeline_smoke.sh
```

- Supporting docs:
  - `docs/templates/RUN_SUMMARY_TEMPLATE.md`
  - `docs/PIPELINE_EXPERIMENT_MATRIX.md`
  - `docs/PIPELINE_FAILURE_RUNBOOK.md`
  - `docs/OUTPUT_VALIDATION_CHECKLIST.md`

For production runtimes, pre-bake 3DGRUT build dependencies into the image (tiny-cuda-nn submodules and fused_ssim built against the image's torch) to avoid rebuild delays during retries.

### Scene Cleaning Runbook

- Stage location: candidate-scoped scene cleaning runs in swap orchestrator Stage C.5 (after candidate selection, before materialization).
- Enable modes:
  - `SCENE_CLEANING_MODE=off`: disabled (default).
  - `SCENE_CLEANING_MODE=auto`: run best-effort; explicit skip reason is written when prerequisites/deps are missing.
  - `SCENE_CLEANING_MODE=force`: hard fail on missing prerequisites or cleaner failure.
- Expected additional runtime: typically ~25-40 minutes/scene on RTX 4090-class GPUs (resolution-dependent).
- NuRec prerequisite artifacts (when cleaning mode is not `off`):
  - `instance_masks/` (uint16 SAM3 instance masks in requested export space)
  - `colmap_undistorted/images/`
  - `colmap_undistorted/sparse/0/`
- Orchestrator outputs:
  - `pipeline/scene_cleaning_report.json` (status/reason/targets)
  - optional `nurec/inpainted_visual_mesh.glb` (when cleaning succeeds)
  - `nurec_outputs.artifacts.inpainted_visual_mesh_glb` is injected when available

## Omniverse Preview (Advanced Geometry Compatibility)

These helpers are optional compatibility tooling for the explicit advanced-geometry lane. The default modern handoff from this repo is scene-memory plus preview-prep artifacts, not USD-first viewer export.

Primary visual asset routing is configured by `NUREC_VISUAL_PRIMARY` when advanced geometry is present:
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
1. resolved advanced-geometry visual asset,
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
- `TEXT_ASSET_GENERATION_PROVIDER_CHAIN` (default: `sam3d,hunyuan3d`; supports `ttt_lrm` as experimental provider)
- `TEXT_SAM3D_API_HOST` + `TEXT_SAM3D_API_KEY` (or `SAM3D_API_HOST` + `SAM3D_API_KEY`)
- `TEXT_HUNYUAN_API_HOST` + `TEXT_HUNYUAN_API_KEY` (or `HUNYUAN_API_HOST` + `HUNYUAN_API_KEY`)
- `STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND` (or `STAGE_D_TTT_LRM_IMAGE_TO_3D_COMMAND`; preferred for local image-conditioned `ttt_lrm` execution)
- `TEXT_TTTLRM_API_HOST` + `TEXT_TTTLRM_API_KEY` (or `TEXT_TTT_LRM_API_HOST` + `TEXT_TTT_LRM_API_KEY`; for remote/API `ttt_lrm` setups)

Concrete `ttt_lrm` command templates:

```bash
# Stage D provider command used by the swap pipeline
export STAGE_D_TTTLRM_IMAGE_TO_3D_COMMAND="bash /app/scripts/run_ttt_lrm_stage_d.sh \
  {REFERENCE_IMAGE} {OUTPUT_GLB} {OUTPUT_DIR} {SCENE_ID} {OBJECT_ID} {ROOM_TYPE}"

# Stage 1 reconstruction backend command used by reconstruction_backend_router.py
export TTT_LRM_CMD_TEMPLATE="bash /app/scripts/run_ttt_lrm_reconstruction.sh \
  {INPUT_VIDEO} {OUTPUT_DIR} {SCENE_ID} {CAPTURE_ID} {JOB_SPEC_PATH}"

# Local runtime binary mode (preferred)
export TTT_LRM_STAGE_D_BIN="/opt/tttLRM/bin/tttlrm_stage_d"
export TTT_LRM_RECON_BIN="/opt/tttLRM/bin/tttlrm_reconstruct"

# Container runtime mode (if binaries are only available in a GPU container)
export TTT_LRM_STAGE_D_CONTAINER_IMAGE="nijelhunt/blueprint-capture-pipeline:latest"
export TTT_LRM_STAGE_D_CONTAINER_BIN="/opt/tttLRM/bin/tttlrm_stage_d"
export TTT_LRM_RECON_CONTAINER_IMAGE="nijelhunt/blueprint-capture-pipeline:latest"
export TTT_LRM_RECON_CONTAINER_BIN="/opt/tttLRM/bin/tttlrm_reconstruct"
```

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
11. Completion writes `qualification_quality_report.json`, `pipeline_summary.json`, and `.qualification_pipeline_complete` plus temporary `swap_*` aliases for compatibility.

## Tests

```bash
pytest -q
```
