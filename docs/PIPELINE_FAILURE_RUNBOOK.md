# Pipeline Failure Runbook

Fast recovery guide for `scripts/run_full_pipeline.sh`.

## Stage 1: Reconstruction Backend
- Failure Signal: missing required NuRec outputs (`export_last.usdz`, `visual_mesh.glb`, `nvblox_mesh.ply`, `mesh_manifest.json`, `occupancy.bin`, `object_point_cloud_index.json`).
- Action:
  - Inspect `full_pipeline/nurec.log`.
  - Confirm `scripts/reconstruction_backend_router.py` exists.
  - If `RECONSTRUCTION_BACKEND=ttt_lrm`, confirm either `TTT_LRM_CMD_TEMPLATE` or `TTT_LRM_EXECUTABLE` is set and valid.
  - Re-run with `--resume` if Stage 1-4 assets already exist.
  - If refinement stack caused failure, rerun with `POST_STAGE4_REFINE=off` once to isolate baseline health.
  - If using compare mode, inspect `reconstruction_compare_report.json` and `reconstruction_backend_meta.json`.

## Stage 2: Directory/Contract Assembly
- Failure Signal: `capture_descriptor.json`/`.nurec_complete` missing or malformed.
- Action:
  - Verify `object_point_cloud_index.json` parses and contains an `objects` array (or empty list).
  - Check free disk space in `WORKSPACE` and `GCS_ROOT`.
  - Re-run smoke wiring: `bash /Users/nijelhunt_1/workspace/BlueprintCapturePipeline/scripts/run_pipeline_smoke.sh`.

## Stage 3: Swap Orchestrator
- Failure Signal: orchestrator non-zero exit or `ORCHESTRATOR_STATUS` is `skipped_missing_dependencies`/`failed_best_effort`.
- Action:
  - Open `full_pipeline/orchestrator.log` and `full_pipeline/orchestrator_run_report.json`.
  - For missing provider credentials, set the required host/key pairs before rerun.
  - In strict mode, confirm BlueprintPipeline runtime scripts exist under `BLUEPRINTPIPELINE_ROOT`.

## Quality Gate Failure
- Failure Signal: `swap_quality_report.json` has status other than `passed`.
- Action:
  - Review failing gates in `swap_quality_report.json`.
  - Check `runtime_preflight_report.json` for missing dependencies or unsafe overrides.
  - If mesh quality gate fails after Delaunay fallback, inspect `mesh_method.txt` and registration ratios in `capture_quality_report.json`.

## Full-Required Assembly Failure
- Failure Signal: wrapper exits with `scene.usda is a standalone stub`.
- Action:
  - Confirm downstream jobs exist in `BLUEPRINTPIPELINE_ROOT`:
    - `replicator-job/generate_replicator_bundle.py`
    - `variation-asset-pipeline-job/run_variation_asset_pipeline.py`
    - `genie-sim-export-job/export_to_geniesim.py`
  - Ensure strict mode is not forcing standalone fallback (`PIPELINE_STANDALONE_MODE` must be `false`).

## Standard Triage Sequence
1. Read `full_pipeline/run_summary.json` for params/runtime/failure context.
2. Read `full_pipeline/log_summary.md` for condensed error lines and timings.
3. Resolve first hard failure, then re-run with the same input and only one parameter change.
