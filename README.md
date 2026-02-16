# BlueprintCapturePipeline

NuRec-first orchestration for converting BlueprintCapture descriptors into sim-ready scenes with swappable assets.

## Scope

This repo is intentionally thin. It orchestrates one path:

`capture_descriptor.json` -> NuRec reconstruction -> swap candidate policy -> SAM3D-first asset materialization -> interactive articulation validation/fallback -> simready + USD assembly.

It reuses existing jobs and helper logic from:

- `/Users/nijelhunt_1/workspace/BlueprintPipeline`

## Entry Point

Run directly:

```bash
python -m blueprint_pipeline.swap_orchestrator \
  --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json
```

Or via installed script:

```bash
blueprint-capture-swap --descriptor-gcs-uri gs://<bucket>/scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json
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
- `swap_quality_report.json`
- `.swap_pipeline_complete` or `.swap_pipeline_failed.json`

Scene artifacts written under:

- `scenes/<scene_id>/assets/scene_manifest.json`
- `scenes/<scene_id>/layout/scene_layout_scaled.json`
- `scenes/<scene_id>/seg/inventory.json`

## Environment

- `GCS_ROOT` (default: `/mnt/gcs`)
- `BLUEPRINTPIPELINE_ROOT` (default: `/Users/nijelhunt_1/workspace/BlueprintPipeline`)
- `NUREC_WORKER_COMMAND` (optional command template; receives `{JOB_SPEC_PATH}`)
- `BLUEPRINTPIPELINE_COMMIT_HASH` (optional pin)
- `FAIL_ON_BLUEPRINTPIPELINE_COMMIT_MISMATCH` (`true` by default)
- `TEXT_ASSET_GENERATION_PROVIDER_CHAIN` (default: `sam3d,hunyuan3d`)

## Tests

```bash
pytest -q
```
