# Capture Bridge Contract (NuRec-First Swap)

This document defines the descriptor + orchestration contract consumed by `BlueprintCapturePipeline`.

## 1) Input Trigger

Object finalize trigger:

`scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json`

## 2) Descriptor Schema (`capture_descriptor.json`)

Required:

- `schema_version` (`v1`)
- `scene_id`
- `capture_id`
- `raw_prefix_uri`
- `frames_index_uri`

Common optional fields:

- `nurec_mode`
- `swap_focus`
- `quality`
- `qa_report_uri`
- `manipulation_candidates`
- `articulation_hints`

Supported aliases:

- `intended_space_type` -> `environment_type_hint`
- `capture_bundle.arkit_poses_uri` -> `arkit_poses_uri`
- `capture_bundle.arkit_intrinsics_uri` -> `arkit_intrinsics_uri`

## 3) Intake Requirements

- QA report must exist and have `status = passed`.
- Raw manifest must exist at `<raw_prefix_uri>/manifest.json`.
- Object index must resolve from `manifest.object_point_cloud_index`.

## 4) Pipeline Outputs

All orchestration artifacts are emitted under:

`scenes/<scene_id>/captures/<capture_id>/pipeline/`

- `nurec_job_spec.json` (v1)
- `nurec_outputs.json` (v1)
- `swap_candidates.json` (v1)
- `swap_execution_report.json` (v1)
- `swap_quality_report.json` (v1)
- `.swap_pipeline_complete` or `.swap_pipeline_failed.json`

## 5) Downstream Scene Artifacts

- `scenes/<scene_id>/assets/scene_manifest.json`
- `scenes/<scene_id>/layout/scene_layout_scaled.json`
- `scenes/<scene_id>/seg/inventory.json`

These are authored for direct compatibility with BlueprintPipeline interactive/simready/usd-assembly jobs.
