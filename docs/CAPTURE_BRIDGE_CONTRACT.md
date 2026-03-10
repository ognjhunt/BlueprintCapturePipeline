# Capture Bridge Contract (Qualification-First)

This document defines the descriptor + orchestration contract consumed by `BlueprintCapturePipeline`.

## 1) Input Trigger

Object finalize trigger:

`scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json`

Raw-upload materialization trigger:

`scenes/<scene_id>/captures/<capture_id>/raw/capture_upload_complete.json`

The storage trigger is enqueue-first (async). Dispatch backends:

- Pub/Sub (`SWAP_TRIGGER_DISPATCH_MODE=pubsub`)
- Cloud Tasks (`SWAP_TRIGGER_DISPATCH_MODE=cloud_tasks`)

## 2) Descriptor Schema (`capture_descriptor.json`)

Required:

- `schema_version` (`v1`)
- `scene_id`
- `capture_id`
- `raw_prefix_uri`
- `frames_index_uri`

Common optional fields:

- `nurec_mode`
- `requested_lanes`
- `swap_focus`
- `quality`
- `qa_report_uri`
- `capture_modality`
- `scaffolding_used`
- `intake_packet_uri`
- `coverage_plan`
- `calibration_assets`
- `uncertainty_priors`
- `manipulation_candidates`
- `articulation_hints`

Supported aliases:

- `intended_space_type` -> `environment_type_hint`
- `capture_bundle.arkit_poses_uri` -> `arkit_poses_uri`
- `capture_bundle.arkit_intrinsics_uri` -> `arkit_intrinsics_uri`

## 3) Intake Requirements

Qualification lane:

- descriptor must parse successfully
- QA report, raw manifest, and object index are inspected when present
- missing or failed evidence produces qualification outputs with `need_more_evidence` instead of forcing geometry generation

Advanced geometry lane:

- QA report must exist and have `status = passed`
- raw manifest must exist at `<raw_prefix_uri>/manifest.json`
- object index must resolve from `manifest.object_point_cloud_index`

## 4) Pipeline Outputs

All orchestration artifacts are emitted under:

`scenes/<scene_id>/captures/<capture_id>/pipeline/`

Default qualification artifacts:

- `site_intake.json` (v1)
- `capture_package_manifest.json` (v1)
- `capture_qa_scorecard.json` (v1)
- `task_scope_record.json` (v1)
- `qualification_record.json` (v1)
- `qualification_brief.json` (v1)
- `opportunity_handoff.json` (v1)
- `task_targets.json` (v1)
- `runtime_preflight_report.json` (v1)
- `swap_quality_report.json` (compatibility alias, lane=`qualification`)
- `.swap_pipeline_complete` or `.swap_pipeline_failed.json`

Advanced geometry artifacts when explicitly requested:

- `nurec_job_spec.json` (v1)
- `nurec_outputs.json` (v1)
- `swap_candidates.json` (v1)
- `swap_execution_report.json` (v1)
- `advanced_quality_report.json` (v1)
- `advanced_geometry/advanced_geometry_bundle.json` (v1)
- `advanced_geometry/labels.json` (v1)
- `advanced_geometry/structure.json` (v1)
- `advanced_geometry/task_targets.synthetic.json` (v1)
- optional `advanced_geometry/3dgs_compressed.ply`

## 5) Downstream Scene Artifacts

- `scenes/<scene_id>/assets/scene_manifest.json`
- `scenes/<scene_id>/layout/scene_layout_scaled.json`
- `scenes/<scene_id>/seg/inventory.json`

These are authored only by the explicit advanced geometry lane for direct compatibility with BlueprintPipeline interactive/simready/usd-assembly jobs.
