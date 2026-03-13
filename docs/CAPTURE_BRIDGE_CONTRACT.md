# Capture Bridge Contract

This document defines the kept product boundary for `BlueprintCapturePipeline`.

## Input Trigger

Raw-upload materialization trigger:

`scenes/<scene_id>/captures/<capture_id>/raw/capture_upload_complete.json`

Descriptor trigger:

`scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json`

## Primary Outputs

All pipeline artifacts are emitted under:

`scenes/<scene_id>/captures/<capture_id>/pipeline/`

Retained output families:

- `scene_memory/scene_memory_manifest.json`
- `scene_memory/scene_memory_readiness.json`
- `scene_memory/conditioning_bundle.json`
- `scene_memory/adapter_manifests/gen3c.json`
- `scene_memory/adapter_manifests/neoverse.json`
- `scene_memory/adapter_manifests/cosmos_transfer.json`
- `presentation_world/presentation_world_manifest.json`
- `presentation_world/runtime_demo_manifest.json`
- `evaluation_prep/scene_memory_bundle_manifest.json`
- `evaluation_prep/site_world_spec.json`
- `evaluation_prep/site_world_registration.json`
- `evaluation_prep/site_world_health.json`
- `evaluation_prep/evaluation_prep_manifest.json`

Qualification analysis remains available and may emit:

- `opportunity_handoff.json`
- `task_scope_record.json`
- `task_targets.json`
- `agent_review_bundle.json`

These qualification artifacts support review and agent analysis. The canonical downstream runtime boundary remains the built site-world package plus presentation-world artifacts.

## Default Flow

The supported modern flow is:

`raw capture -> qualification analysis -> scene_memory -> presentation_world -> evaluation_prep -> site_world runtime`

## Shared Contracts

Shared validation/versioning logic lives in `BlueprintContracts`:

- `handoff_contract`
- `site_world_contract`
- `runtime_layer_contract`
- `canonical_package`
