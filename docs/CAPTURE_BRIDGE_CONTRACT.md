# Capture Bridge Contract

This document defines the kept product boundary for `BlueprintCapturePipeline`.

`BlueprintCapturePipeline` is capture-first and world-model-product-first. It
preserves raw capture truth, emits canonical site packages and hosted/provider
support artifacts, and keeps qualification/readiness outputs as compatibility
and trust layers rather than the product center.

## Input Trigger

Raw-upload materialization trigger:

`scenes/<scene_id>/captures/<capture_id>/raw/capture_upload_complete.json`

Descriptor trigger:

`scenes/<scene_id>/captures/<capture_id>/capture_descriptor.json`

## Output Families

All pipeline artifacts are emitted under:

`scenes/<scene_id>/captures/<capture_id>/pipeline/`

Canonical package outputs:

- `site_package/canonical_site_package.json`
- `site_package/provider_adapter_inputs/world_labs_marble.json`

These are the current provider-agnostic product package and provider-specific
adapter projection. The canonical package is derived from raw capture and
descriptor truth; it must not invent or overwrite raw capture evidence.

Support / trust outputs:

- `qualification_summary.json`
- `capture_quality_summary.json`
- `rights_and_compliance_summary.json`
- `buyer_trust_score.json`
- `world_model_fit_summary.json`
- `capturer_payout_recommendation.json`
- `recapture_requirements.json`
- `provider_preview_status.json`
- `provenance_summary.json`
- `gemini_capture_fidelity_review.json`
- `opportunity_handoff.json`
- `task_scope_record.json`
- `task_targets.json`
- `agent_review_bundle.json`

These artifacts are consumed by `Blueprint-WebApp` for trust, review, pricing,
readiness, and compatibility with existing `qualification` naming. They are
support records, not the product center. They must stay grounded in raw capture
truth and canonical package truth.

Downstream derived output families:

- `scene_memory/scene_memory_manifest.json`
- `scene_memory/scene_memory_readiness.json`
- `scene_memory/conditioning_bundle.json`
- `presentation_world/presentation_bundle.json`
- `presentation_world/presentation_world_manifest.json`
- `presentation_world/runtime_demo_manifest.json`
- `presentation_demo_preflight_report.json`
- `evaluation_prep/scene_memory_bundle_manifest.json`
- `evaluation_prep/site_world_spec.json`
- `evaluation_prep/site_world_registration.json`
- `evaluation_prep/site_world_health.json`
- `evaluation_prep/evaluation_prep_manifest.json`

Downstream artifacts are derived only. They do not define raw capture success,
and they do not rewrite capture, rights, privacy, provenance, or canonical
package truth. They are not required for `preview_simulation` success, which
currently means the World Labs preview path only.

## Presentation Bundle Contract

`presentation_bundle.json` is the concrete non-authoritative presentation artifact. It must remain grounded in canonical and scene-memory inputs and carries:

- canonical source linkage
- derivation and hallucination policy
- capture orientation
- camera behavior for interactive pose-driven rendering
- renderable input references for downstream consumers

`presentation_world_manifest.json` is the family index and `runtime_demo_manifest.json` is the interactive demo contract.

`runtime_demo_manifest.json` is only stage-6 ready when `interactive_demo.readiness_state == "ready"` and it includes a truthful demo UI endpoint:

- `ui_base_url`
- or `public_ui_base_url`

Without one of those fields, the manifest remains a valid bundle-backed contract, but WebApp presentation-demo launch should be treated as blocked rather than launchable.

## Orientation Preservation

Capture orientation is preserved from raw capture metadata when available, then from video probe metadata, and finally from encoded dimensions as a last resort. The resolved `capture_orientation` contract is propagated through scene-memory, presentation-world, and evaluation-prep outputs so portrait captures are not silently collapsed into landscape semantics.

## Default Flow

The supported modern package flow is:

`BlueprintCapture raw bundle -> materialized descriptor -> optional trust/qualification analysis -> privacy-safe walkthrough -> canonical site package -> provider adapter input -> optional World Labs preview -> WebApp sync`

Optional follow-on lanes:

`canonical site package -> scene_memory -> presentation_world -> evaluation_prep -> site_world runtime`

## Shared Contracts

Shared validation/versioning logic lives in `BlueprintContracts`:

- `handoff_contract`
- `site_world_contract`
- `runtime_layer_contract`
- `canonical_package`
