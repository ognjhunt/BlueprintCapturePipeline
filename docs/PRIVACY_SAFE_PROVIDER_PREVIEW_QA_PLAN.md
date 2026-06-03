# Privacy-Safe Provider Preview QA Plan

This plan keeps provider preview QA local, deterministic, and fail-closed until
an operator explicitly authorizes live provider execution.

No step in this plan runs World Labs, uploads raw media, reads production
secrets, mutates WebApp production state, or claims live runtime readiness.

## Scope

The QA target is the current preview path:

```text
raw capture -> privacy processing -> privacy/final_walkthrough.* ->
worldlabs_input -> canonical site package -> World Labs request manifest ->
provider preview status -> WebApp sync projection
```

The validator described here proves artifact shape, privacy-safe input lineage,
raw-bypass blocking, and handoff labels. It does not prove provider execution,
hosted session fulfillment, rights clearance, payments, payouts, or operational
launch readiness.

## Required Artifacts

For a provider preview to be eligible for production provider submission, the
capture root must contain:

- `privacy/final_walkthrough.mov` or `privacy/final_walkthrough.mp4`
- `pipeline/privacy_processing_manifest.json`
- `pipeline/privacy_verification_report.json`
- `pipeline/worldlabs_input/worldlabs_input_manifest.json`
- `pipeline/worldlabs_input_audit.json`
- `pipeline/site_package/canonical_site_package.json`
- `pipeline/site_package/provider_adapter_inputs/world_labs_marble.json`
- `pipeline/worldlabs_request_manifest.json`
- `pipeline/provider_preview_status.json`
- `pipeline/provider_run_manifest.json`
- `pipeline/preview_manifest.json`
- `pipeline/webapp_sync_result.json` when WebApp projection is required

When non-ARKit depth conditioning is used, the same packet must also include:

- `pipeline/privacy_depth/depth_manifest.json`
- `pipeline/privacy_depth/confidence_manifest.json`

When geometry is referenced by any preview or WebApp projection, include:

- `pipeline/geometry/geometry_summary.json`
- `pipeline/geometry/logs/provider_result.json`

## Privacy And Redaction Proof

The privacy proof must come from deterministic artifacts, not reviewer memory.

Minimum checks:

- `privacy_processing_manifest.json.fail_closed == true`
- `privacy_processed_video_uri` and `world_model_video_uri` resolve to
  `privacy/final_walkthrough.*`
- `privacy_verification_report.json.status` matches the privacy manifest status
- SAM3 initial detection completed or the privacy path failed closed
- depth conditioning is present as `arkit` or `depth_anything`
- if no people were detected, the final walkthrough was emitted from the
  no-people path
- if people were detected, VIP removal must be followed by SAM3 verification
- if DeepPrivacy2 fallback is used, `face_anonymized_segments` must be non-empty
  and the fallback verification must be recorded

Blocked privacy states:

- privacy pipeline disabled in a production proof context
- missing `privacy/final_walkthrough.*`
- missing privacy verification report
- SAM3, Depth Anything, VIP, or DeepPrivacy2 failure when fail-closed is required
- privacy manifest status `failed_closed` used as preview-ready proof
- privacy processing completion used as rights or commercialization clearance

## Raw-Path Blocking

Raw walkthrough input is never production provider proof.

The validator must block provider-ready status when any of these are true:

- `BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS=true` in production mode
- `worldlabs_input_manifest.json.selected_video_source_id == "raw_video_uri"`
- `worldlabs_input_audit.json.raw_video_bypass_used == true`
- `worldlabs_request_manifest.json.input_labeling.raw_video_bypass_used == true`
- `worldlabs_request_manifest.json.privacy_safe_input != true`
- `worldlabs_request_manifest.json.worldlabs_input_audit_uri` is missing
- the audit `output_video_uri` does not match the request manifest
  `selected_video_uri`
- the selected input is not `privacy/final_walkthrough.*` or a verified derivative
- WebApp sync artifacts expose raw walkthrough media as buyer media or
  `world_model_video_uri`

Internal demos may keep the raw bypass path only when the output is explicitly
labeled:

- `review_state = "non_production_unredacted_raw_preview"`
- `non_production = true`
- `unredacted_input = true`
- `privacy_safe_input = false`

That label is a blocker for production launch, not an exception.

## Preview Labels

A production-eligible preview packet must carry:

- `review_state = "standard_privacy_safe_preview"`
- `privacy_safe_input = true`
- `raw_video_bypass_used = false`
- `non_production` absent or false
- `unredacted_input` absent or false
- `provider_preview_status.labeling` consistent with the World Labs input audit

Provider output remains generated preview state. A successful
`provider_preview_status.status` does not prove raw capture truth, rights
clearance, hosted access, live runtime readiness, or operational launch
readiness.

## Deterministic Manifest Validator

Add a future local validator as:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_provider_preview_packet.py \
  --capture-root /path/to/scenes/<scene_id>/captures/<capture_id> \
  --mode production
```

The validator should write:

```text
pipeline/provider_preview_qa_manifest.json
```

Required output fields:

- `schema_version`
- `scene_id`
- `capture_id`
- `mode`
- `status`: `passed`, `blocked`, or `review_required`
- `claim_ceiling`: `local_repo_proof` or `provider_proof_pending`
- `privacy_artifacts`
- `redaction_proof`
- `worldlabs_input_lineage`
- `raw_path_policy`
- `canonical_package_match`
- `provider_adapter_match`
- `request_manifest_validation`
- `geometry_labels`
- `webapp_sync_projection`
- `blocked_claims`
- `next_required_live_gates`

Required deterministic checks:

1. Privacy manifest and verification report agree on a completed safe path.
2. `worldlabs_input_audit.output_video_uri` matches
   `worldlabs_request_manifest.selected_video_uri`.
3. Request manifest checksums match the audit payload when present.
4. Canonical package `conditioning.rgb_video.privacy_safe_world_model_input.uri`
   matches the World Labs selected input.
5. Provider adapter input `conditioning_inputs.rgb_video.uri` matches the
   canonical package and request manifest.
6. Any raw bypass label forces `status = blocked` in production mode.
7. Fallback or local geometry labels never set live world-model readiness.
8. WebApp sync is blocked unless upstream ids are real and non-placeholder when
   sync is required.
9. Missing provider operation/world manifests keep live provider proof pending.
10. Missing hosted/runtime/access evidence keeps hosted proof pending.

The validator must exit non-zero for `blocked` in production mode and zero for
`blocked` in advisory mode only when the output manifest clearly names the
blocked claims.

## Agent Review Role

Agent review is advisory and must run after the deterministic validator.

Allowed agent work:

- summarize privacy and provider-preview blockers
- compare artifact URIs across the manifest, canonical package, adapter input,
  provider run manifest, and WebApp sync payload
- draft a human-readable review memo for `rights-provenance-agent`,
  `pipeline-codex`, or `webapp-codex`
- identify missing owner-system proof and the next safe command

Disallowed agent work:

- overriding validator failures
- approving raw bypass for production
- treating privacy completion as rights clearance
- treating request manifests as provider execution proof
- treating WebApp sync or public UI polish as hosted-session fulfillment
- calling providers, uploading raw media, or reading production secrets

## Live Provider Gates

World Labs submission remains blocked until all of these are true:

- deterministic validator status is `passed`
- `BLUEPRINT_ALLOW_RAW_WORLDLABS_BYPASS` is false or unset
- `worldlabs_request_manifest.json.privacy_safe_input == true`
- input audit URI, output URI, and checksums are present
- explicit operator approval exists for the provider job
- provider key/token is supplied through approved secret handling and is not
  logged

After a live provider job is approved and executed, provider proof requires:

- provider name and model
- provider run or operation id
- selected input URI and checksum
- output artifact URI and checksum when available
- operation/world manifest
- terminal status and failure reason, if any
- cost and latency when returned by the provider

Geometry proof remains separate. `world_model_ready_claim_allowed` requires
`geometry_source = "video_to_world"`, `fallback_used = false`,
`provider_native_result = true`, `ready_for_world_model = true`, and
`geometry_live_ready = true`.

Hosted proof remains separate. It requires WebApp attachment sync with real
upstream ids, resolvable artifacts, runtime handle/health, entitlement or access
proof, and at least one recorded session state/read/render/export artifact.

## Safe Local Verification Queue

These checks are local fixture checks and do not run live providers:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest \
  tests/test_privacy_processing.py \
  tests/test_privacy_runner_service.py \
  tests/test_qualification_alpha.py \
  tests/test_launch_bundle.py \
  tests/test_webapp_sync.py \
  tests/test_geometry_stage.py \
  tests/test_retrieval_index_geometry_source.py -q
```

Do not run `BLUEPRINT_PREVIEW_PROVIDER=world_labs ...`, privacy/GPU runner live
commands, deploy scripts, Terraform, WebApp production sync, or any command that
uses production secrets unless a separate issue explicitly authorizes that live
side effect.

## Acceptance Criteria

This plan is complete when the future validator can:

- pass a privacy-safe fixture that routes World Labs input from
  `privacy/final_walkthrough.*`
- block a production fixture with raw World Labs bypass enabled
- block a fixture with missing privacy verification
- block fallback geometry from live world-model readiness
- block WebApp sync projection when upstream ids are generated or placeholder
- emit a manifest that names the exact missing owner-system proof
- keep claim ceilings separate for local repo proof, provider proof, hosted
  proof, rights/privacy proof, human-gated proof, and operational launch proof
