# Capture Bridge Contract

This document defines the kept product boundary for `BlueprintCapturePipeline`.

`BlueprintCapturePipeline` is capture-first and real-site robot-evaluation/data-package first. It
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

## Frames Sub-Layout (`frames_index.v1` / `frames_index.v2`)

Extracted frames live under:

`scenes/<scene_id>/captures/<capture_id>/frames/`

with `frames/index.jsonl` carrying one entry per frame (`frame_id`,
`t_video_sec`, optional `arkit_pose` / `arkit_frame` linkage) in both
versions. The keyframe object and the `capture_bridge_handoff.v1` message
shape are identical in both versions; this revision is scoped to the frames
sub-layout only.

**v1 (legacy, per-object).** One GCS object per extracted JPEG
(`frames/000001.jpg`, …). Implied by the absence of a
`frames/packing_manifest.json` and of per-entry `archive` fields.

**v2 (packed, SCALE2-03).** Frames are packed into deterministic USTAR
archives `frames/frames_NNN.tar` (default 200 frames per archive; ~900
objects per 3-minute capture become ~6). The layout is declared explicitly:

- `frames/packing_manifest.json`:
  `{ "schema_version": "frames_index.v2", "packaging": "tar",
     "frames_per_archive": N, "frame_count": M,
     "archives": [{ "archive": "frames_000.tar", "member_count": … }] }`
- each `index.jsonl` entry additionally carries
  `"packaging": "tar"`, `"archive": "frames_NNN.tar"`,
  `"archive_member": "<frame_id>.jpg"`.

Readers must go through `blueprint_pipeline.frames_layout`
(`load_frames_layout` / `iter_frame_payloads`), which supports both versions
transparently and fails closed on unknown `schema_version` / `packaging`
values or packed entries missing archive linkage.

Producer rollout: native and legacy captures emit v2 when
`BLUEPRINT_EXTRACT_FRAMES_PACKING_ENABLED=1`. Validated selected browser
deliveries use the same packer without changing that global flag, retaining
every extracted frame within the existing finite membership row bound.
Legacy v1 captures remain readable.

### Selected browser delivery inputs

A validated website producer receipt binds the original video, manifest and
completion marker generations. Capture publishes immutable derivatives under
`deliveries/<semantic-delivery-key>/`, including the frame index, packing
manifest, archives, descriptor and QA. Its handoff URI is accepted only when
that exact key matches the validated source-finalize identity and membership
selector. Native/app envelopes retain canonical URIs without selected-browser
authority. No URI pattern or historical receipt grants current consent.

Pipeline stages only the finite generation-pinned membership. Existing readers
consume selected derivatives in place through verified byte snapshots and
refuse missing, changed or cross-delivery members. A saved frame layout stays
bound to its selection across later reads. The validated browser descriptor
records `metadata.capture_entry_source=browser_self_capture`, selecting the
existing website task-context and current-consent preparation branch.

Consent, withdrawal, raw input and lane-resume roots remain the canonical
capture root. Qualification enrichment writes the existing pipeline-owned
canonical descriptor; it preserves the immutable producer descriptor in the
selected delivery. Retained birth evidence is historical provenance, never
execution, deletion, billing or payout permission.

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

## Upload hand-off: what the repositories say and what production does (2026-11-11)

Every website upload's completion marker
(`scenes/<scene_id>/captures/<capture_id>/raw/capture_upload_complete.json`) is finalized
in the shared bucket, and two storage triggers fire on it.

- **BlueprintCapture `extractFrames`** (`cloud/extract-frames/src/index.ts`, recorded at
  commit `870ad89d1eade7db0a1b98e86c1051728dd16b82` in
  `tests/fixtures/capture_bridge_contract_v1.json`). For a browser delivery it writes the
  frames, descriptor, QA report and `deliveries/<key>/capture_delivery_membership.json`. It also
  writes the immutable `deliveries/<key>/pipeline_handoff.json`. It then publishes the full message,
  which is the only message that carries `source_membership_selector`. By default it publishes to
  `blueprint-capture-pipeline-handoff` (`BLUEPRINT_CAPTURE_PIPELINE_TOPIC` overrides this).
- **This Pipeline's `storage-trigger`** (`functions/storage_trigger.py`). It publishes a short
  message to `blueprint-capture-bridge-handoff`. That message has `source_finalize` but no
  selector, and it names `captures/<id>/pipeline_handoff.json`. For a browser delivery,
  extractFrames does not write that path.

The host listener's subscription (`blueprint-pipeline-handoff-listener`) reads only the bridge
topic. A website delivery stages only through its selector (`capture_delivery_staging.py`).
The short message therefore ends `capture_original_birth_unavailable_retryable` and reaches the
dead-letter queue after five deliveries. The swap dispatch worker (`on_swap_dispatch`) reads the
other topic.

**Repository defaults.** Under the Terraform and `deploy.sh` defaults, a website job never reaches
the listener with its selector. A rebuild from those defaults alone would start no website job,
unless extractFrames publishes its full message to the topic that the listener's subscription reads.

**Production.** The host's delivery evidence
(`/var/lib/blueprint/pubsub-handoffs/.pubsub_delivery_evidence`, read 2026-11-11) shows that the
listener receives extractFrames' full message (9-11 KB payloads). None of those records has a
short-message blocker. How the message gets there is not recorded in any of the three repositories
and has not been verified. Possible routes are a topic override on extractFrames, a second
subscription, or something else. Treat it as unknown.

**Read-only checks for the owner** (none has been run from here):

1. `gcloud functions describe extractFrames --region us-central1 --format='value(serviceConfig.environmentVariables)'`:
   the values of `BLUEPRINT_CAPTURE_PIPELINE_TOPIC` and `BLUEPRINT_LARGE_VIDEO_INGEST_TOPIC`.
2. `gcloud pubsub topics list-subscriptions` for each of `blueprint-capture-pipeline-handoff`,
   `blueprint-capture-bridge-handoff` and `blueprint-large-video-ingest`.
3. `gcloud functions describe swap-dispatch-worker` and `gcloud functions describe storage-trigger`:
   whether they are deployed, and their environment.
4. `gcloud run jobs executions list --job blueprint-pipeline --region us-central1`: whether any
   recent execution ran for a `walkthrough-` or `supplement-` capture.
5. `gcloud run jobs describe blueprint-pipeline`: which provider keys its environment holds.
6. The message count in `pipeline-trigger-dlq-retained`.
7. Objects under `gs://<bucket>/_pipeline_failures/`. A failed job run writes there outside any
   site's prefix, and withdrawal cleanup does not cover it.

### The capture app's path was removed

The capture app is not in use (owner, 2026-11-11), and its processing code was deleted the same
day: the device branch of the qualification pipeline and the modules only it used, including the
six paid call sites that could not see the account's monthly limit. There is no switch to turn it
back on (`BLUEPRINT_DEVICE_CAPTURE_PIPELINE` is not read).

What is deployed stays as it is, because how production forwards extractFrames' message to the
listener is not verified (the checks above):

- **The swap dispatch worker** (`functions/storage_trigger.py`, unchanged) still acknowledges and
  drops a website capture, and still publishes the bridge-topic handoff on a raw upload
  completion. For any other payload it launches nothing and runs nothing.
- **The `blueprint-pipeline` Cloud Run job** keeps its entry (`capture_orchestrator.main`). It
  accepts the job's arguments, writes one record (`device_capture_pipeline_closed.v1`, reason
  `device_capture_pipeline_removed`, `removed_on` 2026-11-11) and exits 0.
- **The host listener** ends a handoff named outside the WebApp's website namespace (a scene not
  starting `site-`, a capture not starting `walkthrough-` or `supplement-`) before claiming or
  staging anything. It writes one permanent-invalid delivery record (blocker
  `device_capture_pipeline_removed`) and acknowledges the message: no retry, no dead letter, no
  page.

Readers of artifacts that the capture app's path wrote before (for example
`privacy_processing_manifest.json`, `task_scope_record.json`, `canonical_site_package.json`) stay,
so stored records remain readable.

### Large videos

The WebApp accepts an upload up to `SELF_CAPTURE_MAX_UPLOAD_BYTES` (2 GiB by default).
extractFrames extracts frames inline only up to 1,000,000,000 bytes by default
(`BLUEPRINT_EXTRACT_FRAMES_MAX_INLINE_VIDEO_BYTES`). The value deployed in production is not known.
For a larger video, extractFrames writes `large_video_ingest_blocked.json` beside the capture and
publishes only to `blueprint-large-video-ingest`, which has no subscription. No message reaches
the listener for such a job.

The WebApp's no-grant sweep reads that marker on every pass. It ends the job at once, with the ask
to film a shorter clip or record at 1080p (`websiteLargeVideoBlock.ts`). The marker is the
authority; no second limit is configured on either side.

### Supplements

A supplement (`supplement-<40 hex>`) is uploaded through the same WebApp upload route. Its marker
goes under its own capture prefix, so it takes the same route and the same guards described above.
