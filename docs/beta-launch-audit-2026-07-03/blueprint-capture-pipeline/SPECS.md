# BlueprintCapturePipeline Beta Blockers — Specs

The pipeline is the healthiest of the three repos at the unit level: `python -m blueprint_pipeline.run_e2e` runs and **fails closed** on missing raw media; all 235 modules import cleanly; 3278 tests collect with zero import errors and the 120-test critical-path subset passes; the upstream-id guard and the `proof_pack`/`proof_path` **delivery** gates are genuinely fail-closed. The blockers below are **privacy/rights fail-open projections** that could let unverified or un-redacted material be represented as ready or reach a buyer-facing artifact — which matters because doctrine makes privacy/rights/provenance authoritative. These are **new** relative to the repo's existing self-audits (`last_24h_launch_audit_2026-06-26.md`, `READINESS_MATRIX.md`, `PAID_MARKETPLACE_BETA_LAUNCH_GATE.md`), which this audit does not re-litigate. All were adversarially re-verified.

> Note on the cross-repo ingest blockers: the pipeline's Pub/Sub handoff listener (`pubsub_handoff_listener.py`) and the deployed storage-trigger bridge are covered in **cross-repo** [[XR-02]], [[XR-03]], [[XR-04]] because the fix spans `functions/`, `deploy/`, and this repo.

---

## PIPE-01 — `site_world_spec` / `launchable_export_bundle` are marked launchable and embed the raw un-redacted walkthrough with no privacy-complete or rights-cleared gate

| Field | Value |
| --- | --- |
| Severity | **high** (verified) |
| Category | privacy / rights / provenance |
| Blocks bar | #4 (privacy/rights authoritative; raw capture must not reach buyers un-cleared) |
| Resolution path | code |

### Problem
`_canonical_site_world_runtime_status` computes `launchable` purely from `runtime_service_url` + required runtime-artifact paths (only `protected_regions_manifest`, `canonical_render_policy`, `presentation_variance_policy`). Privacy status and downstream-eligibility are demoted to **warnings only**, never blockers; rights are never consulted. That `launchable` flows into `site_world_health.launchable` → `_build_launchable_export_bundle` → bundle status `ready`. Meanwhile `_primary_runtime_render_descriptor` falls back to `local_paths['raw_video_path']` — the **un-redacted** `walkthrough.mov` (unconditionally populated when the file exists) — as `raw_video_ref`, and `_build_site_world_spec` embeds that render source plus the full `local_paths` block (including `raw_video_path`) into the buyer-facing `site_world_spec` conditioning. Both artifacts sync to the WebApp, and `launchable_export_bundle` "ready" satisfies the launch-gate `buyer_fulfillment_bundle_ready` check.

### Evidence
- `evaluation_prep_stage.py:1823-1869` — `_canonical_site_world_runtime_status`; required runtime paths `:3818-3822`, `:3906-3910`; privacy/eligibility demoted to warnings `:1841-1846`.
- Flow to launchable: `:3507-3508` (`world_model_runtime.launchable`) → bundle status `:3566-3572`.
- Raw fallback: `_primary_runtime_render_descriptor` `:1607-1639` falls back to `local_paths['raw_video_path']` (populated `:1493`/`:1505`); embedded into `site_world_spec` `:2226-2229`, `:2259`.
- Sync + gate satisfaction: `alpha_readiness.py:1083-1088` (synced), `:763-766`, `:846-853` (satisfies `buyer_fulfillment_bundle_ready`).

### Why it blocks beta
A buyer-deliverable/hosted runtime spec can be marked launchable and reference the raw un-redacted capture even when privacy post-processing is `failed_closed`/`not_run` and rights are not cleared. A capturer's raw walkthrough (people/PII, or lacking derived-generation rights) can therefore reach a buyer-facing artifact — a direct violation of the "privacy/rights authoritative" bar.

### Acceptance criteria
- [ ] `_canonical_site_world_runtime_status` adds **hard blockers** for `privacy_processing.status` not in the cleared set and for `rights_review.status != "cleared"`.
- [ ] `_primary_runtime_render_descriptor` / conditioning `local_paths` never fall back to `raw_video_path`; only privacy-safe world-model/privacy-processed URIs are accepted.
- [ ] A `needs_review`/`not_run` capture yields `launchable=false` and a `site_world_spec` with no raw render source.

### Implementation plan
1. Add privacy + rights hard blockers to `_canonical_site_world_runtime_status` (mirror the `privacy_world_model_ready` gate `qualification.py:4815-4833`).
2. Remove the `raw_video_path` fallback from the render descriptor / conditioning; require a privacy-safe URI.
3. Add tests for `needs_review`/`not_run`/`failed_closed` → non-launchable + no raw source.

### Verification
Reproduce with a `needs_review` capture: assert `launchable=false`, no `raw_video_path` in `site_world_spec`, and `buyer_fulfillment_bundle_ready` not satisfied.

### Notes
Strongest pipeline finding — it is the path by which raw media could actually reach a buyer artifact, whereas [[PIPE-02]] is a status projection and [[PIPE-03]] a weaker-gate advisory. Fix together with [[PIPE-02]].

---

## PIPE-02 — `site_package_manifest` / `hosted_review_readiness` project "ready" on rights `needs_review`; the WebApp consumer gates on artifact **presence**, not the rights verdict

| Field | Value |
| --- | --- |
| Severity | **high** (CONFIRMED_WITH_NUANCE — cross-cutting fix) |
| Category | rights / privacy / webapp_sync |
| Blocks bar | #4 (privacy/rights authoritative) |
| Resolution path | code (pipeline + webapp) |

### Problem
`build_site_package_manifest` adds a blocker only when `rights_review.status == "blocked"` — `needs_review` adds no blocker and the status becomes `ready`. `build_hosted_review_readiness` never inspects rights at all. Both are synced to the WebApp as authoritative state. And the WebApp consumer does **not** read these status/verdict fields: `pipelineStateMachine.ts` derives `qualified_ready` from the mere **presence** of a rights-report URI (a `needs_review` capture still emits that URI) and stamps `proof_pack_delivered_at`/`hosted_review_ready_at` from artifact-presence counts — never from the rights verdict. So progression to handoff/hosted-review is presence-driven, with no compensating gate.

### Evidence
- `proof_contracts.py:170-184` (site-package blocks only on `"blocked"`); `:226-243` (hosted-review ignores rights). Reproduced: `rights_provenance_review` → `needs_review`, but `SITE_PACKAGE.status=ready`, `HOSTED_REVIEW.status=ready`, while `PROOF_PACK.status=blocked`, `PROOF_PATH.rights_cleared=false`.
- Synced: `evaluation_prep_stage.py:4989-4990` → `sync_webapp_evaluation_prep` → `alpha_readiness.py:1184-1189` (`authoritative_state_update=True`).
- WebApp consumer: `pipelineStateMachine.ts:440-458` (`qualified_ready` from `hasRightsReport` = URI presence), `:556-573` (proof_pack/hosted-review timestamps from artifact-presence counts), `:1041-1069` (`checkHostedReviewReadiness` checks only preview/worldlabs presence, no rights).

### Why it blocks beta
Two "ready" signals for a capture with **no privacy processing and unverified consent** are synced to the WebApp and consumed as progression triggers, so a `needs_review` capture can advance to handoff/hosted-review and read as "ready" to a reviewer/buyer. The final `proof_pack`/`proof_path` gate is correctly rights-gated (so *ultimate delivery* is safe), but the intermediate ready projections are not — and the WebApp doesn't consult the safe gate.

### Acceptance criteria
- [ ] `build_site_package_manifest` and `build_hosted_review_readiness` treat rights `needs_review` (and privacy `needs_review`/`not_run`) as a blocker (or downgrade status to `needs_review`), matching `build_proof_pack_manifest`.
- [ ] The WebApp `pipelineStateMachine.ts` gates progression on the rights **verdict** (`proof_path_status.rights_cleared` / `proof_pack_manifest.status`), not artifact presence.
- [ ] Tests cover the `needs_review` case in `proof_contracts` (today only `blocked` is tested — `tests/test_proof_contracts.py:64-81`).

### Implementation plan
1. Pipeline: add `needs_review`/`not_run` as blockers in the two builders; add tests.
2. WebApp: change `pipelineStateMachine.ts` to read the rights verdict fields (already on the wire — `rights_provenance_review`, `proof_path_status` are synced) instead of presence.
3. Verify both ends with a `needs_review` fixture.

### Verification
`needs_review` capture → `site_package`/`hosted_review` not `ready`; WebApp does not advance to handoff/hosted-review.

### Notes
A complete fix requires **both** ends; fixing only the pipeline still leaves the presence-based WebApp gate. Pairs with [[PIPE-01]].

---

## PIPE-03 — Privacy pipeline is off by default; the qualification `privacy_postprocess_gate` passes on `not_run`

| Field | Value |
| --- | --- |
| Severity | **medium** (CONFIRMED_WITH_NUANCE — delivery gates compensate) |
| Category | privacy |
| Blocks bar | #4 (defense-in-depth for redaction) |
| Resolution path | code |

### Problem
`PRIVACY_PIPELINE_ENABLED` defaults **False** outside production launch mode, so privacy status is `not_run` (processed URI `None`), and the qualification `privacy_postprocess_gate` **passes** when status is `not_run`. Nothing in-repo forces production mode for a buyer-facing run, so the qualification-level privacy gate is effectively advisory.

### Evidence
- `privacy_processing.py:671` (`default=False`), `:742-755` (`not_run`, uri `None`).
- `qualification.py:4530-4543` (gate passes on `not_run`); production-only redaction `:3068-3070`.
- `launch_proof_policy.py:20-28` — production mode is purely env-driven.

### Why it blocks beta
The qualification privacy gate provides no guarantee on its own. **Nuance (why medium, not hard):** downstream *delivery* gates DO block on `not_run` — buyer world-model media is generated only when privacy is complete (`qualification.py:4815-4833`), the raw-video WorldLabs bypass is production-blocked, and `provider_preview_qa` / `production_handoff_readiness` propagate a `privacy_manifest_or_verification_not_complete` blocker. So un-redacted media does not reach a buyer by this path **by default** — except via the raw-fallback hole in [[PIPE-01]], which is the real exposure. This spec hardens the defense-in-depth.

### Acceptance criteria
- [ ] For any run that will produce buyer/reviewer-facing artifacts, production launch mode (or `PRIVACY_PIPELINE_ENABLED=true`) is a hard precondition, not an advisory env check.
- [ ] `privacy_postprocess_gate` treats `not_run` as non-passing when delivery artifacts will be built.

### Implementation plan
1. Make production mode / privacy-enabled a hard precondition for buyer-facing runs (upgrade the advisory check in `alpha_readiness.py:308-309`).
2. Make `not_run` non-passing in `privacy_postprocess_gate` for delivery runs.
3. Add tests.

### Verification
A default-mode buyer-facing run refuses to proceed (or blocks the privacy gate) instead of passing on `not_run`.

### Notes
Confidence medium; the compensating delivery gates are the reason this is not hard. Close alongside [[PIPE-01]] (which is the path that actually bypasses these compensations).

---

## PIPE-04 — WorldLabs preview video is generated with no rights/consent gate

| Field | Value |
| --- | --- |
| Severity | **medium** (verified) |
| Category | rights |
| Blocks bar | #4 (derived artifacts require rights clearance) |
| Resolution path | code |

### Problem
`_prepare_worldlabs_input_video` is invoked purely on `preview_requested_for_worldlabs` (i.e., `requested_outputs` containing `preview`/`preview_simulation`). There is no check on `rights["derived_scene_generation_allowed"]` or `consent_status` before generating this derived, reviewer-facing preview video — unlike canonical scene-memory readiness, which correctly requires `derived_scene_generation_allowed`.

### Evidence
- `qualification.py:4557-4576` — `_prepare_worldlabs_input_video` gated only on preview request.
- Contrast `qualification.py:951` — scene-memory readiness requires `derived_scene_generation_allowed`.

### Why it blocks beta
A derived preview video (a transformation of the capture) is produced even when the capture is not rights-cleared for derived scene generation. Its privacy safety additionally depends on [[PIPE-03]]/[[PIPE-01]].

### Acceptance criteria
- [ ] `_prepare_worldlabs_input_video` is gated on `_capture_rights(metadata)["derived_scene_generation_allowed"]` (skip/block with a clear reason when rights are absent).

### Implementation plan
1. Add the rights gate mirroring scene-memory readiness; add a test for rights-absent → no preview.

### Verification
A capture without derived-generation rights produces no WorldLabs preview.

---

## PIPE-05 — Full test suite is impractically long for CI (no fast lane)

| Field | Value |
| --- | --- |
| Severity | **medium** (verified) |
| Category | tests |
| Blocks bar | launch confidence |
| Resolution path | code (test infra) |

### Problem
`python -m pytest` on the 3278-test suite spawns real subprocesses (Isaac/provider/render/module-entrypoint tests) and did not complete after 23+ minutes; `CLAUDE.md` advertises bare `pytest` as a key command. No fast core lane exists.

### Evidence
- Full-suite run did not finish in 23 min; the ~120-test critical-path subset runs in ~50s.

### Why it blocks beta
A green suite that nobody can run to completion is a launch-confidence gap; slow/subprocess tests hide regressions (the reference audit itself notes the full suite "was not rerun").

### Acceptance criteria
- [ ] Subprocess/heavy render tests are marked `slow`/`gpu` and default-deselected.
- [ ] A documented fast core lane exists; `CLAUDE.md` reflects the split.

### Implementation plan
1. Add `slow`/`gpu` markers; configure default deselection.
2. Document the fast lane in `CLAUDE.md`.

### Verification
`pytest -m "not slow"` completes quickly and green; full suite runs in a nightly/opt-in lane.

---

## PIPE-06 — `request_id` upstream link is aliased to `site_submission_id` in the WebApp sync (4-key guard is effectively 3-key)

| Field | Value |
| --- | --- |
| Severity | **low** (verified) |
| Category | webapp_sync / provenance |
| Blocks bar | upstream-link integrity (minor) |
| Resolution path | code or doc |

### Problem
`sync_webapp_pipeline_attachment(..., request_id=site_submission_id, ...)` — `request_id` is always a copy of `site_submission_id`, so the four-way "independent upstream links" guard (`webapp_sync.py:363`) is really three-way in this path. Not fail-open (the guard still rejects missing/placeholder/generated ids), but it provides no additional integrity beyond `site_submission_id`.

### Evidence
- `alpha_readiness.py:1193-1196` — `request_id=site_submission_id`.
- `webapp_sync.py:341-373` — validates all four keys.

### Why it blocks beta
It doesn't, materially — but the gate doc treats `request_id` as an independent verification, which it isn't here.

### Acceptance criteria
- [ ] Either source a real distinct `request_id` from the opportunity handoff, or document explicitly that `request_id == site_submission_id` by contract.

### Implementation plan
1. Decide independent vs aliased; implement or document accordingly.

### Verification
The four-key guard's semantics match reality (four independent, or documented alias).
