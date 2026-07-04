# Site Operator Proof Packet — Example

> EXAMPLE DOCUMENT. All ids, URIs, and values below are illustrative. A real
> packet is generated only from pipeline artifacts; every claim cites the
> artifact and field it comes from. If any cited artifact is missing, the
> packet is not producible and the site stays blocked — missing evidence is
> never launch evidence.

**Site:** Eastside Fulfillment — Dock B corridor
**Scene / capture:** `scene-7f3a` / `capture-02`
**Generated:** 2026-07-04T15:20:00Z
**Prepared for:** Site operator of record (Dana R., facilities)
**Packet status:** `proof_pack_ready: true` (`proof_pack_manifest.json.status = "ready"`)

---

## What this packet is — and is not

This packet proves that the pipeline's launch gate passed with evidence on
file for this one capture. It is **not** a safety approval, not a deployment
authorization, and not proof of live payments or payouts. Items listed under
"Operator checklist still open" remain yours to complete before any external
claim is made.

---

## Launch gate: 16/16 checks passed (`launch_gate_summary.json`)

| Evidence area | Check | Result | Source |
|---|---|---|---|
| WebApp linkage | `inbound_request_linked` | `site_submission_id = subm_9Sk2…` (real record, not placeholder/derived) | `capture_descriptor.json.site_submission_id` |
| WebApp linkage | `buyer_request_linked` | `buyer_request_id = breq_4Hn8…` | `opportunity_handoff.json.buyer_request_id` |
| WebApp linkage | `approved_marketplace_capture_job_linked` | `capture_job_id = cjob_1Vt5…` | `capture_descriptor.json.capture_job_id` |
| WebApp sync truth | `webapp_sync_completed` | succeeded, upstream links verified, no placeholder fallback, `synced_at = 2026-07-04T14:58:11Z` (age 0.4 h < 24 h max) | `webapp_sync_result.json.syncs.evaluation_prep` |
| Consent & rights | `rights_provenance_review_cleared` | status `cleared`; consent `documented`; permission document on file | `rights_provenance_review.json.rights` |
| Consent & rights | rights packet completeness | `derived_scene_generation_allowed = true`; `permission_document_uri = gs://…/rights/dock-b-consent-packet.pdf`; scope: `["dock B corridor"]` | `rights_and_compliance_summary.json` |
| Privacy lineage | `privacy_safe_buyer_media_ready` | buyer media derives from `privacy/final_walkthrough.mp4`; pipeline status `person_removed` | `privacy_processing_manifest.json` |
| Raw-bypass blocker | `raw_worldlabs_bypass_not_used` | `raw_video_bypass_used = false` — world-model input derived from privacy-safe media only | `worldlabs_input_audit.json.input_labeling` |
| Provenance | `provenance_summary_grounded` | status `grounded`, `record.canonical_truth = true` | `provenance_summary.json` |
| Recapture | `recapture_not_required` | `required = false`; no missing evidence recorded | `recapture_requirements.json` |
| Qualification | `qualification_authoritative` | state `qualified_ready`, confidence 0.87 | `qualification_record.json` |
| Fulfillment | `buyer_fulfillment_bundle_ready` | bundle status `ready` (explicit; statusless bundles fail) | `evaluation_prep/launchable_export_bundle.json.status` |
| Runtime | `native_runtime_capability_ready` | hosted runtime healthy and launchable | `evaluation_prep/site_world_health.json` |
| Revenue share | `capturer_payout_transition_ready` | `eligible_for_payout = true` (explicit decision; amounts below) | `capturer_payout_recommendation.json` |
| Upload | `mobile_upload_completed` | upload completion receipt on file | `raw/capture_upload_complete.json` |
| Claim context | `mobile_claim_context_captured` | source `iphone`, quoted payout retained | `capture_descriptor.json` |

## Your media, your rights

- The raw walkthrough never leaves capture storage and is never a runtime
  render source (`retrieval_index` marks raw media `privacy_safe: false`;
  runtime render source requires privacy-safe media).
- Derived world outputs exist only because your rights packet explicitly
  allows them (`derived_scene_generation_allowed = true`). If you revoke or
  narrow consent, the rights review stops clearing and the package blocks.
- Revenue share: recommended payout 71.50 USD against base 65.00 USD
  (`capturer_payout_recommendation.json`: `recommended_payout_cents = 7150`,
  `base_payout_cents = 6500`, bonus breakdown attached). Final authority for
  the actual payment is human ops review (`final_authority =
  "webapp_ops_review"`); this packet does not claim money moved.

## Operator checklist still open (not claimed by this packet)

1. Live buyer payment settlement evidence (Stripe).
2. Live capturer payout settlement and connected-account readiness.
3. Real-device claim-flow evidence for this `capture_job_id`.
4. Named finance owner for payout exceptions.

Full list with required evidence: `launch_gate_summary.json.operator_required_checks`.

---

*Fail-closed guarantee: each row above fails — and the packet is not issued —
if its source artifact is missing, stale, placeholder-linked, raw-bypassed,
or lacking explicit consent/rights values. See
`tests/test_site_operator_fail_closed_gates.py`.*
