# Cross-Repo Beta Blockers — Specs

These blockers span two or more repos. They are the ones most likely to be missed by a single-repo review because each repo looks internally consistent — the break is at the seam. All were adversarially re-verified against real code/config.

Severity legend: **hard_blocker** = cannot launch a truthful beta without it · **high** = must fix before external users · **medium/low** = fix soon / track.

---

## XR-01 — Two repos deploy conflicting `firestore.rules` to the same Firebase project; a WebApp rules deploy can break capturer job discovery

| Field | Value |
| --- | --- |
| Repos | Blueprint-WebApp + BlueprintCapture → project `blueprint-8c1ca` |
| Severity | **hard_blocker** (CONFIRMED_WITH_NUANCE — conditional on deploy order) |
| Category | deploy / flow_integrity |
| Confidence | high |
| Blocks bar | #1 (capturer can claim a job) + capturer onboarding |
| Resolution path | config + process (single canonical ruleset) |

### Problem
`BlueprintCapture/.firebaserc` and `Blueprint-WebApp/.firebaserc` both default to project `blueprint-8c1ca`, and neither `firebase.json` sets a named `database`, so both deploy whole-project rules to the **(default)** Firestore instance. The two rulesets diverge: the iOS ruleset grants `match /capture_jobs/{jobId} { allow read: if isSignedIn(); }` (and rules for `referralCodes`, `reservations`, `target_state`, `sessions`, `sessionEvents`), while the WebApp ruleset has **no `capture_jobs` rule at all** and terminates in `match /{document=**} { allow read, write: if false; }`. `firebase deploy --only firestore:rules` is **last-writer-wins** for the whole project.

### Evidence
- WebApp `firestore.rules:165-167` — terminal `match /{document=**} { allow read, write: if false; }`; no `capture_jobs` match anywhere (grep only hits the iOS file).
- iOS `firestore.rules:85-88` — `match /capture_jobs/{jobId} { allow read: if isSignedIn(); allow write: if false; }`.
- iOS `BlueprintCapture/BlueprintCapture/Services/JobsRepository.swift:43-47` — job discovery is a **client-SDK** read: `db.collection("capture_jobs").whereField("active", isEqualTo: true)...getDocuments()`; lines 52-67 explicitly catch `PERMISSION_DENIED` (code 7) and, with mock fallback off by default, rethrow.
- `Blueprint-WebApp/docs/firestore-rules.md` instructs operators to `firebase deploy --only firestore:rules` from the WebApp repo.
- The Cloud Function `cloud/referral-earnings/src/index.ts:684` writes `capture_jobs` via **admin SDK** (bypasses rules) — that is the writer path; the app's *discovery read* is client-side and rule-governed, so the "iOS uses admin SDK" refutation fails.

### Why it blocks beta
If the WebApp's rules are deployed after the iOS rules (the documented WebApp workflow), the project loses the `capture_jobs` allow. Every capturer's job-list read returns `PERMISSION_DENIED`, the iOS app shows no jobs to claim, and the publish→claim link dies. The blast radius is wider than job discovery: the WebApp catch-all also denies `referralCodes` (invite-code onboarding), `reservations`, `target_state`, and telemetry `sessions`/`sessionEvents`. It is a latent config collision that any WebApp rules deploy triggers.

### Acceptance criteria
- [ ] A single canonical `firestore.rules` (superset of both repos' needs) is the only source deployed to `blueprint-8c1ca`, OR the WebApp ruleset includes every collection the iOS client reads/writes (`capture_jobs`, `referralCodes`, `reservations`, `target_state`, `sessions`, `sessionEvents`, …).
- [ ] A CI check fails if the two repos' rules diverge on any collection the iOS client touches.
- [ ] After a WebApp `firestore:rules` deploy, a signed-in client can still read `capture_jobs where active == true`.

### Implementation plan
1. Diff the two rulesets; enumerate every collection each client actually accesses (iOS via client SDK; WebApp via client SDK — admin-SDK access is exempt).
2. Produce one canonical ruleset that is a superset; decide the single owning repo/CI job that deploys it.
3. Add a CI diff/guard so a future divergence fails the build.
4. Verify with the Firestore emulator: run the iOS discovery query and each iOS write against the canonical rules.

### Verification
Emulator test: `capture_jobs` read as `isSignedIn()`, `capture_submissions` create as owner, `referralCodes`/`reservations` reads — all pass under the canonical ruleset deployed from the single source.

### Notes
Reverse direction is less dangerous (WebApp collections mostly exist in the iOS ruleset too), so the specifically dangerous event is **a WebApp rules deploy landing after an iOS deploy**. Related: [[CAP-03]] relies on `capture_submissions` rules; keep both rulesets' `capture_submissions` in sync.

---

## XR-02 — Deployed storage-trigger config routes iPhone uploads to a handoff path that ignores the event the app actually emits → no pipeline dispatch

| Field | Value |
| --- | --- |
| Repos | BlueprintCapture (producer) → BlueprintCapturePipeline (`functions/storage_trigger.py`, `deploy/`) |
| Severity | **hard_blocker** (verified) |
| Category | cross-repo wiring / ingest |
| Confidence | verified |
| Blocks bar | #1 (uploaded bundle flows into the pipeline) |
| Resolution path | config + code |

### Problem
The deployed environment sets `SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF="true"` (in `deploy/scripts/deploy.sh:324` and `deploy/terraform/main.tf:1080`), which makes `_capture_bridge_handoff_primary()` true in prod. In `functions/storage_trigger.py`, `on_storage_finalize` handles two object types — `capture_descriptor.json` (branch at `:357-380`) and `raw/capture_upload_complete.json` (branch at `:382-430`) — and **both branches early-return** when the capture-bridge handoff is primary (`:359-365` and `:392-398`). But the iOS app only ever uploads the **raw bundle** (`scenes/{scene}/captures/{capture}/raw/…`, including `capture_upload_complete.json`); it **never** uploads `capture_descriptor.json`. So the one finalize event a real capture emits is ignored, and the object that would dispatch is never produced.

### Evidence
- `functions/storage_trigger.py:359-365` (descriptor branch early-return) and `:392-398` (raw-complete branch early-return when `_capture_bridge_handoff_primary() and _dispatch_mode() != 'direct'`; deployed dispatch mode is pubsub); guard helper `:82-83`.
- `deploy/scripts/deploy.sh:324` and `deploy/terraform/main.tf:1080` set `SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF="true"`.
- iOS uploads only the raw prefix: `CaptureUploadService.swift` upload directory + `CaptureBundleContext.rawBasePath` = `scenes/{scene}/captures/{capture}/raw/`; grep of BlueprintCapture for `capture_descriptor` returns nothing.

### Why it blocks beta
A real iPhone upload completes to Storage/Firestore but emits **zero** pipeline dispatch. Capture→pipeline continuity is silently broken for every external capture — the pipeline never learns a bundle exists.

### Acceptance criteria
- [ ] A raw iPhone bundle upload (manifest.json + walkthrough.mov + `raw/capture_upload_complete.json`) results in exactly one pipeline dispatch.
- [ ] An integration test uploads only the iOS raw bundle (no `capture_descriptor.json`) and asserts a dispatch is emitted.

### Implementation plan
1. Decide the ingest path: either (a) set `SWAP_TRIGGER_USE_CAPTURE_BRIDGE_HANDOFF=false` to re-enable the raw-upload-complete materialize+dispatch branch (which only needs `manifest.json` + `walkthrough.mov` that iOS already writes), or (b) keep the bridge handoff primary and make it actually produce+publish a valid handoff for iOS raw uploads (requires fixing [[XR-03]] and [[XR-04]]).
2. Implement the chosen path.
3. Add the raw-bundle-only integration test.

### Verification
Emulate a finalize on `raw/capture_upload_complete.json` with the iOS bundle layout; assert the pipeline receives a dispatch with the correct `scene_id`/`capture_id`/`capture_job_id`.

### Notes
This is one of three layers breaking capture→pipeline ingest; see [[XR-03]] and [[XR-04]]. Even after this, ingest still fails at the next two layers. The *data contract* itself is intact (see INDEX "verified healthy: capture_job_id continuity") — the break is entirely deploy-time wiring.

---

## XR-03 — Capture-bridge handoff listener hard-requires `pipeline_handoff.json`, which no repo ever writes → ingest staging aborts

| Field | Value |
| --- | --- |
| Repos | BlueprintCapturePipeline (listener) + BlueprintCapture (producer) |
| Severity | **hard_blocker** (verified) |
| Category | missing required artifact / ingest |
| Confidence | verified |
| Blocks bar | #1 (uploaded bundle flows into the pipeline) |
| Resolution path | code |

### Problem
`pubsub_handoff_listener.stage_handoff_capture` downloads the capture Storage prefix and then raises `PipelineError` unless `(capture_root / 'pipeline_handoff.json').is_file()`. This listener is the consumer wired by the capture-bridge-primary config. But `pipeline_handoff.json` is **not** in the iOS bundle namelist and is **never written** by any capture-path code or WebApp Cloud Function.

### Evidence
- `src/blueprint_pipeline/pubsub_handoff_listener.py:95-99` — raises `PipelineError` when `pipeline_handoff.json` is absent.
- Subscription wired at `deploy/terraform/main.tf:455-457` (`blueprint-pipeline-handoff-listener` on topic `blueprint-capture-pipeline-handoff`).
- iOS required-files set fixed by `CaptureRawContractV3Validator.swift` (manifest.json, hashes.json, capture_upload_complete.json, arkit/frames.jsonl, walkthrough.mov) — `pipeline_handoff.json` not present; grep of BlueprintCapture for `pipeline_handoff` finds nothing.
- In the pipeline, `pipeline_handoff.json` is only ever **read** (`first_gpu_run_packet.py:1436`, `first_gpu_e2e_readiness.py:189/217`, `cross_repo_first_gpu_readiness.py:204`); no capture-path writer; `functions/index.js` only forwards to Paperclip (no handoff-file writer).

### Why it blocks beta
Even if a handoff message reaches the listener (after [[XR-02]]/[[XR-04]] are fixed), staging aborts with `PipelineError` because the required file is absent from every real iOS bundle. Ingest fails at the staging step.

### Acceptance criteria
- [ ] A real iOS capture bundle stages successfully through `stage_handoff_capture` without a hand-authored file.
- [ ] Test: stage a bundle containing only the iOS namelist and assert success.

### Implementation plan
1. Choose one: (a) have the iOS capture upload — or the pipeline's storage-finalize trigger (`functions/storage_trigger.py`, which already fires on `capture_upload_complete`) — write `scenes/{scene}/captures/{capture}/pipeline_handoff.json` with the `owner_system`/`request_id`/`site_submission_id`/`buyer_request_id`/`capture_job_id` block the pipeline expects; or (b) relax `stage_handoff_capture` to **synthesize** the handoff from `raw/manifest.json` + `capture_context.json` (which already carry `scene_id`/`capture_id`/`capture_job_id`/`site_submission_id`/`buyer_request_id`).
2. Implement + test.

### Verification
Stage a captured bundle end-to-end; assert no `PipelineError` and a well-formed handoff object.

### Notes
Prefer (b) (synthesize) to avoid adding another required upload artifact and another place for the contract to drift. Coupled with [[XR-02]] and [[XR-04]].

---

## XR-04 — Handoff Pub/Sub topic has two consumers expecting incompatible payload schemas → messages dead-letter

| Field | Value |
| --- | --- |
| Repos | BlueprintCapturePipeline (`functions/storage_trigger.py`, `pubsub_handoff_listener.py`, `deploy/terraform`) |
| Severity | **high** (verified) |
| Category | schema mismatch / ingest |
| Confidence | verified |
| Blocks bar | #1 (uploaded bundle flows into the pipeline) |
| Resolution path | code + infra |

### Problem
Topic `blueprint-capture-pipeline-handoff` has **two** consumers with different payload contracts: the `on_swap_dispatch` Cloud Function (event-triggered on the topic) and the `pipeline_handoff_listener` pull subscription. The only publisher (`storage_trigger._build_dispatch_payload`) emits `{descriptor_gcs_uri, bucket, scene_id, capture_id}`, but `pubsub_handoff_listener.parse_handoff_payload` **requires** `raw_prefix_uri == gs://{bucket}/scenes/{scene}/captures/{capture}/raw` and reads `pipeline_handoff_uri` — fields the publisher never emits. The listener's subscription has `dead_letter_policy max_delivery_attempts=5`, so any such message is rejected and dead-lettered.

### Evidence
- Topic `deploy/terraform/main.tf:442-443`; two consumers: `on_swap_dispatch` (`main.tf:1110`, event filter topic=`pipeline_trigger`, `main.tf:1140-1145`) and pull sub `pipeline_handoff_listener` (`main.tf:455-457`).
- Publisher `functions/storage_trigger.py:94-102` (`_build_dispatch_payload` → `{descriptor_gcs_uri, bucket, scene_id, capture_id}`); reader `storage_trigger.py:442-444` uses `payload.descriptor_gcs_uri`.
- Listener `pubsub_handoff_listener.py:49-64` requires `raw_prefix_uri` and `pipeline_handoff_uri`.
- Dead-letter policy `deploy/terraform/main.tf:466-469`.

### Why it blocks beta
The "capture bridge handoff" automation the deployed config makes primary is internally schema-inconsistent; the publisher and listener disagree on payload shape, so a correctly-triggered dispatch cannot be consumed by the listener. This is the connective tissue for capture→pipeline and it cannot carry a valid message.

### Acceptance criteria
- [ ] The handoff listener has a dedicated topic/subscription distinct from the `on_swap_dispatch` descriptor topic.
- [ ] One canonical handoff payload schema (`bucket`, `scene_id`, `capture_id`, `raw_prefix_uri`, `pipeline_handoff_uri`) is emitted by exactly one publisher and accepted by `parse_handoff_payload`.
- [ ] Contract test: `parse_handoff_payload` accepts exactly what the publisher produces.

### Implementation plan
1. Split the topics so the listener and `on_swap_dispatch` no longer share a subscription source.
2. Define the canonical handoff schema; update the single publisher to emit it.
3. Add the publisher↔parser contract test.

### Verification
Publish a real handoff message; assert the listener parses and stages it (0 dead-letters).

### Notes
Fix alongside [[XR-02]] and [[XR-03]]; the three together restore capture→pipeline ingest. Decide first (in [[XR-02]]) whether the bridge-handoff path or the direct raw-complete path is the beta ingest path — that decision determines whether XR-03/XR-04 need fixing at all or the path is retired.

---

## XR-05 — Live-provider / operator evidence required for a truthful *paid* beta is still open (payments, payouts, KYC, real-device claim)

| Field | Value |
| --- | --- |
| Repos | Blueprint-WebApp + BlueprintCapture + BlueprintCapturePipeline |
| Severity | **hard_blocker** for a *paid* beta (documented, not code) |
| Category | live-provider / payment / identity / hardware |
| Confidence | verified (from the repo's own launch gate) |
| Blocks bar | #2 (buyer pays live), #3 (capturer payout), truthful messaging |
| Resolution path | human_decision + live_provider + payment + hardware evidence |

### Problem
The automated launch gate (`docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md`) proves the paid path only at **contract level**. Several items require live/operator evidence before a truthful paid beta and are **still open**. This spec exists so the beta plan does not mistake "contracts passed" for "operationally launch-ready."

### Evidence
From `BlueprintCapturePipeline/docs/PAID_MARKETPLACE_BETA_LAUNCH_GATE.md` (still-open operator evidence):
- `iphone_real_device_claim_flow` — screen recording of discovery→reservation→upload with a stable `capture_job_id` on a real iPhone. **(Blocked upstream by [[CAP-01]]/[[CAP-04]] today.)**
- `buyer_payment_settlement` — live Stripe checkout/payment-intent evidence.
- `capturer_payout_settlement` + `stripe_connected_account_live_readiness` — live Connect account, `payouts_enabled=true`, no blocking requirements, webhook reconciliation, ledger match.
- `payout_exception_monitor_live` — live monitor for `payout.failed`/`canceled`/`disbursement_failed`/overdue `finance_review`.
- `identity_kyc_provider_decision` + `background_check_provider_decision` — explicit decisions (Stripe Connect-only vs Persona/Stripe Identity; Checkr or none).
- `human_finance_review_owner` — named finance owner + review queue before enabling live payout execution.
- `buyer_artifact_access` — authenticated buyer session proving artifact/fulfillment access after purchase. (Code path exists via the hosted-session entitlement flow — see [[WEB-12]] — but needs live post-purchase proof.)

### Why it blocks beta
A paid beta that advertises live buyer payments or capturer payouts without this evidence would be an untruthful claim. Per platform doctrine, provenance/rights/commercialization claims must be proof-bounded.

### Acceptance criteria
- [ ] Each open evidence id above has a recorded artifact or an explicit written decision.
- [ ] The beta's public messaging matches what is proven (iPhone = strongest external path; live payments/payouts only claimed once settled).

### Implementation plan
1. Assign an owner to each evidence id.
2. Sequence: fix the code blockers this audit found ([[CAP-01]]…[[CAP-04]], [[XR-01]]…[[XR-04]], [[WEB-01]]) → run the live flows → capture evidence.
3. Make the KYC/background-check/finance-owner decisions explicitly.

### Verification
Re-run `scripts/run_paid_marketplace_launch_gate.py`; confirm each operator-evidence id is satisfied or has a signed decision.

### Notes
This is the one "known/documented" item included for completeness. It is **downstream** of the code blockers — most live evidence cannot even be collected until [[CAP-01]]/[[CAP-04]] and [[XR-01]]…[[XR-04]] are fixed.
