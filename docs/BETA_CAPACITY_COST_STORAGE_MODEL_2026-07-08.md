# Beta Capacity, Cost, and Storage Model

Status: launch gate artifact for 100-user external beta.

This model covers the storage and ingest side of the beta. It keeps raw capture
truth available long enough for QA, package generation, disputes, and buyer
review, while putting hard ceilings on upload size, capture duration, retention,
and provider fan-out.

## Hard Per-Capture Limits

- Max capture upload payload: 20 GiB (`21474836480` bytes).
- Max capture duration: 45 minutes (`2700` seconds).
- Inline extractFrames max raw walkthrough video: `1500000000` bytes.
- Raw videos above the inline extract limit must not be downloaded by the
  2 GiB Cloud Function. They are blocked with a QA/status artifact and routed to
  `large_video_cloud_run_ingest`.

The iOS upload path enforces the 20 GiB / 45 minute limits before Firebase
Storage transfer. Firebase Storage rules separately cap raw uploads at 20 GiB,
and extractFrames has its own pre-download video-size gate.

## 100-User Model

The machine-readable model is
`docs/beta_capacity_cost_storage_model_2026-07-08.json`.

- 100 external users.
- 3 captures per user per month.
- 300 captures per modeled month.
- 25 concurrent uploaders for the soak target.
- 15 minute intake soak target.
- Cloud Run task dispatch is set to at least 25 through Terraform
  `max_concurrent_jobs`.
- Cloud Tasks queue-depth alerting trips above 50 queued tasks for 5 minutes,
  rather than waiting for the old 100-task / 10-minute backlog.
- GPU privacy/video-to-world runners are intentionally capped below dispatch
  concurrency to bound spend: SAM3=3, VIP=2, DeepPrivacy2=2,
  video-to-world=2.
- Firestore `captures.createdAt` composite-index hotspotting is a monitored
  scale-up risk, not treated as a 100-user beta blocker. Terraform declares
  sharded `createdAtShard` companion indexes and a Firestore p99 request
  latency alert for soak/load evidence.
- Oversize raw videos are handed to the private
  `blueprint-large-video-ingest` topic for disk-backed Cloud Run processing
  instead of the 2 GiB inline function path.
- p50 raw capture size: 1.2 GiB.
- p95 raw capture size: 8 GiB.
- p99 / absolute cap: 20 GiB.
- Derived artifact multiplier: 2.5x raw capture bytes.

At p50, the month adds about 360 GiB raw and 900 GiB derived artifacts. If every
capture landed at the p95 assumption, the month adds about 2400 GiB raw and
8400 GiB raw plus derived artifacts. If every capture hits the hard cap, raw
ingest is bounded at 6000 GiB before retention and lifecycle policies.

## Cost Per Capture

The machine-readable model includes `blueprint.beta_cost_per_capture_model.v1`.
This is a planning estimate, not live billing proof.

| Metric | Value |
| --- | --- |
| Budget cap per capture | `$16.67` (`$5000 / 300 captures`) |
| Provider-spend review threshold per capture | `$8.33` (`$2500 / 300 captures`) |
| p50 storage per capture | `4.2 GiB` (`1.2 GiB` raw + `3.0 GiB` derived) |
| p95 storage per capture | `28 GiB` (`8 GiB` raw + `20 GiB` derived) |
| Estimated egress per capture | `1.0 GiB` |
| Estimated GPU seconds per capture | `1200` |
| p50 planning estimate | `$3.56` per capture / `$1068` per modeled month |
| p50 monthly budget headroom | `$3932` before the `$5000` hard stop |

The planning unit costs are explicit in JSON (`gpu_hour_usd`,
`storage_gib_month_usd`, `egress_gib_usd`, `pipeline_cpu_per_capture_usd`, and
`ops_buffer_per_capture_usd`) so finance can replace assumptions with billing
exports without changing the report schema.

## Budget Guardrails

- Cohort provider-spend review threshold: `$2500`.
- Cohort hard-stop threshold: `$5000`.
- `scripts/gpu_spend_guard.py` persists the rolling
  `gpu_spend_ledger.v1` and `gpu_fleet_budget_guard.v1` daily/total fleet
  budget status, reconciles it against a current complete provider billing
  export, and writes the production `blueprint.paid_spend_admission_lock.v1`
  consumed by the shared paid-lane chokepoint. `$5000.00` is blocked, not
  admitted.
- Terraform's optional project-scoped
  `google_billing_budget.gpu_fleet_beta` remains an alerting input, not an
  admission control. Set `billing_account_id` for production, but do not call
  the budget resource itself a hard stop.
- The systemd guard stops new paid work, emits the page event, and records
  controlled drain/provider-confirmed teardown state. A short-lived override
  requires two-person approval, a durable ticket, safe file permissions, and a
  maximum four-hour validity interval. See
  `docs/PAID_SPEND_ADMISSION_LOCK.md`.

## Storage Lifecycle

The primary capture bucket lifecycle policy is checked in at
`deploy/storage/primary-capture-bucket-lifecycle.json` and applied with:

```bash
scripts/apply_primary_capture_bucket_lifecycle.sh "$BLUEPRINT_PRIMARY_CAPTURE_BUCKET"
```

Per-data-class policy:

| Data class | Prefixes | Lifecycle action |
| --- | --- | --- |
| Raw capture truth | `scenes/`, `targets/` | Nearline after 30 days, Coldline after 90 days, delete after 180 days. |
| Temporary processing | `tmp/`, `staging/`, `debug/` | Delete after 14 days. |
| Buyer/eval/hosted artifacts | `buyer_delivery/`, `marketplace/`, `hosted_sessions/`, `robot_eval_jobs/` | Delete after 365 days unless a contract-specific retention hold supersedes it. |

Arena delivery packages emit `arena_delivery_retention_policy.v1` with the same
data classes, the checked-in lifecycle policy path, the apply script, and an
explicit `primary_capture_bucket_lifecycle_apply_proof_missing` blocker until a
live bucket apply is archived.

The beta data retention policy artifact is
`docs/beta_data_retention_policy_2026-07-09.json`
(`blueprint.beta_data_retention_policy.v1`). Launch readiness packets include it
as `beta_data_retention_policy_json`, and
`scripts/validate_beta_capacity_storage.py` checks that the policy matches the
bucket lifecycle classes, local retention limits, support evidence window, and
manual `operator_dpa_data_processing_terms` boundary.

This lifecycle policy is a cost and hygiene control. It is not a substitute for
legal deletion workflows, user-request deletion, backup/PITR, or contract hold
handling.

## Local `robot_eval_jobs/` Cache

The repo-root `robot_eval_jobs/` directory is local cache and provider-run
scratch space. It is not launch proof by itself. The July 2026 audit found it
at about 2.9 GiB. For the 100-user beta model:

| Metric | Value |
| --- | ---: |
| Modeled captures/month | 300 |
| Average robot-eval runs/capture | 0.25 |
| Planned robot-eval jobs/month | 75 |
| Local review threshold | 25 GiB |
| Local hard stop | 50 GiB |
| Default local job retention | 30 days |

Before operator handoff, run a dry-run inventory:

```bash
python scripts/manage_output_artifact_retention.py \
  --output-root robot_eval_jobs \
  --manifest-path output/robot_eval_jobs_retention_manifest.json
```

Only execute deletion with the explicit acknowledgement described in
`docs/runbooks/output-artifact-retention.md`. A repo-root `robot_eval_jobs/`
entry is launch evidence only if the current launch readiness packet references
it or it has been copied into a current operator evidence bundle. This local
cache policy does not prove GCS lifecycle, legal deletion, or live provider
result validity.

## Buyer Delivery Egress (SCALE2-04)

Buyer delivery downloads are direct GCS signed URLs today (~$0.12/GiB at
~1 GiB/delivery). A Cloud CDN design — CDN-rate egress (~$0.04–0.08/GiB) +
edge caching, entitlement gating unchanged — is validated and feature-flagged
(`BLUEPRINT_DELIVERY_CDN_ENABLED`, default off, GCS fallback preserved) in
`docs/BUYER_DELIVERY_CDN_DESIGN_2026-07-20.md`, which carries the full cost
table (~$1,229/mo direct vs ~$430–1,000/mo CDN at 10k deliveries/month).
Provisioning the CDN backend (LB, hostname, signing key) is an owner
decision executed via a separate reviewed Terraform change; until then the
direct-GCS path and cost line stand.

## Firestore CreatedAt Hotspot Guard

Corrected in scaling round 2 (SCALE2-07): earlier revisions of this model
declared four Terraform composite indexes on a literal `captures` collection
that no code in any Blueprint repo has ever written. Those phantom resources
are deleted from `deploy/terraform/main.tf`. The real capture-record
collection is `creatorCaptures`, owned and written by Blueprint-WebApp
(`server/routes/creator.ts`), whose registration writer has populated the
`createdAtShard` hotspot-guard field since scaling round 1
(`server/utils/captureShard.ts`, sha256(capture_id) mod 16). The matching
composite indexes now live where the collection's owner deploys them —
`Blueprint-WebApp/firestore.indexes.json`:

| Contract | Value |
| --- | --- |
| Collection | `creatorCaptures` (Blueprint-WebApp owned) |
| Shard field | `createdAtShard` (16-way, sha256 full-digest mod) |
| Current-reader composite | `creator_id ASC, created_at DESC` |
| Sharded scale-up composites | `creator_id ASC, createdAtShard ASC, created_at DESC`; `status ASC, createdAtShard ASC, created_at ASC` |
| Index manifest | `Blueprint-WebApp/firestore.indexes.json` |
| Runtime alert (this repo) | `google_monitoring_alert_policy.firestore_request_latency` |
| Firestore latency metric | `serviceruntime.googleapis.com/api/request_latencies` |
| Alert threshold | p99 above `0.25s` for `300s` |
| Soak report field | `firestore_latency_observation` |

Before scaling beyond the beta model, readers must aggregate per-shard
`created_at` results before any removal of the legacy composite. The
checked-in index manifest and alert are not live Firestore latency proof and
do not prove readers already fan out per-shard queries.

## Verification

Run:

```bash
python scripts/validate_beta_capacity_storage.py
python scripts/run_beta_intake_soak_test.py --dry-run
```

Before external beta, also run the soak harness against the real staging or
production intake endpoint with the target concurrency and archive the JSON
report under `output/beta_capacity/`:

```bash
python scripts/run_beta_intake_soak_test.py \
  --target-url "$BLUEPRINT_INTAKE_SOAK_URL" \
  --bearer-token-env BLUEPRINT_LIVE_PIPELINE_INTAKE_TOKEN \
  --require-firestore-latency \
  --firestore-p99-latency-seconds "$BLUEPRINT_FIRESTORE_P99_LATENCY_SECONDS" \
  --firestore-latency-source "$BLUEPRINT_FIRESTORE_LATENCY_SOURCE" \
  --duration-seconds 900 \
  --concurrency 25 > output/beta_capacity/intake_soak_report.json
```

That executed report is the load/soak evidence. The dry-run output only proves
the planned concurrency, duration, cost model shape, and Firestore observation
schema.
