# Blueprint Cross-Surface Data Retention Policy

Audit finding: **R048 (P1)** — data-retention existed only as an agent/doc notion scoped to
WebApp Firestore, was unenforced, and did not reach pipeline output artifacts, hosted/derived
world models, or storage. There was no single, enforced, cross-surface retention contract.

This document is the **canonical, cross-surface** retention policy for all Blueprint
authoritative and derived data (GCP project `blueprint-8c1ca`). It is backed by a committed,
machine-readable config and a fail-closed validator, so retention is **enforced**, not just prose.

- Machine-readable contract: [`configs/data_retention_policy.json`](../configs/data_retention_policy.json)
- Validator (fail-closed): [`scripts/validate_data_retention_policy.py`](../scripts/validate_data_retention_policy.py)
  (+ [`scripts/validate_data_retention_policy_tests.py`](../scripts/validate_data_retention_policy_tests.py))
- Raw-capture bucket lifecycle it builds on (R042):
  [`BlueprintCapture/storage.lifecycle.json`](../../BlueprintCapture/storage.lifecycle.json) +
  `BlueprintCapture/docs/STORAGE_RETENTION_POLICY_2026-07-09.md`
- Backup / disaster recovery for the same data (R053):
  `Blueprint-WebApp/docs/runbooks/DATA_BACKUP_AND_DR_RUNBOOK.md`

This validator mirrors the fail-closed idiom of `BlueprintCapture/scripts/validate_storage_lifecycle.py`.
Retention here is a **policy contract**; the *bucket-level* deletion/tiering of raw capture truth is
executed by the committed GCS lifecycle file in BlueprintCapture, which this policy references and
must stay consistent with.

## Capture-truth invariants (enforced by the validator)

1. **Raw capture bundles are authoritative and the longest-lived class.** `scenes/` retention must
   be `>= capture_truth_floor_days` (2555 = 7 years), and **no other data class may be retained
   longer than raw**.
2. **Derived / hosted artifacts get strictly shorter retention than raw.** World models, delivered
   packages, and pipeline intermediates are regenerable from raw, so they expire sooner.
3. **Financial / legal-hold data is never auto-deleted** — only `review_then_delete` /
   `retain_indefinite`, and it clears the 7-year financial floor.
4. **Firestore TTL entries are consistent** — a TTL-managed collection names its `ttl_field` and
   uses `ttl_delete`, so this policy and the applied Firestore TTL policy stay in lock-step.
5. Deleting a Firestore metadata record (e.g. `creatorCaptures`) **never** deletes the underlying
   raw capture truth in GCS, which keeps its own 10-year floor independently.

## Data classes → retention → action

### Storage (GCS bucket `blueprint-8c1ca.appspot.com`)

| Prefix | Class | Retention | Action | Enforced by |
|--------|-------|----------:|--------|-------------|
| `scenes/` | raw_capture_authoritative | 3650d (10y) | tier→delete | **GCS lifecycle** (`BlueprintCapture/storage.lifecycle.json`) |
| `site-worlds/` | derived_world_model | 730d (2y) | delete | **GCS lifecycle** |
| `marketplace-artifacts/` | derived_hosted_delivery | 365d (1y) | delete | **GCS lifecycle** |
| `scenes/*/captures/*/pipeline/` | derived_pipeline_intermediate | 180d | delete | **scheduled job** (see below) |

> `scenes/.../pipeline/` intermediates live **mid-path**. GCS lifecycle conditions only
> prefix-match the *start* of an object name, so these cannot be isolated by a bucket lifecycle
> rule and must be swept by a scheduled cleanup job — a documented human/ops step, not committed
> lifecycle. They are regenerable from raw.

### Firestore (project `blueprint-8c1ca`)

| Collection | Class | Retention | Action | Enforced by |
|------------|-------|----------:|--------|-------------|
| `creatorPayouts` | financial | 2555d (7y) | review→delete | manual review (legal hold) |
| `buyerOrders` | financial | 2555d (7y) | review→delete | manual review (legal hold) |
| `marketplaceEntitlements` | access_grant | 730d | review→delete | manual review |
| `waitlistSubmissions` | pii_lead | 365d | ttl_delete | **Firestore TTL** (`expireAt`) |
| `inboundRequests` | pii_lead | 365d | ttl_delete | **Firestore TTL** (`expireAt`) |
| `contactRequests` | pii_lead | 730d | ttl_delete | **Firestore TTL** (`expireAt`) |
| `leadEnrichmentDossiers` | derived_lead | 365d | delete | scheduled job |
| `capture_jobs` | operational | 730d | review→delete | manual review |
| `creatorCaptures` | creator_operational | 90d after delisting | review→delete | manual review |
| `hostedSessions` | hosted_session_ephemeral | 90d | ttl_delete | **Firestore TTL** (`expiresAt`) |
| `action_ledger` | audit | 730d | delete | scheduled job |
| `answer_cache` | cache | 30d | ttl_delete | **Firestore TTL** (`expiresAt`) |

Any Firestore collection **not** listed inherits `default_firestore_retention` (2-year
review-then-delete). Add an explicit entry to override.

The conceptual classes map to real collections: **leads** = `waitlistSubmissions` /
`inboundRequests`; **orders** = `buyerOrders`; **entitlements** = `marketplaceEntitlements`.

## Enforced by committed config vs human/dashboard step

| Mechanism | Status | What it covers |
|-----------|--------|----------------|
| GCS lifecycle (`scenes/`) | **Committed + applied** (R042) | Raw capture tier/delete |
| GCS lifecycle (`site-worlds/`, `marketplace-artifacts/`) | **Committed policy** in this config; a matching bucket lifecycle rule must be applied | Derived artifact deletion |
| Firestore TTL | **Committed policy** (field + days here); TTL policy must be **applied per field** (human/dashboard step below) | PII leads, hosted sessions, caches |
| Scheduled cleanup jobs | **Documented human/ops step** | `scenes/.../pipeline/` intermediates, `leadEnrichmentDossiers`, `action_ledger` |
| Manual review batches | **Documented human/ops step** | financial, access grants, operational, creator metadata |

This policy file and validator are **code/config-enforced** (they fail CI closed on drift).
Actually *applying* Firestore TTL policies, creating the scheduled cleanup jobs, and extending
the GCS lifecycle to derived prefixes are **human/dashboard/gcloud steps** — they are called out
here so nothing reads as "done" when it is only specified.

## Apply / verify commands

```bash
# Validate the retention contract (also run in the pipeline pytest fast-lane gate):
python3 scripts/validate_data_retention_policy.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/validate_data_retention_policy_tests.py

# Apply a Firestore TTL policy for each ttl_delete collection (human/dashboard step).
# Example for waitlistSubmissions.expireAt:
gcloud firestore fields ttl update expireAt \
    --collection-group=waitlistSubmissions \
    --enable-ttl \
    --project=blueprint-8c1ca --database="(default)"

# Read back an applied TTL policy:
gcloud firestore fields ttl describe expireAt \
    --collection-group=waitlistSubmissions \
    --project=blueprint-8c1ca --database="(default)"

# Raw-capture bucket lifecycle (already committed in BlueprintCapture, R042):
gsutil lifecycle get gs://blueprint-8c1ca.appspot.com
```

> Changing any retention floor, the capture-truth floor, or turning a `review_then_delete`
> class into an auto-delete is a **deliberate policy change**: update this doc and the guardrail
> constants in `scripts/validate_data_retention_policy.py` together, re-run the validator, and get
> sign-off before applying. The validator fails closed on any drift toward premature deletion of
> capture truth.
