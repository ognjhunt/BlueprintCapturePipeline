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
- Cloud Run task dispatch and GPU privacy/video-to-world runner max instances
  are set to at least 25 through Terraform `max_concurrent_jobs`.
- p50 raw capture size: 1.2 GiB.
- p95 raw capture size: 8 GiB.
- p99 / absolute cap: 20 GiB.
- Derived artifact multiplier: 2.5x raw capture bytes.

At p50, the month adds about 360 GiB raw and 900 GiB derived artifacts. If every
capture landed at the p95 assumption, the month adds about 2400 GiB raw and
8400 GiB raw plus derived artifacts. If every capture hits the hard cap, raw
ingest is bounded at 6000 GiB before retention and lifecycle policies.

## Storage Lifecycle

The primary capture bucket lifecycle policy is checked in at
`deploy/storage/primary-capture-bucket-lifecycle.json` and applied with:

```bash
scripts/apply_primary_capture_bucket_lifecycle.sh "$BLUEPRINT_PRIMARY_CAPTURE_BUCKET"
```

Policy summary:

- `scenes/` and `targets/`: Nearline after 30 days, Coldline after 90 days,
  delete after 180 days.
- `tmp/`, `staging/`, and `debug/`: delete after 14 days.
- buyer delivery, marketplace, hosted-session, and robot-eval artifacts:
  delete after 365 days unless a contract-specific retention hold supersedes it.

This lifecycle policy is a cost and hygiene control. It is not a substitute for
legal deletion workflows, user-request deletion, backup/PITR, or contract hold
handling.

## Verification

Run:

```bash
python scripts/validate_beta_capacity_storage.py
python scripts/run_beta_intake_soak_test.py --dry-run
```

Before external beta, also run the soak harness against the real staging or
production intake endpoint with the target concurrency and archive the JSON
report under `output/beta_capacity/`.
