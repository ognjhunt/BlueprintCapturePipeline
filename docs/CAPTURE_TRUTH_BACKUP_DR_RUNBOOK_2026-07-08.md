# Capture Truth Backup and DR Runbook

Status: required before 100-user external beta.

Capture truth is non-regenerable. Firestore `capture_submissions`, hosted
session state, marketplace entitlement state, raw capture bucket objects, and
package artifacts must have a recovery path that is separate from lifecycle
cost controls.

## Recovery Targets

- Firestore RPO: 24 hours through scheduled daily backups, plus point-in-time
  recovery where supported.
- Firestore RTO: 4 hours for a documented restore into a non-production
  verification project.
- Primary capture bucket RPO: object versioning plus 30 day soft delete.
- Primary capture bucket RTO: 4 hours to restore a deleted raw capture prefix
  into a non-production verification prefix.

## Apply Controls

Dry-run first:

```bash
scripts/apply_capture_truth_backup_policy.sh \
  --dry-run \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --bucket "$BLUEPRINT_PRIMARY_CAPTURE_BUCKET" \
  --database "(default)"
```

Apply intentionally:

```bash
scripts/apply_capture_truth_backup_policy.sh \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --bucket "$BLUEPRINT_PRIMARY_CAPTURE_BUCKET" \
  --database "(default)"
```

The script enables:

- `gcloud firestore databases update --enable-pitr --delete-protection`
- `gcloud firestore backups schedules create --recurrence daily --retention 14d`
- `gcloud storage buckets update --versioning --soft-delete-duration 30d`

## Restore Drill Evidence

Before external beta, archive one drill under `output/beta_capacity/backup_drill/`
with a `capture_truth_restore_drill.v1` JSON artifact:

- source project id and non-production restore project id
- Firestore backup id or PITR timestamp used
- restored collection sample: `capture_submissions/<capture_id>`
- restored bucket object generation for
  `scenes/<scene_id>/captures/<capture_id>/raw/manifest.json`
- restore command transcript with secrets redacted
- validation result showing restored Firestore ids and storage object checksums

Minimum artifact shape:

```json
{
  "schema_version": "capture_truth_restore_drill.v1",
  "status": "passed",
  "source_project_id": "blueprint-prod",
  "restore_project_id": "blueprint-restore-drill",
  "non_production_restore_project": true,
  "firestore_restore": {
    "backup_id": "projects/.../backups/...",
    "validation_status": "passed",
    "restored_document_paths": ["capture_submissions/example-capture"]
  },
  "storage_restore": {
    "bucket": "gs://primary-capture-bucket",
    "restored_object": "restore-drill/scenes/example/captures/example/raw/manifest.json",
    "raw_manifest_generation": "1700000000000000",
    "restored_checksum_sha256": "redacted-example-checksum",
    "validation_status": "passed"
  },
  "transcript": {
    "path": "output/beta_capacity/backup_drill/transcript.redacted.txt",
    "secrets_redacted": true
  },
  "claim_boundary": {
    "live_restore_drill_executed": true,
    "production_restore_performed": false
  }
}
```

Validate the archived drill before claiming readiness:

```bash
python scripts/validate_capture_truth_backup_policy.py \
  --require-restore-drill \
  --restore-drill-artifact output/beta_capacity/backup_drill/capture_truth_restore_drill.json
```

Do not claim backup readiness from this runbook alone. Backup readiness requires
the apply script to be run against the real project and a restore drill artifact
to be archived.
