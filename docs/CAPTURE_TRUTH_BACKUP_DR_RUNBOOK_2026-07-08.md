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
with:

- source project id and non-production restore project id
- Firestore backup id or PITR timestamp used
- restored collection sample: `capture_submissions/<capture_id>`
- restored bucket object generation for
  `scenes/<scene_id>/captures/<capture_id>/raw/manifest.json`
- restore command transcript with secrets redacted
- validation result showing restored Firestore ids and storage object checksums

Do not claim backup readiness from this runbook alone. Backup readiness requires
the apply script to be run against the real project and a restore drill artifact
to be archived.
