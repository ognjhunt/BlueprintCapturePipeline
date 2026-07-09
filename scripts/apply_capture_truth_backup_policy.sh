#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/apply_capture_truth_backup_policy.sh [--dry-run] --project <gcp-project> --bucket <bucket> [--database "(default)"]

Enables the minimum capture-truth durability controls:
  - Firestore point-in-time recovery
  - Firestore daily backup schedule
  - Firestore database delete protection
  - GCS object versioning on the primary capture bucket
  - GCS soft delete on the primary capture bucket

This script does not perform a restore drill. Use the runbook in
docs/CAPTURE_TRUTH_BACKUP_DR_RUNBOOK_2026-07-08.md after applying controls.
USAGE
}

dry_run=false
project="${GOOGLE_CLOUD_PROJECT:-${GCLOUD_PROJECT:-}}"
database="(default)"
bucket="${BLUEPRINT_PRIMARY_CAPTURE_BUCKET:-${PIPELINE_BUCKET:-}}"
backup_retention="14d"
soft_delete_duration="30d"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=true
      shift
      ;;
    --project)
      project="${2:-}"
      shift 2
      ;;
    --database)
      database="${2:-}"
      shift 2
      ;;
    --bucket)
      bucket="${2:-}"
      shift 2
      ;;
    --backup-retention)
      backup_retention="${2:-}"
      shift 2
      ;;
    --soft-delete-duration)
      soft_delete_duration="${2:-}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$project" || -z "$bucket" || -z "$database" ]]; then
  usage >&2
  echo "error: --project, --bucket, and --database are required" >&2
  exit 2
fi

bucket="${bucket#gs://}"

echo "project=$project"
echo "database=$database"
echo "bucket=gs://$bucket"
echo "backup_retention=$backup_retention"
echo "soft_delete_duration=$soft_delete_duration"
echo "dry_run=$dry_run"

if [[ "$dry_run" == "true" ]]; then
  echo "command: gcloud firestore databases update --project '$project' --database '$database' --enable-pitr --delete-protection"
  echo "command: gcloud firestore backups schedules create --project '$project' --database '$database' --retention '$backup_retention' --recurrence daily"
  echo "command: gcloud storage buckets update 'gs://$bucket' --project '$project' --versioning --soft-delete-duration '$soft_delete_duration'"
  exit 0
fi

gcloud firestore databases update \
  --project "$project" \
  --database "$database" \
  --enable-pitr \
  --delete-protection

gcloud firestore backups schedules create \
  --project "$project" \
  --database "$database" \
  --retention "$backup_retention" \
  --recurrence daily

gcloud storage buckets update "gs://$bucket" \
  --project "$project" \
  --versioning \
  --soft-delete-duration "$soft_delete_duration"
