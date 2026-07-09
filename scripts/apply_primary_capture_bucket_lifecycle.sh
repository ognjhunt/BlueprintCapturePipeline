#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/apply_primary_capture_bucket_lifecycle.sh [--dry-run] <bucket-name-or-gs-uri>

Applies deploy/storage/primary-capture-bucket-lifecycle.json to the primary
capture bucket. The bucket may also be provided through
BLUEPRINT_PRIMARY_CAPTURE_BUCKET or PIPELINE_BUCKET.
USAGE
}

dry_run=false
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=true
  shift
fi

bucket="${1:-${BLUEPRINT_PRIMARY_CAPTURE_BUCKET:-${PIPELINE_BUCKET:-}}}"
if [[ -z "$bucket" ]]; then
  usage >&2
  echo "error: missing bucket name" >&2
  exit 2
fi

bucket="${bucket#gs://}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lifecycle_file="$repo_root/deploy/storage/primary-capture-bucket-lifecycle.json"

if [[ ! -f "$lifecycle_file" ]]; then
  echo "error: lifecycle file not found: $lifecycle_file" >&2
  exit 2
fi

echo "bucket=gs://$bucket"
echo "lifecycle_file=$lifecycle_file"

if [[ "$dry_run" == "true" ]]; then
  echo "dry_run=true"
  echo "command: gcloud storage buckets update gs://$bucket --lifecycle-file=$lifecycle_file"
  exit 0
fi

gcloud storage buckets update "gs://$bucket" --lifecycle-file="$lifecycle_file"
