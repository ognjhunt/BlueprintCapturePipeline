#!/usr/bin/env python3
"""Fail closed when the backup / disaster-recovery config drifts away from durability.

Finding **R053 (P1)**: there was no documented/scheduled backup or DR plan for the
authoritative data (Firestore DB + GCS buckets). This validator guards the committed
backup/DR config (``configs/firestore_backup_schedule.json``) so it stays well-formed and
actually covers the authoritative surfaces, mirroring the fail-closed idiom of
``BlueprintCapture/scripts/validate_storage_lifecycle.py`` (R042).

Durability guardrails (see Blueprint-WebApp/docs/runbooks/DATA_BACKUP_AND_DR_RUNBOOK.md):

  * A Firestore managed export MUST be configured (project, destination bucket, schedule,
    an all-collections export, and a real ``gcloud firestore export`` command template).
  * The Firestore export destination MUST be a DIFFERENT bucket than the primary
    authoritative data bucket -- a primary-bucket incident must not also destroy the backups.
  * The primary authoritative capture bucket MUST be listed, with object versioning enabled
    (versioning is the bucket-level DR mechanism), and its versioning retention must clear the
    capture-truth floor so raw capture truth stays recoverable.
  * RPO/RTO targets must be present and sane, and backup retention must at least cover the RPO.

This validator does NOT touch live infrastructure; enabling versioning and creating the
scheduled export job are human/dashboard steps documented in the runbook.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PRIMARY_ROLE = "primary_authoritative"
BACKUP_ROLE = "backup_target"
_ALLOWED_ROLES = {PRIMARY_ROLE, BACKUP_ROLE}

# Capture-truth floor (7 years) -- mirrors capture_truth_floor_days in the retention policy
# and MIN_DELETE_AGE_DAYS in BlueprintCapture/scripts/validate_storage_lifecycle.py.
CAPTURE_TRUTH_FLOOR_DAYS = 2555

# Maximum acceptable Recovery Point Objective. A larger RPO means a documented policy change.
MAX_RPO_HOURS = 24


def fail(message: str) -> None:
    print(f"Backup/DR config validation failed: {message}", file=sys.stderr)
    sys.exit(1)


def _positive_number(value: object, name: str, source: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        fail(f"{source}: '{name}' must be a positive number")
    return float(value)


def _positive_int(value: object, name: str, source: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        fail(f"{source}: '{name}' must be a positive integer")
    return value


def _cron_ok(expr: object) -> bool:
    return isinstance(expr, str) and len([f for f in expr.split(" ") if f]) == 5


def validate_backup_config(config: object, source: str) -> None:
    if not isinstance(config, dict):
        fail(f"{source}: top-level backup config must be a JSON object")

    rpo = _positive_number(config.get("rpo_hours"), "rpo_hours", source)
    _positive_number(config.get("rto_hours"), "rto_hours", source)
    if rpo > MAX_RPO_HOURS:
        fail(f"{source}: rpo_hours {rpo} exceeds the {MAX_RPO_HOURS}h maximum RPO target")

    # ── Firestore export ──────────────────────────────────────────────────────
    export = config.get("firestore_export")
    if not isinstance(export, dict):
        fail(f"{source}: 'firestore_export' must be an object (Firestore is authoritative and must be backed up)")

    for field in ("project_id", "database", "destination_bucket", "destination_prefix"):
        if not isinstance(export.get(field), str) or not export.get(field):
            fail(f"{source}: firestore_export.{field} must be a non-empty string")

    collection_ids = export.get("collection_ids")
    if not isinstance(collection_ids, list):
        fail(f"{source}: firestore_export.collection_ids must be a list ([] = export all collections)")
    if collection_ids:
        fail(
            f"{source}: firestore_export.collection_ids must be [] (export the WHOLE authoritative "
            "control-plane); a partial export would silently leave collections un-backed-up"
        )

    if not _cron_ok(export.get("schedule_cron")):
        fail(f"{source}: firestore_export.schedule_cron must be a 5-field cron expression")

    backup_retention = _positive_int(
        export.get("backup_retention_days"), "firestore_export.backup_retention_days", source
    )
    if backup_retention < (rpo / 24.0):
        fail(f"{source}: firestore_export.backup_retention_days {backup_retention} is shorter than the RPO")

    template = export.get("gcloud_command_template")
    if not isinstance(template, str) or "gcloud firestore export" not in template:
        fail(f"{source}: firestore_export.gcloud_command_template must invoke 'gcloud firestore export'")
    for placeholder in ("{destination_bucket}", "{destination_prefix}", "{project_id}"):
        if placeholder not in template:
            fail(f"{source}: firestore_export.gcloud_command_template must contain {placeholder}")

    export_bucket = export["destination_bucket"]

    # ── Storage buckets ───────────────────────────────────────────────────────
    buckets = config.get("storage_buckets")
    if not isinstance(buckets, list) or not buckets:
        fail(f"{source}: 'storage_buckets' must be a non-empty array")

    primary = None
    backup_target = None
    for index, bucket in enumerate(buckets):
        label = f"{source}: storage_buckets[{index}]"
        if not isinstance(bucket, dict):
            fail(f"{label}: each bucket entry must be an object")
        name = bucket.get("bucket")
        if not isinstance(name, str) or not name:
            fail(f"{label}: 'bucket' must be a non-empty string")
        role = bucket.get("role")
        if role not in _ALLOWED_ROLES:
            fail(f"{label}: role {role!r} must be one of {sorted(_ALLOWED_ROLES)}")
        if not isinstance(bucket.get("object_versioning"), bool):
            fail(f"{label}: 'object_versioning' must be a boolean")

        if role == PRIMARY_ROLE:
            if primary is not None:
                fail(f"{label}: multiple {PRIMARY_ROLE!r} buckets; there must be exactly one authoritative primary")
            primary = bucket
        elif role == BACKUP_ROLE:
            backup_target = bucket

    if primary is None:
        fail(f"{source}: no {PRIMARY_ROLE!r} bucket listed; the authoritative capture bucket must be covered")

    # ── Durability guardrails ────────────────────────────────────────────────
    # Backups must not live in the primary bucket.
    if export_bucket == primary["bucket"]:
        fail(
            f"{source}: Firestore export destination_bucket {export_bucket!r} must differ from the "
            "primary authoritative bucket; a primary-bucket incident must not also destroy the backups"
        )

    # A dedicated backup-target bucket should carry the exports.
    if backup_target is not None and export_bucket != backup_target["bucket"]:
        fail(
            f"{source}: Firestore export destination_bucket {export_bucket!r} does not match the declared "
            f"backup_target bucket {backup_target['bucket']!r}"
        )

    # Primary bucket must have versioning ON with retention clearing the capture-truth floor.
    if not primary["object_versioning"]:
        fail(
            f"{source}: primary authoritative bucket {primary['bucket']!r} must have object_versioning=true "
            "(versioning is the bucket-level DR mechanism for capture truth)"
        )
    versioning_retention = _positive_int(
        primary.get("versioning_retention_days"),
        "storage_buckets(primary).versioning_retention_days",
        source,
    )
    if versioning_retention < CAPTURE_TRUTH_FLOOR_DAYS:
        fail(
            f"{source}: primary bucket versioning_retention_days {versioning_retention} is below the "
            f"capture-truth floor of {CAPTURE_TRUTH_FLOOR_DAYS}d; raw capture truth must stay recoverable"
        )

    # Restore path must be documented with an import command template.
    restore = config.get("restore")
    if not isinstance(restore, dict):
        fail(f"{source}: 'restore' must be an object documenting the restore procedure")
    import_template = restore.get("firestore_import_command_template")
    if not isinstance(import_template, str) or "gcloud firestore import" not in import_template:
        fail(f"{source}: restore.firestore_import_command_template must invoke 'gcloud firestore import'")

    print(
        "Backup/DR config validation passed: Firestore full export -> gs://"
        f"{export_bucket}/{export['destination_prefix']} (cron '{export['schedule_cron']}', "
        f"kept {backup_retention}d); primary bucket {primary['bucket']} versioned "
        f">= {CAPTURE_TRUTH_FLOOR_DAYS}d; RPO {rpo:.0f}h/RTO {config['rto_hours']:.0f}h; "
        "backups isolated from primary; restore import path documented."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the committed Firestore + storage backup/DR config (finding R053)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to the backup config JSON (defaults to configs/firestore_backup_schedule.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = args.config or (repo_root / "configs" / "firestore_backup_schedule.json")

    if not config_path.exists():
        fail(f"{config_path} is missing")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"{config_path} is not valid JSON: {exc}")

    validate_backup_config(config, config_path.name)


if __name__ == "__main__":
    main()
