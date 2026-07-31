#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from blueprint_pipeline.artifact_storage import default_evidence_root  # noqa: E402


REQUIRED_SCRIPT_SNIPPETS = [
    "gcloud firestore databases update",
    "--enable-pitr",
    "--delete-protection",
    "gcloud firestore backups schedules create",
    "--recurrence daily",
    "--retention \"$backup_retention\"",
    "gcloud storage buckets update",
    "--versioning",
    "--soft-delete-duration \"$soft_delete_duration\"",
]

REQUIRED_RUNBOOK_SNIPPETS = [
    "Firestore RPO: 24 hours",
    "Firestore RTO: 4 hours",
    "object versioning plus 30 day soft delete",
    "scripts/apply_capture_truth_backup_policy.sh",
    "output/beta_capacity/backup_drill/",
    "capture_truth_restore_drill.v1",
    "Do not claim backup readiness from this runbook alone",
]


def _require(path: Path, snippets: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in text:
            raise AssertionError(f"{path} is missing required snippet: {snippet}")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string(value: Any) -> str:
    return str(value or "").strip()


def validate_restore_drill_artifact(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise AssertionError(f"restore drill artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AssertionError(f"restore drill artifact is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise AssertionError("restore drill artifact must be a JSON object")
    if payload.get("schema_version") != "capture_truth_restore_drill.v1":
        raise AssertionError("restore drill artifact must use capture_truth_restore_drill.v1")
    if payload.get("status") != "passed":
        raise AssertionError("restore drill artifact status must be passed")
    if payload.get("non_production_restore_project") is not True:
        raise AssertionError("restore drill must use a non-production restore project")
    if not _string(payload.get("source_project_id")):
        raise AssertionError("restore drill must name source_project_id")
    if not _string(payload.get("restore_project_id")):
        raise AssertionError("restore drill must name restore_project_id")
    if payload.get("source_project_id") == payload.get("restore_project_id"):
        raise AssertionError("restore drill source and restore projects must differ")

    firestore = _mapping(payload.get("firestore_restore"))
    storage = _mapping(payload.get("storage_restore"))
    transcript = _mapping(payload.get("transcript"))
    claim_boundary = _mapping(payload.get("claim_boundary"))

    if firestore.get("validation_status") != "passed":
        raise AssertionError("firestore_restore.validation_status must be passed")
    if not (firestore.get("backup_id") or firestore.get("pitr_timestamp")):
        raise AssertionError("firestore restore must include backup_id or pitr_timestamp")
    restored_documents = firestore.get("restored_document_paths")
    if not isinstance(restored_documents, list) or not restored_documents:
        raise AssertionError("firestore restore must include restored_document_paths")
    if not any(str(path).startswith("capture_submissions/") for path in restored_documents):
        raise AssertionError("firestore restore must include capture_submissions sample")

    if storage.get("validation_status") != "passed":
        raise AssertionError("storage_restore.validation_status must be passed")
    if not _string(storage.get("bucket")):
        raise AssertionError("storage restore must name bucket")
    if not _string(storage.get("restored_object")):
        raise AssertionError("storage restore must name restored_object")
    if not _string(storage.get("restored_checksum_sha256")):
        raise AssertionError("storage restore must include restored_checksum_sha256")
    if not _string(storage.get("raw_manifest_generation")):
        raise AssertionError("storage restore must include raw_manifest_generation")

    if transcript.get("secrets_redacted") is not True:
        raise AssertionError("restore transcript must be marked secrets_redacted=true")
    if claim_boundary.get("live_restore_drill_executed") is not True:
        raise AssertionError("claim boundary must prove live_restore_drill_executed=true")
    if claim_boundary.get("production_restore_performed") is not False:
        raise AssertionError("restore drill must not perform a production restore")

    return {
        "status": "passed",
        "restore_drill_artifact": str(path),
        "source_project_id": str(payload["source_project_id"]),
        "restore_project_id": str(payload["restore_project_id"]),
    }


def validate_backup_policy(
    repo_root: Path,
    restore_drill_artifact: Path | None = None,
    *,
    require_restore_drill: bool = False,
) -> dict[str, str]:
    script = repo_root / "scripts" / "apply_capture_truth_backup_policy.sh"
    runbook = repo_root / "docs" / "CAPTURE_TRUTH_BACKUP_DR_RUNBOOK_2026-07-08.md"
    _require(script, REQUIRED_SCRIPT_SNIPPETS)
    _require(runbook, REQUIRED_RUNBOOK_SNIPPETS)
    result = {
        "status": "passed",
        "script": str(script),
        "runbook": str(runbook),
        "restore_drill_required": str(require_restore_drill).lower(),
    }
    if restore_drill_artifact is not None:
        result.update(validate_restore_drill_artifact(restore_drill_artifact))
    elif require_restore_drill:
        default_path = (
            default_evidence_root()
            / "beta_capacity"
            / "backup_drill"
            / "capture_truth_restore_drill.json"
        )
        result.update(validate_restore_drill_artifact(default_path))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate capture truth backup/DR scripts and runbook.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--restore-drill-artifact")
    parser.add_argument("--require-restore-drill", action="store_true")
    args = parser.parse_args()
    restore_drill_artifact = (
        Path(args.restore_drill_artifact).expanduser().resolve()
        if args.restore_drill_artifact
        else None
    )
    result = validate_backup_policy(
        Path(args.repo_root).resolve(),
        restore_drill_artifact,
        require_restore_drill=args.require_restore_drill,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
