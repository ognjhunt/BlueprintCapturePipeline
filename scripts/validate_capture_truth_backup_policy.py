#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


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
    "Do not claim backup readiness from this runbook alone",
]


def _require(path: Path, snippets: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for snippet in snippets:
        if snippet not in text:
            raise AssertionError(f"{path} is missing required snippet: {snippet}")


def validate_backup_policy(repo_root: Path) -> dict[str, str]:
    script = repo_root / "scripts" / "apply_capture_truth_backup_policy.sh"
    runbook = repo_root / "docs" / "CAPTURE_TRUTH_BACKUP_DR_RUNBOOK_2026-07-08.md"
    _require(script, REQUIRED_SCRIPT_SNIPPETS)
    _require(runbook, REQUIRED_RUNBOOK_SNIPPETS)
    return {
        "status": "passed",
        "script": str(script),
        "runbook": str(runbook),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate capture truth backup/DR scripts and runbook.")
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    result = validate_backup_policy(Path(args.repo_root).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
