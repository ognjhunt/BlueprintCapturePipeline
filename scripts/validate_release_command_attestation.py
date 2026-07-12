#!/usr/bin/env python3
"""Validate the signed release-command subject after OIDC bundle verification."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def validate_subject(
    *,
    subject_path: Path,
    repository_sha: str,
    image_digest: str,
    release_id: str,
    workflow_run_ids: dict[str, str],
    evidence_root: Path | None,
) -> list[str]:
    try:
        subject = json.loads(subject_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["release_command_attestation_subject_unreadable"]
    blockers = []
    expected = {
        "schema_version": "blueprint.release_command_attestation_subject.v1",
        "repository_sha": repository_sha,
        "image_digest": image_digest,
        "release_id": release_id,
        "workflow_run_ids": workflow_run_ids,
    }
    for field, value in expected.items():
        if subject.get(field) != value:
            blockers.append(f"release_command_attestation_binding_mismatch:{field}")
    declared = subject.get("artifact_sha256s")
    if not isinstance(declared, dict) or not declared:
        blockers.append("release_command_attestation_artifact_hashes_missing")
    elif evidence_root is not None:
        for relative, expected_digest in declared.items():
            path = (evidence_root / str(relative)).resolve()
            if not path.is_relative_to(evidence_root.resolve()) or not path.is_file():
                blockers.append(f"release_command_attestation_artifact_missing:{relative}")
                continue
            observed = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
            if observed != expected_digest:
                blockers.append(f"release_command_attestation_artifact_mismatch:{relative}")
    return sorted(set(blockers))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True, type=Path)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--ci-run-id", required=True)
    parser.add_argument("--full-test-run-id", required=True)
    parser.add_argument("--codeql-run-id", required=True)
    parser.add_argument("--evidence-root", type=Path)
    args = parser.parse_args()
    blockers = validate_subject(
        subject_path=args.subject,
        repository_sha=args.repository_sha,
        image_digest=args.image_digest,
        release_id=args.release_id,
        workflow_run_ids={
            "ci": args.ci_run_id,
            "full_test": args.full_test_run_id,
            "codeql": args.codeql_run_id,
        },
        evidence_root=args.evidence_root,
    )
    for blocker in blockers:
        print(blocker)
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
