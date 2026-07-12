#!/usr/bin/env python3
"""Build the deterministic subject attested by GitHub's external OIDC signer."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

SHA = re.compile(r"^[0-9a-f]{40}$")
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build_attestation_subject(
    *,
    repository: str,
    repository_sha: str,
    image_digest: str,
    release_id: str,
    workflow_run_ids: dict[str, str],
    evidence_root: Path,
) -> dict:
    if SHA.fullmatch(repository_sha) is None:
        raise ValueError("release_repository_sha_invalid")
    if DIGEST.fullmatch(image_digest) is None:
        raise ValueError("release_image_digest_invalid")
    if not repository or not release_id:
        raise ValueError("release_repository_or_id_missing")
    if set(workflow_run_ids) != {"ci", "full_test", "codeql"} or any(
        not str(value).isdigit() for value in workflow_run_ids.values()
    ):
        raise ValueError("release_workflow_run_ids_invalid")
    artifacts = {
        path.relative_to(evidence_root).as_posix(): _sha256(path)
        for path in sorted(evidence_root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    if not artifacts:
        raise ValueError("release_evidence_artifacts_missing")
    return {
        "schema_version": "blueprint.release_command_attestation_subject.v1",
        "repository": repository,
        "repository_sha": repository_sha,
        "image_digest": image_digest,
        "release_id": release_id,
        "workflow_run_ids": workflow_run_ids,
        "artifact_sha256s": artifacts,
        "claim_boundary": {
            "subject_requires_external_oidc_attestation": True,
            "subject_file_alone_is_not_signature_evidence": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--ci-run-id", required=True)
    parser.add_argument("--full-test-run-id", required=True)
    parser.add_argument("--codeql-run-id", required=True)
    parser.add_argument("--evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = build_attestation_subject(
        repository=args.repository,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
