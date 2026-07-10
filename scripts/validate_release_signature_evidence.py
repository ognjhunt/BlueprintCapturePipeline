#!/usr/bin/env python3
"""Validate a release-bound signature envelope and its raw proof artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit


SHA_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
ALLOWED_EVIDENCE_URI_SCHEMES = {"gs", "https", "oci", "s3"}


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _evidence_uri_is_durable(value: object) -> bool:
    text = str(value or "").strip()
    parsed = urlsplit(text)
    return (
        parsed.scheme.lower() in ALLOWED_EVIDENCE_URI_SCHEMES
        and bool(parsed.netloc)
        and not parsed.username
        and not parsed.password
        and not parsed.fragment
    )


def validate_signature_evidence(
    *,
    evidence_dir: Path,
    repository_sha: str,
    image_digest: str,
    now: datetime,
) -> dict[str, Any]:
    blockers: list[str] = []
    repository_sha = repository_sha.strip().lower()
    image_digest = image_digest.strip().lower()
    if SHA_PATTERN.fullmatch(repository_sha) is None:
        blockers.append("release_repository_sha_invalid")
    if IMAGE_PATTERN.fullmatch(image_digest) is None:
        blockers.append("release_image_digest_invalid")
    envelope_path = evidence_dir / "artifact_signature.json"
    if envelope_path.is_symlink() or not envelope_path.is_file():
        blockers.append("signature_envelope_missing_or_symlink")
        envelope: dict[str, Any] = {}
    else:
        try:
            raw_envelope = json.loads(envelope_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            blockers.append("signature_envelope_malformed")
            envelope = {}
        else:
            envelope = dict(raw_envelope) if isinstance(raw_envelope, Mapping) else {}
            if not envelope:
                blockers.append("signature_envelope_not_object")
    expected = {
        "schema_version": "blueprint.release_evidence.v1",
        "evidence_id": "artifact_signature",
        "evidence_schema_version": "blueprint.release_signature_verification.v1",
        "status": "verified",
        "repository_sha": repository_sha,
        "image_digest": image_digest,
    }
    for key, value in expected.items():
        if envelope.get(key) != value:
            blockers.append(f"signature_envelope_binding_invalid:{key}")
    source_digest = str(envelope.get("source_artifact_digest") or "")
    if DIGEST_PATTERN.fullmatch(source_digest) is None:
        blockers.append("signature_source_digest_invalid")
    if not _evidence_uri_is_durable(envelope.get("evidence_uri")):
        blockers.append("signature_evidence_uri_not_durable")
    generated_at = _parse_time(envelope.get("generated_at"))
    expires_at = _parse_time(envelope.get("expires_at"))
    current = now.astimezone(timezone.utc)
    if generated_at is None or expires_at is None or generated_at >= expires_at:
        blockers.append("signature_validity_interval_malformed")
    else:
        if generated_at > current + timedelta(minutes=5):
            blockers.append("signature_evidence_from_future")
        if expires_at <= current:
            blockers.append("signature_evidence_expired")
        if expires_at - generated_at > timedelta(hours=24):
            blockers.append("signature_validity_exceeds_release_policy")
    proof_digests: set[str] = set()
    if evidence_dir.is_dir() and not evidence_dir.is_symlink():
        for path in sorted(evidence_dir.rglob("*")):
            if path.is_symlink():
                blockers.append(
                    f"signature_evidence_symlink:{path.relative_to(evidence_dir).as_posix()}"
                )
                continue
            if not path.is_file() or path.name in {
                "artifact_signature.json",
                "run.json",
                "gate.json",
            }:
                continue
            try:
                proof_digests.add(_sha256(path))
            except OSError:
                blockers.append(
                    f"signature_proof_unreadable:{path.relative_to(evidence_dir).as_posix()}"
                )
    else:
        blockers.append("signature_evidence_directory_invalid")
    if source_digest not in proof_digests:
        blockers.append("signature_source_proof_missing_or_digest_mismatch")
    blockers = sorted(set(blockers))
    return {
        "schema_version": "blueprint.release_signature_evidence_gate.v1",
        "status": "passed" if not blockers else "blocked",
        "repository_sha": repository_sha or None,
        "image_digest": image_digest or None,
        "source_artifact_digest": source_digest or None,
        "proof_artifact_count": len(proof_digests),
        "blockers": blockers,
        "claim_boundary": {
            "gate_validates_supplied_signature_evidence_not_live_registry_state": True,
            "deployed_digest_readback_requires_separate_live_evidence": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--repository-sha", required=True)
    parser.add_argument("--image-digest", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = validate_signature_evidence(
        evidence_dir=args.evidence_dir.resolve(),
        repository_sha=args.repository_sha,
        image_digest=args.image_digest,
        now=datetime.now(timezone.utc),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(f"[release-signature-evidence] status={result['status']}")
    for blocker in result["blockers"]:
        print(f"[release-signature-evidence] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
