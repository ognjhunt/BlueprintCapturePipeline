#!/usr/bin/env python3
"""Upload a release bundle to versioned S3 Object Lock COMPLIANCE storage."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import tarfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.parse import urlsplit


POLICY_SCHEMA = "blueprint.release_evidence_retention_policy.v1"
MINIMUM_RETENTION_DAYS = 2555
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
IMAGE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class S3Client(Protocol):
    def get_object_lock_configuration(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def put_object(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def head_object(self, **kwargs: Any) -> Mapping[str, Any]: ...
    def get_object(self, **kwargs: Any) -> Mapping[str, Any]: ...


def _sha256(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    raw = digest.digest()
    return digest.hexdigest(), base64.b64encode(raw).decode("ascii")


def _readback_digest(body: Any) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        while True:
            chunk = body.read(1024 * 1024)
            if not chunk:
                break
            if not isinstance(chunk, bytes):
                raise TypeError("archive readback body must return bytes")
            digest.update(chunk)
            size += len(chunk)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    return digest.hexdigest(), size


def _validate_bundle_archive(
    bundle_path: Path,
    external_manifest: Mapping[str, Any],
) -> list[str]:
    blockers: list[str] = []
    try:
        with tarfile.open(bundle_path, mode="r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                blockers.append("bundle_archive_duplicate_member")
            if any(
                Path(name).is_absolute() or ".." in Path(name).parts for name in names
            ):
                blockers.append("bundle_archive_unsafe_member_path")
            manifest_members = [member for member in members if member.name == "manifest.json"]
            if len(manifest_members) != 1 or not manifest_members[0].isfile():
                blockers.append("bundle_embedded_manifest_missing_or_invalid")
                return blockers
            if manifest_members[0].size > 16 * 1024 * 1024:
                blockers.append("bundle_embedded_manifest_too_large")
                return blockers
            manifest_handle = archive.extractfile(manifest_members[0])
            if manifest_handle is None:
                blockers.append("bundle_embedded_manifest_unreadable")
                return blockers
            embedded = json.loads(manifest_handle.read().decode("utf-8"))
            if not isinstance(embedded, Mapping):
                blockers.append("bundle_embedded_manifest_not_object")
                return blockers
            expected_external = dict(external_manifest)
            for field in (
                "bundle_filename",
                "bundle_size_bytes",
                "bundle_sha256",
            ):
                expected_external.pop(field, None)
            if dict(embedded) != expected_external:
                blockers.append("bundle_embedded_manifest_mismatch")

            raw_entries = embedded.get("entries")
            entries = raw_entries if isinstance(raw_entries, list) else []
            expected_entries: dict[str, Mapping[str, Any]] = {}
            for raw_entry in entries:
                entry = raw_entry if isinstance(raw_entry, Mapping) else {}
                path = str(entry.get("path") or "")
                if not path or path in expected_entries:
                    blockers.append("bundle_manifest_entry_invalid_or_duplicate")
                    continue
                expected_entries[path] = entry
            if embedded.get("entry_count") != len(expected_entries):
                blockers.append("bundle_manifest_entry_count_mismatch")
            actual_members = {
                member.name: member for member in members if member.name != "manifest.json"
            }
            if set(actual_members) != set(expected_entries):
                blockers.append("bundle_archive_entry_set_mismatch")
            for name in sorted(set(actual_members) & set(expected_entries)):
                member = actual_members[name]
                if not member.isfile():
                    blockers.append(f"bundle_archive_entry_not_file:{name}")
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    blockers.append(f"bundle_archive_entry_unreadable:{name}")
                    continue
                digest, size = _readback_digest(handle)
                entry = expected_entries[name]
                if entry.get("sha256") != digest:
                    blockers.append(f"bundle_archive_entry_digest_mismatch:{name}")
                if entry.get("size_bytes") != size or member.size != size:
                    blockers.append(f"bundle_archive_entry_size_mismatch:{name}")
    except (OSError, tarfile.TarError, UnicodeError, json.JSONDecodeError):
        blockers.append("bundle_archive_malformed")
    return sorted(set(blockers))


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlsplit(uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError("archive URI must be s3://bucket/prefix without query or fragment")
    prefix = parsed.path.strip("/")
    return parsed.netloc, prefix


def archive_bundle(
    *,
    client: S3Client,
    bundle_path: Path,
    bundle_manifest: Mapping[str, Any],
    archive_uri: str,
    policy: Mapping[str, Any],
    now: datetime,
) -> dict[str, Any]:
    blockers: list[str] = []
    if policy.get("schema_version") != POLICY_SCHEMA:
        blockers.append("retention_policy_schema_invalid")
    if policy.get("object_lock_mode") != "COMPLIANCE":
        blockers.append("retention_policy_object_lock_mode_invalid")
    for field in (
        "require_bucket_object_lock",
        "require_version_id",
        "require_sha256_checksum_readback",
    ):
        if policy.get(field) is not True:
            blockers.append(f"retention_policy_control_disabled:{field}")
    if bundle_manifest.get("schema_version") != "blueprint.release_evidence_bundle.v1":
        blockers.append("bundle_manifest_schema_invalid")
    if bundle_manifest.get("status") != "ready_to_archive":
        blockers.append("bundle_not_ready_to_archive")
    repository_sha = str(bundle_manifest.get("repository_sha") or "").lower()
    image_digest = str(bundle_manifest.get("image_digest") or "").lower()
    if SHA_PATTERN.fullmatch(repository_sha) is None:
        blockers.append("bundle_repository_sha_invalid")
    if IMAGE_PATTERN.fullmatch(image_digest) is None:
        blockers.append("bundle_image_digest_invalid")
    if not bundle_path.is_file() or bundle_path.is_symlink():
        blockers.append("bundle_file_invalid")
        bundle_hex = bundle_b64 = ""
    else:
        bundle_hex, bundle_b64 = _sha256(bundle_path)
        if bundle_manifest.get("bundle_sha256") != bundle_hex:
            blockers.append("bundle_digest_mismatch")
        if bundle_manifest.get("bundle_size_bytes") != bundle_path.stat().st_size:
            blockers.append("bundle_size_mismatch")
        blockers.extend(_validate_bundle_archive(bundle_path, bundle_manifest))
    try:
        bucket, prefix = _parse_s3_uri(archive_uri)
    except ValueError as exc:
        blockers.append(str(exc))
        bucket = prefix = ""
    retention_days = policy.get("minimum_retention_days")
    if not isinstance(retention_days, int) or retention_days < MINIMUM_RETENTION_DAYS:
        blockers.append("minimum_retention_days_invalid")
        retention_days = 0
    retain_until = now.astimezone(timezone.utc) + timedelta(days=retention_days)
    key = "/".join(
        part
        for part in (
            prefix,
            repository_sha,
            image_digest.removeprefix("sha256:"),
            f"release-evidence-{bundle_hex}.tar.gz" if bundle_hex else "invalid",
        )
        if part
    )
    version_id: str | None = None
    if not blockers:
        try:
            lock = client.get_object_lock_configuration(Bucket=bucket)
        except Exception as exc:  # noqa: BLE001 - provider errors become evidence blockers
            blockers.append(f"archive_object_lock_check_failed:{type(exc).__name__}")
        else:
            configuration = lock.get("ObjectLockConfiguration")
            if not isinstance(configuration, Mapping) or configuration.get("ObjectLockEnabled") != "Enabled":
                blockers.append("archive_bucket_object_lock_not_enabled")
    if not blockers:
        try:
            with bundle_path.open("rb") as handle:
                response = client.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=handle,
                    ContentLength=bundle_path.stat().st_size,
                    ChecksumAlgorithm="SHA256",
                    ChecksumSHA256=bundle_b64,
                    ObjectLockMode="COMPLIANCE",
                    ObjectLockRetainUntilDate=retain_until,
                    Metadata={
                        "repository-sha": repository_sha,
                        "image-digest": image_digest,
                        "bundle-sha256": bundle_hex,
                    },
                )
        except Exception as exc:  # noqa: BLE001 - provider errors become evidence blockers
            blockers.append(f"archive_upload_failed:{type(exc).__name__}")
        else:
            version_id = str(response.get("VersionId") or "") or None
            if not version_id:
                blockers.append("archive_version_id_missing")
            response_checksum = str(response.get("ChecksumSHA256") or "")
            if response_checksum and response_checksum != bundle_b64:
                blockers.append("archive_put_checksum_mismatch")
    head: Mapping[str, Any] = {}
    if not blockers and version_id is not None:
        try:
            head = client.head_object(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
                ChecksumMode="ENABLED",
            )
        except Exception as exc:  # noqa: BLE001 - provider errors become evidence blockers
            blockers.append(f"archive_readback_failed:{type(exc).__name__}")
        else:
            if head.get("ObjectLockMode") != "COMPLIANCE":
                blockers.append("archive_readback_object_lock_mode_invalid")
            readback_until = head.get("ObjectLockRetainUntilDate")
            if not isinstance(readback_until, datetime) or readback_until < retain_until - timedelta(seconds=1):
                blockers.append("archive_readback_retention_too_short")
            if head.get("ContentLength") != bundle_path.stat().st_size:
                blockers.append("archive_readback_size_mismatch")
            if str(head.get("ChecksumSHA256") or "") != bundle_b64:
                blockers.append("archive_readback_checksum_mismatch")
            metadata = head.get("Metadata")
            expected_metadata = {
                "repository-sha": repository_sha,
                "image-digest": image_digest,
                "bundle-sha256": bundle_hex,
            }
            if not isinstance(metadata, Mapping) or any(
                metadata.get(key) != value for key, value in expected_metadata.items()
            ):
                blockers.append("archive_readback_metadata_mismatch")
    restore_readback_verified = False
    if not blockers and version_id is not None:
        try:
            response = client.get_object(
                Bucket=bucket,
                Key=key,
                VersionId=version_id,
                ChecksumMode="ENABLED",
            )
            body = response.get("Body")
            if body is None:
                raise ValueError("archive readback body missing")
            readback_hex, readback_size = _readback_digest(body)
        except Exception as exc:  # noqa: BLE001 - provider errors become evidence blockers
            blockers.append(f"archive_restore_readback_failed:{type(exc).__name__}")
        else:
            if readback_hex != bundle_hex:
                blockers.append("archive_restore_readback_digest_mismatch")
            if readback_size != bundle_path.stat().st_size:
                blockers.append("archive_restore_readback_size_mismatch")
            response_checksum = str(response.get("ChecksumSHA256") or "")
            if response_checksum and response_checksum != bundle_b64:
                blockers.append("archive_restore_readback_checksum_mismatch")
            restore_readback_verified = not blockers
    blockers = sorted(set(blockers))
    return {
        "schema_version": "blueprint.immutable_release_evidence_receipt.v1",
        "generated_at": now.astimezone(timezone.utc).isoformat(),
        "status": "archived_immutable" if not blockers else "blocked",
        "repository_sha": repository_sha or None,
        "image_digest": image_digest or None,
        "bundle_sha256": bundle_hex or None,
        "archive_uri": f"s3://{bucket}/{key}" if bucket and key else None,
        "version_id": version_id,
        "object_lock_mode": "COMPLIANCE" if not blockers else None,
        "retain_until": retain_until.isoformat() if retention_days else None,
        "readback_verified": not blockers,
        "restore_readback_verified": restore_readback_verified and not blockers,
        "blockers": blockers,
        "claim_boundary": {
            "receipt_proves_archive_object_lock_not_release_correctness": True,
            "legal_hold_and_eventual_deletion_are_separate_owner_controls": True,
        },
    }


def build_retention_envelope(
    *,
    receipt: Mapping[str, Any],
    source_artifact_digest: str,
    now: datetime,
) -> dict[str, Any]:
    """Wrap an archive receipt for direct release-evidence graph ingestion."""

    generated_at = now.astimezone(timezone.utc)
    return {
        "schema_version": "blueprint.release_evidence.v1",
        "evidence_id": "immutable_retention",
        "evidence_schema_version": "blueprint.immutable_release_evidence_receipt.v1",
        "status": receipt.get("status"),
        "repository_sha": receipt.get("repository_sha"),
        "image_digest": receipt.get("image_digest"),
        "generated_at": generated_at.isoformat(),
        "expires_at": (generated_at + timedelta(hours=24)).isoformat(),
        "source_artifact_digest": source_artifact_digest,
        "evidence_uri": receipt.get("archive_uri"),
        "archive_version_id": receipt.get("version_id"),
        "claim_boundary": {
            "envelope_wraps_receipt_without_upgrading_its_status": True,
            "archive_receipt_is_not_release_correctness": True,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bundle-manifest", type=Path, required=True)
    parser.add_argument("--archive-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--envelope-output", type=Path)
    parser.add_argument(
        "--policy", type=Path, default=Path("docs/release_evidence_retention_policy.json")
    )
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.bundle_manifest.read_text(encoding="utf-8"))
        policy = json.loads(args.policy.read_text(encoding="utf-8"))
        import boto3
    except (OSError, UnicodeError, json.JSONDecodeError, ImportError) as exc:
        print(f"[release-evidence-archive] ERROR unavailable_input_or_runtime:{exc}", file=sys.stderr)
        return 1
    result = archive_bundle(
        client=boto3.client("s3"),
        bundle_path=args.bundle.resolve(),
        bundle_manifest=dict(manifest) if isinstance(manifest, Mapping) else {},
        archive_uri=args.archive_uri,
        policy=dict(policy) if isinstance(policy, Mapping) else {},
        now=datetime.now(timezone.utc),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.envelope_output is not None:
        receipt_hex, _receipt_b64 = _sha256(args.output)
        envelope = build_retention_envelope(
            receipt=result,
            source_artifact_digest=f"sha256:{receipt_hex}",
            now=datetime.now(timezone.utc),
        )
        args.envelope_output.parent.mkdir(parents=True, exist_ok=True)
        args.envelope_output.write_text(
            json.dumps(envelope, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(f"[release-evidence-archive] status={result['status']} output={args.output}")
    for blocker in result["blockers"]:
        print(f"[release-evidence-archive] blocker={blocker}", file=sys.stderr)
    return 0 if result["status"] == "archived_immutable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
