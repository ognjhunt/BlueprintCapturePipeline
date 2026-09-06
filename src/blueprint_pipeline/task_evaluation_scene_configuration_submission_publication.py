"""Publish a validated scene submission without uploading publisher source bytes."""
from __future__ import annotations

import argparse
import fcntl
import stat
from contextlib import contextmanager
import hashlib
import json
import os
import pwd
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .decision_evidence_contracts import canonical_digest
from .public_scene_host_input_intake import _verified_checkout_head
from .task_evaluation_configured_scene_object_store import _object_missing, _streaming_readback
from .task_evaluation_launch_preparation_contract import (
    launch_preparation_request_digest,
    validate_launch_preparation_request,
)
from .task_evaluation_launch_preparation_worker import _s3_client, collect_preparation_references

SCHEMA = "task_evaluation_scene_configuration_submission_publication.v1"
MANIFEST = "bundle_manifest.v1.json"
REQUEST = "scene_configuration_preparation_request.v1.json"
PART_BYTES = 8 * 1024 * 1024
DEFAULT_LOCK_ROOT = Path("/var/lib/blueprint/pipeline-control-plane/submission-publication-locks")
_SAFE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class SubmissionPublicationError(ValueError):
    """A publication failed before submission or provider execution."""


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise SubmissionPublicationError("submission_publication_" + reason)


def _path(root: Path, relative: str) -> Path:
    item = Path(relative)
    _require(not item.is_absolute() and bool(item.parts)
             and all(_SAFE.fullmatch(part) for part in item.parts), "path_invalid")
    path = root / item
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink_forbidden")
    _require(path.resolve(strict=True).is_relative_to(root) and path.is_file(), "path_invalid")
    return path


def _sha(path: Path) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _json(path: Path) -> dict:
    _require(path.stat().st_size <= 8 * 1024 * 1024, "json_size_limit")
    value = json.loads(path.read_text())
    _require(isinstance(value, dict), "json_invalid")
    return value


def _validated_inventory(root: Path, expected_commit: str) -> tuple[dict, list[dict]]:
    manifest = _json(_path(root, MANIFEST))
    _require(manifest.get("schema_version") ==
             "task_evaluation_scene_configuration_submission_manifest.v1"
             and manifest.get("status") == "validated_pending_production_publication_and_submission"
             and manifest.get("manifest_digest") == canonical_digest(manifest, digest_field="manifest_digest"),
             "manifest_invalid")
    _require(manifest.get("source_commit") == expected_commit
             and re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "commit_mismatch")
    _require(manifest.get("raw_source_upload_allowed") is False
             and manifest.get("provider_allocated") is False, "raw_upload_forbidden")
    namespace = manifest.get("input_namespace")
    _require(isinstance(namespace, str) and _SAFE.fullmatch(namespace) is not None, "namespace_invalid")
    prefix = f"s3://blueprint/task-evaluation/production-inputs/{namespace}/"
    rows = manifest.get("files")
    _require(isinstance(rows, list) and 1 <= len(rows) <= 1024, "inventory_invalid")
    owned = manifest.get("source") == "owner_provided_completed_asset"
    if owned:
        from .task_evaluation_completed_scene_publication import verified_owner_source_inventory
        raw = verified_owner_source_inventory(root, manifest)
    else:
        publisher = _json(_path(root, "provenance/publisher_intake.v1.json"))
        _require(publisher.get("source_uploaded_by_blueprint") is False
                 and publisher.get("public_redistribution_allowed") is False, "publisher_rights_invalid")
        artifacts = publisher.get("artifacts")
        _require(isinstance(artifacts, list) and bool(artifacts), "publisher_inventory_invalid")
        raw = {(row["publisher_url"], row["sha256"], row["size_bytes"]) for row in artifacts}
    raw_digests = {row[1] for row in raw}
    paths, uris, retained = set(), set(), set()
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {
            "relative_path", "uri", "digest", "size_bytes", "publication_allowed",
        }, "inventory_row_invalid")
        relative, uri = row["relative_path"], row["uri"]
        _require(isinstance(relative, str) and isinstance(uri, str)
                 and relative not in paths and uri not in uris, "inventory_duplicate")
        path = _path(root, relative)
        digest, size = row["digest"], row["size_bytes"]
        _require(isinstance(digest, str) and _DIGEST.fullmatch(digest) is not None
                 and type(size) is int and size > 0 and _sha(path) == (digest, size),
                 "local_readback_mismatch")
        paths.add(relative)
        uris.add(uri)
        if row["publication_allowed"] is False:
            identity = (uri, digest, size)
            _require(relative.startswith("source/") and identity in raw
                     and (owned or urlsplit(uri).scheme == "https"), "raw_reference_invalid")
            retained.add(identity)
        else:
            _require(row["publication_allowed"] is True and not relative.startswith("source/")
                     and digest not in raw_digests and uri == prefix + relative,
                     "raw_upload_or_destination_forbidden")
    _require(retained == raw, "raw_inventory_incomplete")
    observed = set()
    for path in root.rglob("*"):
        _require(not path.is_symlink(), "symlink_forbidden")
        if not path.is_dir():
            observed.add(path.relative_to(root).as_posix())
    _require(observed == paths | {MANIFEST}, "directory_inventory_mismatch")
    request = validate_launch_preparation_request(_json(_path(root, REQUEST)))
    _require(request["run_mode"] == "scene_configuration"
             and request["expected_production_commit"] == expected_commit
             and request["publication"]["input_namespace"] == namespace
             and launch_preparation_request_digest(request) == manifest.get("request_digest"),
             "request_binding_mismatch")
    by_uri = {row["uri"]: row for row in rows}
    for ref in collect_preparation_references(request):
        row = by_uri.get(ref["uri"])
        _require(row is not None and all(row[key] == ref[key] for key in ("uri", "digest", "size_bytes")),
                 "request_reference_missing")
    return manifest, rows


def _existing(client: Any, bucket: str, key: str, digest: str, size: int) -> bool:
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _object_missing(exc):
            return False
        raise SubmissionPublicationError("submission_publication_remote_head_failed") from exc
    _require(head.get("ContentLength") == size, "existing_object_mismatch")
    _require(_streaming_readback(client=client, bucket=bucket, key=key, maximum_size_bytes=size)
             == (digest, size), "existing_object_mismatch")
    return True


def _upload(client: Any, bucket: str, key: str, path: Path, size: int) -> None:
    # Caller holds the production-host namespace lock. No unsupported
    # provider conditional-write headers or global atomicity claim are used.
    if size <= PART_BYTES:
        with path.open("rb") as source:
            client.put_object(Bucket=bucket, Key=key, Body=source, ContentLength=size)
        return
    upload_id = client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
    try:
        parts = []
        with path.open("rb") as source:
            while chunk := source.read(PART_BYTES):
                number = len(parts) + 1
                result = client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id,
                                            PartNumber=number, Body=chunk)
                parts.append({"PartNumber": number, "ETag": result["ETag"]})
        client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id,
                                         MultipartUpload={"Parts": parts})
    except Exception:
        client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        raise


def _publish_locked(
    *, manifest_path: str | Path, receipt_path: str | Path,
    expected_source_commit: str, service_account: str = "blueprint", client: Any | None = None,
) -> dict:
    """Publish admitted inventory only; this function never submits a run."""
    manifest_path, receipt_path = Path(manifest_path), Path(receipt_path)
    _require(manifest_path.name == MANIFEST
             and not any(p.is_symlink() for p in (manifest_path, *manifest_path.parents)),
             "manifest_path_invalid")
    root = manifest_path.parent.resolve(strict=True)
    _require(not receipt_path.resolve().is_relative_to(root)
             and not any(p.is_symlink() for p in (receipt_path, *receipt_path.parents)),
             "receipt_path_invalid")
    _require(pwd.getpwnam(service_account).pw_uid == os.geteuid(), "service_identity_mismatch")
    _require(_verified_checkout_head() == expected_source_commit, "execution_commit_mismatch")
    manifest, rows = _validated_inventory(root, expected_source_commit)
    if manifest.get("source") == "owner_provided_completed_asset":
        from .task_evaluation_owner_source_store import install_source
        for row in rows:
            if row["publication_allowed"] is False:
                install_source(source=_path(root, row["relative_path"]), uri=row["uri"],
                               digest=row["digest"], size_bytes=row["size_bytes"])
    manifest_hash, manifest_size = _sha(manifest_path)
    prior = _json(receipt_path) if receipt_path.exists() else None
    if prior is not None:
        _require(prior.get("schema_version") == SCHEMA and prior.get("status") == "published_and_read_back"
                 and prior.get("manifest_sha256") == manifest_hash
                 and prior.get("receipt_digest") == canonical_digest(prior, digest_field="receipt_digest"),
                 "receipt_conflict")
    active_client = client if client is not None else _s3_client("blueprint")
    publish_rows = [row for row in rows if row["publication_allowed"]]
    prefix = f"s3://blueprint/task-evaluation/production-inputs/{manifest['input_namespace']}/"
    publish_rows.append({"relative_path": MANIFEST, "uri": prefix + MANIFEST,
                         "digest": manifest_hash, "size_bytes": manifest_size})
    results = []
    for row in publish_rows:
        uri = urlsplit(row["uri"])
        path = _path(root, row["relative_path"])
        digest, size = row["digest"], row["size_bytes"]
        _require(_sha(path) == (digest, size), "source_changed_during_publication")
        reused = _existing(active_client, uri.netloc, uri.path.lstrip("/"), digest, size)
        uploaded = False
        if not reused:
            try:
                _upload(active_client, uri.netloc, uri.path.lstrip("/"), path, size)
                uploaded = True
            except Exception as exc:
                # An uncertain upload response may still have stored exact
                # bytes. Adopt only full-byte readback; this does not establish
                # exclusion of arbitrary external object-store writers.
                if not _existing(active_client, uri.netloc, uri.path.lstrip("/"), digest, size):
                    raise SubmissionPublicationError("submission_publication_upload_failed") from exc
            _require(_existing(active_client, uri.netloc, uri.path.lstrip("/"), digest, size),
                     "remote_readback_missing")
        results.append({**row, "upload_performed": uploaded,
                        "full_byte_service_account_readback_passed": True})
    result = {
        "schema_version": SCHEMA, "status": "published_and_read_back",
        "source_commit": expected_source_commit, "service_account": service_account,
        "input_namespace": manifest["input_namespace"], "manifest_sha256": manifest_hash,
        "manifest_digest": manifest["manifest_digest"], "request_digest": manifest["request_digest"],
        "published_objects": results, "host_only_source_objects": [
            row for row in rows if not row["publication_allowed"]],
        "raw_source_uploaded": False, "provider_allocated": False,
        "run_submitted": False, "full_byte_service_account_readback_passed": True,
        "single_writer_scope": "production_host_namespace_flock",
        "global_atomic_create_claimed": False,
    }
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    if prior is not None:
        return prior
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    with receipt_path.open("x") as output:
        json.dump(result, output, sort_keys=True, separators=(",", ":"))
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    return result


@contextmanager
def _namespace_lock(root: Path, namespace: str):
    _require(_SAFE.fullmatch(namespace) is not None, "namespace_invalid")
    _require(not any(p.is_symlink() for p in (root, *root.parents)), "lock_path_invalid")
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / f"{namespace}.lock",
                         os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        observed = os.fstat(descriptor)
        _require(stat.S_ISREG(observed.st_mode) and observed.st_uid == os.geteuid()
                 and not observed.st_mode & 0o077,
                 "lock_identity_invalid")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SubmissionPublicationError("submission_publication_namespace_writer_active") from exc
        yield
    finally:
        os.close(descriptor)


def publish_scene_configuration_submission(
    *, manifest_path: str | Path, receipt_path: str | Path,
    expected_source_commit: str, service_account: str = "blueprint",
    client: Any | None = None, lock_root: str | Path = DEFAULT_LOCK_ROOT,
) -> dict:
    """Serialize cooperating production publishers, then verify and publish.

    The operator must exclusively own this fresh namespace. This is not an
    object-store global compare-and-swap; arbitrary external writers are outside
    the claim. Existing differing bytes are always refused when observed.
    """
    path = Path(manifest_path)
    _require(not any(p.is_symlink() for p in (path, *path.parents)), "manifest_path_invalid")
    namespace = _json(path).get("input_namespace")
    _require(isinstance(namespace, str) and _SAFE.fullmatch(namespace) is not None,
             "namespace_invalid")
    with _namespace_lock(Path(lock_root), namespace):
        return _publish_locked(
            manifest_path=manifest_path, receipt_path=receipt_path,
            expected_source_commit=expected_source_commit,
            service_account=service_account, client=client,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--receipt-out", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--service-account", default="blueprint")
    args = parser.parse_args()
    result = publish_scene_configuration_submission(
        manifest_path=args.manifest, receipt_path=args.receipt_out,
        expected_source_commit=args.expected_source_commit, service_account=args.service_account,
    )
    print(json.dumps({"status": result["status"], "receipt_digest": result["receipt_digest"],
                      "raw_source_uploaded": False, "run_submitted": False}))


if __name__ == "__main__":
    main()
