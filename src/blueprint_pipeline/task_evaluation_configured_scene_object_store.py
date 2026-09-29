"""Durably publish configured-scene artifacts with exact S3 readback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

from .paid_resource_admission import PaidResourceAdmissionGrant, require_paid_resource_admission_grant


DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/configured-scenes"
LARGE_ARTIFACT_KEY_PREFIX = f"{DEFAULT_KEY_PREFIX}/artifacts"
# Runtime-source wrapper layers are published under this artifact kind; the
# wrapper builder embeds the resulting URI, so the two must agree exactly.
EXTERNAL_LAYER_ARTIFACT_KIND = "native-runtime-source-layer"
_RANGE_READBACK_THRESHOLD_BYTES = 32 * 1024 * 1024
_RANGE_READBACK_CHUNK_BYTES = 8 * 1024 * 1024
_RANGE_READBACK_CONCURRENCY = 4
_SAFE_KEY_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")

_ARTIFACT_STORE_FILE_ENV = {
    "access_key": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ACCESS_KEY_ID_FILE",
    "secret_key": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_SECRET_ACCESS_KEY_FILE",
    "bucket": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_BUCKET_FILE",
    "endpoint": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_ENDPOINT_URL_FILE",
    "region": "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_REGION_FILE",
}
_LEGACY_OBJECT_STORE_FILE_ENV = {
    "access_key": "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE",
    "secret_key": "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE",
    "bucket": "BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE",
    "endpoint": "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE",
    "region": "BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE",
}
_EXPECTED_ARTIFACT_BUCKET_ENV = (
    "BLUEPRINT_TASK_EVALUATION_ARTIFACT_STORE_EXPECTED_BUCKET"
)


class TaskEvaluationConfiguredSceneObjectStoreError(RuntimeError):
    """A configured-scene object could not be published and read back."""


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _private_file_value(environment_name: str, *, required: bool) -> str:
    raw_path = str(os.getenv(environment_name) or "").strip()
    if not raw_path:
        if required:
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                f"configured_scene_object_store_configuration_missing:{environment_name}"
            )
        return ""
    path = Path(raw_path).expanduser()
    descriptor = -1
    try:
        if path.is_symlink():
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_object_store_secret_file_unsafe"
            )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        metadata = os.fstat(descriptor)
        mode = stat.S_IMODE(metadata.st_mode)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or mode & ~0o640
            or not mode & 0o440
        ):
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_object_store_secret_file_unsafe"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read(4097)
    except TaskEvaluationConfiguredSceneObjectStoreError:
        raise
    except OSError as exc:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_secret_file_unavailable"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(payload) > 4096:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_secret_file_unsafe"
        )
    try:
        value = payload.decode("utf-8").strip()
    except UnicodeError as exc:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_secret_file_unavailable"
        ) from exc
    if required and not value:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            f"configured_scene_object_store_configuration_missing:{environment_name}"
        )
    return value


def _client_from_file_environment(
    names: Mapping[str, str], *, require_endpoint_and_region: bool = False,
    checksums_when_required: bool = False, path_style: bool = False,
) -> tuple[Any, str]:
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.client import Config  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_client_unavailable"
        ) from exc
    access_key = _private_file_value(names["access_key"], required=True)
    secret_key = _private_file_value(names["secret_key"], required=True)
    bucket = _private_file_value(names["bucket"], required=True)
    endpoint = _private_file_value(
        names["endpoint"], required=require_endpoint_and_region
    )
    region = _private_file_value(
        names["region"], required=require_endpoint_and_region
    )
    # B2 rejects botocore's default flexible checksums on presigned PUTs;
    # remote-CPU clients send and validate them only when an API requires it.
    checksums = (
        {"request_checksum_calculation": "when_required",
         "response_checksum_validation": "when_required"}
        if checksums_when_required else {}
    )
    # A remote-CPU worker pins ``https://<B2 host>/<bucket>/<key prefix>/``,
    # so its URLs are path style whatever botocore's default becomes.
    addressing = {"s3": {"addressing_style": "path"}} if path_style else {}
    kwargs: dict[str, Any] = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": region or "us-east-1",
        "config": Config(signature_version="s3v4", **addressing, **checksums),
    }
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs), bucket


def _object_store_client() -> tuple[Any, str]:
    """Return the existing configured-scene/Spaces client unchanged."""

    return _client_from_file_environment(_LEGACY_OBJECT_STORE_FILE_ENV)


def _artifact_object_store_client() -> tuple[Any, str]:
    """Return the dedicated large-artifact client, or the legacy fallback."""

    # If any dedicated binding is present, require all five dedicated values;
    # never borrow a missing value from the legacy store. This allows durable
    # bundle/output bytes to move to B2 without rerouting configured-scene
    # publication or transient WAM objects away from Spaces.
    dedicated = any(
        str(os.getenv(name) or "").strip()
        for name in _ARTIFACT_STORE_FILE_ENV.values()
    )
    names = _ARTIFACT_STORE_FILE_ENV if dedicated else _LEGACY_OBJECT_STORE_FILE_ENV
    # Endpoint and region are part of the exact B2 account identity; unlike
    # the legacy AWS-compatible fallback, both are mandatory here.
    client, bucket = _client_from_file_environment(
        names, require_endpoint_and_region=dedicated
    )
    expected_bucket = str(os.getenv(_EXPECTED_ARTIFACT_BUCKET_ENV) or "").strip()
    if expected_bucket and bucket != expected_bucket:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_store_bucket_identity_mismatch"
        )
    return client, bucket


def remote_cpu_object_store() -> tuple[Any, str, str]:
    """The dedicated B2 store for remote CPU jobs (plan 14): client, bucket and region.

    Only the dedicated artifact-store binding qualifies, never the legacy
    fallback, because its region is part of the admitted data location.
    """

    if not any(str(os.getenv(name) or "").strip() for name in _ARTIFACT_STORE_FILE_ENV.values()):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "remote_cpu_object_store_not_dedicated"
        )
    client, bucket = _client_from_file_environment(
        _ARTIFACT_STORE_FILE_ENV, require_endpoint_and_region=True, checksums_when_required=True,
        path_style=True,
    )
    expected_bucket = str(os.getenv(_EXPECTED_ARTIFACT_BUCKET_ENV) or "").strip()
    if expected_bucket and bucket != expected_bucket:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_store_bucket_identity_mismatch"
        )
    # The admitted data location is the region the endpoint serves, not a self-declared one.
    region = _private_file_value(_ARTIFACT_STORE_FILE_ENV["region"], required=True)
    endpoint = urlsplit(_private_file_value(_ARTIFACT_STORE_FILE_ENV["endpoint"], required=True))
    if endpoint.scheme != "https" or region not in str(endpoint.hostname or "").split("."):
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_object_store_region_unbound")
    return client, bucket, region


def _safe_object_name(value: str) -> PurePosixPath:
    path = PurePosixPath(str(value or ""))
    if (
        path.is_absolute()
        or not path.parts
        or any(_SAFE_KEY_COMPONENT.fullmatch(part) is None for part in path.parts)
    ):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_object_name_invalid"
        )
    return path


def _object_missing(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    response = response if isinstance(response, dict) else {}
    metadata = response.get("ResponseMetadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    error = response.get("Error", {})
    error = error if isinstance(error, dict) else {}
    return (
        int(metadata.get("HTTPStatusCode") or 0) == 404
        or str(error.get("Code") or "").lower()
        in {"404", "nosuchkey", "notfound"}
        or isinstance(exc, KeyError)
    )


def _streaming_readback(
    *, client: Any, bucket: str, key: str, maximum_size_bytes: int
) -> tuple[str, int]:
    try:
        response = client.get_object(Bucket=bucket, Key=key)
        body = response["Body"]
        digest = hashlib.sha256()
        size = 0
        try:
            while True:
                chunk = body.read(min(1024 * 1024, maximum_size_bytes + 1 - size))
                if not chunk:
                    break
                size += len(chunk)
                if size > maximum_size_bytes:
                    raise TaskEvaluationConfiguredSceneObjectStoreError(
                        "configured_scene_artifact_readback_exceeds_limit"
                    )
                digest.update(chunk)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
    except TaskEvaluationConfiguredSceneObjectStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - S3-compatible clients vary
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_readback_failed"
        ) from exc
    return "sha256:" + digest.hexdigest(), size


def _ranged_readback(
    *, client: Any, bucket: str, key: str, size: int, etag: str
) -> tuple[str, int]:
    """Hash every byte in order using bounded, object-pinned range requests.

    Some S3-compatible stores stall a single multipart-object GET even when
    ranges are fast. At most four 8-MiB responses are retained at once. A
    changed object, ignored range, short body, or oversized body fails closed.
    """
    if not etag or size <= 0:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_range_identity_missing"
        )

    def read_range_once(start: int) -> bytes:
        end = min(start + _RANGE_READBACK_CHUNK_BYTES, size) - 1
        response = client.get_object(
            Bucket=bucket, Key=key, Range=f"bytes={start}-{end}", IfMatch=etag
        )
        body = response["Body"]
        try:
            expected_size = end - start + 1
            if (
                response.get("ETag") != etag
                or response.get("ContentRange") != f"bytes {start}-{end}/{size}"
                or response.get("ContentLength") != expected_size
                or response.get("ResponseMetadata", {}).get("HTTPStatusCode") != 206
            ):
                raise TaskEvaluationConfiguredSceneObjectStoreError(
                    "configured_scene_artifact_range_identity_mismatch"
                )
            payload = bytearray()
            # A stream read may return fewer bytes without reaching EOF.
            # Read one extra byte so an oversized response still fails closed.
            while len(payload) <= expected_size:
                chunk = body.read(expected_size + 1 - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) != expected_size:
                raise TaskEvaluationConfiguredSceneObjectStoreError(
                    "configured_scene_artifact_range_size_mismatch"
                )
            return bytes(payload)
        finally:
            body.close()

    def read_range(start: int) -> bytes:
        for attempt in range(3):
            try:
                return read_range_once(start)
            except TaskEvaluationConfiguredSceneObjectStoreError:
                raise  # Identity, size and content refusals must not become transient success.
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(0.25 * (attempt + 1))
        raise AssertionError("unreachable range retry state")

    digest = hashlib.sha256()
    read_size = 0
    batch_bytes = _RANGE_READBACK_CHUNK_BYTES * _RANGE_READBACK_CONCURRENCY
    try:
        with ThreadPoolExecutor(max_workers=_RANGE_READBACK_CONCURRENCY) as executor:
            for batch_start in range(0, size, batch_bytes):
                starts = range(
                    batch_start, min(batch_start + batch_bytes, size),
                    _RANGE_READBACK_CHUNK_BYTES,
                )
                # map preserves byte order; batching bounds completed responses
                # even if the first request in a batch finishes last.
                for payload in executor.map(read_range, starts):
                    digest.update(payload)
                    read_size += len(payload)
    except TaskEvaluationConfiguredSceneObjectStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - S3-compatible clients vary
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            f"configured_scene_artifact_readback_failed:{type(exc).__name__}"
        ) from exc
    return "sha256:" + digest.hexdigest(), read_size


def publish_configured_scene_artifact(
    *,
    path: str | Path,
    artifact_kind: str,
    client: Any | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Publish a large immutable artifact once and prove exact remote bytes.

    The key is independent of the run and source path, so identical provider
    bundles, provider outputs, and diagnostic checkpoint archives are reused.
    An existing object is never overwritten: its size and digest metadata must
    agree before a full streaming readback is accepted.
    """

    unresolved = Path(path)
    if unresolved.is_symlink():
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_source_invalid"
        )
    source = unresolved.expanduser().resolve()
    if not source.is_file():
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_source_invalid"
        )
    digest, size = _sha256_and_size(source)
    def upload(resolved_client, resolved_bucket, key, metadata):
        resolved_client.upload_file(str(source), resolved_bucket, key, ExtraArgs=metadata)
    return _publish_configured_scene_data(digest=digest, size=size, filename=source.name,
        artifact_kind=artifact_kind, upload=upload, client=client, bucket=bucket)


def publish_configured_scene_stream(
    *, write_stream: Callable, digest: str, size_bytes: int, filename: str,
    artifact_kind: str, client: Any | None = None, bucket: str | None = None,
) -> dict[str, Any]:
    """Upload a repeatable stream without a local archive, then read back every byte."""
    if (re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest)) is None
            or type(size_bytes) is not int or size_bytes < 1
            or _SAFE_KEY_COMPONENT.fullmatch(filename) is None):
        raise TaskEvaluationConfiguredSceneObjectStoreError('configured_scene_artifact_stream_invalid')
    from .object_store_multipart_stream import MultipartStream
    def upload(resolved_client, resolved_bucket, key, metadata):
        sink = MultipartStream(client=resolved_client, bucket=resolved_bucket, key=key,
            metadata=metadata, expected_digest=digest, expected_size=size_bytes)
        try:
            write_stream(sink)
            sink.finish()
        except BaseException:
            try:
                sink.abort()
            except Exception:
                pass  # Preserve the publication failure; no local evidence is evicted.
            raise
    return _publish_configured_scene_data(digest=digest, size=size_bytes, filename=filename,
        artifact_kind=artifact_kind, upload=upload, client=client, bucket=bucket)


def _publish_configured_scene_data(*, digest, size, filename, artifact_kind, upload, client, bucket):
    kind = _safe_object_name(artifact_kind)
    if len(kind.parts) != 1:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_kind_invalid"
        )
    digest_hex = digest.removeprefix("sha256:")
    key = str(
        PurePosixPath(LARGE_ARTIFACT_KEY_PREFIX)
        / kind
        / "sha256"
        / digest_hex
        / filename
    )
    resolved_client, resolved_bucket = (
        _artifact_object_store_client()
        if client is None or bucket is None
        else (client, bucket)
    )
    cache_hit = False
    upload_performed = False
    try:
        try:
            head = resolved_client.head_object(Bucket=resolved_bucket, Key=key)
            cache_hit = True
        except Exception as exc:  # noqa: BLE001 - provider exception shapes vary
            if not _object_missing(exc):
                raise
            upload(resolved_client, resolved_bucket, key, {
                "Metadata": {"sha256": digest_hex},
                "ContentType": "application/octet-stream",
            })
            upload_performed = True
            head = resolved_client.head_object(Bucket=resolved_bucket, Key=key)
        metadata = head.get("Metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        if (
            int(head.get("ContentLength") or -1) != size
            or metadata.get("sha256") != digest_hex
        ):
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_artifact_existing_identity_mismatch"
            )
        if size >= _RANGE_READBACK_THRESHOLD_BYTES:
            readback_digest, readback_size = _ranged_readback(
                client=resolved_client, bucket=resolved_bucket, key=key,
                size=size, etag=str(head.get("ETag") or ""),
            )
        else:
            readback_digest, readback_size = _streaming_readback(
                client=resolved_client,
                bucket=resolved_bucket,
                key=key,
                maximum_size_bytes=size,
            )
    except TaskEvaluationConfiguredSceneObjectStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - S3-compatible clients vary
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_publication_failed"
        ) from exc
    if readback_digest != digest or readback_size != size:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_readback_mismatch"
        )
    remote_verified_at = datetime.now(UTC)
    last_modified = head.get("LastModified")
    if isinstance(last_modified, datetime) and last_modified.tzinfo is not None:
        remote_verified_at = last_modified.astimezone(UTC)
    return {
        "schema_version": "task_evaluation_scene_artifact_reference.v1",
        "status": "remote_verified",
        "artifact_kind": str(kind),
        "uri": f"s3://{resolved_bucket}/{key}",
        "digest": digest,
        "size_bytes": size,
        "cache_hit": cache_hit,
        "upload_performed": upload_performed,
        "content_addressed_key": True,
        "remote_identity_verified": True,
        "full_byte_service_account_readback_passed": True,
        "remote_verified_at": remote_verified_at.isoformat().replace("+00:00", "Z"),
        "readback_digest": readback_digest,
        "readback_size_bytes": readback_size,
        "raw_secret_values_recorded": False,
    }


def publish_runtime_source_external_layers(
    receipt: Mapping[str, Any],
    *,
    client: Any | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Publish every external layer a runtime-source build receipt names.

    Each layer must land at exactly the URI the wrapper embeds; a bucket or
    prefix that disagrees with the build is refused rather than republished.
    """

    layers = receipt.get("external_layers") if isinstance(receipt, Mapping) else None
    if (
        receipt.get("schema_version") != "task_evaluation_adapter_bundle_build_receipt.v1"
        or receipt.get("role") != "runtime_source"
        or not isinstance(layers, list)
    ):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_runtime_source_receipt_invalid"
        )
    published: list[dict[str, Any]] = []
    for row in layers:
        if not isinstance(row, Mapping):
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_runtime_source_receipt_invalid"
            )
        reference = publish_configured_scene_artifact(
            path=str(row.get("store_path") or ""),
            artifact_kind=EXTERNAL_LAYER_ARTIFACT_KIND,
            client=client,
            bucket=bucket,
        )
        if (
            reference["uri"] != row.get("uri")
            or reference["digest"] != row.get("sha256")
            or reference["size_bytes"] != row.get("size_bytes")
        ):
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_runtime_source_layer_uri_mismatch"
            )
        published.append({**reference, "relative_path": row.get("relative_path")})
    return {
        "schema_version": "task_evaluation_runtime_source_layer_publication.v1",
        "status": "remote_verified",
        "wrapper_sha256": receipt.get("sha256"),
        "layer_count": len(published),
        "layers": published,
        "raw_secret_values_recorded": False,
    }


def presign_configured_scene_artifact(
    *, reference: Mapping[str, Any], expiration_seconds: int
) -> str:
    """Issue a bounded GET URL for one exact verified CAS reference."""

    if (
        not isinstance(expiration_seconds, int)
        or isinstance(expiration_seconds, bool)
        or expiration_seconds < 1
        or expiration_seconds > 7 * 24 * 60 * 60
    ):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_presign_expiration_invalid"
        )
    uri = str(reference.get("uri") or "")
    parsed = urlsplit(uri)
    digest = str(reference.get("digest") or "")
    kind = str(reference.get("artifact_kind") or "")
    key = parsed.path.lstrip("/")
    prefix = LARGE_ARTIFACT_KEY_PREFIX.strip("/") + "/"
    if (
        reference.get("schema_version")
        != "task_evaluation_scene_artifact_reference.v1"
        or reference.get("status") != "remote_verified"
        or parsed.scheme != "s3"
        or not parsed.netloc
        or not key.startswith(prefix)
        or _SAFE_KEY_COMPONENT.fullmatch(kind) is None
        or f"/{kind}/sha256/" not in "/" + key
        or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        or f"/sha256/{digest.removeprefix('sha256:')}/" not in key
        or reference.get("content_addressed_key") is not True
        or reference.get("remote_identity_verified") is not True
        or reference.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_reference_invalid"
        )
    client, bucket = _artifact_object_store_client()
    if parsed.netloc != bucket:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_reference_invalid"
        )
    try:
        return str(
            client.generate_presigned_url(
                "get_object",
                Params={
                    "Bucket": bucket,
                    "Key": key,
                    "ResponseCacheControl": "no-store, max-age=0",
                },
                ExpiresIn=expiration_seconds,
                HttpMethod="GET",
            )
        )
    except Exception as exc:  # noqa: BLE001 - S3-compatible clients vary
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_presign_failed"
        ) from exc


def _reference_location(
    reference: dict[str, Any],
    *,
    client: Any | None,
    bucket: str | None,
    maximum_size_bytes: int | None = None,
) -> tuple[Any, str, str, str, int]:
    """The client, bucket, key, digest and size of a verified CAS reference.

    Anything else, or one larger than ``maximum_size_bytes``, is
    ``configured_scene_artifact_reference_invalid``, raised before any client
    is built.
    """

    uri = str(reference.get("uri") or "")
    parsed = urlsplit(uri)
    expected_digest = str(reference.get("digest") or "")
    expected_size = reference.get("size_bytes")
    kind = str(reference.get("artifact_kind") or "")
    key = parsed.path.lstrip("/")
    prefix = LARGE_ARTIFACT_KEY_PREFIX.strip("/") + "/"
    if (
        parsed.scheme != "s3"
        or not parsed.netloc
        or not key.startswith(prefix)
        or _SAFE_KEY_COMPONENT.fullmatch(kind) is None
        or f"/{kind}/sha256/" not in "/" + key
        or re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest) is None
        or f"/sha256/{expected_digest.removeprefix('sha256:')}/" not in key
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
        or (maximum_size_bytes is not None and expected_size > maximum_size_bytes)
        or reference.get("remote_identity_verified") is not True
        or reference.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_reference_invalid"
        )
    resolved_client, resolved_bucket = (
        _artifact_object_store_client()
        if client is None or bucket is None
        else (client, bucket)
    )
    if parsed.netloc != resolved_bucket:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_reference_invalid"
        )
    return resolved_client, resolved_bucket, key, expected_digest, expected_size


def verify_configured_scene_artifact(
    *,
    reference: dict[str, Any],
    client: Any | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Check with one HEAD request, reading no bytes, that a CAS reference's object is still there.

    Its size and ``sha256`` metadata must be the reference's
    (``configured_scene_artifact_existing_identity_mismatch``); an object that is
    gone is ``configured_scene_artifact_missing``, and any other failure
    ``configured_scene_artifact_head_failed``.
    """

    resolved_client, resolved_bucket, key, expected_digest, expected_size = _reference_location(
        reference, client=client, bucket=bucket
    )
    try:
        head = resolved_client.head_object(Bucket=resolved_bucket, Key=key)
        metadata = head.get("Metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        matches = (
            int(head.get("ContentLength") or -1) == expected_size
            and metadata.get("sha256") == expected_digest.removeprefix("sha256:")
        )
    except Exception as exc:  # noqa: BLE001 - provider exception shapes vary
        if _object_missing(exc):
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_artifact_missing"
            ) from exc
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_head_failed"
        ) from exc
    if not matches:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_existing_identity_mismatch"
        )
    return {
        "schema_version": "task_evaluation_scene_artifact_verification.v1",
        "status": "remote_present",
        "digest": expected_digest,
        "size_bytes": expected_size,
        "bytes_read": 0,
    }


def materialize_configured_scene_artifact(
    *,
    reference: dict[str, Any],
    destination: str | Path,
    maximum_size_bytes: int,
    client: Any | None = None,
    bucket: str | None = None,
) -> dict[str, Any]:
    """Stream one CAS artifact into bounded same-filesystem staging.

    The destination is exposed only after the declared size and digest match.
    A partial transfer is removed and can never be mistaken for retained
    evidence.
    """

    if not isinstance(maximum_size_bytes, int) or isinstance(maximum_size_bytes, bool) or maximum_size_bytes < 1:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_materialization_limit_invalid"
        )
    resolved_client, resolved_bucket, key, expected_digest, expected_size = _reference_location(
        reference, client=client, bucket=bucket, maximum_size_bytes=maximum_size_bytes
    )
    target = Path(destination).expanduser().absolute()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
    if target.exists() or target.is_symlink():
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_destination_exists"
        )
    temporary_path: Path | None = None
    try:
        response = resolved_client.get_object(Bucket=resolved_bucket, Key=key)
        body = response["Body"]
        digest = hashlib.sha256()
        size = 0
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".partial", dir=target.parent
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                while True:
                    chunk = body.read(min(1024 * 1024, maximum_size_bytes + 1 - size))
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > maximum_size_bytes:
                        raise TaskEvaluationConfiguredSceneObjectStoreError(
                            "configured_scene_artifact_materialization_exceeds_limit"
                        )
                    digest.update(chunk)
                    stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
        observed_digest = "sha256:" + digest.hexdigest()
        if size != expected_size or observed_digest != expected_digest:
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_artifact_materialization_mismatch"
            )
        temporary_path.chmod(0o440)
        os.replace(temporary_path, target)
        temporary_path = None
    except TaskEvaluationConfiguredSceneObjectStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - S3-compatible clients vary
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_materialization_failed"
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    if _sha256_and_size(target) != (expected_digest, expected_size):
        target.unlink(missing_ok=True)
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_materialization_readback_failed"
        )
    return {
        "schema_version": "task_evaluation_scene_artifact_materialization.v1",
        "status": "completed",
        "path": str(target),
        "digest": expected_digest,
        "size_bytes": expected_size,
        "bounded_staging_maximum_bytes": maximum_size_bytes,
        "local_full_byte_readback_passed": True,
        "raw_secret_values_recorded": False,
    }


def configured_scene_object_store_publisher(
    *, key_prefix: str = DEFAULT_KEY_PREFIX
) -> Callable[..., dict[str, Any]]:
    """Return the production publisher consumed by configured-scene sealing."""

    client, bucket = _object_store_client()
    prefix = PurePosixPath(str(key_prefix).strip("/"))
    if not prefix.parts or any(
        _SAFE_KEY_COMPONENT.fullmatch(part) is None for part in prefix.parts
    ):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_prefix_invalid"
    )

    def publish(*, path: Path, object_name: str) -> dict[str, Any]:
        unresolved = Path(path)
        if unresolved.is_symlink():
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_object_store_source_invalid"
            )
        source = unresolved.resolve()
        if not source.is_file():
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_object_store_source_invalid"
            )
        relative = _safe_object_name(object_name)
        digest, size = _sha256_and_size(source)
        digest_hex = digest.removeprefix("sha256:")
        key = str(
            prefix
            / relative.parent
            / "sha256"
            / digest_hex
            / relative.name
        )
        try:
            client.upload_file(str(source), bucket, key)
            response = client.get_object(Bucket=bucket, Key=key)
            body = response["Body"]
            observed = hashlib.sha256()
            observed_size = 0
            try:
                for chunk in iter(lambda: body.read(1024 * 1024), b""):
                    observed.update(chunk)
                    observed_size += len(chunk)
            finally:
                close = getattr(body, "close", None)
                if callable(close):
                    close()
        except Exception as exc:  # noqa: BLE001 - S3-compatible clients vary
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_object_store_upload_or_readback_failed"
            ) from exc
        readback_digest = "sha256:" + observed.hexdigest()
        if readback_digest != digest or observed_size != size:
            raise TaskEvaluationConfiguredSceneObjectStoreError(
                "configured_scene_object_store_readback_mismatch"
            )
        return {
            "uri": f"s3://{bucket}/{key}",
            "digest": digest,
            "size_bytes": size,
            "full_byte_service_account_readback_passed": True,
            "readback_digest": readback_digest,
            "readback_size_bytes": observed_size,
            "content_addressed_key": True,
            "raw_secret_values_recorded": False,
        }

    return publish


def validate_configured_scene_object_store_configuration(
    *, key_prefix: str = DEFAULT_KEY_PREFIX
) -> dict[str, Any]:
    """Validate the local publication client without contacting object storage.

    This proves that the deployed caller can read its file-backed credentials,
    construct the configured S3 client, and resolve the exact safe namespace.
    It deliberately does not claim remote bucket or IAM authority; those remain
    proven only by the publisher's byte-for-byte upload/readback receipt.
    """

    _client, bucket = _object_store_client()
    prefix = PurePosixPath(str(key_prefix).strip("/"))
    if not prefix.parts or any(
        _SAFE_KEY_COMPONENT.fullmatch(part) is None for part in prefix.parts
    ):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_prefix_invalid"
        )
    if _SAFE_KEY_COMPONENT.fullmatch(bucket) is None:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_bucket_invalid"
        )
    return {
        "schema_version": "task_evaluation_configured_scene_object_store_readiness.v1",
        "status": "locally_configured",
        "key_prefix": str(prefix),
        "credential_files_validated": True,
        "client_constructed": True,
        "remote_bucket_authority_verified": False,
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
    }


def read_configured_scene_object(
    *, reference: dict[str, Any], maximum_size_bytes: int = 16 * 1024 * 1024
) -> bytes:
    """Read one digest-bound configured-scene object from the canonical store."""

    uri = str(reference.get("uri") or "")
    parsed = urlsplit(uri)
    expected_digest = str(reference.get("digest") or "")
    expected_size = reference.get("size_bytes")
    client, configured_bucket = _object_store_client()
    if parsed.netloc != configured_bucket and any(os.getenv(name) for name in _ARTIFACT_STORE_FILE_ENV.values()):
        client, configured_bucket = _artifact_object_store_client()
    key = parsed.path.lstrip("/")
    prefix = DEFAULT_KEY_PREFIX.strip("/") + "/"
    if (
        parsed.scheme != "s3"
        or parsed.netloc != configured_bucket
        or not key.startswith(prefix)
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_digest)
        or f"/sha256/{expected_digest.removeprefix('sha256:')}/" not in key
        or not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 1
        or expected_size > maximum_size_bytes
    ):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_read_reference_invalid"
        )
    try:
        response = client.get_object(Bucket=configured_bucket, Key=key)
        body = response["Body"]
        try:
            payload = body.read(maximum_size_bytes + 1)
        finally:
            close = getattr(body, "close", None)
            if callable(close):
                close()
    except Exception as exc:  # noqa: BLE001 - S3-compatible clients vary
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_readback_failed"
        ) from exc
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    if len(payload) != expected_size or digest != expected_digest:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_readback_mismatch"
        )
    return payload


REMOTE_CPU_STAGING_KEY_PREFIX = f"{DEFAULT_KEY_PREFIX}/remote-cpu/staging/"
REMOTE_CPU_ARTIFACT_KINDS = frozenset(
    {"remote-cpu-input", "remote-cpu-output", "remote-cpu-source", "remote-cpu-sentinel"}
)
_REMOTE_CPU_ATTEMPT = re.compile(r"(rcj-[a-z]{2}-[0-9a-f]{24})/\1-a[1-9][0-9]{0,2}-[0-9a-f]{32}")
# The longest remote-CPU attempt the contract allows: start allowance, a one-hour task and the grace.
MAX_REMOTE_CPU_PUT_SECONDS = 4 * 3600
REMOTE_CPU_RESOURCE_CLASS = "cloud_run_cpu_job"
_MAX_SINGLE_COPY_BYTES = 5 * 1024**3
_COPY_PART_BYTES = 512 * 1024**2


def _remote_cpu_key(uri: str, *, bucket: str, cas: bool = False, prefix: bool = False) -> str:
    """The key of one object in one attempt's staging prefix (or that prefix), or of one CAS object.

    Every remote-CPU write authority names ``…/remote-cpu/staging/<job>/<attempt>/<name>`` and
    nothing else; only a presigned GET may also name a content-addressed input.
    """

    parsed = urlsplit(str(uri or ""))
    key = parsed.path.lstrip("/")
    if cas:
        parts = key.removeprefix(LARGE_ARTIFACT_KEY_PREFIX + "/").split("/")
        valid = (key.startswith(LARGE_ARTIFACT_KEY_PREFIX + "/") and len(parts) == 4 and parts[1] == "sha256"
                 and re.fullmatch(r"[0-9a-f]{64}", parts[2]) is not None
                 and all(_SAFE_KEY_COMPONENT.fullmatch(part) for part in (parts[0], parts[3])))
    else:
        parts = key.removeprefix(REMOTE_CPU_STAGING_KEY_PREFIX).split("/")
        valid = (key.startswith(REMOTE_CPU_STAGING_KEY_PREFIX) and len(parts) == 3
                 and _REMOTE_CPU_ATTEMPT.fullmatch("/".join(parts[:2])) is not None
                 and (parts[2] == "" if prefix else _SAFE_KEY_COMPONENT.fullmatch(parts[2]) is not None))
    if parsed.scheme != "s3" or parsed.netloc != bucket or parsed.query or parsed.fragment or not valid:
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_object_name_invalid")
    return key


def _remote_cpu_presign(client: Any, method: str, *, key: str, bucket: str, expires_in_seconds: int,
                        **params: Any) -> str:
    if (not isinstance(expires_in_seconds, int) or isinstance(expires_in_seconds, bool)
            or not 1 <= expires_in_seconds <= 7 * 24 * 60 * 60):
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_presign_expiration_invalid")
    try:
        return str(client.generate_presigned_url(
            method, Params={"Bucket": bucket, "Key": key, **params}, ExpiresIn=expires_in_seconds,
            HttpMethod="PUT" if method == "put_object" else "GET"))
    except Exception:  # noqa: BLE001 - the refusal is typed and never carries a URL
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_presign_failed") from None


def presign_remote_cpu_put(*, grant: PaidResourceAdmissionGrant | None, binding_digest: str, staging_uri: str,
                           expires_in_seconds: int, client: Any, bucket: str) -> str:
    """A PUT URL for one object of one attempt's staging prefix, and nothing else.

    A presigned PUT is write authority held by whoever has the URL, so it exists only under the
    ``cloud_run_cpu_job`` grant bound to the admitted attempt, and never outlives the longest
    attempt the contract allows.
    """

    require_paid_resource_admission_grant(grant, resource_class=REMOTE_CPU_RESOURCE_CLASS,
                                          allocation_binding_digest=binding_digest, require_allocation_binding=True)
    key = _remote_cpu_key(staging_uri, bucket=bucket)
    if not isinstance(expires_in_seconds, int) or expires_in_seconds > MAX_REMOTE_CPU_PUT_SECONDS:
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_presign_expiration_invalid")
    return _remote_cpu_presign(client, "put_object", key=key, bucket=bucket, expires_in_seconds=expires_in_seconds)


def presign_remote_cpu_get(*, uri: str, expires_in_seconds: int, client: Any, bucket: str) -> str:
    """A GET URL for one content-addressed input or one object of an attempt's staging prefix."""

    try:
        key = _remote_cpu_key(uri, bucket=bucket, cas=True)
    except TaskEvaluationConfiguredSceneObjectStoreError:
        key = _remote_cpu_key(uri, bucket=bucket)
    return _remote_cpu_presign(client, "get_object", key=key, bucket=bucket,
                               expires_in_seconds=expires_in_seconds, ResponseCacheControl="no-store, max-age=0")


def read_remote_cpu_staging_object(*, staging_uri: str, maximum_size_bytes: int, client: Any,
                                   bucket: str) -> bytes | None:
    """One staging object's bytes (a heartbeat or a receipt), or ``None`` while it does not exist."""

    key = _remote_cpu_key(staging_uri, bucket=bucket)
    try:
        body = client.get_object(Bucket=bucket, Key=key)["Body"]
    except Exception as exc:  # noqa: BLE001 - provider exception shapes vary
        if _object_missing(exc):
            return None
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_staging_read_failed") from None
    chunks, size = [], 0
    try:
        while size <= maximum_size_bytes:
            chunk = body.read(maximum_size_bytes + 1 - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    finally:
        close = getattr(body, "close", None)
        if callable(close):
            close()
    if size > maximum_size_bytes:
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_staging_object_too_large")
    return b"".join(chunks)


def _precondition_failed(exc: Exception) -> bool:
    response = getattr(exc, "response", {})
    response = response if isinstance(response, dict) else {}
    status = int((response.get("ResponseMetadata") or {}).get("HTTPStatusCode") or 0)
    return status == 412 or str((response.get("Error") or {}).get("Code") or "") == "PreconditionFailed"


def _server_side_copy(client: Any, *, bucket: str, source: str, key: str, size: int, etag: str,
                      metadata: Mapping[str, str], single_copy_limit: int, part_bytes: int) -> None:
    copy_source = {"Bucket": bucket, "Key": source}
    common = {"Metadata": dict(metadata), "ContentType": "application/octet-stream"}
    if size <= single_copy_limit:
        client.copy_object(Bucket=bucket, Key=key, CopySource=copy_source, CopySourceIfMatch=etag,
                           MetadataDirective="REPLACE", **common)
        return
    upload_id = client.create_multipart_upload(Bucket=bucket, Key=key, **common)["UploadId"]
    try:
        parts = []
        for number, start in enumerate(range(0, size, part_bytes), start=1):
            end = min(start + part_bytes, size) - 1
            response = client.upload_part_copy(
                Bucket=bucket, Key=key, UploadId=upload_id, PartNumber=number, CopySource=copy_source,
                CopySourceRange=f"bytes={start}-{end}", CopySourceIfMatch=etag)
            parts.append({"PartNumber": number, "ETag": response["CopyPartResult"]["ETag"]})
        client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id,
                                         MultipartUpload={"Parts": parts})
    except BaseException:
        try:
            client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        except Exception:  # noqa: BLE001 - keep the copy failure; B2 lifecycle removes the parts
            pass
        raise


def copy_remote_cpu_staging_to_cas(
    *, staging_uri: str, digest: str, size_bytes: int, etag: str, artifact_kind: str, filename: str,
    client: Any, bucket: str, single_copy_limit: int = _MAX_SINGLE_COPY_BYTES, part_bytes: int = _COPY_PART_BYTES,
) -> dict[str, Any]:
    """Promote one staging object into CAS server-side (plan 14 §9); no byte crosses the host.

    ``CopyObject`` (``UploadPartCopy`` above the single-copy limit) replaces the metadata with the
    digest and is guarded by the staging ETag, so a changed staging object is
    ``remote_cpu_staging_changed``.  An existing CAS object must already carry the same identity.
    The collector still owes the one streaming readback of the promoted bytes.
    """

    source = _remote_cpu_key(staging_uri, bucket=bucket)
    if (artifact_kind not in REMOTE_CPU_ARTIFACT_KINDS or _SAFE_KEY_COMPONENT.fullmatch(str(filename)) is None
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(digest)) is None or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool) or size_bytes < 1 or not isinstance(etag, str) or not etag):
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_promotion_invalid")
    hexdigest = digest.removeprefix("sha256:")
    key = f"{LARGE_ARTIFACT_KEY_PREFIX}/{artifact_kind}/sha256/{hexdigest}/{filename}"
    copied = False
    try:
        try:
            head = client.head_object(Bucket=bucket, Key=key)
        except Exception as exc:  # noqa: BLE001 - provider exception shapes vary
            if not _object_missing(exc):
                raise
            # PR 4: this stamps Metadata.sha256 before any byte is read back; the collector's one
            # streaming readback deletes the CAS object when the bytes do not match it.
            _server_side_copy(client, bucket=bucket, source=source, key=key, size=size_bytes, etag=etag,
                              metadata={"sha256": hexdigest}, single_copy_limit=single_copy_limit,
                              part_bytes=part_bytes)
            copied = True
            head = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:  # noqa: BLE001 - typed, never echoing the provider message
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "remote_cpu_staging_changed" if _precondition_failed(exc)
            else f"remote_cpu_promotion_failed:{type(exc).__name__}") from None
    metadata = head.get("Metadata") if isinstance(head.get("Metadata"), dict) else {}
    if int(head.get("ContentLength") or -1) != size_bytes or metadata.get("sha256") != hexdigest:
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_promotion_identity_mismatch")
    return {
        "schema_version": "remote_cpu_cas_promotion.v1", "status": "copied" if copied else "already_present",
        "artifact_kind": artifact_kind, "uri": f"s3://{bucket}/{key}", "digest": digest, "size_bytes": size_bytes,
        "remote_identity_verified": True, "full_byte_service_account_readback_passed": False,
        "bytes_through_host": 0,
    }


def delete_remote_cpu_staging_versions(*, staging_prefix: str, client: Any, bucket: str,
                                       page_size: int = 1000) -> dict[str, Any]:
    """Delete every version and delete marker under one attempt's staging prefix, then list again.

    B2 only hides an object deleted without a version id, so neither a HEAD 404 nor a plain
    delete proves absence: the proof is a complete ``ListObjectVersions`` that comes back empty.
    """

    prefix = _remote_cpu_key(staging_prefix, bucket=bucket, prefix=True)

    def listing() -> tuple[list[tuple[str, str]], int]:
        entries, pages, markers = [], 0, {}
        while True:
            page = client.list_object_versions(Bucket=bucket, Prefix=prefix, MaxKeys=page_size, **markers)
            pages += 1
            for row in [*(page.get("Versions") or []), *(page.get("DeleteMarkers") or [])]:
                if not str(row.get("Key") or "").startswith(prefix) or not row.get("VersionId"):
                    raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_staging_listing_invalid")
                entries.append((row["Key"], row["VersionId"]))
            if not page.get("IsTruncated"):
                return entries, pages
            markers = {"KeyMarker": page["NextKeyMarker"], "VersionIdMarker": page["NextVersionIdMarker"]}
            if pages >= 100_000:
                raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_staging_listing_unbounded")

    try:
        entries, _ = listing()
        for key, version in entries:
            client.delete_object(Bucket=bucket, Key=key, VersionId=version)
        remaining, pages = listing()
    except TaskEvaluationConfiguredSceneObjectStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - typed, never echoing the provider message
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            f"remote_cpu_staging_delete_failed:{type(exc).__name__}") from None
    return {"versions_deleted": len(entries), "versions_remaining": len(remaining), "listing_complete": True,
            "listing_pages": pages}


def discard_remote_cpu_output_object(*, uri: str, digest: str, client: Any, bucket: str) -> dict[str, Any]:
    """Remove every version of one promoted ``remote-cpu-output`` CAS object whose readback failed (plan 14 §9).

    ``copy_remote_cpu_staging_to_cas`` stamps ``Metadata.sha256`` before any byte is read back, so an object
    whose bytes do not hash to its key would otherwise answer every later promotion of that digest.
    """

    key = _remote_cpu_key(uri, bucket=bucket, cas=True)
    parts = key.removeprefix(LARGE_ARTIFACT_KEY_PREFIX + "/").split("/")
    if parts[0] != "remote-cpu-output" or f"sha256:{parts[2]}" != digest:
        raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_discard_invalid")

    def versions() -> list[str]:
        page = client.list_object_versions(Bucket=bucket, Prefix=key, MaxKeys=1000)
        if page.get("IsTruncated"):
            raise TaskEvaluationConfiguredSceneObjectStoreError("remote_cpu_discard_listing_unbounded")
        return [row["VersionId"] for row in [*(page.get("Versions") or []), *(page.get("DeleteMarkers") or [])]
                if row.get("Key") == key and row.get("VersionId")]

    try:
        found = versions()
        for version in found:
            client.delete_object(Bucket=bucket, Key=key, VersionId=version)
        remaining = versions()
    except TaskEvaluationConfiguredSceneObjectStoreError:
        raise
    except Exception as exc:  # noqa: BLE001 - typed, never echoing the provider message
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            f"remote_cpu_discard_failed:{type(exc).__name__}") from None
    return {"versions_deleted": len(found), "versions_remaining": len(remaining)}


def _presigned_put(url: str, data: bytes) -> int:
    import urllib.error

    from .safe_outbound_http import presigned_transfer_policy, request

    try:
        return request(url, method="PUT", data=data, policy=presigned_transfer_policy(url), timeout_seconds=60).status
    except urllib.error.HTTPError as exc:
        return exc.code


def remote_cpu_object_store_sentinel(*, grant: PaidResourceAdmissionGrant | None, binding_digest: str,
                                     staging_prefix: str, attempt_id: str, client: Any, bucket: str,
                                     put: Callable[[str, bytes], int] | None = None) -> dict[str, Any]:
    """Prove on B2, inside one attempt's staging prefix, what remote-CPU teardown relies on.

    A presigned PUT (checksums only when required), a ``CopyObject`` into CAS guarded by the
    staging ETag, a plain delete that only hides the object, and version deletes that leave the
    prefix's ``ListObjectVersions`` empty.  The staging versions are deleted even when a step fails.
    """

    require_paid_resource_admission_grant(grant, resource_class=REMOTE_CPU_RESOURCE_CLASS,
                                          allocation_binding_digest=binding_digest, require_allocation_binding=True)
    _remote_cpu_key(staging_prefix, bucket=bucket, prefix=True)
    uri = staging_prefix + "object-store-sentinel.json"
    key = _remote_cpu_key(uri, bucket=bucket)
    data = json.dumps({"schema_version": "remote_cpu_object_store_sentinel.v1", "attempt_id": attempt_id},
                      sort_keys=True).encode("utf-8")
    checks = dict.fromkeys(("presigned_put", "copy_object", "delete_hides", "version_delete"), False)
    failures: list[str] = []
    try:
        url = presign_remote_cpu_put(grant=grant, binding_digest=binding_digest, staging_uri=uri,
                                     expires_in_seconds=300, client=client, bucket=bucket)
        checks["presigned_put"] = 200 <= int((put or _presigned_put)(url, data)) < 300
        etag = str(client.head_object(Bucket=bucket, Key=key).get("ETag") or "")
        promoted = copy_remote_cpu_staging_to_cas(
            staging_uri=uri, digest="sha256:" + hashlib.sha256(data).hexdigest(), size_bytes=len(data), etag=etag,
            artifact_kind="remote-cpu-sentinel", filename="object-store-sentinel.json", client=client, bucket=bucket)
        checks["copy_object"] = promoted["remote_identity_verified"] is True
        client.delete_object(Bucket=bucket, Key=key)
        try:
            client.head_object(Bucket=bucket, Key=key)
            hidden = False
        except Exception as exc:  # noqa: BLE001 - provider exception shapes vary
            hidden = _object_missing(exc)
        listed = client.list_object_versions(Bucket=bucket, Prefix=key)
        checks["delete_hides"] = hidden and bool(listed.get("Versions")) and bool(listed.get("DeleteMarkers"))
    except TaskEvaluationConfiguredSceneObjectStoreError as exc:
        failures.append(str(exc))
    except Exception as exc:  # noqa: BLE001 - typed, never echoing a URL
        failures.append(f"remote_cpu_object_store_sentinel_failed:{type(exc).__name__}")
    try:
        deletion = delete_remote_cpu_staging_versions(staging_prefix=staging_prefix, client=client, bucket=bucket)
        checks["version_delete"] = deletion["versions_deleted"] >= 2 and deletion["versions_remaining"] == 0
    except TaskEvaluationConfiguredSceneObjectStoreError as exc:
        failures.append(str(exc))
    failures.extend(f"remote_cpu_object_store_sentinel_failed:{name}" for name, passed in checks.items() if not passed)
    return {"schema_version": "remote_cpu_object_store_sentinel.v1", "status": "blocked" if failures else "passed",
            "checks": checks, "blockers": sorted(set(failures))}


def main(argv: list[str] | None = None) -> int:
    """Materialize one verified configured-scene artifact from a JSON reference."""

    parser = argparse.ArgumentParser(
        description="Materialize one digest-bound configured-scene artifact."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--maximum-size-bytes", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        reference = json.loads(args.reference.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_reference_unreadable"
        ) from exc
    if not isinstance(reference, dict):
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_reference_invalid"
        )
    result = materialize_configured_scene_artifact(
        reference=reference,
        destination=args.destination,
        maximum_size_bytes=args.maximum_size_bytes,
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


__all__ = [
    "DEFAULT_KEY_PREFIX",
    "EXTERNAL_LAYER_ARTIFACT_KIND",
    "LARGE_ARTIFACT_KEY_PREFIX",
    "REMOTE_CPU_STAGING_KEY_PREFIX",
    "TaskEvaluationConfiguredSceneObjectStoreError",
    "configured_scene_object_store_publisher",
    "copy_remote_cpu_staging_to_cas",
    "delete_remote_cpu_staging_versions",
    "discard_remote_cpu_output_object",
    "materialize_configured_scene_artifact",
    "presign_configured_scene_artifact",
    "presign_remote_cpu_get",
    "presign_remote_cpu_put",
    "publish_configured_scene_artifact",
    "publish_runtime_source_external_layers",
    "read_configured_scene_object",
    "read_remote_cpu_staging_object",
    "remote_cpu_object_store",
    "remote_cpu_object_store_sentinel",
    "validate_configured_scene_object_store_configuration",
]


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())
