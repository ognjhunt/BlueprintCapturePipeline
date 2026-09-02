"""Durably publish configured-scene artifacts with exact S3 readback."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/configured-scenes"
LARGE_ARTIFACT_KEY_PREFIX = f"{DEFAULT_KEY_PREFIX}/artifacts"
# Runtime-source wrapper layers are published under this artifact kind; the
# wrapper builder embeds the resulting URI, so the two must agree exactly.
EXTERNAL_LAYER_ARTIFACT_KIND = "native-runtime-source-layer"
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
    names: Mapping[str, str], *, require_endpoint_and_region: bool = False
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
    kwargs: dict[str, Any] = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "region_name": region or "us-east-1",
        "config": Config(signature_version="s3v4"),
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
    kind = _safe_object_name(artifact_kind)
    if len(kind.parts) != 1:
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_artifact_kind_invalid"
        )
    digest, size = _sha256_and_size(source)
    digest_hex = digest.removeprefix("sha256:")
    key = str(
        PurePosixPath(LARGE_ARTIFACT_KEY_PREFIX)
        / kind
        / "sha256"
        / digest_hex
        / source.name
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
            resolved_client.upload_file(
                str(source),
                resolved_bucket,
                key,
                ExtraArgs={
                    "Metadata": {"sha256": digest_hex},
                    "ContentType": "application/octet-stream",
                },
            )
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
        or expected_size > maximum_size_bytes
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
    "TaskEvaluationConfiguredSceneObjectStoreError",
    "configured_scene_object_store_publisher",
    "materialize_configured_scene_artifact",
    "presign_configured_scene_artifact",
    "publish_configured_scene_artifact",
    "publish_runtime_source_external_layers",
    "read_configured_scene_object",
    "validate_configured_scene_object_store_configuration",
]


if __name__ == "__main__":  # pragma: no cover - exercised through module CLI
    raise SystemExit(main())
