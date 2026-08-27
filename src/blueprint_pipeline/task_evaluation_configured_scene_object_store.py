"""Durably publish configured-scene artifacts with exact S3 readback."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Callable
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit


DEFAULT_KEY_PREFIX = "blueprint/arm-decision-proof-v1/configured-scenes"
_SAFE_KEY_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}")


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


def _object_store_client() -> tuple[Any, str]:
    try:
        import boto3  # type: ignore[import-not-found]
        from botocore.client import Config  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise TaskEvaluationConfiguredSceneObjectStoreError(
            "configured_scene_object_store_client_unavailable"
        ) from exc
    access_key = _private_file_value(
        "BLUEPRINT_WAM_OBJECT_STORE_ACCESS_KEY_ID_FILE", required=True
    )
    secret_key = _private_file_value(
        "BLUEPRINT_WAM_OBJECT_STORE_SECRET_ACCESS_KEY_FILE", required=True
    )
    bucket = _private_file_value(
        "BLUEPRINT_WAM_OBJECT_STORE_BUCKET_FILE", required=True
    )
    endpoint = _private_file_value(
        "BLUEPRINT_WAM_OBJECT_STORE_ENDPOINT_URL_FILE", required=False
    )
    region = _private_file_value(
        "BLUEPRINT_WAM_OBJECT_STORE_REGION_FILE", required=False
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


__all__ = [
    "DEFAULT_KEY_PREFIX",
    "TaskEvaluationConfiguredSceneObjectStoreError",
    "configured_scene_object_store_publisher",
    "read_configured_scene_object",
]
