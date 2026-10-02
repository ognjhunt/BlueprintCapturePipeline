"""Explicit S3-compatible transport without AWS endpoint or credential discovery."""
from __future__ import annotations
import os
import stat
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def require_compatible_endpoint(endpoint: str | None) -> str:
    """Validate locally, before SDK construction or any network request."""
    value = str(endpoint or "").strip()
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not host or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("s3_compatible_explicit_https_endpoint_required")
    if any(host == suffix or host.endswith("." + suffix) for suffix in
           ("amazonaws.com", "amazonaws.com.cn", "api.aws", "aws.amazon.com")):
        raise ValueError("aws_object_storage_integration_removed")
    return value


def s3_compatible_client(sdk: Any, **kwargs: Any) -> Any:
    endpoint = require_compatible_endpoint(kwargs.get("endpoint_url"))
    if not kwargs.get("aws_access_key_id") or not kwargs.get("aws_secret_access_key"):
        raise ValueError("s3_compatible_explicit_credentials_required")
    return sdk.client("s3", **{**kwargs, "endpoint_url": endpoint})


def read_private_credential(path: str | Path) -> str:
    """Read an owner-only regular file without exposing its path or contents."""
    fd = os.open(Path(path).expanduser(), os.O_RDONLY | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(fd)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077 or not 0 < metadata.st_size <= 65536:
            raise ValueError("s3_compatible_private_credential_file_required")
        with os.fdopen(fd, "r", encoding="utf-8", closefd=False) as handle:
            value = handle.read().strip()
        if not value:
            raise ValueError("s3_compatible_private_credential_file_required")
        return value
    finally:
        os.close(fd)
