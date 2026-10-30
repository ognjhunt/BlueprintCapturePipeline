"""Sealed, offline-installable Cosmos DROID inference dependency inventory.

The policy server uses its own CPython 3.13/CUDA 12.8 environment. Isaac's
CPython 3.11/3.12 client receives only the pinned protocol wheels. This module
uses only the standard library and never downloads or installs anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEPENDENCY_MANIFEST_FILENAME = "cosmos-droid-dependencies.v1.json"
DEPENDENCY_MANIFEST_SHA256 = "sha256:b999ed98b7c73da5c06f1c7f47ec5bea50148dbf1d422435686f2a1f59fa3ba6"
DEPENDENCY_MANIFEST_DIGEST = "sha256:3eb229d970c775596eefe3f9cec35fb9e9232649e7a87b616a55ee9c03dc5460"
TOTAL_DOWNLOAD_BYTES = 5652022212
_VENDOR = Path(__file__).resolve().parent / "_vendor"
_MAX_MANIFEST_BYTES = 1024 * 1024


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def verified_dependency_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Open the exact committed bytes, including when staged outside the package."""
    source = Path(path) if path is not None else _VENDOR / DEPENDENCY_MANIFEST_FILENAME
    if source.is_symlink() or not source.is_file() or source.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("cosmos_dependency_manifest_file_invalid")
    raw = source.read_bytes()
    if _digest(raw) != DEPENDENCY_MANIFEST_SHA256:
        raise ValueError("cosmos_dependency_manifest_bytes_mismatch")
    manifest = json.loads(raw)
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    digest = _digest(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if (manifest.get("schema") != "cosmos_droid_dependency_manifest.v1"
            or digest != DEPENDENCY_MANIFEST_DIGEST
            or manifest.get("manifest_digest") != digest):
        raise ValueError("cosmos_dependency_manifest_digest_mismatch")
    external = {}
    for row in [manifest["interpreter"], *manifest["server_wheels"], *manifest["thin_client_wheels"]]:
        if "url" in row:
            if row["url"] in external and external[row["url"]] != row["size_bytes"]:
                raise ValueError("cosmos_dependency_duplicate_download_mismatch")
            external[row["url"]] = row["size_bytes"]
    if sum(external.values()) != TOTAL_DOWNLOAD_BYTES or manifest["total_download_bytes"] != TOTAL_DOWNLOAD_BYTES:
        raise ValueError("cosmos_dependency_download_budget_mismatch")
    return manifest


def stage_dependency_manifest(runtime: str | Path) -> Path:
    """Copy verified manifest and prebuilt pure-Python wheel into a provider bundle."""
    manifest = verified_dependency_manifest()
    root = Path(runtime)
    root.mkdir(parents=True, exist_ok=True)
    for row in [*manifest["server_wheels"], *manifest["thin_client_wheels"]]:
        if "bundled_path" not in row:
            continue
        source = _VENDOR / row["filename"]
        if source.is_symlink() or not source.is_file() or source.stat().st_size != row["size_bytes"]:
            raise ValueError("cosmos_dependency_bundled_wheel_invalid")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("cosmos_dependency_bundled_wheel_digest_mismatch")
        destination = root / row["bundled_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.parent.is_symlink():
            raise ValueError("cosmos_dependency_staging_symlink")
        destination.write_bytes(raw)
    destination = root / DEPENDENCY_MANIFEST_FILENAME
    if destination.is_symlink():
        raise ValueError("cosmos_dependency_staging_symlink")
    destination.write_bytes((_VENDOR / DEPENDENCY_MANIFEST_FILENAME).read_bytes())
    verified_dependency_manifest(destination)
    return destination
