"""Sealed, offline-installable MolmoAct2 DROID inference dependency inventory.

The policy server uses its own CPython 3.12/CUDA 12.8 environment. Isaac's
native Python client uses the existing NumPy and standard-library HTTP client. This module
uses only the standard library and never downloads or installs anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEPENDENCY_MANIFEST_FILENAME = "molmoact2-droid-dependencies.v1.json"
DEPENDENCY_MANIFEST_SHA256 = "sha256:1650f97ba5cf3d9857b4075407ac265305bb55a6f4ff7e1ada7d58549cd49b40"
DEPENDENCY_MANIFEST_DIGEST = "sha256:d709e18fd1eb5d69ea97dd02ab9775665857edb27c26c18742d150ead1b4d36e"
TOTAL_DOWNLOAD_BYTES = 4_045_566_007
_VENDOR = Path(__file__).resolve().parent / "_vendor"
_MAX_MANIFEST_BYTES = 1024 * 1024


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def verified_dependency_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Open the exact committed bytes, including when staged outside the package."""
    source = Path(path) if path is not None else _VENDOR / DEPENDENCY_MANIFEST_FILENAME
    if source.is_symlink() or not source.is_file() or source.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("molmoact2_dependency_manifest_file_invalid")
    raw = source.read_bytes()
    if _digest(raw) != DEPENDENCY_MANIFEST_SHA256:
        raise ValueError("molmoact2_dependency_manifest_bytes_mismatch")
    manifest = json.loads(raw)
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    digest = _digest(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if (manifest.get("schema") != "molmoact2_droid_dependency_manifest.v1"
            or digest != DEPENDENCY_MANIFEST_DIGEST
            or manifest.get("manifest_digest") != digest):
        raise ValueError("molmoact2_dependency_manifest_digest_mismatch")
    external = {}
    for row in [manifest["interpreter"], *manifest["server_wheels"], *manifest["thin_client_wheels"]]:
        if "url" in row:
            if row["url"] in external and external[row["url"]] != row["size_bytes"]:
                raise ValueError("molmoact2_dependency_duplicate_download_mismatch")
            external[row["url"]] = row["size_bytes"]
    if sum(external.values()) != TOTAL_DOWNLOAD_BYTES or manifest["total_download_bytes"] != TOTAL_DOWNLOAD_BYTES:
        raise ValueError("molmoact2_dependency_download_budget_mismatch")
    return manifest


def stage_dependency_manifest(runtime: str | Path) -> Path:
    """Copy the verified inference manifest into a provider bundle."""
    verified_dependency_manifest()
    root = Path(runtime)
    root.mkdir(parents=True, exist_ok=True)
    destination = root / DEPENDENCY_MANIFEST_FILENAME
    if destination.is_symlink():
        raise ValueError("molmoact2_dependency_staging_symlink")
    destination.write_bytes((_VENDOR / DEPENDENCY_MANIFEST_FILENAME).read_bytes())
    verified_dependency_manifest(destination)
    return destination
