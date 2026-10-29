"""Sealed, offline-installable OpenWAM PiPER inference dependency inventory.

The policy server uses its own CPython 3.12/CUDA 12.8 environment. Isaac's
CPython 3.11 client receives only its dedicated WebSocket wheel. This module
uses only the standard library and never downloads or installs anything.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

DEPENDENCY_MANIFEST_FILENAME = "openwam-piper-dependencies.v1.json"
DEPENDENCY_MANIFEST_SHA256 = "sha256:18f688806ebdceea660235f2150d9dc3c833c75433a3dbb6a0c31c08ff4898d0"
DEPENDENCY_MANIFEST_DIGEST = "sha256:92558c8efed216e0ccf103c2194ead8e7d521ee40b3d6ffd1200afbe5735b68d"
TOTAL_DOWNLOAD_BYTES = 4_119_355_954
_VENDOR = Path(__file__).resolve().parent / "_vendor"
_MAX_MANIFEST_BYTES = 1024 * 1024


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def verified_dependency_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Open the exact committed bytes, including when staged outside the package."""
    source = Path(path) if path is not None else _VENDOR / DEPENDENCY_MANIFEST_FILENAME
    if source.is_symlink() or not source.is_file() or source.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("openwam_dependency_manifest_file_invalid")
    raw = source.read_bytes()
    if _digest(raw) != DEPENDENCY_MANIFEST_SHA256:
        raise ValueError("openwam_dependency_manifest_bytes_mismatch")
    manifest = json.loads(raw)
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    digest = _digest(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if (manifest.get("schema") != "openwam_piper_dependency_manifest.v1"
            or digest != DEPENDENCY_MANIFEST_DIGEST
            or manifest.get("manifest_digest") != digest):
        raise ValueError("openwam_dependency_manifest_digest_mismatch")
    external = {}
    for row in [manifest["interpreter"], *manifest["server_wheels"], *manifest["thin_client_wheels"]]:
        if "url" in row:
            if row["url"] in external and external[row["url"]] != row["size_bytes"]:
                raise ValueError("openwam_dependency_duplicate_download_mismatch")
            external[row["url"]] = row["size_bytes"]
    if sum(external.values()) != TOTAL_DOWNLOAD_BYTES or manifest["total_download_bytes"] != TOTAL_DOWNLOAD_BYTES:
        raise ValueError("openwam_dependency_download_budget_mismatch")
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
            raise ValueError("openwam_dependency_bundled_wheel_invalid")
        raw = source.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row["sha256"]:
            raise ValueError("openwam_dependency_bundled_wheel_digest_mismatch")
        destination = root / row["bundled_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_symlink() or destination.parent.is_symlink():
            raise ValueError("openwam_dependency_staging_symlink")
        destination.write_bytes(raw)
    destination = root / DEPENDENCY_MANIFEST_FILENAME
    if destination.is_symlink():
        raise ValueError("openwam_dependency_staging_symlink")
    destination.write_bytes((_VENDOR / DEPENDENCY_MANIFEST_FILENAME).read_bytes())
    verified_dependency_manifest(destination)
    return destination
