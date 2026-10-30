"""Exact offline wheel inventory for the pinned HumanoidArena beta policy server.

This standard-library helper neither downloads nor installs packages. The native
source packet owns the large Torch/CUDA wheels; callers must reuse those exact
bytes. Hardware drivers and training entrypoints remain outside this profile.
"""
from __future__ import annotations

import hashlib
import json
import math
import stat
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

DEPENDENCY_MANIFEST_FILENAME = "g1-beta-inference-dependencies.v1.json"
DEPENDENCY_MANIFEST_SHA256 = "sha256:0ca77b40b8cc2ea9c0fae5581f323011424377a2aa8602a969e8586fee511785"
DEPENDENCY_MANIFEST_DIGEST = "sha256:5ad43668b6903fa2682680c59f90f6daf3c4ee3a07b48930f74d96a53206a5de"
TOTAL_DOWNLOAD_BYTES = 261_777_389
REUSED_RUNTIME_WHEEL_BYTES = 4_132_469_228
MAXIMUM_EXPANDED_BYTES = 16 * 1024**3
_VENDOR = Path(__file__).resolve().parent / "_vendor"
_MAX_MANIFEST_BYTES = 1024 * 1024


def _digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def artifact_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Return only the exact committed profile, including externally staged copies."""
    source = Path(path) if path is not None else _VENDOR / DEPENDENCY_MANIFEST_FILENAME
    if source.is_symlink() or not source.is_file() or source.stat().st_size > _MAX_MANIFEST_BYTES:
        raise ValueError("g1_beta_dependency_manifest_file_invalid")
    raw = source.read_bytes()
    if _digest(raw) != DEPENDENCY_MANIFEST_SHA256:
        raise ValueError("g1_beta_dependency_manifest_bytes_mismatch")
    manifest = json.loads(raw)
    body = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    digest = _digest(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if (manifest.get("schema") != "native_g1_beta_dependency_manifest.v1"
            or digest != DEPENDENCY_MANIFEST_DIGEST or manifest.get("manifest_digest") != digest):
        raise ValueError("g1_beta_dependency_manifest_digest_mismatch")
    wheels = manifest["wheels"]
    external = sum(row["size_bytes"] for row in wheels if row["source"] == "external_download")
    reused = sum(row["size_bytes"] for row in wheels if row["source"] == "runtime_source_packet")
    if (external != TOTAL_DOWNLOAD_BYTES or manifest["total_download_bytes"] != external
            or reused != REUSED_RUNTIME_WHEEL_BYTES or manifest["reused_runtime_wheel_bytes"] != reused
            or manifest["maximum_expanded_bytes"] != MAXIMUM_EXPANDED_BYTES):
        raise ValueError("g1_beta_dependency_byte_budget_mismatch")
    return manifest


def stage_dependency_manifest(runtime_root: str | Path) -> Path:
    """Stage the small sealed profile; no wheel bytes are duplicated into the bundle."""
    artifact_manifest()
    root = Path(runtime_root)
    if root.is_symlink():
        raise ValueError("g1_beta_dependency_staging_symlink")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / DEPENDENCY_MANIFEST_FILENAME
    if destination.is_symlink():
        raise ValueError("g1_beta_dependency_staging_symlink")
    destination.write_bytes((_VENDOR / DEPENDENCY_MANIFEST_FILENAME).read_bytes())
    artifact_manifest(destination)
    return destination


def verify_wheelhouse(
    paths: Mapping[str, Path], *, deadline_epoch: float | None = None,
) -> dict[str, Any]:
    """Verify every reused/downloaded wheel and its expanded installation bound.

    Keys are exact filenames. A missing native wheel is an explicit incomplete
    source-packet error; there is no unquoted external-download fallback.
    """
    if deadline_epoch is not None and (
        isinstance(deadline_epoch, bool) or not isinstance(deadline_epoch, (int, float))
        or (isinstance(deadline_epoch, float) and not math.isfinite(deadline_epoch))
        or deadline_epoch <= 0
    ):
        raise ValueError("g1_beta_dependency_deadline_invalid")

    def check_deadline() -> None:
        if deadline_epoch is not None and time.time() >= deadline_epoch:
            raise TimeoutError("g1_beta_dependency_verification_deadline_exceeded")

    check_deadline()
    manifest = artifact_manifest()
    expected = {row["filename"]: row for row in manifest["wheels"]}
    if set(paths) != set(expected):
        raise ValueError("g1_beta_dependency_wheel_set_mismatch")
    expanded = 0
    members = 0
    for filename, row in expected.items():
        check_deadline()
        source = Path(paths[filename])
        if source.is_symlink() or not source.is_file() or source.stat().st_size != row["size_bytes"]:
            raise ValueError(f"g1_beta_dependency_wheel_file_invalid:{filename}")
        hasher = hashlib.sha256()
        with source.open("rb") as stream:
            while block := stream.read(4 * 1024**2):
                check_deadline()
                hasher.update(block)
        if hasher.hexdigest() != row["sha256"]:
            raise ValueError(f"g1_beta_dependency_wheel_digest_mismatch:{filename}")
        check_deadline()
        with zipfile.ZipFile(source) as archive:
            for entry in archive.infolist():
                check_deadline()
                member = PurePosixPath(entry.filename)
                mode = stat.S_IFMT(entry.external_attr >> 16)
                if (member.is_absolute() or ".." in member.parts or "\\" in entry.filename
                        or ":" in entry.filename or mode not in {0, stat.S_IFREG, stat.S_IFDIR}):
                    raise ValueError(f"g1_beta_dependency_wheel_member_invalid:{filename}")
                expanded += entry.file_size
                members += 1
        if expanded > MAXIMUM_EXPANDED_BYTES:
            raise ValueError("g1_beta_dependency_expanded_byte_budget_exceeded")
    return {
        "schema": "native_g1_beta_wheelhouse_verification.v1",
        "manifest_digest": manifest["manifest_digest"],
        "wheel_count": len(expected),
        "expanded_bytes": expanded,
        "archive_members": members,
        "claim_ceiling": "dependency_bytes_only",
    }
