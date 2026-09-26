"""Private, SHA-pinned G1 checkpoint transfer prepared before paid compute."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_development_pair import PAIR_ORDER
from .wam_provider_object_store import (
    RUNTIME_DEPENDENCY_URL_FILENAME,
    close_cached_runtime_dependency_staging,
    stage_cached_runtime_dependency_object_store,
)


CACHE_ROOT_ENV = "BLUEPRINT_G1_CHECKPOINT_CACHE_ROOT"
PROVIDER_CACHE_FILE_ENV = "BLUEPRINT_G1_CHECKPOINT_CACHE_FILE"
TRANSFER_SCHEMA = "native_g1_private_checkpoint_transfer.v1"


def _fetcher() -> Any:
    repository = Path(__file__).resolve().parents[2]
    path = repository / "scripts/fetch_g1_humanoidarena_checkpoint.py"
    spec = importlib.util.spec_from_file_location("g1_checkpoint_cache_verifier", path)
    if spec is None or spec.loader is None:
        raise ValueError("g1_checkpoint_cache_verifier_missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inventory_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "configs/g1_humanoidarena_checkpoint_inventory.v1.json"
    )


def verify_local_g1_checkpoint_cache(root: Path) -> list[dict[str, Any]]:
    """Rehash all pinned files before a provider URL can be created."""

    cache = Path(root)
    if not cache.is_absolute() or cache.is_symlink() or not cache.is_dir():
        raise ValueError("g1_checkpoint_cache_root_invalid")
    fetcher = _fetcher()
    rows = []
    for candidate in PAIR_ORDER:
        receipt = fetcher.materialize_candidate(
            inventory_path=_inventory_path(), candidate_id=candidate,
            output_dir=cache, verify_only=True,
        )
        if receipt.get("status") != "checkpoint_bytes_verified":
            raise ValueError("g1_checkpoint_cache_candidate_invalid:" + candidate)
        rows.extend(receipt["files"])
    return rows


def stage_g1_checkpoint_cache(
    *, cache_root: Path, job_dir: Path, key_prefix: str,
    expiration_seconds: int,
) -> dict[str, Any]:
    """Upload immutable bytes and seal expiring GET URLs in a private file."""

    rows = verify_local_g1_checkpoint_cache(cache_root)
    job = Path(job_dir)
    if not job.is_absolute() or job.is_symlink() or job.exists():
        raise ValueError("g1_checkpoint_cache_job_invalid")
    job.mkdir(mode=0o700)
    transfer_rows = []
    safe_rows = []
    cache_hits = 0
    uploads = 0
    try:
        for index, row in enumerate(rows):
            source = Path(cache_root) / row["relative_path"]
            staging_dir = job / f"file-{index:02d}"
            result = stage_cached_runtime_dependency_object_store(
                job_dir=staging_dir,
                dependency_path=source,
                expected_sha256=row["sha256"],
                key_prefix=key_prefix,
                expiration_seconds=expiration_seconds,
                artifact_kind="g1_checkpoint",
            )
            if result.get("status") != "completed" or not result.get("remote_identity_verified"):
                raise ValueError("g1_checkpoint_cache_remote_staging_blocked")
            url = (staging_dir / RUNTIME_DEPENDENCY_URL_FILENAME).read_text(
                encoding="utf-8"
            ).strip()
            if not url.startswith("https://"):
                raise ValueError("g1_checkpoint_cache_signed_url_invalid")
            transfer_rows.append({**row, "url": url})
            safe_rows.append(row)
            cache_hits += int(result.get("cache_hit") is True)
            uploads += int(result.get("upload_performed") is True)
        private_path = job / "g1_checkpoint_transfer_urls.json"
        descriptor = os.open(private_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({
                "schema_version": TRANSFER_SCHEMA,
                "files": transfer_rows,
            }, stream, sort_keys=True)
            stream.write("\n")
        result = {
            "schema_version": "native_g1_checkpoint_cache_staging.v1",
            "status": "completed",
            "file_count": len(rows),
            "cache_hit_count": cache_hits,
            "upload_count": uploads,
            "inventory_rows_digest": canonical_digest({"files": safe_rows}),
            "signed_url_file_path": str(private_path),
            "raw_signed_urls_recorded": False,
        }
        return result
    except BaseException:
        close_g1_checkpoint_cache(job)
        raise


def close_g1_checkpoint_cache(job_dir: Path) -> dict[str, Any]:
    """Remove expiring URL files while retaining verified immutable objects."""

    job = Path(job_dir)
    private_path = job / "g1_checkpoint_transfer_urls.json"
    private_path.unlink(missing_ok=True)
    for child in sorted(job.glob("file-[0-9][0-9]")):
        if child.is_dir() and not child.is_symlink():
            close_cached_runtime_dependency_staging(child)
    return {
        "schema_version": "native_g1_checkpoint_cache_closeout.v1",
        "status": "completed" if not private_path.exists() else "blocked",
        "signed_url_file_removed": not private_path.exists(),
        "content_addressed_cache_objects_retained": True,
    }
