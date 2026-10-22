"""What any reader of an Atlas preview job may know, without the Atlas controller.

The intake service (owner downloads) and withdrawal deletion read jobs, locks and
archive names from here. Only ``website_atlas_preview`` runs Atlas, so neither
reaches the provider or GPU lanes through this module (ADP-040 / Day 28).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA = "website_atlas_preview_job.v1"
JOB_FILE = "preview_job.json"
LOCK_FILE = "preview.lock"
ARTIFACT_KIND = "atlas-scene-bundle"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


class PreviewUnavailable(ValueError):
    """No archived preview this caller may read; never says why."""


def bundle_filename(capture_id: str) -> str:
    return f"{capture_id}.zip"


def read_job(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError("website_atlas_preview_job_unsafe")
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
        raise ValueError("website_atlas_preview_job_invalid")
    return value


def capture_withdrawn(capture_root: Path) -> bool:
    from .consent_takedown import read_consent_state
    consent = read_consent_state(capture_root)
    return consent["state"] == "revoked" or "website_withdrawal/tombstone.json" in (consent.get("source_path") or "")


def open_bundle(*, request_id: str, capture_id: str, digest: str, storage_root: Path | None,
                client: Any = None, bucket: str | None = None) -> tuple[Any, int]:
    """The archived bundle body and size for an owner download the WebApp already authorized.

    While the workspace exists its own job must be ready with this exact digest
    and unwithdrawn; after retirement only an object named for this capture and
    digest exists to read.
    """
    if (not _ID.fullmatch(request_id) or capture_id != f"walkthrough-{request_id}"
            or not _DIGEST.fullmatch(digest)):
        raise PreviewUnavailable("website_atlas_preview_unavailable")
    if storage_root is not None:
        roots = [root for root in Path(storage_root).glob(f"*/scenes/site-{request_id}/captures/{capture_id}")
                 if root.is_dir()]
        if len(roots) > 1:
            raise PreviewUnavailable("website_atlas_preview_unavailable")
        if roots:
            path = roots[0] / "pipeline" / "atlas" / JOB_FILE
            job = read_job(path) if path.is_file() else None
            if (capture_withdrawn(roots[0]) or job is None or job["state"] != "ready"
                    or (job.get("archive") or {}).get("digest") != digest):
                raise PreviewUnavailable("website_atlas_preview_unavailable")
    from .task_evaluation_configured_scene_object_keys import LARGE_ARTIFACT_KEY_PREFIX
    from .task_evaluation_configured_scene_object_store import _artifact_object_store_client, _object_missing
    if client is None or bucket is None:
        client, bucket = _artifact_object_store_client()
    key = f"{LARGE_ARTIFACT_KEY_PREFIX}/{ARTIFACT_KIND}/sha256/{digest[7:]}/{bundle_filename(capture_id)}"
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except Exception as error:  # noqa: BLE001 - provider exception shapes vary
        if _object_missing(error):
            raise PreviewUnavailable("website_atlas_preview_unavailable") from None
        raise
    if (head.get("Metadata") or {}).get("sha256") != digest[7:]:
        raise PreviewUnavailable("website_atlas_preview_unavailable")
    return client.get_object(Bucket=bucket, Key=key)["Body"], int(head["ContentLength"])
