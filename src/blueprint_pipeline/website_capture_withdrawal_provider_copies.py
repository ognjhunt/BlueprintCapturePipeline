"""A withdrawn site's copies held by outside providers: what can be deleted there, and the provider's answer.

Beta journey step: site withdrawal. What a website job leaves at each provider, from the code:

- Google Gemini Files API: review images uploaded for a large completed-background review
  (``website_image_completion._review_contents``). Each upload's ``name`` is recorded at
  creation in ``review-media-<digest>.json`` (``website_review_media.v1``), with whether the
  job's own cleanup deleted it. The SDK client offers ``files.delete``: called here.
- Meta (the hosted SAM 3.1 API): the walkthrough video uploaded for segmentation
  (``meta_sam31._upload_video``). Its ``file_id`` is recorded in ``uploaded-video.json``, and
  the job's cleanup in ``video-cleanup.json``. ``DELETE /v1/files/<id>`` exists: called here.
  An upload with an intent but no recorded id (``video-upload-intent.json`` alone) is unknown.
- World Labs (Atlas previews): the job's assets, named by ``website_atlas_preview.atlas_asset_ids``
  in its retained records; the job deletes them itself (``provider_cleanup``) and
  ``DELETE /assets/<id>`` is called here for any it did not. Operation records stay at
  World Labs 30 days with no delete endpoint.
- OpenAI: requests go through the Agents SDK with ``store=False`` (no stored response) and
  image edits return bytes; no object is left that an endpoint deletes. Not deleted at provider.
- Gemini ``generate_content``: no stored object, no delete endpoint. Not deleted at provider.
- Vast: rented instances end at teardown, which each run proves; disk erasure on the host is
  not observable and there is no delete. Not deleted at provider.
- Object storage (B2, the source bucket): deleted by the main plan, not here.

Every recorded id is copied into the site's withdrawal journal
(``website_withdrawal/provider_copies.json``) when planned, so the ids survive the local
deletion of the records that held them and a failed delete is retried by the next run.
A delete is recorded only from the provider's answer: ``deleted_at_provider``,
``absent_at_provider`` (a 404), or ``delete_failed`` with the error. Nothing is claimed
deleted otherwise.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .core.common import write_json

LEDGER_SCHEMA = "website_withdrawal_provider_copies.v1"
LEDGER_FILENAME = "provider_copies.json"
DONE = frozenset({"deleted_by_job", "deleted_at_provider", "absent_at_provider"})
# Providers whose API offers a delete for the copies a website job leaves there.
DELETABLE_PROVIDERS = frozenset({"gemini", "meta", "world_labs"})
NOT_DELETED_AT_PROVIDER = (
    {"provider": "openai", "reason": "requests are sent with store=False and image edits return bytes; OpenAI "
     "offers no endpoint to delete request data it retains under its own terms"},
    {"provider": "gemini", "kind": "generate_content", "reason": "a generate_content request leaves no stored "
     "object and the API has no delete for request data"},
    {"provider": "world_labs", "kind": "operations", "reason": "operation records stay at World Labs for 30 days; "
     "the API has no delete for them"},
    {"provider": "vast", "reason": "instances end at teardown, proven per run; erasure of the host's disk is not "
     "observable and there is no delete"},
)


class ProviderAnswer(Exception):
    """A provider's refusal with its HTTP status (404 means the copy is already gone)."""

    def __init__(self, status: int | None, code: str):
        super().__init__(code)
        self.status = status


def _read(path: Path) -> dict[str, Any] | None:
    try:
        if path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
            return None
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _discover(site: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Every provider copy the site's own records name, and copies whose id was never recorded."""

    copies: dict[str, dict[str, Any]] = {}
    unknown: list[dict[str, Any]] = []
    journal = site / "website_withdrawal"
    for path in sorted(site.rglob("review-media-*.json")):
        if journal in path.parents:
            continue
        receipt = _read(path)
        if not receipt or receipt.get("schema_version") != "website_review_media.v1":
            continue
        failed = {row.get("name") for row in receipt.get("cleanup_errors") or [] if isinstance(row, dict)}
        for image in receipt.get("images") or []:
            name = image.get("name") if isinstance(image, dict) else None
            if isinstance(name, str) and name:
                # The job's cleanup tried every upload ("deleted", or "incomplete" naming each that failed);
                # media kept after an uncertain request, or a run killed before cleanup, are not done.
                done = receipt.get("cleanup_status") in {"deleted", "incomplete"} and name not in failed
                copies[f"gemini:{name}"] = {"provider": "gemini", "kind": "file", "id": name, "source": str(path),
                                            "status": "deleted_by_job" if done else "pending"}
    for path in sorted(site.rglob("uploaded-video.json")):
        if journal in path.parents:
            continue
        receipt = _read(path) or {}
        file_id = receipt.get("file_id")
        if isinstance(file_id, str) and file_id:
            cleanup = _read(path.parent / "video-cleanup.json") or {}
            done = cleanup.get("file_id") == file_id and cleanup.get("deleted") is True
            copies[f"meta:{file_id}"] = {"provider": "meta", "kind": "file", "id": file_id, "source": str(path),
                                         "status": "deleted_by_job" if done else "pending"}
    for path in sorted(site.rglob("video-upload-intent.json")):
        if journal not in path.parents and not (path.parent / "uploaded-video.json").exists():
            unknown.append({"provider": "meta", "kind": "file", "source": str(path),
                            "reason": "an upload was started but no file id was recorded, so it cannot be deleted"})
    from .website_atlas_preview import atlas_asset_ids
    from .website_atlas_preview_store import JOB_FILE

    for job_path in sorted(site.glob(f"captures/*/pipeline/atlas/{JOB_FILE}")):
        job = _read(job_path) or {}
        deleted = set((job.get("provider_cleanup") or {}).get("deleted") or [])
        for asset in atlas_asset_ids(job_path.parent):
            copies[f"world_labs:{asset}"] = {"provider": "world_labs", "kind": "asset", "id": asset,
                                             "source": str(job_path),
                                             "status": "deleted_by_job" if asset in deleted else "pending"}
    return copies, unknown


def plan_providers(site: Path) -> dict[str, Any]:
    """The site's provider copies: discovered now, merged into the journal's ledger (which never forgets an id)."""

    ledger_path = site / "website_withdrawal" / LEDGER_FILENAME
    ledger = _read(ledger_path) or {"schema_version": LEDGER_SCHEMA, "copies": {}}
    discovered, unknown = _discover(site)
    changed = False
    for key, copy in discovered.items():
        known = ledger["copies"].get(key)
        if known is None:
            ledger["copies"][key], changed = copy, True
        elif copy["status"] == "deleted_by_job" and known["status"] not in DONE:
            known["status"], changed = "deleted_by_job", True
    if changed:
        write_json(ledger_path, ledger)
    return {"ledger": str(ledger_path),
            "copies": [dict(ledger["copies"][key], key=key) for key in sorted(ledger["copies"])],
            "unknown": unknown, "not_deleted_at_provider": [dict(row) for row in NOT_DELETED_AT_PROVIDER]}


def production_deleters() -> dict[str, Callable[[str], Any]]:
    """Delete one copy by id through each provider's existing client; raises ``ProviderAnswer`` on a refusal."""

    def gemini(name: str) -> Any:
        from google import genai
        from google.genai import errors, types

        from .clean_plate_removal_analysis_gemini import _api_key
        with genai.Client(api_key=_api_key()[0], http_options=types.HttpOptions(
                timeout=30_000, retry_options=types.HttpRetryOptions(attempts=1))) as client:
            try:
                client.files.delete(name=name)
            except errors.APIError as error:
                raise ProviderAnswer(getattr(error, "code", None), "gemini_files_delete_refused") from None
        return {"deleted": True}

    def meta(file_id: str) -> Any:
        from urllib.error import HTTPError
        from urllib.request import Request

        from .meta_sam31 import meta_api_key
        from .semantic_teacher_image_edit_worker import _open_no_redirect
        request = Request("https://api.meta.ai/v1/files/" + file_id, method="DELETE",
                          headers={"Authorization": "Bearer " + meta_api_key()})
        try:
            with _open_no_redirect(request, timeout=30) as response:
                value = json.loads(response.read(100_000))
        except HTTPError as error:
            raise ProviderAnswer(error.code, "meta_files_delete_refused") from None
        if value.get("deleted") is not True:
            raise ProviderAnswer(None, "meta_files_delete_not_confirmed")
        return {"deleted": True}

    def world_labs(asset_id: str) -> Any:
        from . import website_atlas_pose as pose
        base, key = pose.atlas_api_config()
        try:
            pose.AtlasClient(base_url=base, api_key=key).call("DELETE", f"/assets/{pose._safe_id(asset_id)}")
        except pose.AtlasHttpError as error:
            raise ProviderAnswer(error.status, "world_labs_asset_delete_refused") from None
        return {"deleted": True}

    return {"gemini": gemini, "meta": meta, "world_labs": world_labs}


def apply_providers(section: Mapping[str, Any], deleters: Mapping[str, Callable[[str], Any]] | None) -> list[dict]:
    """Delete every pending or failed copy at its provider and record each answer in the ledger."""

    ledger_path = Path(section["ledger"])
    ledger = _read(ledger_path) or {"schema_version": LEDGER_SCHEMA, "copies": {}}
    results = []
    for key, copy in sorted(ledger["copies"].items()):
        if copy["status"] in DONE:
            continue
        stamp = datetime.now(timezone.utc).isoformat()
        deleter = (deleters or {}).get(copy["provider"])
        copy["attempts"] = int(copy.get("attempts") or 0) + 1
        if deleter is None:
            copy.update(status="delete_failed", last_error="provider_client_unavailable", answered_at=stamp)
        else:
            try:
                answer = deleter(copy["id"])
            except ProviderAnswer as error:
                gone = error.status == 404
                copy.update(status="absent_at_provider" if gone else "delete_failed",
                            last_error=None if gone else f"{error}:{error.status}", answered_at=stamp)
            except Exception as error:  # noqa: BLE001 - recorded and retried next run; the text may echo URLs
                copy.update(status="delete_failed", last_error=type(error).__name__, answered_at=stamp)
            else:
                copy.update(status="deleted_at_provider", last_error=None, answered_at=stamp,
                            answer=json.loads(json.dumps(answer, default=str)) if answer is not None else None)
        write_json(ledger_path, ledger)
        results.append(dict(copy, key=key))
    return results


def blocking(section: Mapping[str, Any], applied: list[dict] | None = None) -> list[dict[str, Any]]:
    """Provider copies that keep the deletion incomplete: not yet deleted, failed, or with no recorded id."""

    latest = {row["key"]: row for row in applied or []}
    rows = []
    pending = [latest.get(copy["key"], copy) for copy in section.get("copies") or []]
    failed = [copy for copy in pending if copy["status"] not in DONE]
    if failed:
        rows.append({"holder": "provider_copies_not_deleted", "copies": [
            {key: copy.get(key) for key in ("provider", "kind", "id", "status", "last_error", "attempts")}
            for copy in failed], "reason": "these provider copies are not yet deleted; the next run retries them"})
    missing = [row for row in section.get("unknown") or [] if row.get("provider") in DELETABLE_PROVIDERS]
    if missing:
        rows.append({"holder": "provider_copy_id_not_recorded", "copies": missing,
                     "reason": "these providers offer a delete, but no id was recorded for these copies, so they "
                               "cannot be deleted and whether they exist is unknown"})
    return rows


def retention_remains(section: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Copies that stay at a provider under its own retention: no delete endpoint exists for them."""

    rows = [dict(row) for row in section.get("not_deleted_at_provider") or []]
    rows.extend({**row, "reason": row.get("reason", "") + "; this provider offers no delete"}
                for row in section.get("unknown") or [] if row.get("provider") not in DELETABLE_PROVIDERS)
    return rows


__all__ = ["DELETABLE_PROVIDERS", "DONE", "retention_remains", "LEDGER_FILENAME", "NOT_DELETED_AT_PROVIDER", "ProviderAnswer", "apply_providers", "blocking",
           "plan_providers", "production_deleters"]
