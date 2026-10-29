"""Signed WebApp operations for source-bound owned-upload Atlas preparation.

The WebApp verifies the actual signed-in source owner. The native service
resolves the retained plan, source bytes, work directory and private archive;
clients cannot supply provider settings, local paths or invented site records.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .owned_capture_atlas_worker import enqueue_owned_atlas, inspect_owned_atlas, open_owned_bundle, retry_owned_atlas
from .website_atlas_preview_store import PreviewUnavailable

logger = logging.getLogger(__name__)


def register_owned_capture_atlas_routes(app: FastAPI, *, require_admission: Callable,
                                        roots: Callable[[], tuple[Path, Path]]) -> None:
    def require_webapp(request: Request) -> None:
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") != "blueprint-webapp"):
            raise HTTPException(status_code=403, detail="owned Atlas issuer not authorized")

    def refusal(error: Exception) -> HTTPException:
        if isinstance(error, FileNotFoundError):
            return HTTPException(status_code=404, detail="owned_capture_atlas_job_not_found")
        code = str(error).partition(":")[0]
        if not code or len(code) > 120 or any(ch not in "abcdefghijklmnopqrstuvwxyz_0123456789" for ch in code):
            code = "owned_capture_atlas_operation_unavailable"
        return HTTPException(status_code=410 if code == "capture_revoked_or_expired" else 409, detail=code)

    async def command(request: Request, schema: str, fields: set[str]) -> dict[str, str]:
        require_webapp(request)
        try:
            body = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="invalid JSON body") from None
        expected = fields | {"schema_version"}
        if (not isinstance(body, Mapping) or set(body) != expected or body.get("schema_version") != schema
                or any(not isinstance(body[key], str) for key in expected)):
            raise HTTPException(status_code=422, detail="owned Atlas operation submission invalid")
        return {key: body[key] for key in fields}

    @app.post("/api/live-pipeline/reconstructions/atlas-execute", dependencies=[Depends(require_admission)])
    async def execute(request: Request):
        body = await command(request, "owned_capture_atlas_execution_submission.v1",
            {"plan_id", "task_context_digest", "source_video_digest", "owner_user_id", "idempotency_key"})
        state_root, capture_store_root = roots()
        try:
            return await run_in_threadpool(enqueue_owned_atlas, state_root=state_root,
                capture_store_root=capture_store_root, **body)
        except (ValueError, OSError) as error:
            raise refusal(error) from None

    @app.post("/api/live-pipeline/reconstructions/atlas-inspect", dependencies=[Depends(require_admission)])
    async def inspect(request: Request):
        body = await command(request, "owned_capture_atlas_inspection_submission.v1", {"plan_id", "owner_user_id"})
        state_root, capture_store_root = roots()
        try:
            return await run_in_threadpool(inspect_owned_atlas, state_root=state_root,
                capture_store_root=capture_store_root, **body)
        except (ValueError, OSError) as error:
            raise refusal(error) from None

    @app.post("/api/live-pipeline/reconstructions/atlas-retry", dependencies=[Depends(require_admission)])
    async def retry(request: Request):
        body = await command(request, "owned_capture_atlas_retry_submission.v1",
            {"plan_id", "task_context_digest", "source_video_digest", "owner_user_id", "idempotency_key"})
        state_root, capture_store_root = roots()
        try:
            return await run_in_threadpool(retry_owned_atlas, state_root=state_root,
                                           capture_store_root=capture_store_root, **body)
        except (ValueError, OSError) as error:
            raise refusal(error) from None

    @app.post("/api/live-pipeline/reconstructions/atlas-bundle", dependencies=[Depends(require_admission)])
    async def bundle(request: Request):
        value = await command(request, "owned_capture_atlas_bundle_submission.v1", {"plan_id", "owner_user_id", "digest"})
        state_root, capture_store_root = roots()
        try:
            body, size = await run_in_threadpool(open_owned_bundle, state_root=state_root,
                capture_store_root=capture_store_root, **value)
        except PreviewUnavailable:
            raise HTTPException(status_code=404, detail="owned_capture_atlas_bundle_unavailable") from None
        except (ValueError, FileNotFoundError) as error:
            raise refusal(error) from None
        except Exception as error:  # noqa: BLE001 - provider errors can contain signed object URLs
            logger.warning("owned_atlas_bundle_read_failed type=%s", type(error).__name__)
            raise HTTPException(status_code=503, detail="owned_capture_atlas_storage_unavailable") from None

        def chunks():
            try:
                yield from body.iter_chunks(1024 * 1024)
            finally:
                body.close()
        return StreamingResponse(chunks(), media_type="application/zip", headers={
            "Content-Length": str(size), "X-Blueprint-Artifact-SHA256": value["digest"],
            "Content-Disposition": f'attachment; filename="{value["plan_id"]}.zip"',
            "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})
