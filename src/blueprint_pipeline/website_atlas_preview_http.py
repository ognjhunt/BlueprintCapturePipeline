"""Owner downloads of an archived Atlas preview through the WebApp's signed client (D6, 2026-10-22).

The WebApp checks the signed-in owner and current consent, then streams these
bytes on; callers name a capture and digest, never a path or object key.
"""
import logging
import os
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .website_atlas_preview import PreviewUnavailable, open_bundle
from .website_preparation_status_http import HANDOFF_STORAGE_ROOT_ENV

logger = logging.getLogger(__name__)
CHUNK_BYTES = 1024 * 1024


def register_website_atlas_preview_routes(app: FastAPI, *, require_admission: Callable) -> None:
    @app.get("/api/live-pipeline/website-atlas-previews/{capture_id}/bundle",
             dependencies=[Depends(require_admission)])
    async def download_preview(request: Request, capture_id: str, request_id: str = "",
                               digest: str = "") -> StreamingResponse:
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") != "blueprint-webapp"):
            raise HTTPException(status_code=403, detail="website atlas preview issuer not authorized")
        configured = os.getenv(HANDOFF_STORAGE_ROOT_ENV, "").strip()
        storage_root = Path(configured) if configured and Path(configured).is_absolute() else None
        try:
            body, size = await run_in_threadpool(open_bundle, request_id=request_id, capture_id=capture_id,
                                                 digest=digest, storage_root=storage_root)
        except PreviewUnavailable:
            raise HTTPException(status_code=404, detail="website_atlas_preview_unavailable") from None
        except Exception as exc:  # noqa: BLE001 - storage errors can carry signed URLs; name only the class
            logger.warning("website_atlas_preview_download_unavailable type=%s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="website_atlas_preview_storage_unavailable") from None

        def chunks():
            try:
                yield from body.iter_chunks(CHUNK_BYTES)
            finally:
                body.close()
        return StreamingResponse(chunks(), media_type="application/zip", headers={
            "Content-Length": str(size), "X-Blueprint-Artifact-SHA256": digest,
            "Content-Disposition": f'attachment; filename="atlas-preview-{capture_id}.zip"',
            "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})
