"""Owner downloads of an archived Atlas preview through the WebApp's signed client (D6, 2026-10-22).

The WebApp checks the signed-in owner and current consent, then streams these
bytes on; callers name a capture and digest, never a path or object key. The
web-sized scene preview a site opens from its job page (``website_scene_preview``)
is read the same way.
"""
import logging
import os
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .website_atlas_preview_store import PreviewUnavailable, open_bundle
from .website_preparation_status_http import HANDOFF_STORAGE_ROOT_ENV
from .website_scene_preview import ScenePreviewUnavailable, open_scene_preview

logger = logging.getLogger(__name__)
CHUNK_BYTES = 1024 * 1024


def _storage_root() -> Path | None:
    configured = os.getenv(HANDOFF_STORAGE_ROOT_ENV, "").strip()
    return Path(configured) if configured and Path(configured).is_absolute() else None


def _webapp_only(request: Request) -> None:
    if (not request.headers.get("x-blueprint-pipeline-signature")
            or getattr(request.state, "intake_client_id", "") != "blueprint-webapp"):
        raise HTTPException(status_code=403, detail="website preview issuer not authorized")


def register_website_atlas_preview_routes(app: FastAPI, *, require_admission: Callable) -> None:
    @app.get("/api/live-pipeline/website-scene-previews/{capture_id}/scene.ply",
             dependencies=[Depends(require_admission)])
    async def read_scene_preview(request: Request, capture_id: str, request_id: str = "",
                                 digest: str = "") -> StreamingResponse:
        _webapp_only(request)
        try:
            body, size = await run_in_threadpool(open_scene_preview, request_id=request_id, capture_id=capture_id,
                                                 digest=digest, storage_root=_storage_root())
        except ScenePreviewUnavailable:
            raise HTTPException(status_code=404, detail="website_scene_preview_unavailable") from None
        except Exception as exc:  # noqa: BLE001 - storage errors can carry signed URLs; name only the class
            logger.warning("website_scene_preview_read_unavailable type=%s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="website_scene_preview_storage_unavailable") from None

        def chunks():
            try:
                yield from body.iter_chunks(CHUNK_BYTES)
            finally:
                body.close()
        return StreamingResponse(chunks(), media_type="application/octet-stream", headers={
            "Content-Length": str(size), "X-Blueprint-Artifact-SHA256": digest, "Content-Disposition": "inline",
            "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"})

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
