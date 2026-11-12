"""Owner downloads of an archived Atlas preview through the WebApp's signed client (D6, 2026-10-22).

The WebApp checks the signed-in owner and current consent, then streams these
bytes on; callers name a capture and digest, never a path or object key. The
web-sized scene preview a site opens from its job page (``website_scene_preview``)
is read the same way, and so is the silent clip an approved deployment company
watches (``website_company_clip``), which also answers one byte range.
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
    @app.get("/api/live-pipeline/website-destination-pictures/{capture_id}/destination.place.png",
             dependencies=[Depends(require_admission)])
    async def read_place_picture(request: Request, capture_id: str, request_id: str = "", digest: str = ""):
        """The place picture (``website_destination_choice``), read like the scene preview; 404 while switch
        (iii) is off."""
        from fastapi.responses import Response

        from .website_destination_choice import PlacePictureUnavailable, open_place_picture
        _webapp_only(request)
        try:
            body = await run_in_threadpool(open_place_picture, request_id=request_id, capture_id=capture_id,
                                           digest=digest, storage_root=_storage_root())
        except PlacePictureUnavailable:
            raise HTTPException(status_code=404, detail="website_place_picture_unavailable") from None
        return Response(body, media_type="image/png", headers={
            "X-Blueprint-Artifact-SHA256": digest, "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff"})

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

    @app.get("/api/live-pipeline/website-company-clips/{capture_id}/clip.mp4",
             dependencies=[Depends(require_admission)])
    async def read_company_clip(request: Request, capture_id: str, request_id: str = "",
                                digest: str = "") -> StreamingResponse:
        """The silent company clip (``website_company_clip``), read like the scene preview, by range."""
        from fastapi.responses import JSONResponse

        from .website_company_clip import CompanyClipUnavailable, RangeNotSatisfiable, open_company_clip
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") != "blueprint-webapp"):
            raise HTTPException(status_code=403, detail="website company clip issuer not authorized")
        try:
            body, size, served = await run_in_threadpool(
                open_company_clip, request_id=request_id, capture_id=capture_id, digest=digest,
                storage_root=_storage_root(), range_header=request.headers.get("range"))
        except CompanyClipUnavailable:
            raise HTTPException(status_code=404, detail="website_company_clip_unavailable") from None
        except RangeNotSatisfiable as exc:
            return JSONResponse({"detail": "website_company_clip_range_not_satisfiable"}, status_code=416, headers={
                "Content-Range": f"bytes */{exc.size}", "Cache-Control": "private, no-store"})
        except Exception as exc:  # noqa: BLE001 - storage errors can carry signed URLs; name only the class
            logger.warning("website_company_clip_read_unavailable type=%s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="website_company_clip_storage_unavailable") from None

        def chunks():
            try:
                yield from body.iter_chunks(CHUNK_BYTES)
            finally:
                body.close()
        headers = {"Accept-Ranges": "bytes", "X-Blueprint-Artifact-SHA256": digest, "Content-Disposition": "inline",
                   "Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"}
        if served is None:
            return StreamingResponse(chunks(), media_type="video/mp4", headers={**headers, "Content-Length": str(size)})
        first, last = served
        return StreamingResponse(chunks(), status_code=206, media_type="video/mp4", headers={
            **headers, "Content-Length": str(last - first + 1), "Content-Range": f"bytes {first}-{last}/{size}"})

    @app.post("/api/live-pipeline/website-company-clips/{capture_id}/recheck",
              dependencies=[Depends(require_admission)])
    async def recheck_company_clip(request: Request, capture_id: str, request_id: str = ""):
        """A job's permission changed: the next clip pass reads the live rights again (``permission_recheck``).
        It only marks; it tells the caller nothing about the capture."""
        from fastapi.responses import JSONResponse

        from .website_company_clip import RECHECK_SCHEMA, request_company_clip_recheck
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") != "blueprint-webapp"):
            raise HTTPException(status_code=403, detail="website company clip issuer not authorized")
        outcome = await run_in_threadpool(request_company_clip_recheck, request_id=request_id,
                                          capture_id=capture_id, storage_root=_storage_root())
        if outcome == "too_soon":
            return JSONResponse({"detail": "website_company_clip_recheck_too_soon"}, status_code=429,
                                headers={"Retry-After": "60", "Cache-Control": "private, no-store"})
        if outcome != "accepted":
            raise HTTPException(status_code=404, detail="website_company_clip_unavailable")
        return JSONResponse({"schema_version": RECHECK_SCHEMA, "accepted": True}, status_code=202,
                            headers={"Cache-Control": "private, no-store"})

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
