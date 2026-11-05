"""The WebApp's signed read of one retained screening review video (``beta_screening_review_videos``).

Like the Atlas preview read (``website_atlas_preview_http``): the WebApp checks the
signed-in owner and current consent, then fetches these bytes once, verifies them
against the published digest and size, and serves them view-only. Callers name a
capture and a digest, never a path. A withdrawn capture, or a digest no sealed
manifest of that capture names, is 404.
"""
import logging
import os
from collections.abc import Callable
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool

from .beta_screening_review_videos import open_video
from .website_atlas_preview_store import PreviewUnavailable
from .website_preparation_status_http import HANDOFF_STORAGE_ROOT_ENV

logger = logging.getLogger(__name__)
CHUNK_BYTES = 1024 * 1024


def register_beta_screening_review_video_routes(app: FastAPI, *, require_admission: Callable) -> None:
    @app.get("/api/live-pipeline/beta-screening-review-videos/{video_sha256}",
             dependencies=[Depends(require_admission)])
    async def read_review_video(request: Request, video_sha256: str, request_id: str = "",
                                capture_id: str = "", manifest_digest: str = "") -> StreamingResponse:
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") != "blueprint-webapp"):
            raise HTTPException(status_code=403, detail="screening review video issuer not authorized")
        configured = os.getenv(HANDOFF_STORAGE_ROOT_ENV, "").strip()
        storage_root = Path(configured) if configured and Path(configured).is_absolute() else None
        try:
            stream, size = await run_in_threadpool(open_video, request_id=request_id, capture_id=capture_id,
                                                   digest="sha256:" + video_sha256, storage_root=storage_root,
                                                   manifest_digest=manifest_digest or None)
        except PreviewUnavailable:
            raise HTTPException(status_code=404, detail="beta_screening_review_video_unavailable") from None
        except Exception as exc:  # noqa: BLE001 - name only the class
            logger.warning("beta_screening_review_video_unavailable type=%s", type(exc).__name__)
            raise HTTPException(status_code=503, detail="beta_screening_review_video_storage_unavailable") from None

        def chunks():
            try:
                yield from iter(lambda: stream.read(CHUNK_BYTES), b"")
            finally:
                stream.close()
        return StreamingResponse(chunks(), media_type="video/mp4", headers={
            "Content-Length": str(size), "X-Blueprint-Artifact-SHA256": "sha256:" + video_sha256,
            "Content-Disposition": "inline", "Cache-Control": "private, no-store",
            "X-Content-Type-Options": "nosniff"})
