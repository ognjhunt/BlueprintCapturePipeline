"""Signed readback on the existing intake app; callers cannot select local paths."""
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .common import read_json_any
from .website_preparation_status import PreparationStatusUnavailable, read_preparation_status, validate_selectors


def register_website_preparation_status_routes(app: FastAPI, *, require_admission: Callable,
        manifest_path_provider: Callable[[], Path], resolve_client_root: Callable) -> None:
    @app.post("/api/live-pipeline/website-preparation-status", dependencies=[Depends(require_admission)])
    async def read_status(request: Request) -> JSONResponse:
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") != "blueprint-webapp"):
            raise HTTPException(status_code=403, detail="website preparation status issuer not authorized")
        try:
            selected = validate_selectors(await request.json())
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="website_preparation_status_selectors_invalid") from exc
        try:
            manifest_path = manifest_path_provider()
            manifest = read_json_any(manifest_path) if manifest_path.is_file() else {}
            manifest_root = manifest.get("capture_root") if isinstance(manifest, Mapping) else None
            root = resolve_client_root(payload={**selected, "site_submission_id": selected["request_id"]},
                    client_id=request.state.intake_client_id,
                    manifest_capture_root=Path(manifest_root).expanduser().absolute()
                        if type(manifest_root) is str and manifest_root else None)
            if root is None:
                raise PreparationStatusUnavailable("website_preparation_status_unavailable")
            result = await run_in_threadpool(read_preparation_status, capture_root=root, selectors=selected)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail="website_preparation_status_unavailable") from exc
        return JSONResponse(result, headers={"Cache-Control": "no-store"})
