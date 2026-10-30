"""Signed readback on the existing intake app; callers cannot select local paths."""
import os
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .common import read_json_any
from .website_preparation_status import (
    VERSION_HEADER, PreparationStatusUnavailable, read_preparation_status, requested_status_version,
    validate_selectors,
)

# The listener's tree, from the shared control-plane environment file. Unset
# means no fallback: the explicit capture-root mapping alone answers, as before.
HANDOFF_STORAGE_ROOT_ENV = "BLUEPRINT_PUBSUB_HANDOFF_STORAGE_ROOT"


def website_capture_root(selected: Mapping[str, str]) -> Path | None:
    """The listener's own workspace for these validated selectors, or None.

    Server-owned: the storage root comes from this service's environment and
    the names from validate_selectors, which admits no path syntax. Exactly one
    bucket may hold the capture; read_preparation_status still proves the
    ledger, birth, owner and context before anything is returned.
    """
    configured = os.getenv(HANDOFF_STORAGE_ROOT_ENV, "").strip()
    storage = Path(configured)
    if not configured or not storage.is_absolute() or not storage.is_dir():
        return None
    matches = [path for path in storage.glob(f"*/scenes/{selected['scene_id']}/captures/{selected['capture_id']}")
               if path.is_dir()]
    return matches[0] if len(matches) == 1 else None


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
        schema_version = requested_status_version(request.headers.get(VERSION_HEADER))
        try:
            manifest_path = manifest_path_provider()
            manifest = read_json_any(manifest_path) if manifest_path.is_file() else {}
            manifest_root = manifest.get("capture_root") if isinstance(manifest, Mapping) else None
            root = resolve_client_root(payload={**selected, "site_submission_id": selected["request_id"]},
                    client_id=request.state.intake_client_id,
                    manifest_capture_root=Path(manifest_root).expanduser().absolute()
                        if type(manifest_root) is str and manifest_root else None)
            # A configured map or control-plane manifest names at most one
            # capture; every other website capture lives in the listener's tree.
            if root is None or Path(root).name != selected["capture_id"]:
                root = website_capture_root(selected)
            if root is None:
                raise PreparationStatusUnavailable("website_preparation_status_unavailable")
            result = await run_in_threadpool(read_preparation_status, capture_root=root, selectors=selected,
                                             schema_version=schema_version)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=503, detail="website_preparation_status_unavailable") from exc
        return JSONResponse(result, headers={"Cache-Control": "no-store"})

    # Separate opt-in companion: does not extend any v1-v4 receipt or byte cap.
    from .website_candidate_readiness_http import register_candidate_readiness_routes
    register_candidate_readiness_routes(app, require_admission=require_admission,
        manifest_path_provider=manifest_path_provider, resolve_client_root=resolve_client_root)
