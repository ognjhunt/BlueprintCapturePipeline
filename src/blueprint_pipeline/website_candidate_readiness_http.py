"""Bounded signed companion; existing preparation status schema/limit stays unchanged."""

from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .common import read_json_any
from .website_candidate_readiness import (
    PATH,
    CandidateReadinessUnavailable,
    read_candidate_readiness,
)
from .website_preparation_status import validate_selectors
from .website_preparation_status_http import website_capture_root


def register_candidate_readiness_routes(
    app: FastAPI,
    *,
    require_admission: Callable,
    manifest_path_provider: Callable,
    resolve_client_root: Callable,
) -> None:
    @app.post(PATH, dependencies=[Depends(require_admission)])
    async def read_candidates(request: Request) -> JSONResponse:
        if (
            not request.headers.get("x-blueprint-pipeline-signature")
            or getattr(request.state, "intake_client_id", "") != "blueprint-webapp"
        ):
            raise HTTPException(
                status_code=403, detail="website candidate readiness issuer not authorized"
            )
        try:
            selected = validate_selectors(await request.json())
        except (ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=422, detail="website_candidate_readiness_selectors_invalid"
            ) from exc
        try:
            path = manifest_path_provider()
            manifest = read_json_any(path) if path.is_file() else {}
            configured = manifest.get("capture_root") if isinstance(manifest, Mapping) else None
            root = resolve_client_root(
                payload={**selected, "site_submission_id": selected["request_id"]},
                client_id=request.state.intake_client_id,
                manifest_capture_root=Path(configured).expanduser().absolute()
                if type(configured) is str and configured
                else None,
            )
            if root is None or Path(root).name != selected["capture_id"]:
                root = website_capture_root(selected)
            if root is None:
                raise CandidateReadinessUnavailable("website_candidate_readiness_unavailable")
            value = await run_in_threadpool(
                read_candidate_readiness, capture_root=Path(root), selectors=selected
            )
        except CandidateReadinessUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503, detail="website_candidate_readiness_unavailable"
            ) from exc
        return JSONResponse(value, headers={"Cache-Control": "no-store"})
