"""Signed no-spend preparation and the existing owner spend-admission surface."""
from __future__ import annotations
import json
from typing import Callable
from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool
from .beta_policy_finalists import prepare_finalist, inspect_finalist, authorize_finalist


def register_beta_policy_finalist_routes(app: FastAPI, *, require_admission: Callable) -> None:
    async def body(request: Request) -> dict:
        raw = bytearray()
        async for part in request.stream():
            raw.extend(part)
            if len(raw) > 32_768:
                raise HTTPException(413, "beta_finalist_request_too_large")
        try:
            value = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(400, "beta_finalist_invalid_json") from exc
        if not isinstance(value, dict):
            raise HTTPException(422, "beta_finalist_object_required")
        return value

    def invoke(action, *args):
        try:
            return action(*args)
        except FileNotFoundError as exc:
            raise HTTPException(404, "beta_finalist_not_found") from exc
        except (ValueError, KeyError, TypeError) as exc:
            code = str(exc)
            raise HTTPException(409, code if code.startswith("beta_") else "beta_finalist_source_unavailable") from exc

    @app.post("/api/live-pipeline/beta-policy-finalists", dependencies=[Depends(require_admission)])
    async def prepare(request: Request):
        return await run_in_threadpool(invoke, prepare_finalist, await body(request))

    @app.get("/api/live-pipeline/beta-policy-finalists/{fid}", dependencies=[Depends(require_admission)])
    async def inspect(fid: str):
        return await run_in_threadpool(invoke, inspect_finalist, fid)

    @app.post("/api/live-pipeline/beta-policy-finalists/{fid}/authorize", dependencies=[Depends(require_admission)])
    async def authorize(fid: str, request: Request):
        return await run_in_threadpool(invoke, authorize_finalist, fid, await body(request))
