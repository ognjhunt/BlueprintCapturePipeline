"""Signed owner-intent routes on the existing Pipeline intake application."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .task_evaluation_scene_intake import CLIENTS_ENV, ROOT_ENV, SceneIntakeError, stage_scene_intent


def register_scene_intake_routes(app: FastAPI, require_admission: Callable,
                                 deployment_identity: Callable) -> None:
    @app.post("/api/live-pipeline/task-evaluation-scene-intents",
              dependencies=[Depends(require_admission)])
    async def intake_task_evaluation_scene_intent(request: Request) -> JSONResponse:
        # This grants bounded future execution, unlike preparation-only intake.
        # Legacy bearer admission is deliberately insufficient here.
        if not request.headers.get("x-blueprint-pipeline-signature"):
            raise HTTPException(status_code=401, detail="scene intake requires signed owner authority")
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping):
            raise HTTPException(status_code=400, detail="expected JSON object")
        root = os.getenv(ROOT_ENV, "").strip()
        if not root:
            raise HTTPException(status_code=503, detail="scene intake queue not configured")
        if "launch_preparation" in (deployment_identity().get("disk_headroom", {}).get("refused_roles") or []):
            raise HTTPException(status_code=503, detail="scene intake disk admission refused")
        trusted = {item.strip() for item in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",")
                   if item.strip()}
        try:
            receipt = await run_in_threadpool(
                stage_scene_intent, value=payload, queue_root=root,
                authenticated_client=str(getattr(request.state, "intake_client_id", "")),
                trusted_clients=trusted,
            )
        except SceneIntakeError as exc:
            code = str(exc)
            return JSONResponse(status_code=(403 if code.endswith("issuer_not_authorized")
                else 409 if code.endswith("idempotency_conflict") else 422),
                content={"status": "rejected", "blockers": [code],
                         "provider_mutation_performed_inside_http_request": False})
        return JSONResponse(status_code=202, content=receipt)
