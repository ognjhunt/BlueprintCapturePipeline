"""Signed owner-intent routes on the existing Pipeline intake application."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .task_evaluation_scene_intake import (
    CLIENTS_ENV, ROOT_ENV, SceneIntakeError, stage_scene_intent, scene_intent_status, revoke_scene_intent,
)


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

    @app.get("/api/live-pipeline/task-evaluation-scene-intents/{intent_id}",
             dependencies=[Depends(require_admission)])
    async def inspect_task_evaluation_scene_intent(intent_id: str, request: Request) -> JSONResponse:
        trusted = {v.strip() for v in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if v.strip()}
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") not in trusted):
            raise HTTPException(status_code=403, detail="scene intent issuer not authorized")
        root = os.getenv(ROOT_ENV, "").strip()
        if not root:
            raise HTTPException(status_code=503, detail="scene intake queue not configured")
        try:
            result = await run_in_threadpool(scene_intent_status, queue_root=root, intent_id=intent_id)
        except SceneIntakeError as exc:
            raise HTTPException(status_code=404 if str(exc).endswith("record_unreadable") else 409,
                                detail=str(exc)) from exc
        return JSONResponse(content=result, headers={"Cache-Control": "no-store"})

    @app.post("/api/live-pipeline/task-evaluation-scene-intents/{intent_id}/revoke",
              dependencies=[Depends(require_admission)])
    async def revoke_task_evaluation_scene_intent(intent_id: str, request: Request) -> JSONResponse:
        trusted = {v.strip() for v in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if v.strip()}
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") not in trusted):
            raise HTTPException(status_code=403, detail="scene intent issuer not authorized")
        root = os.getenv(ROOT_ENV, "").strip()
        if not root:
            raise HTTPException(status_code=503, detail="scene intake queue not configured")
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, Mapping) or set(payload) != {"intent_digest", "owner"}:
            raise HTTPException(status_code=422, detail="revocation request invalid")
        try:
            result = await run_in_threadpool(revoke_scene_intent, queue_root=root, intent_id=intent_id,
                intent_digest=payload["intent_digest"], owner=payload["owner"])
        except SceneIntakeError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return JSONResponse(content=result, headers={"Cache-Control": "no-store"})
