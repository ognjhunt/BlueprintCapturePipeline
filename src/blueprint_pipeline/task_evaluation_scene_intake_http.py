"""Signed owner-intent routes on the existing Pipeline intake application."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from .decision_evidence_contracts import cross_runtime_canonical_digest
from .native_g1_team_campaign_intake import (
    QUEUE_ENV as G1_QUEUE_ENV,
    REGISTRY_ENV as G1_REGISTRY_ENV,
    list_g1_team_campaign_setups,
    stage_g1_team_campaign,
)
from .task_evaluation_scene_intake import (
    CLIENTS_ENV, ROOT_ENV, SceneIntakeError, stage_scene_intent, scene_intent_status, revoke_scene_intent,
)


def register_scene_intake_routes(app: FastAPI, require_admission: Callable,
                                 deployment_identity: Callable) -> None:
    @app.post("/api/live-pipeline/native-g1-team-campaign-setups",
              dependencies=[Depends(require_admission)])
    async def inspect_native_g1_team_campaign_setups(request: Request) -> JSONResponse:
        trusted = {v.strip() for v in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if v.strip()}
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") not in trusted):
            raise HTTPException(status_code=403, detail="G1 team catalog issuer not authorized")
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, dict) or set(payload) != {"owner"}:
            raise HTTPException(status_code=422, detail="G1 team catalog request invalid")
        registry = os.getenv(G1_REGISTRY_ENV, "").strip()
        if not registry:
            raise HTTPException(status_code=503, detail="G1 team catalog not configured")
        try:
            catalog = await run_in_threadpool(
                list_g1_team_campaign_setups,
                registry_path=Path(registry), owner=payload["owner"],
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except (OSError, KeyError, TypeError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail="G1 team catalog unavailable") from exc
        return JSONResponse(content=catalog, headers={"Cache-Control": "private, no-store"})

    @app.post("/api/live-pipeline/native-g1-team-campaigns",
              dependencies=[Depends(require_admission)])
    async def intake_native_g1_team_campaign(request: Request) -> JSONResponse:
        if not request.headers.get("x-blueprint-pipeline-signature"):
            raise HTTPException(status_code=401, detail="G1 team intake requires signed owner authority")
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="expected JSON object")
        registry = os.getenv(G1_REGISTRY_ENV, "").strip()
        root = os.getenv(G1_QUEUE_ENV, "").strip()
        if not registry or not root:
            raise HTTPException(status_code=503, detail="G1 team intake not configured")
        if "launch_preparation" in (deployment_identity().get("disk_headroom", {}).get("refused_roles") or []):
            raise HTTPException(status_code=503, detail="G1 team intake disk admission refused")
        trusted = {v.strip() for v in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if v.strip()}
        try:
            receipt = await run_in_threadpool(
                stage_g1_team_campaign, value=payload, registry_path=Path(registry),
                queue_root=Path(root),
                authenticated_client=str(getattr(request.state, "intake_client_id", "")),
                trusted_clients=trusted,
            )
        except ValueError as exc:
            code = str(exc)
            return JSONResponse(status_code=(403 if code.endswith("issuer_not_authorized")
                else 409 if code.endswith("idempotency_conflict") else 422),
                content={"status": "rejected", "blockers": [code],
                         "provider_mutation_performed_inside_http_request": False})
        except (OSError, KeyError, TypeError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail="G1 team intake unavailable") from exc
        return JSONResponse(status_code=202, content=receipt,
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/live-pipeline/task-evaluation-team-context",
              dependencies=[Depends(require_admission)])
    async def inspect_team_evaluation_context(request: Request) -> JSONResponse:
        trusted = {v.strip() for v in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if v.strip()}
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") not in trusted):
            raise HTTPException(status_code=403, detail="scene source issuer not authorized")
        from .task_evaluation_team_run_context import evaluation_context
        try:
            payload = await request.json()
            if not isinstance(payload, dict) or set(payload) != {'source_launch_id', 'owner'}:
                raise ValueError('evaluation_context_request_invalid')
            result = await run_in_threadpool(evaluation_context, **payload)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=409, detail="evaluation context unavailable") from exc
        return JSONResponse(content=result, headers={"Cache-Control":"no-store"})

    @app.get("/api/live-pipeline/task-evaluation-public-scene-sources",
             dependencies=[Depends(require_admission)])
    async def inspect_public_scene_sources(request: Request) -> JSONResponse:
        trusted = {v.strip() for v in os.getenv(CLIENTS_ENV, "blueprint-webapp").split(",") if v.strip()}
        if (not request.headers.get("x-blueprint-pipeline-signature")
                or getattr(request.state, "intake_client_id", "") not in trusted):
            raise HTTPException(status_code=403, detail="scene source issuer not authorized")
        from .task_evaluation_public_scene_catalog import load_catalog
        try:
            value = await run_in_threadpool(load_catalog)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise HTTPException(status_code=503, detail="public scene source catalog unavailable") from exc
        return JSONResponse(content=value, headers={"Cache-Control": "no-store"})

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
        headroom = deployment_identity().get("disk_headroom", {})
        refused_roles = sorted(str(role) for role in (headroom.get("refused_roles") or []))
        capacity: dict[str, object] = {"state": "available"}
        if "launch_preparation" in refused_roles:
            from .control_plane_capacity_controller import _read_attention_summary, capacity_eta

            target = next((row for row in headroom.get("targets") or []
                           if isinstance(row, Mapping) and row.get("role") == "launch_preparation"), {})
            footprint = (headroom.get("footprints") or {}).get("launch_preparation") or {}
            required = footprint.get("bytes")
            available = target.get("available_bytes", headroom.get("available_bytes"))
            shortfall = (max(0, int(required) - int(available))
                         if isinstance(required, (int, float)) and isinstance(available, (int, float))
                         else None)
            now = time.time()
            summary_path = Path(os.getenv("BLUEPRINT_CAPACITY_SUMMARY_PATH",
                                          "/var/lib/blueprint/pipeline-control-plane/capacity/summary.json"))
            summary = _read_attention_summary(summary_path)
            observed = summary.get("observed_at_epoch") if isinstance(summary, Mapping) else None
            if (not isinstance(observed, (int, float)) or now - observed > 7200
                    or observed > now + 300 or shortfall is None):
                summary = None
            capacity = {"state": "queued_for_capacity", "refused_roles": refused_roles,
                        **capacity_eta(shortfall or 0, summary=summary, now=now)}
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
        response_receipt = {**receipt, "capacity": capacity}
        response_receipt["receipt_digest"] = cross_runtime_canonical_digest(
            response_receipt, digest_field="receipt_digest")
        return JSONResponse(status_code=202, content=response_receipt)

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
