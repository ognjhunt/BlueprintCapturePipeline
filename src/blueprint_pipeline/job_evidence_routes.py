"""Small admitted HTTP surface for the existing job operator flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .job_evidence import MAX_REQUEST_BYTES, JobEvidenceError, JobEvidenceStore
from .job_evidence_webapp_sync import sync_job_evidence_to_webapp


def register_job_evidence_routes(
    app: FastAPI, *, require_admission: Callable,
    state_root: Callable[[], Path], run_state_root: Callable[[], Path],
) -> None:
    def store() -> JobEvidenceStore:
        return JobEvidenceStore(state_root(), run_state_root())

    async def body(request: Request) -> dict:
        chunks = bytearray()
        async for chunk in request.stream():
            chunks.extend(chunk)
            if len(chunks) > MAX_REQUEST_BYTES:
                raise HTTPException(413, "job_evidence_request_too_large")
        try:
            value = json.loads(chunks)
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(400, "job_evidence_invalid_json") from exc
        if not isinstance(value, dict):
            raise HTTPException(422, "job_evidence_object_required")
        return value

    def invoke(action: Callable, *args) -> dict:
        try:
            publication = action(*args)
        except (JobEvidenceError, ValueError) as exc:
            code = str(exc)
            status = 404 if code == "job_evidence_not_found" else 409
            raise HTTPException(status, code) from exc
        return {"publication": publication, "webapp_sync": sync_job_evidence_to_webapp(publication)}

    @app.post("/api/live-pipeline/job-evidence-requests", dependencies=[Depends(require_admission)])
    async def submit(request: Request) -> dict:
        payload = await body(request)
        return await run_in_threadpool(invoke, store().submit, payload)

    @app.get("/api/live-pipeline/job-evidence-requests/{request_id}", dependencies=[Depends(require_admission)])
    async def inspect(request_id: str) -> dict:
        return await run_in_threadpool(invoke, store().inspect, request_id)

    @app.post("/api/live-pipeline/job-evidence-requests/{request_id}/execute", dependencies=[Depends(require_admission)])
    async def execute(request_id: str, request: Request) -> dict:
        payload = await body(request)
        if set(payload) != {"request_digest", "authorized_by"}:
            raise HTTPException(422, "job_evidence_execution_fields_invalid")
        return await run_in_threadpool(
            invoke, store().execute, request_id, payload["request_digest"], payload["authorized_by"]
        )

    @app.post("/api/live-pipeline/job-evidence-requests/{request_id}/cancel", dependencies=[Depends(require_admission)])
    async def cancel(request_id: str, request: Request) -> dict:
        payload = await body(request)
        if set(payload) != {"request_digest", "cancelled_by"}:
            raise HTTPException(422, "job_evidence_cancellation_fields_invalid")
        return await run_in_threadpool(
            invoke, store().cancel, request_id, payload["request_digest"], payload["cancelled_by"]
        )
