"""Authenticated task selection and status; provider events only wake work."""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from .contracts import AgentExecutionError
from .production import configured_service, _read_private


def register_agent_execution_routes(app: FastAPI, *, require_admission, service_factory=configured_service):
    def fail(exc):
        code = str(exc) if isinstance(exc, AgentExecutionError) else "agent_task_configuration_invalid"
        status = 403 if code == "agent_task_client_not_authorized" else 409
        if code in {"agent_production_not_configured", "agent_production_release_mismatch"}:
            status = 503
        return HTTPException(status_code=status, detail=code)

    def client(request):
        value = getattr(request.state, "intake_client_id", None)
        if not isinstance(value, str) or not value:
            raise HTTPException(status_code=401, detail="agent_client_identity_missing")
        return value

    @app.get("/api/live-pipeline/agents/health", dependencies=[Depends(require_admission)])
    def health():
        try:
            return service_factory().health()
        except (AgentExecutionError, ValueError, TypeError) as exc:
            raise fail(exc) from None

    @app.post("/api/live-pipeline/agents/tasks/{task_id}/enqueue", dependencies=[Depends(require_admission)])
    async def enqueue(task_id: str, request: Request):
        # Reject caller-supplied prompts/authority even when syntactically JSON.
        if await request.body() not in (b"", b"{}"):
            raise HTTPException(status_code=400, detail="agent_task_uses_server_admission_only")
        try:
            return await run_in_threadpool(service_factory().enqueue, task_id, client(request))
        except (AgentExecutionError, ValueError, TypeError) as exc:
            raise fail(exc) from None

    @app.get("/api/live-pipeline/agents/tasks/{task_id}", dependencies=[Depends(require_admission)])
    def task_status(task_id: str, request: Request):
        try:
            return service_factory().status(task_id, client(request))
        except (AgentExecutionError, ValueError, TypeError) as exc:
            raise fail(exc) from None

    @app.post("/api/live-pipeline/agents/tasks/{task_id}/{action}", dependencies=[Depends(require_admission)])
    async def action(task_id: str, action: str, request: Request):
        if await request.body() not in (b"", b"{}"):
            raise HTTPException(status_code=400, detail="agent_action_arguments_not_allowed")
        try:
            return await run_in_threadpool(service_factory().act, task_id, client(request), action)
        except (AgentExecutionError, ValueError, TypeError) as exc:
            raise fail(exc) from None

    @app.post("/api/live-pipeline/agents/webhook")
    async def webhook(request: Request):
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > 64_000:
                raise HTTPException(status_code=413, detail="agent_webhook_too_large")
        try:
            service = service_factory()
            path = service.config.webhook_secret_file
            if not path:
                raise AgentExecutionError("agent_webhook_not_configured")
            secret = _read_private(Path(path), limit=16_000, secret=True).decode().strip()
            return await run_in_threadpool(service.service.receive_webhook, bytes(payload),
                                          dict(request.headers), signing_secret=secret)
        except (AgentExecutionError, ValueError, TypeError) as exc:
            if str(exc) == "agent_webhook_signature_invalid":
                raise HTTPException(status_code=401, detail=str(exc)) from None
            raise fail(exc) from None
