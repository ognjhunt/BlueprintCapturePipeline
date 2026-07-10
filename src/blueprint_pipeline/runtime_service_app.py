"""Shared FastAPI app factory for native site-world runtime backends."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import threading
from typing import Any, Dict, Mapping, Protocol

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from blueprint_contracts.site_world_contract import normalize_trajectory_payload

from .logging_utils import log_event


logger = logging.getLogger(__name__)


class RuntimeBackend(Protocol):
    base_url: str
    ws_base_url: str

    def runtime_info(self, *, service_version: str) -> Dict[str, Any]:
        ...

    def register_site_world_package(
        self,
        *,
        spec: Dict[str, Any],
        registration: Dict[str, Any],
        health: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...

    def load_site_world(self, site_world_id: str) -> Dict[str, Any]:
        ...

    def load_site_world_health(self, site_world_id: str) -> Dict[str, Any]:
        ...

    def create_session(self, site_world_id: str, **kwargs: Any) -> Dict[str, Any]:
        ...

    def reset_session(self, session_id: str, **kwargs: Any) -> Dict[str, Any]:
        ...

    def step_session(self, session_id: str, *, action: Any) -> Dict[str, Any]:
        ...

    def session_state(self, session_id: str) -> Dict[str, Any]:
        ...

    def control_session(self, session_id: str, *, control: Dict[str, Any]) -> Dict[str, Any]:
        ...

    def render_bytes(self, session_id: str, camera_id: str) -> bytes:
        ...

    def media_response(self, session_id: str, *, camera_id: str, chunk_id: str | None) -> Dict[str, Any]:
        ...

    def drain_media_events(self, session_id: str) -> list[Dict[str, Any]]:
        ...

    def explorer_render(
        self,
        session_id: str,
        *,
        camera_id: str,
        pose: Dict[str, Any],
        viewport_width: int | None,
        viewport_height: int | None,
        refine_mode: str | None,
    ) -> Dict[str, Any]:
        ...

    def explorer_frame_bytes(self, session_id: str, camera_id: str) -> bytes:
        ...


class SessionCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str | None = None
    robot_profile_id: str
    task_id: str
    scenario_id: str
    start_state_id: str
    requested_backend: str | None = None
    notes: str = ""
    canonical_package_uri: str | None = None
    canonical_package_version: str | None = None
    prompt: str | None = None
    trajectory: Dict[str, Any] | str | None = None
    presentation_model: str | None = None


class SessionResetRequest(BaseModel):
    task_id: str | None = None
    scenario_id: str | None = None
    start_state_id: str | None = None


class SessionStepRequest(BaseModel):
    action: list[float] | Dict[str, Any] = Field(default_factory=list)


class SessionControlRequest(BaseModel):
    seq: int | None = None
    tClientMs: int | None = None
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yawRate: float = 0.0
    pitchRate: float = 0.0
    durationMs: int = 1200


class ExplorerPoseRequest(BaseModel):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0


class ExplorerRenderRequest(BaseModel):
    camera_id: str = "head_rgb"
    pose: ExplorerPoseRequest = Field(default_factory=ExplorerPoseRequest)
    viewport_width: int | None = None
    viewport_height: int | None = None
    refine_mode: str | None = None


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _runtime_auth_tokens_from_env() -> dict[str, str]:
    """Return bearer-token -> tenant-id without persisting raw tokens."""

    configured: dict[str, str] = {}
    raw = str(os.getenv("BLUEPRINT_RUNTIME_AUTH_TOKENS_JSON") or "").strip()
    if raw:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError("BLUEPRINT_RUNTIME_AUTH_TOKENS_JSON must be valid JSON") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError("BLUEPRINT_RUNTIME_AUTH_TOKENS_JSON must be an object")
        for tenant_id, token in payload.items():
            tenant = str(tenant_id or "").strip()
            secret = str(token or "").strip()
            if not tenant or not secret:
                raise RuntimeError("runtime auth token entries require nonempty tenant and token")
            configured[secret] = tenant
    single = str(
        os.getenv("BLUEPRINT_RUNTIME_AUTH_TOKEN")
        or os.getenv("SITE_WORLD_RUNTIME_SERVICE_API_KEY")
        or ""
    ).strip()
    if single:
        tenant = str(
            os.getenv("BLUEPRINT_RUNTIME_TENANT_ID")
            or os.getenv("SITE_WORLD_RUNTIME_TENANT_ID")
            or "default"
        ).strip()
        if not tenant:
            raise RuntimeError("BLUEPRINT_RUNTIME_TENANT_ID must be nonempty")
        configured[single] = tenant
    return configured


def runtime_auth_required_from_env() -> bool:
    return bool(
        _env_truthy("BLUEPRINT_RUNTIME_REQUIRE_AUTH")
        or _env_truthy("NATIVE_WORLD_MODEL_PRODUCTION_GRADE")
        or _runtime_auth_tokens_from_env()
    )


def validate_runtime_service_exposure(*, host: str) -> None:
    """Prevent a non-loopback runtime from starting without bearer auth."""

    exposed = host.strip().lower() not in {"127.0.0.1", "::1", "localhost"}
    if exposed and not _runtime_auth_tokens_from_env():
        raise RuntimeError(
            "non-loopback runtime service requires BLUEPRINT_RUNTIME_AUTH_TOKEN "
            "or BLUEPRINT_RUNTIME_AUTH_TOKENS_JSON (SITE_WORLD_RUNTIME_SERVICE_API_KEY "
            "is accepted for the single-tenant client contract)"
        )


def create_runtime_app(
    *,
    backend: RuntimeBackend,
    title: str,
    auth_tokens: Mapping[str, str] | None = None,
    require_auth: bool | None = None,
) -> FastAPI:
    app = FastAPI(title=title, version="1.0.0")
    raw_auth_tokens = dict(auth_tokens) if auth_tokens is not None else _runtime_auth_tokens_from_env()
    resolved_auth_tokens = {
        str(token).strip(): str(tenant_id).strip()
        for token, tenant_id in raw_auth_tokens.items()
    }
    if any(
        not str(token).strip()
        or not str(tenant_id).strip()
        or len(str(tenant_id).strip()) > 256
        for token, tenant_id in resolved_auth_tokens.items()
    ):
        raise RuntimeError("runtime auth tokens require nonempty tokens and tenant identifiers")
    auth_required = runtime_auth_required_from_env() if require_auth is None else require_auth
    auth_enabled = bool(auth_required or resolved_auth_tokens)
    site_world_tenants: dict[str, str] = {}
    session_tenants: dict[str, str] = {}
    resource_owner_lock = threading.RLock()

    def _authenticate_header(value: str | None) -> str | None:
        header = str(value or "").strip()
        if not header.startswith("Bearer "):
            return None
        candidate = header[7:]
        for token, tenant_id in resolved_auth_tokens.items():
            if hmac.compare_digest(candidate.encode("utf-8"), token.encode("utf-8")):
                return str(tenant_id)
        return None

    @app.middleware("http")
    async def authenticate_request(request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path == "/healthz" or not auth_enabled:
            return await call_next(request)
        tenant_id = _authenticate_header(request.headers.get("Authorization"))
        if tenant_id is None:
            return JSONResponse(status_code=401, content={"detail": "Unauthorized"})
        request.state.runtime_tenant_id = tenant_id
        return await call_next(request)

    def _tenant_context(request: Request) -> str | None:
        return getattr(request.state, "runtime_tenant_id", None)

    def _site_world_owner(site_world_id: str) -> str | None:
        if site_world_id in site_world_tenants:
            return site_world_tenants[site_world_id]
        owner_loader = getattr(backend, "site_world_tenant_id", None)
        if callable(owner_loader):
            owner = str(owner_loader(site_world_id) or "").strip() or None
        else:
            owner = str(backend.load_site_world(site_world_id).get("tenant_id") or "").strip() or None
        if owner:
            site_world_tenants[site_world_id] = owner
        return owner

    def _session_owner(session_id: str) -> str | None:
        if session_id in session_tenants:
            return session_tenants[session_id]
        owner_loader = getattr(backend, "session_tenant_id", None)
        if callable(owner_loader):
            owner = str(owner_loader(session_id) or "").strip() or None
        else:
            owner = str(backend.session_state(session_id).get("tenant_id") or "").strip() or None
        if owner:
            session_tenants[session_id] = owner
        return owner

    def _authorize_owner(owner: str | None, tenant_id: str | None) -> None:
        if not auth_enabled:
            return
        if (
            not owner
            or not tenant_id
            or not hmac.compare_digest(owner.encode("utf-8"), tenant_id.encode("utf-8"))
        ):
            raise HTTPException(status_code=403, detail="Forbidden")

    def _authorized_session_context(
        session_id: str,
        tenant_id: str | None = Depends(_tenant_context),
    ) -> str | None:
        if auth_enabled:
            try:
                _authorize_owner(_session_owner(session_id), tenant_id)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail="Session not found") from exc
        return tenant_id

    def _http_error(
        *,
        route: str,
        status_code: int,
        detail: str,
        exc: Exception,
        **fields: Any,
    ) -> HTTPException:
        log_event(
            logger,
            logging.WARNING,
            "runtime_service.request_failed",
            route=route,
            status_code=status_code,
            detail=detail,
            error_type=type(exc).__name__,
            **fields,
        )
        return HTTPException(status_code=status_code, detail=detail)

    @app.on_event("startup")
    async def prewarm_backend() -> None:
        if auth_required and not resolved_auth_tokens:
            raise RuntimeError("runtime authentication is required but no bearer token is configured")
        prewarm = getattr(backend, "prewarm_runtime", None)
        if callable(prewarm):
            log_event(logger, logging.INFO, "runtime_service.prewarm_started", title=title)
            try:
                result = await asyncio.to_thread(prewarm)
            except Exception as exc:
                log_event(
                    logger,
                    logging.ERROR,
                    "runtime_service.prewarm_failed",
                    title=title,
                    error_type=type(exc).__name__,
                    reason=str(exc),
                )
                raise
            log_event(
                logger,
                logging.INFO,
                "runtime_service.prewarm_completed",
                title=title,
                result_status=dict(result).get("status") if isinstance(result, dict) else None,
            )

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        runtime = backend.runtime_info(service_version=app.version)
        readiness = dict(runtime.get("readiness") or {})
        return {
            "status": "ok",
            "service": runtime.get("service") or "site-world-runtime",
            "version": app.version,
            "runtime_kind": runtime.get("runtime_kind"),
            "production_grade": runtime.get("production_grade"),
            "model_ready": bool(readiness.get("model_ready", False)),
            "checkpoint_ready": bool(readiness.get("checkpoint_ready", False)),
        }

    @app.get("/v1/runtime")
    def runtime_info(_tenant_id: str | None = Depends(_tenant_context)) -> Dict[str, Any]:
        return backend.runtime_info(service_version=app.version)

    @app.post("/v1/site-worlds")
    def register_site_world(
        payload: Dict[str, Any],
        tenant_id: str | None = Depends(_tenant_context),
    ) -> Dict[str, Any]:
        try:
            if not all(isinstance(payload.get(key), dict) for key in ("spec", "registration", "health")):
                raise ValueError("site-world registration requires spec + registration + health payloads")
            registration_payload = dict(payload["registration"])
            supplied_tenant = str(registration_payload.get("tenant_id") or "").strip() or None
            if auth_enabled and supplied_tenant not in {None, tenant_id}:
                raise PermissionError("registration tenant does not match authenticated tenant")
            if tenant_id:
                registration_payload["tenant_id"] = tenant_id
            site_world_id = str(registration_payload.get("site_world_id") or "").strip()
            with resource_owner_lock:
                if auth_enabled and site_world_id:
                    try:
                        _authorize_owner(_site_world_owner(site_world_id), tenant_id)
                    except FileNotFoundError:
                        pass
                registration = dict(
                    backend.register_site_world_package(
                        spec=dict(payload["spec"]),
                        registration=registration_payload,
                        health=dict(payload["health"]),
                    )
                )
            health = dict(backend.load_site_world_health(str(registration.get("site_world_id") or "")))
        except PermissionError as exc:
            raise _http_error(
                route="register_site_world",
                status_code=403,
                detail="Forbidden",
                exc=exc,
            ) from exc
        except HTTPException:
            raise
        except Exception as exc:
            raise _http_error(
                route="register_site_world",
                status_code=400,
                detail=str(exc),
                exc=exc,
            ) from exc
        log_event(
            logger,
            logging.INFO,
            "runtime_service.site_world_registered",
            site_world_id=registration.get("site_world_id"),
            health_status=health.get("status"),
        )
        if tenant_id:
            site_world_tenants[str(registration.get("site_world_id") or "")] = tenant_id
        return {
            **registration,
            "health": health,
        }

    @app.get("/v1/site-worlds/{site_world_id}")
    def get_site_world(
        site_world_id: str,
        tenant_id: str | None = Depends(_tenant_context),
    ) -> Dict[str, Any]:
        try:
            _authorize_owner(_site_world_owner(site_world_id), tenant_id)
            payload = dict(backend.load_site_world(site_world_id))
        except FileNotFoundError as exc:
            detail = f"site world not found: {site_world_id}"
            raise _http_error(
                route="get_site_world",
                status_code=404,
                detail=detail,
                exc=exc,
                site_world_id=site_world_id,
            ) from exc
        log_event(
            logger,
            logging.DEBUG,
            "runtime_service.site_world_loaded",
            site_world_id=site_world_id,
        )
        return payload

    @app.get("/v1/site-worlds/{site_world_id}/health")
    def get_site_world_health(
        site_world_id: str,
        tenant_id: str | None = Depends(_tenant_context),
    ) -> Dict[str, Any]:
        try:
            _authorize_owner(_site_world_owner(site_world_id), tenant_id)
            payload = dict(backend.load_site_world_health(site_world_id))
        except FileNotFoundError as exc:
            detail = f"site world not found: {site_world_id}"
            raise _http_error(
                route="get_site_world_health",
                status_code=404,
                detail=detail,
                exc=exc,
                site_world_id=site_world_id,
            ) from exc
        log_event(
            logger,
            logging.DEBUG,
            "runtime_service.site_world_health_loaded",
            site_world_id=site_world_id,
            health_status=payload.get("status"),
        )
        return payload

    @app.post("/v1/site-worlds/{site_world_id}/sessions")
    def create_session(
        site_world_id: str,
        request: SessionCreateRequest,
        tenant_id: str | None = Depends(_tenant_context),
    ) -> Dict[str, Any]:
        try:
            _authorize_owner(_site_world_owner(site_world_id), tenant_id)
            with resource_owner_lock:
                if auth_enabled and request.session_id:
                    try:
                        _session_owner(request.session_id)
                    except FileNotFoundError:
                        pass
                    else:
                        raise HTTPException(status_code=409, detail="Session already exists")
                session = dict(
                    backend.create_session(
                        site_world_id,
                        session_id=request.session_id,
                        robot_profile_id=request.robot_profile_id,
                        task_id=request.task_id,
                        scenario_id=request.scenario_id,
                        start_state_id=request.start_state_id,
                        requested_backend=request.requested_backend,
                        notes=request.notes,
                        canonical_package_uri=request.canonical_package_uri,
                        canonical_package_version=request.canonical_package_version,
                        prompt=request.prompt,
                        trajectory=normalize_trajectory_payload(request.trajectory),
                        presentation_model=request.presentation_model,
                        debug_mode=False,
                        tenant_id=tenant_id,
                    )
                )
        except FileNotFoundError as exc:
            detail = f"site world not found: {site_world_id}"
            raise _http_error(
                route="create_session",
                status_code=404,
                detail=detail,
                exc=exc,
                site_world_id=site_world_id,
            ) from exc
        except HTTPException:
            raise
        except FileExistsError as exc:
            raise _http_error(
                route="create_session",
                status_code=409,
                detail="Session already exists",
                exc=exc,
                site_world_id=site_world_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="create_session",
                status_code=400,
                detail=str(exc),
                exc=exc,
                site_world_id=site_world_id,
            ) from exc
        log_event(
            logger,
            logging.INFO,
            "runtime_service.session_created",
            site_world_id=site_world_id,
            session_id=session.get("session_id"),
            robot_profile_id=request.robot_profile_id,
            task_id=request.task_id,
            scenario_id=request.scenario_id,
            start_state_id=request.start_state_id,
            requested_backend=request.requested_backend,
            debug_mode=False,
        )
        if tenant_id:
            session_tenants[str(session.get("session_id") or "")] = tenant_id
        return session

    @app.post("/v1/sessions/{session_id}/reset")
    def reset_session(
        session_id: str,
        request: SessionResetRequest,
        _tenant_id: str | None = Depends(_authorized_session_context),
    ) -> Dict[str, Any]:
        try:
            payload = dict(
                backend.reset_session(
                    session_id,
                    task_id=request.task_id,
                    scenario_id=request.scenario_id,
                    start_state_id=request.start_state_id,
                )
            )
        except FileNotFoundError as exc:
            detail = f"session not found: {session_id}"
            raise _http_error(
                route="reset_session",
                status_code=404,
                detail=detail,
                exc=exc,
                session_id=session_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="reset_session",
                status_code=400,
                detail=str(exc),
                exc=exc,
                session_id=session_id,
            ) from exc
        log_event(
            logger,
            logging.INFO,
            "runtime_service.session_reset",
            session_id=session_id,
            task_id=request.task_id,
            scenario_id=request.scenario_id,
            start_state_id=request.start_state_id,
        )
        return payload

    @app.post("/v1/sessions/{session_id}/step")
    def step_session(
        session_id: str,
        request: SessionStepRequest,
        _tenant_id: str | None = Depends(_authorized_session_context),
    ) -> Dict[str, Any]:
        try:
            payload = dict(backend.step_session(session_id, action=request.action))
        except FileNotFoundError as exc:
            detail = f"session not found: {session_id}"
            raise _http_error(
                route="step_session",
                status_code=404,
                detail=detail,
                exc=exc,
                session_id=session_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="step_session",
                status_code=400,
                detail=str(exc),
                exc=exc,
                session_id=session_id,
            ) from exc
        log_event(
            logger,
            logging.INFO,
            "runtime_service.session_stepped",
            session_id=session_id,
            action_type=type(request.action).__name__,
        )
        return payload

    @app.get("/v1/sessions/{session_id}/state")
    def session_state(
        session_id: str,
        _tenant_id: str | None = Depends(_authorized_session_context),
    ) -> Dict[str, Any]:
        try:
            return dict(backend.session_state(session_id))
        except FileNotFoundError as exc:
            detail = f"session not found: {session_id}"
            raise _http_error(
                route="session_state",
                status_code=404,
                detail=detail,
                exc=exc,
                session_id=session_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="session_state",
                status_code=400,
                detail=str(exc),
                exc=exc,
                session_id=session_id,
            ) from exc

    @app.get("/v1/sessions/{session_id}/render")
    @app.get("/v1/sessions/{session_id}/render/{camera_id}")
    def render_session(
        session_id: str,
        camera_id: str = "head_rgb",
        _tenant_id: str | None = Depends(_authorized_session_context),
    ) -> Response:
        try:
            payload = backend.render_bytes(session_id, camera_id)
        except FileNotFoundError as exc:
            detail = f"session not found: {session_id}"
            raise _http_error(
                route="render_session",
                status_code=404,
                detail=detail,
                exc=exc,
                session_id=session_id,
                camera_id=camera_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="render_session",
                status_code=400,
                detail=str(exc),
                exc=exc,
                session_id=session_id,
                camera_id=camera_id,
            ) from exc
        return Response(content=payload, media_type="image/png")

    @app.post("/v2/sessions/{session_id}/control")
    def control_session(
        session_id: str,
        request: SessionControlRequest,
        _tenant_id: str | None = Depends(_authorized_session_context),
    ) -> Dict[str, Any]:
        try:
            payload = dict(backend.control_session(session_id, control=request.model_dump()))
        except FileNotFoundError as exc:
            detail = f"session not found: {session_id}"
            raise _http_error(
                route="control_session",
                status_code=404,
                detail=detail,
                exc=exc,
                session_id=session_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="control_session",
                status_code=400,
                detail=str(exc),
                exc=exc,
                session_id=session_id,
            ) from exc
        log_event(
            logger,
            logging.INFO,
            "runtime_service.session_controlled",
            session_id=session_id,
            seq=request.seq,
        )
        return payload

    @app.get("/v2/sessions/{session_id}/media")
    @app.get("/v2/sessions/{session_id}/media/{camera_id}")
    def media_session(
        session_id: str,
        camera_id: str = "head_rgb",
        chunk_id: str | None = None,
        _tenant_id: str | None = Depends(_authorized_session_context),
    ) -> Response:
        try:
            payload = dict(backend.media_response(session_id, camera_id=camera_id, chunk_id=chunk_id))
        except FileNotFoundError as exc:
            detail = f"session not found: {session_id}"
            raise _http_error(
                route="media_session",
                status_code=404,
                detail=detail,
                exc=exc,
                session_id=session_id,
                camera_id=camera_id,
                chunk_id=chunk_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="media_session",
                status_code=400,
                detail=str(exc),
                exc=exc,
                session_id=session_id,
                camera_id=camera_id,
                chunk_id=chunk_id,
            ) from exc
        response = Response(
            content=payload.get("content") or b"",
            media_type=str(payload.get("media_type") or "application/octet-stream"),
        )
        for header_name, header_value in dict(payload.get("headers") or {}).items():
            response.headers[str(header_name)] = str(header_value)
        return response

    @app.get("/v2/sessions/{session_id}/rollout")
    def rollout_session(
        session_id: str,
        _tenant_id: str | None = Depends(_authorized_session_context),
    ) -> Dict[str, Any]:
        try:
            state = dict(backend.session_state(session_id))
        except FileNotFoundError as exc:
            detail = f"session not found: {session_id}"
            raise _http_error(
                route="rollout_session",
                status_code=404,
                detail=detail,
                exc=exc,
                session_id=session_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="rollout_session",
                status_code=400,
                detail=str(exc),
                exc=exc,
                session_id=session_id,
            ) from exc
        rollout = state.get("rollout")
        return dict(rollout) if isinstance(rollout, dict) else {}

    @app.post("/v1/sessions/{session_id}/explorer/render")
    @app.post("/v1/sessions/{session_id}/explorer-render")
    def explorer_render(
        session_id: str,
        request: ExplorerRenderRequest,
        _tenant_id: str | None = Depends(_authorized_session_context),
    ) -> Dict[str, Any]:
        try:
            payload = dict(
                backend.explorer_render(
                    session_id,
                    camera_id=request.camera_id,
                    pose=request.pose.model_dump(),
                    viewport_width=request.viewport_width,
                    viewport_height=request.viewport_height,
                    refine_mode=request.refine_mode,
                )
            )
        except FileNotFoundError as exc:
            detail = f"session not found: {session_id}"
            raise _http_error(
                route="explorer_render",
                status_code=404,
                detail=detail,
                exc=exc,
                session_id=session_id,
                camera_id=request.camera_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="explorer_render",
                status_code=400,
                detail=str(exc),
                exc=exc,
                session_id=session_id,
                camera_id=request.camera_id,
            ) from exc
        log_event(
            logger,
            logging.INFO,
            "runtime_service.explorer_rendered",
            session_id=session_id,
            camera_id=request.camera_id,
            refine_mode=request.refine_mode,
        )
        return payload

    @app.get("/v1/sessions/{session_id}/explorer/frame/{camera_id}")
    @app.get("/v1/sessions/{session_id}/explorer-frame")
    def explorer_frame(
        session_id: str,
        camera_id: str = "head_rgb",
        _tenant_id: str | None = Depends(_authorized_session_context),
    ) -> Response:
        try:
            payload = backend.explorer_frame_bytes(session_id, camera_id)
        except FileNotFoundError as exc:
            detail = f"session not found: {session_id}"
            raise _http_error(
                route="explorer_frame",
                status_code=404,
                detail=detail,
                exc=exc,
                session_id=session_id,
                camera_id=camera_id,
            ) from exc
        except Exception as exc:
            raise _http_error(
                route="explorer_frame",
                status_code=400,
                detail=str(exc),
                exc=exc,
                session_id=session_id,
                camera_id=camera_id,
            ) from exc
        return Response(content=payload, media_type="image/png")

    @app.websocket("/v1/sessions/{session_id}/stream")
    async def stream_session(session_id: str, websocket: WebSocket) -> None:
        """Stream session state + media events to connected clients.

        Each message is a JSON object with a `type` field:
          { "type": "state",       "payload": <session_state_dict> }
          { "type": "media_event", "payload": { "event": "chunk_ready", ... } }

        Media events are drained from the per-session queue that background
        chunk-generation threads push to. This allows sub-50ms notification
        latency for chunk transitions instead of the 250ms state-poll period.
        """
        if auth_enabled:
            tenant_id = _authenticate_header(websocket.headers.get("Authorization"))
            if tenant_id is None:
                await websocket.close(code=4401)
                return
            try:
                _authorize_owner(_session_owner(session_id), tenant_id)
            except FileNotFoundError:
                await websocket.close(code=4404)
                return
            except HTTPException:
                await websocket.close(code=4403)
                return
        await websocket.accept()
        log_event(
            logger,
            logging.INFO,
            "runtime_service.websocket_connected",
            session_id=session_id,
        )
        try:
            for _ in range(10_000):
                # 1. Emit current session state
                state = dict(backend.session_state(session_id))
                await websocket.send_json({"type": "state", "payload": state})

                # 2. Drain and emit any pending media events (pushed by chunk worker)
                pending_events = list(backend.drain_media_events(session_id))
                for event in pending_events:
                    await websocket.send_json({"type": "media_event", "payload": dict(event)})

                await asyncio.sleep(0.25)
        except FileNotFoundError:
            log_event(
                logger,
                logging.WARNING,
                "runtime_service.websocket_failed",
                session_id=session_id,
                reason="session_not_found",
            )
            await websocket.send_json({"error": f"session not found: {session_id}"})
        except WebSocketDisconnect:
            log_event(
                logger,
                logging.INFO,
                "runtime_service.websocket_disconnected",
                session_id=session_id,
            )
            pass
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app
