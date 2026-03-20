"""Shared FastAPI app factory for native site-world runtime backends."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Protocol

from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from blueprint_contracts.site_world_contract import normalize_trajectory_payload


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
    debug_mode: bool = False
    unsafe_allow_blocked_site_world: bool = False


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


def create_runtime_app(*, backend: RuntimeBackend, title: str) -> FastAPI:
    app = FastAPI(title=title, version="1.0.0")

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
    def runtime_info() -> Dict[str, Any]:
        return backend.runtime_info(service_version=app.version)

    @app.post("/v1/site-worlds")
    def register_site_world(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if not all(isinstance(payload.get(key), dict) for key in ("spec", "registration", "health")):
                raise ValueError("site-world registration requires spec + registration + health payloads")
            registration = dict(
                backend.register_site_world_package(
                    spec=dict(payload["spec"]),
                    registration=dict(payload["registration"]),
                    health=dict(payload["health"]),
                )
            )
            health = dict(backend.load_site_world_health(str(registration.get("site_world_id") or "")))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            **registration,
            "health": health,
        }

    @app.get("/v1/site-worlds/{site_world_id}")
    def get_site_world(site_world_id: str) -> Dict[str, Any]:
        try:
            return dict(backend.load_site_world(site_world_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"site world not found: {site_world_id}") from exc

    @app.get("/v1/site-worlds/{site_world_id}/health")
    def get_site_world_health(site_world_id: str) -> Dict[str, Any]:
        try:
            return dict(backend.load_site_world_health(site_world_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"site world not found: {site_world_id}") from exc

    @app.post("/v1/site-worlds/{site_world_id}/sessions")
    def create_session(site_world_id: str, request: SessionCreateRequest) -> Dict[str, Any]:
        try:
            return dict(
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
                    debug_mode=request.debug_mode,
                    unsafe_allow_blocked_site_world=request.unsafe_allow_blocked_site_world,
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"site world not found: {site_world_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/sessions/{session_id}/reset")
    def reset_session(session_id: str, request: SessionResetRequest) -> Dict[str, Any]:
        try:
            return dict(
                backend.reset_session(
                    session_id,
                    task_id=request.task_id,
                    scenario_id=request.scenario_id,
                    start_state_id=request.start_state_id,
                )
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/sessions/{session_id}/step")
    def step_session(session_id: str, request: SessionStepRequest) -> Dict[str, Any]:
        try:
            return dict(backend.step_session(session_id, action=request.action))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/sessions/{session_id}/state")
    def session_state(session_id: str) -> Dict[str, Any]:
        try:
            return dict(backend.session_state(session_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/sessions/{session_id}/render")
    @app.get("/v1/sessions/{session_id}/render/{camera_id}")
    def render_session(session_id: str, camera_id: str = "head_rgb") -> Response:
        try:
            payload = backend.render_bytes(session_id, camera_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return Response(content=payload, media_type="image/png")

    @app.post("/v2/sessions/{session_id}/control")
    def control_session(session_id: str, request: SessionControlRequest) -> Dict[str, Any]:
        try:
            return dict(backend.control_session(session_id, control=request.model_dump()))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v2/sessions/{session_id}/media")
    @app.get("/v2/sessions/{session_id}/media/{camera_id}")
    def media_session(session_id: str, camera_id: str = "head_rgb", chunk_id: str | None = None) -> Response:
        try:
            payload = dict(backend.media_response(session_id, camera_id=camera_id, chunk_id=chunk_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        response = Response(
            content=payload.get("content") or b"",
            media_type=str(payload.get("media_type") or "application/octet-stream"),
        )
        for header_name, header_value in dict(payload.get("headers") or {}).items():
            response.headers[str(header_name)] = str(header_value)
        return response

    @app.get("/v2/sessions/{session_id}/rollout")
    def rollout_session(session_id: str) -> Dict[str, Any]:
        try:
            state = dict(backend.session_state(session_id))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        rollout = state.get("rollout")
        return dict(rollout) if isinstance(rollout, dict) else {}

    @app.post("/v1/sessions/{session_id}/explorer/render")
    @app.post("/v1/sessions/{session_id}/explorer-render")
    def explorer_render(session_id: str, request: ExplorerRenderRequest) -> Dict[str, Any]:
        try:
            return dict(
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
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/v1/sessions/{session_id}/explorer/frame/{camera_id}")
    @app.get("/v1/sessions/{session_id}/explorer-frame")
    def explorer_frame(session_id: str, camera_id: str = "head_rgb") -> Response:
        try:
            payload = backend.explorer_frame_bytes(session_id, camera_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=f"session not found: {session_id}") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
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
        await websocket.accept()
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
            await websocket.send_json({"error": f"session not found: {session_id}"})
        except WebSocketDisconnect:
            pass
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                pass

    return app
