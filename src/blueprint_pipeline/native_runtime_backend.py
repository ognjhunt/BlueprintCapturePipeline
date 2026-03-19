"""Native site-world runtime backend behind the shared runtime-service contract."""

from __future__ import annotations

import importlib.util
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4

from PIL import Image, ImageDraw

from .model_access_env import normalize_model_access_env
from blueprint_contracts.runtime_service_contract import RuntimeMetadata
from blueprint_contracts.site_world_contract import merge_site_world_definition


normalize_model_access_env()


def _utc_now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def _json_read(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _env_truthy(name: str) -> bool:
    return str(os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _module_available(name: str) -> bool:
    return bool(importlib.util.find_spec(name))


def _optional_existing_path(raw_value: Any) -> Optional[Path]:
    text = str(raw_value or "").strip()
    if not text or text.startswith(("gs://", "http://", "https://")):
        return None
    path = Path(text).expanduser().resolve()
    return path if path.exists() else None


def _runtime_readiness() -> Dict[str, Any]:
    packages = {
        "torch": _module_available("torch"),
        "diffusers": _module_available("diffusers"),
        "cosmos_predict2_5": _module_available("cosmos_predict2_5"),
    }
    model_dir = _optional_existing_path(os.getenv("NATIVE_WORLD_MODEL_PATH"))
    checkpoint_path = _optional_existing_path(os.getenv("NATIVE_WORLD_MODEL_CHECKPOINT_PATH"))
    package_ready = packages["torch"] and (packages["diffusers"] or packages["cosmos_predict2_5"])
    model_ready = bool(model_dir) or _env_truthy("NATIVE_WORLD_MODEL_READY")
    checkpoint_ready = bool(checkpoint_path) or _env_truthy("NATIVE_WORLD_MODEL_CHECKPOINT_READY")
    notes: list[str] = []
    if not package_ready:
        notes.append("missing_native_runtime_packages")
    if not model_ready:
        notes.append("native_model_not_provisioned")
    if not checkpoint_ready:
        notes.append("native_checkpoint_not_provisioned")
    return {
        "ready": package_ready and model_ready and checkpoint_ready,
        "package_ready": package_ready,
        "model_ready": model_ready,
        "checkpoint_ready": checkpoint_ready,
        "packages": packages,
        "model_dir": str(model_dir) if model_dir else "",
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else "",
        "notes": notes,
    }


def _runtime_blockers(site_world: Mapping[str, Any], health: Mapping[str, Any]) -> list[str]:
    blockers = [str(item).strip() for item in list(health.get("blockers") or []) if str(item).strip()]
    runtime_eligibility = (
        dict(site_world.get("runtime_eligibility") or {})
        if isinstance(site_world.get("runtime_eligibility"), Mapping)
        else {}
    )
    blockers.extend(
        str(item).strip()
        for item in list(runtime_eligibility.get("blockers") or [])
        if str(item).strip()
    )
    return list(dict.fromkeys(blockers))


@dataclass(frozen=True)
class NativeRuntimeConfig:
    root_dir: Path
    base_url: str
    ws_base_url: str


class NativeWorldModelRuntimeStore:
    """Disk-backed native runtime service with honest readiness reporting."""

    def __init__(self, config: NativeRuntimeConfig) -> None:
        self.root_dir = config.root_dir.resolve()
        self.base_url = config.base_url.rstrip("/")
        self.ws_base_url = config.ws_base_url.rstrip("/")
        self.site_worlds_dir = self.root_dir / "site_worlds"
        self.sessions_dir = self.root_dir / "sessions"
        self.site_worlds_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _site_world_dir(self, site_world_id: str) -> Path:
        return self.site_worlds_dir / site_world_id

    def _session_dir(self, session_id: str) -> Path:
        return self.sessions_dir / session_id

    def _registration_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_registration.json"

    def _health_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_health.json"

    def _spec_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_spec.json"

    def _session_state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session_state.json"

    def runtime_info(self, *, service_version: str) -> Dict[str, Any]:
        readiness = _runtime_readiness()
        checkpoint_path = str(readiness.get("checkpoint_path") or "")
        model_dir = str(readiness.get("model_dir") or "")
        production_grade = bool(readiness["ready"] and _env_truthy("NATIVE_WORLD_MODEL_PRODUCTION_GRADE"))
        return RuntimeMetadata(
            runtime_kind="native_world_model",
            production_grade=production_grade,
            service="native-site-world-runtime",
            version=service_version,
            runtime_base_url=self.base_url,
            websocket_base_url=self.ws_base_url,
            engine_identity={
                "engine": "native_site_world_runtime",
                "mode": "contract_complete_native_backend",
                "packages": dict(readiness.get("packages") or {}),
            },
            model_identity={
                "model_family": "cosmos_swm_native",
                "model_id": str(os.getenv("COSMOS_MODEL_ID") or os.getenv("NATIVE_WORLD_MODEL_ID") or "unconfigured"),
                "model_dir": model_dir,
                "model_ready": bool(readiness["model_ready"]),
            },
            checkpoint_identity={
                "checkpoint_id": Path(checkpoint_path).name if checkpoint_path else "unconfigured",
                "checkpoint_path": checkpoint_path,
                "checkpoint_ready": bool(readiness["checkpoint_ready"]),
            },
            state_guarantees={
                "authoritative_state": True,
                "restart_safe": True,
                "deterministic_replay": True,
                "render_source": "native_contract_placeholder",
            },
            capabilities={
                "site_world_package_registration": True,
                "site_world_registration": True,
                "session_reset": True,
                "session_step": True,
                "session_render": True,
                "session_state": True,
                "session_stream": True,
                "protected_region_locking": True,
                "runtime_layer_compositing": False,
                "debug_render_outputs": True,
            },
            readiness=readiness,
        ).to_dict()

    def register_site_world_package(
        self,
        *,
        spec: Mapping[str, Any],
        registration: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> Dict[str, Any]:
        site_world_id = str(registration.get("site_world_id") or "").strip()
        if not site_world_id:
            raise RuntimeError("site_world_registration requires site_world_id")
        site_world_dir = self._site_world_dir(site_world_id)
        site_world_dir.mkdir(parents=True, exist_ok=True)
        _json_write(self._registration_path(site_world_id), registration)
        _json_write(self._spec_path(site_world_id), spec)
        _json_write(self._health_path(site_world_id), health)
        return self.load_site_world(site_world_id)

    def load_site_world(self, site_world_id: str) -> Dict[str, Any]:
        registration_path = self._registration_path(site_world_id)
        spec_path = self._spec_path(site_world_id)
        if not registration_path.is_file():
            raise FileNotFoundError(site_world_id)
        registration = _json_read(registration_path)
        spec = _json_read(spec_path) if spec_path.is_file() else {}
        merged = merge_site_world_definition(registration=registration, spec=spec)
        merged["runtime_service_url"] = self.base_url
        return merged

    def load_site_world_health(self, site_world_id: str) -> Dict[str, Any]:
        health_path = self._health_path(site_world_id)
        if not health_path.is_file():
            raise FileNotFoundError(site_world_id)
        return _json_read(health_path)

    def _load_session_state(self, session_id: str) -> Dict[str, Any]:
        state_path = self._session_state_path(session_id)
        if not state_path.is_file():
            raise FileNotFoundError(session_id)
        return _json_read(state_path)

    def _store_session_state(self, session_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        _json_write(self._session_state_path(session_id), payload)
        return dict(payload)

    def create_session(self, site_world_id: str, **kwargs: Any) -> Dict[str, Any]:
        site_world = self.load_site_world(site_world_id)
        health = self.load_site_world_health(site_world_id)
        launchable = bool(health.get("launchable", False))
        if not launchable and not bool(kwargs.get("unsafe_allow_blocked_site_world")):
            blockers = ",".join(_runtime_blockers(site_world, health)) or "site_world_not_launchable"
            raise RuntimeError(f"site world is blocked: {blockers}")

        runtime_eligibility = (
            dict(site_world.get("runtime_eligibility") or {})
            if isinstance(site_world.get("runtime_eligibility"), Mapping)
            else {}
        )
        requested_backend = str(kwargs.get("requested_backend") or "").strip()
        selected_backend = requested_backend or str(runtime_eligibility.get("default_backend") or "native_world_model")
        session_id = str(kwargs.get("session_id") or uuid4())
        state = {
            "session_id": session_id,
            "site_world_id": site_world_id,
            "scene_id": site_world.get("scene_id"),
            "capture_id": site_world.get("capture_id"),
            "status": "ready",
            "runtime_kind": "native_world_model",
            "runtime_base_url": self.base_url,
            "websocket_base_url": self.ws_base_url,
            "runtime_backend_requested": requested_backend or None,
            "runtime_backend_selected": selected_backend,
            "robot_profile_id": kwargs.get("robot_profile_id"),
            "task_id": kwargs.get("task_id"),
            "scenario_id": kwargs.get("scenario_id"),
            "start_state_id": kwargs.get("start_state_id"),
            "notes": kwargs.get("notes") or "",
            "prompt": kwargs.get("prompt"),
            "trajectory": dict(kwargs.get("trajectory") or {}),
            "canonical_package_uri": kwargs.get("canonical_package_uri") or site_world.get("canonical_package_uri"),
            "canonical_package_version": kwargs.get("canonical_package_version") or site_world.get("canonical_package_version"),
            "presentation_model": kwargs.get("presentation_model"),
            "debug_mode": bool(kwargs.get("debug_mode")),
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "step_count": 0,
            "last_action": [],
            "pose": {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0},
        }
        return self._store_session_state(session_id, state)

    def reset_session(self, session_id: str, **kwargs: Any) -> Dict[str, Any]:
        state = self._load_session_state(session_id)
        if kwargs.get("task_id"):
            state["task_id"] = kwargs["task_id"]
        if kwargs.get("scenario_id"):
            state["scenario_id"] = kwargs["scenario_id"]
        if kwargs.get("start_state_id"):
            state["start_state_id"] = kwargs["start_state_id"]
        state["status"] = "ready"
        state["step_count"] = 0
        state["last_action"] = []
        state["pose"] = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0}
        state["updated_at"] = _utc_now_iso()
        return self._store_session_state(session_id, state)

    def step_session(self, session_id: str, *, action: list[float]) -> Dict[str, Any]:
        state = self._load_session_state(session_id)
        pose = dict(state.get("pose") or {})
        if len(action) >= 1:
            pose["x"] = float(pose.get("x", 0.0)) + float(action[0])
        if len(action) >= 2:
            pose["y"] = float(pose.get("y", 0.0)) + float(action[1])
        if len(action) >= 3:
            pose["yaw"] = float(pose.get("yaw", 0.0)) + float(action[2])
        state["pose"] = pose
        state["status"] = "running"
        state["step_count"] = int(state.get("step_count") or 0) + 1
        state["last_action"] = list(action)
        state["updated_at"] = _utc_now_iso()
        return self._store_session_state(session_id, state)

    def session_state(self, session_id: str) -> Dict[str, Any]:
        return self._load_session_state(session_id)

    def _render_png(
        self,
        session_id: str,
        camera_id: str,
        *,
        pose: Mapping[str, Any] | None = None,
        refine_mode: str | None = None,
        size: tuple[int, int] = (960, 540),
    ) -> bytes:
        state = self._load_session_state(session_id)
        width, height = size
        image = Image.new("RGB", size, color=(238, 245, 247))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(24, 24), (width - 24, height - 24)], outline=(15, 118, 110), width=3)
        lines = [
            "Blueprint Native Runtime",
            f"session_id={session_id}",
            f"site_world_id={state.get('site_world_id')}",
            f"camera_id={camera_id}",
            f"backend={state.get('runtime_backend_selected')}",
            f"task_id={state.get('task_id')}",
            f"scenario_id={state.get('scenario_id')}",
            f"step_count={state.get('step_count')}",
        ]
        render_pose = dict(pose or state.get("pose") or {})
        lines.append(
            "pose="
            f"({float(render_pose.get('x', 0.0)):.2f},"
            f"{float(render_pose.get('y', 0.0)):.2f},"
            f"{float(render_pose.get('z', 0.0)):.2f},"
            f" yaw={float(render_pose.get('yaw', 0.0)):.2f},"
            f" pitch={float(render_pose.get('pitch', 0.0)):.2f})"
        )
        if refine_mode:
            lines.append(f"refine_mode={refine_mode}")
        lines.append("render_source=native_contract_placeholder")
        y = 48
        for line in lines:
            draw.text((48, y), line, fill=(22, 45, 52))
            y += 32

        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def render_bytes(self, session_id: str, camera_id: str) -> bytes:
        return self._render_png(session_id, camera_id)

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
        state = self._load_session_state(session_id)
        state["pose"] = dict(pose)
        state["updated_at"] = _utc_now_iso()
        self._store_session_state(session_id, state)
        frame_dir = self._session_dir(session_id) / "explorer_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"{camera_id}.png"
        frame_path.write_bytes(
            self._render_png(
                session_id,
                camera_id,
                pose=pose,
                refine_mode=refine_mode,
                size=(
                    max(320, int(viewport_width or 960)),
                    max(240, int(viewport_height or 540)),
                ),
            )
        )
        return {
            "status": "completed",
            "session_id": session_id,
            "camera_id": camera_id,
            "pose": dict(pose),
            "refine_mode": refine_mode,
            "frame_path": str(frame_path.resolve()),
            "runtime_kind": "native_world_model",
        }

    def explorer_frame_bytes(self, session_id: str, camera_id: str) -> bytes:
        frame_path = self._session_dir(session_id) / "explorer_frames" / f"{camera_id}.png"
        if frame_path.is_file():
            return frame_path.read_bytes()
        return self.render_bytes(session_id, camera_id)


def native_runtime_config_from_env() -> NativeRuntimeConfig:
    host = os.getenv("SITE_WORLD_RUNTIME_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("SITE_WORLD_RUNTIME_SERVICE_PORT", "8791"))
    base_url = os.getenv("SITE_WORLD_RUNTIME_PUBLIC_BASE_URL", f"http://{host}:{port}")
    ws_base_url = os.getenv(
        "SITE_WORLD_RUNTIME_PUBLIC_WS_BASE_URL",
        base_url.replace("http://", "ws://").replace("https://", "wss://"),
    )
    root_dir = Path(os.getenv("SITE_WORLD_NATIVE_RUNTIME_ROOT", "./data/native-site-world-runtime"))
    return NativeRuntimeConfig(root_dir=root_dir, base_url=base_url, ws_base_url=ws_base_url)
