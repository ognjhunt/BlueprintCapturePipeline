"""Native site-world runtime backend behind the shared runtime-service contract."""

from __future__ import annotations

import importlib.util
import io
import json
import math
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import uuid4

from PIL import Image, ImageDraw

from .camera_geometry_validation import (
    validate_camera_calibration,
    validate_se3_matrix,
)
from .model_access_env import normalize_model_access_env
from .security_controls import contained_path, strict_identifier
from blueprint_contracts.runtime_service_contract import RuntimeMetadata
from blueprint_contracts.site_world_contract import merge_site_world_definition


normalize_model_access_env()


def _utc_now_iso() -> str:
    import datetime as _dt

    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def _json_write(path: Path, payload: Mapping[str, Any]) -> None:
    import os as _os
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use a unique temp filename per call so concurrent threads each get their
    # own temp file and don't race to rename the same ".tmp" path.
    tmp = path.parent / f"{path.stem}.{_os.getpid()}.{id(payload)}.tmp"
    try:
        tmp.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)  # clean up if replace failed


def _json_read(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_bytes_if_available(path: Path) -> Optional[bytes]:
    try:
        return path.read_bytes()
    except OSError:
        return None


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
    cosmos_repo = _find_cosmos_repo()
    package_ready = packages["torch"] and (packages["diffusers"] or packages["cosmos_predict2_5"] or bool(cosmos_repo))
    model_ready = bool(model_dir) or _env_truthy("NATIVE_WORLD_MODEL_READY") or bool(cosmos_repo)
    checkpoint_ready = bool(checkpoint_path) or _env_truthy("NATIVE_WORLD_MODEL_CHECKPOINT_READY") or bool(cosmos_repo)
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
        "cosmos_repo": str(cosmos_repo[0]) if cosmos_repo else "",
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


# ---------------------------------------------------------------------------
# Cosmos repo detection
# ---------------------------------------------------------------------------

_COSMOS_REPO_CANDIDATE_PATHS: List[str] = [
    "/root/workspace/cosmos-predict2.5",
    str(Path.home() / "workspace" / "cosmos-predict2.5"),
    str(Path.home() / "cosmos-predict2.5"),
]

# Per-session locks for Cosmos inference (prevent duplicate runs)
_COSMOS_LOCKS: Dict[str, threading.RLock] = {}
_COSMOS_LOCKS_MUTEX = threading.Lock()


# ---------------------------------------------------------------------------
# Per-session media event queue — drained by the WS stream handler each loop
# iteration so the client gets chunk_ready / underrun / generation_started
# notifications within ~50ms instead of the 250ms state-poll period.
# ---------------------------------------------------------------------------

_SESSION_MEDIA_EVENTS: Dict[str, List[Dict[str, Any]]] = {}
_SESSION_EVENTS_LOCK = threading.Lock()


def _push_media_event(session_id: str, event: Dict[str, Any]) -> None:
    with _SESSION_EVENTS_LOCK:
        if session_id not in _SESSION_MEDIA_EVENTS:
            _SESSION_MEDIA_EVENTS[session_id] = []
        _SESSION_MEDIA_EVENTS[session_id].append(event)
        # Cap at 20 events to prevent unbounded growth
        _SESSION_MEDIA_EVENTS[session_id] = _SESSION_MEDIA_EVENTS[session_id][-20:]


def _drain_media_events(session_id: str) -> List[Dict[str, Any]]:
    with _SESSION_EVENTS_LOCK:
        events = list(_SESSION_MEDIA_EVENTS.get(session_id, []))
        _SESSION_MEDIA_EVENTS[session_id] = []
        return events

def _cosmos_session_lock(session_id: str) -> threading.RLock:
    with _COSMOS_LOCKS_MUTEX:
        if session_id not in _COSMOS_LOCKS:
            _COSMOS_LOCKS[session_id] = threading.RLock()
        return _COSMOS_LOCKS[session_id]


def _find_cosmos_repo() -> Optional[Tuple[Path, Path]]:
    """Return (repo_root, python_bin) or None if not found."""
    explicit = os.getenv("COSMOS_OFFICIAL_REPO_ROOT", "").strip()
    candidates = [explicit] + _COSMOS_REPO_CANDIDATE_PATHS if explicit else _COSMOS_REPO_CANDIDATE_PATHS
    for candidate in candidates:
        root = Path(candidate).expanduser()
        inf = root / "examples" / "inference.py"
        py = root / ".venv" / "bin" / "python"
        try:
            if not inf.is_file() or not py.is_file():
                continue
            if not os.access(inf, os.R_OK):
                continue
            if not os.access(py, os.X_OK):
                continue
            return root, py
        except OSError:
            continue
    return None


def _default_storage_root() -> Path:
    configured = str(os.getenv("GCS_ROOT") or "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parents[2] / ".local_runs").resolve()


def _parse_gs_uri(raw_value: Any) -> Optional[Tuple[str, str]]:
    text = str(raw_value or "").strip()
    if not text.startswith("gs://"):
        return None
    remainder = text[5:]
    bucket, _, key = remainder.partition("/")
    if not bucket:
        return None
    return bucket, key


def _bucket_from_site_world(site_world: Mapping[str, Any]) -> str:
    candidates = [
        site_world.get("canonical_package_uri"),
        site_world.get("canonicalPackageUri"),
        site_world.get("site_world_spec_uri"),
        site_world.get("siteWorldSpecUri"),
        site_world.get("site_world_registration_uri"),
        site_world.get("siteWorldRegistrationUri"),
    ]
    for candidate in candidates:
        parsed = _parse_gs_uri(candidate)
        if parsed:
            return parsed[0]
    return str(os.getenv("GCS_BUCKET") or "vast-local").strip() or "vast-local"


# ---------------------------------------------------------------------------
# SE3 pose helpers for interactive session stepping
# ---------------------------------------------------------------------------


def _roty(theta: float):
    """3×3 rotation matrix around Y axis (yaw / horizontal turn)."""
    import numpy as np
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _action_to_delta_T(action: Any):
    """
    Convert an action (dict or float-list) to a 4×4 SE3 delta in camera frame.

    Dict form:  {"type": "move_forward|move_backward|turn_left|turn_right", "magnitude": float}
    List form:  [dx_forward, dx_strafe, dyaw_radians]  (legacy float-array contract)
    """
    import numpy as np
    T = np.eye(4, dtype=np.float64)
    if isinstance(action, dict):
        atype = str(action.get("type") or "move_forward")
        mag = float(action.get("magnitude") or 0.5)
        if atype == "move_forward":
            T[2, 3] = -mag          # camera -Z is forward
        elif atype == "move_backward":
            T[2, 3] = mag
        elif atype == "turn_left":
            T[:3, :3] = _roty(math.radians(mag))
        elif atype == "turn_right":
            T[:3, :3] = _roty(-math.radians(mag))
        elif atype == "move_up":
            T[1, 3] = -mag
        elif atype == "move_down":
            T[1, 3] = mag
    elif isinstance(action, (list, tuple)):
        if len(action) >= 1:
            T[2, 3] = -float(action[0])          # forward
        if len(action) >= 2:
            T[0, 3] = float(action[1])            # strafe
        if len(action) >= 3:
            T[:3, :3] = _roty(float(action[2]))   # yaw radians
    return T


def _apply_action(T_world_camera, action: Any):
    """Apply SE3 action delta (in camera frame) to current world pose → new pose."""
    import numpy as np
    T = np.array(T_world_camera, dtype=np.float64)
    delta = _action_to_delta_T(action)
    return T @ delta


def _pose_from_site_index(site_index_path: Path):
    """Read the first record's T_world_camera from a site reference index JSONL."""
    try:
        with site_index_path.open() as f:
            rec = json.loads(f.readline())
        import numpy as np
        validation = validate_se3_matrix(
            rec.get("T_world_camera"), field="site_index_world_from_camera"
        )
        if not validation["valid"]:
            return None
        return np.array(validation["matrix"], dtype=np.float64)
    except Exception:
        return None


def _intrinsics_from_site_index(site_index_path: Path) -> Tuple[Dict[str, float], int, int]:
    try:
        with site_index_path.open("r", encoding="utf-8") as handle:
            rec = json.loads(handle.readline())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("site_index_camera_calibration_unreadable") from exc
    if not isinstance(rec, Mapping):
        raise RuntimeError("site_index_camera_calibration_record_invalid")
    validation = validate_camera_calibration(
        rec,
        require_extrinsics=True,
        require_frame_metadata=True,
        require_translation_units=True,
        require_reprojection_error=True,
    )
    if validation["status"] != "passed":
        raise RuntimeError(
            "site_index_camera_calibration_blocked:"
            + ",".join(str(item) for item in validation["blockers"])
        )
    intrinsics = dict(validation["intrinsics"])
    return intrinsics, int(intrinsics["height"]), int(intrinsics["width"])


def _resolve_site_index_path(
    site_id: str,
    scene_id: str,
    capture_id: str,
    storage_root: Path,
    bucket: str,
) -> Optional[Path]:
    """Return the site_reference_index.jsonl path if it exists."""
    safe_bucket = strict_identifier(bucket, field="storage_bucket")
    safe_site_id = strict_identifier(site_id, field="site_id")
    p = contained_path(
        storage_root,
        safe_bucket,
        "sites",
        safe_site_id,
        "reference_memory",
        "site_reference_index.jsonl",
        field="site reference index path",
    )
    if p.is_file():
        return p
    # Also try without bucket prefix (flat layout)
    p2 = contained_path(
        storage_root,
        "sites",
        safe_site_id,
        "reference_memory",
        "site_reference_index.jsonl",
        field="flat site reference index path",
    )
    return p2 if p2.is_file() else None


def _resolve_site_reference_manifest_path(
    site_id: str,
    storage_root: Path,
    bucket: str,
) -> Optional[Path]:
    safe_bucket = strict_identifier(bucket, field="storage_bucket")
    safe_site_id = strict_identifier(site_id, field="site_id")
    candidates = [
        contained_path(
            storage_root,
            safe_bucket,
            "sites",
            safe_site_id,
            "reference_memory",
            "site_reference_manifest.json",
            field="site reference manifest path",
        ),
        contained_path(
            storage_root,
            "sites",
            safe_site_id,
            "reference_memory",
            "site_reference_manifest.json",
            field="flat site reference manifest path",
        ),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _resolve_site_id(site_world: Mapping[str, Any], scene_id: str, capture_id: str, storage_root: Path, bucket: str) -> str:
    """Best-effort site_id resolution from site_world metadata or capture_descriptor."""
    site_id = str(site_world.get("site_id") or "").strip()
    if site_id:
        return strict_identifier(site_id, field="site_id")
    # Fall back to reading capture_descriptor
    safe_bucket = strict_identifier(bucket, field="storage_bucket")
    safe_scene_id = strict_identifier(scene_id, field="scene_id")
    safe_capture_id = strict_identifier(capture_id, field="capture_id")
    desc_path = contained_path(
        storage_root,
        safe_bucket,
        "scenes",
        safe_scene_id,
        "captures",
        safe_capture_id,
        "capture_descriptor.json",
        field="capture descriptor path",
    )
    try:
        desc = json.loads(desc_path.read_text())
        meta = desc.get("metadata") or desc
        identity = meta.get("site_identity") or {}
        resolved_site_id = str(identity.get("site_id") or "").strip()
        return strict_identifier(resolved_site_id, field="site_id") if resolved_site_id else ""
    except Exception:
        return ""


def _site_reference_runtime_adapter(
    *,
    site_id: str,
    manifest_path: Optional[Path],
    site_index_path: Optional[Path],
    storage_root: Path,
    bucket: str,
    runtime_readiness: Mapping[str, Any],
) -> Dict[str, Any]:
    blockers: list[str] = []
    manifest = _json_read(manifest_path) if manifest_path and manifest_path.is_file() else {}
    if not site_id:
        blockers.append("missing_site_id")
    if not manifest_path:
        blockers.append("missing_site_reference_manifest")
    if manifest and manifest.get("schema_version") != "site_reference_database.v1":
        blockers.append("site_reference_manifest_schema_invalid")
    if not site_index_path:
        blockers.append("missing_site_reference_index")
    query_reference_count = _site_reference_query_count(
        site_index_path=site_index_path,
        storage_root=storage_root,
        bucket=bucket,
    )
    geometry_state = _site_reference_geometry_state(site_index_path)
    if site_index_path and query_reference_count <= 0:
        blockers.append("site_reference_query_empty")
    blockers.extend(geometry_state["blockers"])
    backend_blockers = [str(item) for item in list(runtime_readiness.get("notes") or []) if str(item)]
    local_contract_ready = bool(site_id and manifest_path and site_index_path) and not any(
        blocker in blockers
        for blocker in (
            "missing_site_id",
            "missing_site_reference_manifest",
            "site_reference_manifest_schema_invalid",
            "missing_site_reference_index",
        )
    )
    retrieval_ready = local_contract_ready and query_reference_count > 0
    return {
        "schema_version": "site_reference_runtime_adapter.v1",
        "site_id": site_id or None,
        "generated_at": _utc_now_iso(),
        "site_reference_manifest_path": str(manifest_path) if manifest_path else None,
        "site_reference_index_path": str(site_index_path) if site_index_path else None,
        "manifest_schema_version": manifest.get("schema_version"),
        "total_reference_frames": manifest.get("total_reference_frames"),
        "capture_count": manifest.get("capture_count"),
        "chunk_count": manifest.get("chunk_count"),
        "local_contract_ready": local_contract_ready,
        "retrieval_ready": retrieval_ready,
        "query_reference_count": query_reference_count,
        "non_arkit_geometry_state": geometry_state["state"],
        "non_arkit_geometry_sources": geometry_state["sources"],
        "provider_native_geometry_ready": geometry_state["provider_native_geometry_ready"],
        "world_model_ready": bool(retrieval_ready and geometry_state["swm_world_model_ready"]),
        "selected_runtime_path": "cosmos_i2w" if runtime_readiness.get("ready") else "splat_only",
        "runtime_adapter_ready": retrieval_ready,
        "backend_ready": bool(runtime_readiness.get("ready")),
        "backend_blockers": backend_blockers,
        "blockers": blockers,
        "claim_policy": "local_adapter_only_no_live_provider_or_hosted_success_claim",
    }


def _site_reference_geometry_state(site_index_path: Optional[Path]) -> Dict[str, Any]:
    if site_index_path is None or not site_index_path.is_file():
        return {
            "state": "blocked",
            "sources": [],
            "provider_native_geometry_ready": False,
            "swm_world_model_ready": False,
            "blockers": [],
        }
    rows: list[Dict[str, Any]] = []
    try:
        with site_index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    rows.append(dict(payload))
    except Exception:
        return {
            "state": "blocked",
            "sources": [],
            "provider_native_geometry_ready": False,
            "swm_world_model_ready": False,
            "blockers": ["site_reference_index_unreadable"],
        }

    geometry_sources = [
        str(row.get("geometry_source") or "").strip() or "unknown"
        for row in rows
    ]
    non_arkit_sources = {
        source
        for source in geometry_sources
        if source not in {"arkit", "arcore"}
    }
    if not non_arkit_sources:
        return {
            "state": "not_applicable",
            "sources": [],
            "provider_native_geometry_ready": True,
            "swm_world_model_ready": True,
            "blockers": [],
        }
    if non_arkit_sources == {"video_to_world"}:
        return {
            "state": "ready",
            "sources": sorted(non_arkit_sources),
            "provider_native_geometry_ready": True,
            "swm_world_model_ready": True,
            "blockers": [],
        }
    return {
        "state": "blocked",
        "sources": sorted(non_arkit_sources),
        "provider_native_geometry_ready": False,
        "swm_world_model_ready": False,
        "blockers": ["provider_native_geometry_missing", "non_arkit_geometry_not_live_video_to_world"],
    }


def _site_reference_query_count(
    *,
    site_index_path: Optional[Path],
    storage_root: Path,
    bucket: str,
) -> int:
    if site_index_path is None:
        return 0
    try:
        from .synthesis.retrieval_query import query_site

        target = _pose_from_site_index(site_index_path)
        if target is None:
            return 0
        return len(
            query_site(
                site_index_path=site_index_path,
                target_T_world_camera=target,
                k=3,
                mode="spatial",
                storage_root=storage_root,
                bucket=bucket,
            )
        )
    except Exception:
        return 0


def _pose_summary_from_matrix(T_world_camera: Any) -> Dict[str, float]:
    try:
        import numpy as np

        T = np.array(T_world_camera, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError("pose must be 4x4")
        yaw = math.atan2(float(T[0, 2]), float(T[0, 0]))
        return {
            "x": float(T[0, 3]),
            "y": float(T[1, 3]),
            "z": float(T[2, 3]),
            "yaw": yaw,
            "pitch": 0.0,
        }
    except Exception:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0}


def _utc_now_ms() -> int:
    return int(time.time() * 1000)


def _compact_reference_record(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "reference_id": str(rec.get("reference_id") or "").strip(),
        "capture_id": str(rec.get("capture_id") or "").strip(),
        "frame_id": str(rec.get("frame_id") or "").strip(),
        "frame_uri": str(rec.get("frame_uri") or "").strip(),
    }


# ---------------------------------------------------------------------------
# Runtime config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NativeRuntimeConfig:
    root_dir: Path
    base_url: str
    ws_base_url: str


# ---------------------------------------------------------------------------
# NativeWorldModelRuntimeStore
# ---------------------------------------------------------------------------


class NativeWorldModelRuntimeStore:
    """Disk-backed native runtime service with honest readiness reporting."""

    def __init__(self, config: NativeRuntimeConfig) -> None:
        self.root_dir = config.root_dir.resolve()
        self.base_url = config.base_url.rstrip("/")
        self.ws_base_url = config.ws_base_url.rstrip("/")
        self.site_worlds_dir = self.root_dir / "site_worlds"
        self.sessions_dir = self.root_dir / "sessions"
        self._generation_owner_lock = threading.Lock()
        self._generation_owner_session_id: Optional[str] = None
        self._generation_owner_acquired_at: Optional[str] = None
        self._prewarm_lock = threading.Lock()
        self._registration_lock = threading.RLock()
        self._prewarm_status: Dict[str, Any] = {"status": "idle"}
        self.site_worlds_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _site_world_dir(self, site_world_id: str) -> Path:
        safe_id = strict_identifier(site_world_id, field="site_world_id")
        return contained_path(self.site_worlds_dir, safe_id, field="site world path")

    def _session_dir(self, session_id: str) -> Path:
        safe_id = strict_identifier(session_id, field="session_id")
        return contained_path(self.sessions_dir, safe_id, field="session path")

    def _registration_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_registration.json"

    def _health_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_health.json"

    def _spec_path(self, site_world_id: str) -> Path:
        return self._site_world_dir(site_world_id) / "site_world_spec.json"

    def _session_state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "session_state.json"

    def _site_reference_runtime_artifact_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "site_reference_runtime_readiness.json"

    def _cosmos_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "cosmos"

    def _cosmos_frames_dir(self, session_id: str) -> Path:
        return self._cosmos_dir(session_id) / "frames"

    def _cosmos_status_path(self, session_id: str) -> Path:
        return self._cosmos_dir(session_id) / "status.json"

    def _video_chunks_dir(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "video_chunks"

    def _chunk_video_path(self, session_id: str, chunk_id: str) -> Path:
        safe_chunk_id = strict_identifier(chunk_id, field="chunk_id")
        return contained_path(
            self._video_chunks_dir(session_id),
            f"{safe_chunk_id}.mp4",
            field="video chunk path",
        )

    def _chunk_tail_path(self, session_id: str, chunk_id: str) -> Path:
        safe_chunk_id = strict_identifier(chunk_id, field="chunk_id")
        return contained_path(
            self._video_chunks_dir(session_id),
            f"{safe_chunk_id}_tail.png",
            field="video chunk tail path",
        )

    def _rollout_defaults(self) -> Dict[str, Any]:
        # COSMOS_CHUNK_SIZE and COSMOS_CHUNK_OVERLAP are the canonical env vars.
        # NATIVE_WORLD_MODEL_CHUNK_FRAMES is kept for backward compatibility.
        chunk_frames = max(
            8,
            int(
                os.getenv("COSMOS_CHUNK_SIZE")
                or os.getenv("NATIVE_WORLD_MODEL_CHUNK_FRAMES")
                or "57"
            ),
        )
        fps = max(1, int(os.getenv("NATIVE_WORLD_MODEL_CHUNK_FPS", "28")))
        chunk_duration_ms = max(400, int(round((chunk_frames / fps) * 1000)))
        output_profile = self._output_profile()
        uses_truthful_preview = self._uses_truthful_preview()
        refinement_enabled = self._cosmos_refinement_enabled()
        return {
            "mode": "chunked_video",
            "status": "idle",
            "output_profile": output_profile,
            "presentation_mode": "truthful_preview" if uses_truthful_preview else "generated_chunk",
            "interactive_path_kind": "retrieval_depth_splat_preview" if uses_truthful_preview else "cosmos_chunk_generation",
            "provenance_mode": "truthful_preview_with_async_refinement" if uses_truthful_preview else "cosmos_primary",
            "target_fps": fps,
            "chunk_frames": chunk_frames,
            "chunk_duration_ms": chunk_duration_ms,
            "target_ready_chunks": self._target_ready_chunks(),
            "refinement_enabled": refinement_enabled,
            "refinement_status": "idle" if refinement_enabled else "disabled",
            "refinement_inflight_chunk_id": None,
            "refined_chunk_ids": [],
            "control_intent": {
                "seq": 0,
                "tClientMs": None,
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "yawRate": 0.0,
                "pitchRate": 0.0,
                "durationMs": chunk_duration_ms,
                "updatedAt": _utc_now_iso(),
            },
            "trajectory_horizon": [],
            "grounding_reference_set": [],
            "lookahead_anchor": None,
            "active_chunk_id": None,
            "queued_chunk_ids": [],
            "buffered_chunk_ids": [],
            "chunks": [],
            "last_chunk_tail_path": None,
            "generation_lag_ms": None,
            "underrun": False,
            "current_media_type": None,
            "current_render_source": None,
            "chunk_count": 0,
            "generation_owner_session_id": None,
            "worker_backend": None,
            "first_chunk_generation_ms": None,
            "last_chunk_generation_ms": None,
            "last_chunk_ready_latency_ms": None,
            "underrun_count": 0,
            "world_state_version": 0,
            "world_state": {
                "state_id": None,
                "state_version": 0,
                "updated_at": _utc_now_iso(),
                "model_state_kind": "external_runtime_state",
                "memory_mode": "retrieval_grounded_preview" if uses_truthful_preview else "chunk_generation_only",
                "pose": None,
                "target_pose": None,
                "control_intent": {},
                "trajectory_horizon": [],
                "grounding_reference_set": [],
                "lookahead_anchor": None,
                "last_committed_chunk_id": None,
            },
        }

    def _ensure_rollout(self, state: Dict[str, Any]) -> Dict[str, Any]:
        rollout = dict(state.get("rollout") or {})
        defaults = self._rollout_defaults()
        merged = {**defaults, **rollout}
        merged["control_intent"] = {
            **dict(defaults.get("control_intent") or {}),
            **dict(rollout.get("control_intent") or {}),
        }
        merged["trajectory_horizon"] = list(rollout.get("trajectory_horizon") or [])
        merged["grounding_reference_set"] = list(rollout.get("grounding_reference_set") or [])
        merged["queued_chunk_ids"] = list(rollout.get("queued_chunk_ids") or [])
        merged["buffered_chunk_ids"] = list(rollout.get("buffered_chunk_ids") or [])
        merged["chunks"] = list(rollout.get("chunks") or [])
        merged["refined_chunk_ids"] = list(rollout.get("refined_chunk_ids") or [])
        merged["world_state"] = {
            **dict(defaults.get("world_state") or {}),
            **dict(rollout.get("world_state") or {}),
        }
        state["rollout"] = merged
        return merged

    def _output_profile(self) -> str:
        explicit = str(os.getenv("NATIVE_WORLD_MODEL_OUTPUT_PROFILE") or "").strip().lower()
        if explicit:
            return explicit
        return "swm_preview_refine"

    def _uses_truthful_preview(self) -> bool:
        explicit = str(os.getenv("NATIVE_WORLD_MODEL_ENABLE_TRUTHFUL_PREVIEW") or "").strip().lower()
        if explicit in {"1", "true", "yes", "on"}:
            return True
        if explicit in {"0", "false", "no", "off"}:
            return False
        return self._output_profile() in {"swm_preview_refine", "truthful_preview", "preview_refine"}

    def _cosmos_refinement_enabled(self) -> bool:
        explicit = str(os.getenv("NATIVE_WORLD_MODEL_ENABLE_COSMOS_REFINEMENT") or "").strip().lower()
        if explicit in {"0", "false", "no", "off"}:
            return False
        if explicit in {"1", "true", "yes", "on"}:
            return bool(_runtime_readiness().get("ready"))
        return self._uses_truthful_preview() and bool(_runtime_readiness().get("ready"))

    def _preview_generation_mode(self) -> str:
        if self._uses_truthful_preview():
            return "splat_only"
        return self._live_synthesis_mode()

    def _target_ready_chunks(self) -> int:
        default = "2" if self._uses_truthful_preview() else "1"
        return max(1, int(os.getenv("NATIVE_WORLD_MODEL_TARGET_READY_CHUNKS", default)))

    def _remaining_ready_chunks(self, rollout: Mapping[str, Any]) -> int:
        buffered = [str(item) for item in list(rollout.get("buffered_chunk_ids") or []) if str(item)]
        active_chunk_id = str(rollout.get("active_chunk_id") or "").strip()
        active_index = buffered.index(active_chunk_id) if active_chunk_id in buffered else -1
        return len(buffered) - max(active_index + 1, 0)

    def _should_queue_more_chunks(self, rollout: Mapping[str, Any]) -> bool:
        if len(list(rollout.get("queued_chunk_ids") or [])) > 0:
            return False
        return self._remaining_ready_chunks(rollout) < max(
            1,
            int(rollout.get("target_ready_chunks") or self._target_ready_chunks()),
        )

    def _cosmos_refinement_settings(self, rollout: Mapping[str, Any]) -> Dict[str, Any]:
        chunk_frames = max(8, int(rollout.get("chunk_frames") or 57))
        width = max(320, int(os.getenv("NATIVE_WORLD_MODEL_COSMOS_WIDTH", "960")))
        height = max(180, int(os.getenv("NATIVE_WORLD_MODEL_COSMOS_HEIGHT", "540")))
        num_steps = max(1, int(os.getenv("NATIVE_WORLD_MODEL_COSMOS_NUM_STEPS", "12")))
        guidance_scale = max(0.0, float(os.getenv("NATIVE_WORLD_MODEL_COSMOS_GUIDANCE", "4.0")))
        return {
            "num_frames": chunk_frames,
            "width": width,
            "height": height,
            "num_steps": num_steps,
            "guidance_scale": guidance_scale,
        }

    def _update_rollout_world_state(
        self,
        *,
        session_id: str,
        rollout: Dict[str, Any],
        pose: Mapping[str, Any],
        target_pose: Mapping[str, Any],
        trajectory_horizon: List[Dict[str, float]],
        grounding_reference_set: List[Dict[str, Any]],
        lookahead_anchor: Dict[str, Any] | None,
        last_committed_chunk_id: str | None,
    ) -> None:
        world_state_version = int(rollout.get("world_state_version") or 0) + 1
        rollout["world_state_version"] = world_state_version
        rollout["world_state"] = {
            "state_id": f"{session_id}-state-{world_state_version:06d}",
            "state_version": world_state_version,
            "updated_at": _utc_now_iso(),
            "model_state_kind": "external_runtime_state",
            "memory_mode": "retrieval_grounded_preview" if self._uses_truthful_preview() else "chunk_generation_only",
            "pose": dict(pose),
            "target_pose": dict(target_pose),
            "control_intent": dict(rollout.get("control_intent") or {}),
            "trajectory_horizon": list(trajectory_horizon),
            "grounding_reference_set": list(grounding_reference_set),
            "lookahead_anchor": dict(lookahead_anchor) if isinstance(lookahead_anchor, Mapping) else None,
            "last_committed_chunk_id": last_committed_chunk_id,
        }

    def _chunk_provenance(
        self,
        *,
        rollout: Mapping[str, Any],
        render_source: str,
        grounding_refs: List[Dict[str, Any]],
        lookahead_refs: List[Dict[str, Any]],
        previous_tail_path: str | None,
        refinement_status: str,
    ) -> Dict[str, Any]:
        return {
            "presentation_mode": str(rollout.get("presentation_mode") or ""),
            "interactive_path_kind": str(rollout.get("interactive_path_kind") or ""),
            "provenance_mode": str(rollout.get("provenance_mode") or ""),
            "render_source": render_source,
            "grounded": True,
            "retrieval_reference_count": len(grounding_refs),
            "lookahead_reference_count": len(lookahead_refs),
            "tail_conditioned": bool(previous_tail_path),
            "future_conditioned": bool(lookahead_refs),
            "refinement_status": refinement_status,
        }

    def _selected_runtime_identity(self, readiness: Mapping[str, Any]) -> Dict[str, Any]:
        """Truthful runtime identity for the actually selected render path.

        ``model_family``/``render_source`` must reflect what a session will
        really render (splat_only truthful preview vs cosmos_i2w vs
        unconfigured), never a hard-coded cosmos label.
        """

        ready = bool(readiness.get("ready"))
        primary_mode = self._preview_generation_mode()
        if primary_mode == "cosmos_i2w":
            if ready:
                identity = {
                    "selected_runtime_path": "cosmos_i2w",
                    "model_family": "cosmos_i2w_native",
                    "render_source": "cosmos_i2w",
                }
            else:
                identity = {
                    "selected_runtime_path": "unconfigured",
                    "model_family": "unconfigured",
                    "render_source": "unconfigured",
                }
        elif primary_mode == "splat_only":
            identity = {
                "selected_runtime_path": "splat_only",
                "model_family": "site_splat_truthful_preview",
                "render_source": "truthful_preview_splat",
            }
        else:
            identity = {
                "selected_runtime_path": primary_mode,
                "model_family": primary_mode,
                "render_source": primary_mode,
            }
        identity["async_cosmos_refinement_enabled"] = self._cosmos_refinement_enabled()
        return identity

    def runtime_info(self, *, service_version: str) -> Dict[str, Any]:
        readiness = _runtime_readiness()
        readiness["prewarm_status"] = dict(self._prewarm_status)
        checkpoint_path = str(readiness.get("checkpoint_path") or "")
        model_dir = str(readiness.get("model_dir") or "")
        cosmos_repo = str(readiness.get("cosmos_repo") or "")
        production_grade = bool(readiness["ready"] and _env_truthy("NATIVE_WORLD_MODEL_PRODUCTION_GRADE"))
        runtime_identity = self._selected_runtime_identity(readiness)
        readiness["selected_runtime_path"] = runtime_identity["selected_runtime_path"]
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
                "selected_runtime_path": runtime_identity["selected_runtime_path"],
                "packages": dict(readiness.get("packages") or {}),
                "cosmos_repo": cosmos_repo,
                "prewarm_status": dict(self._prewarm_status),
            },
            model_identity={
                "model_family": runtime_identity["model_family"],
                "selected_runtime_path": runtime_identity["selected_runtime_path"],
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
                "render_source": runtime_identity["render_source"],
                "async_cosmos_refinement_enabled": runtime_identity[
                    "async_cosmos_refinement_enabled"
                ],
            },
            capabilities={
                "site_world_package_registration": True,
                "site_world_registration": True,
                "session_reset": True,
                "session_step": True,
                "session_control": True,
                "session_render": True,
                "session_media": True,
                "session_state": True,
                "session_stream": True,
                "protected_region_locking": True,
                "runtime_layer_compositing": False,
                "debug_render_outputs": True,
                "truthful_preview_rollout": True,
                "async_cosmos_refinement": True,
            },
            readiness=readiness,
        ).to_dict()

    def _claim_generation_owner(self, session_id: str) -> tuple[bool, Optional[str]]:
        with self._generation_owner_lock:
            owner = self._generation_owner_session_id
            if owner and owner != session_id:
                return False, owner
            self._generation_owner_session_id = session_id
            if not self._generation_owner_acquired_at:
                self._generation_owner_acquired_at = _utc_now_iso()
            return True, session_id

    def _release_generation_owner(self, session_id: str) -> None:
        with self._generation_owner_lock:
            if self._generation_owner_session_id == session_id:
                self._generation_owner_session_id = None
                self._generation_owner_acquired_at = None

    def prewarm_runtime(self) -> Dict[str, Any]:
        with self._prewarm_lock:
            readiness = _runtime_readiness()
            if not bool(readiness.get("ready", False)):
                self._prewarm_status = {
                    "status": "skipped",
                    "reason": "runtime_not_ready",
                    "checked_at": _utc_now_iso(),
                }
                return dict(self._prewarm_status)
            started_at = _utc_now_iso()
            try:
                from .synthesis.cosmos_inference import prewarm_cosmos_model

                result = dict(prewarm_cosmos_model())
                self._prewarm_status = {
                    "status": "ready",
                    "started_at": started_at,
                    "completed_at": _utc_now_iso(),
                    **result,
                }
            except Exception as exc:
                self._prewarm_status = {
                    "status": "failed",
                    "started_at": started_at,
                    "completed_at": _utc_now_iso(),
                    "error": str(exc),
                }
            return dict(self._prewarm_status)

    def register_site_world_package(
        self,
        *,
        spec: Mapping[str, Any],
        registration: Mapping[str, Any],
        health: Mapping[str, Any],
    ) -> Dict[str, Any]:
        if not str(registration.get("site_world_id") or "").strip():
            raise RuntimeError("site_world_registration requires site_world_id")
        site_world_id = strict_identifier(registration.get("site_world_id"), field="site_world_id")
        with self._registration_lock:
            registration_path = self._registration_path(site_world_id)
            if registration_path.is_file():
                existing_tenant = str(
                    _json_read(registration_path).get("tenant_id") or ""
                ).strip() or None
                incoming_tenant = str(registration.get("tenant_id") or "").strip() or None
                if existing_tenant != incoming_tenant:
                    raise PermissionError("site world tenant ownership cannot be changed")
            site_world_dir = self._site_world_dir(site_world_id)
            site_world_dir.mkdir(parents=True, exist_ok=True)
            _json_write(registration_path, registration)
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

    def site_world_tenant_id(self, site_world_id: str) -> str | None:
        registration_path = self._registration_path(site_world_id)
        if not registration_path.is_file():
            raise FileNotFoundError(site_world_id)
        return str(_json_read(registration_path).get("tenant_id") or "").strip() or None

    def session_tenant_id(self, session_id: str) -> str | None:
        return str(self._load_session_state(session_id).get("tenant_id") or "").strip() or None

    def _load_session_state(self, session_id: str) -> Dict[str, Any]:
        state_path = self._session_state_path(session_id)
        if not state_path.is_file():
            raise FileNotFoundError(session_id)
        return _json_read(state_path)

    def _store_session_state(self, session_id: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        with _cosmos_session_lock(f"state:{session_id}"):
            _json_write(self._session_state_path(session_id), payload)
            return dict(payload)

    def _make_observation(
        self,
        session_id: str,
        step_count: int,
        camera_id: str = "head_rgb",
        *,
        render_source: str = "cosmos_zero_ft",
    ) -> Dict[str, Any]:
        """Build an observation dict with a valid worldSnapshot entry (camelCase for WebApp contract)."""
        snapshot_id = f"snap-{session_id[:12]}-{step_count:05d}"
        return {
            # WebApp hasRenderableSnapshot checks camelCase keys
            "primaryCameraId": camera_id,
            "primary_camera_id": camera_id,
            "worldSnapshot": {
                "snapshotId": snapshot_id,
                "snapshot_id": snapshot_id,
                "step": step_count,
                "render_source": render_source,
            },
            # Keep snake_case aliases for broader compatibility
            "world_snapshot": {
                "snapshot_id": snapshot_id,
                "snapshotId": snapshot_id,
                "step": step_count,
                "render_source": render_source,
            },
        }

    def _make_pending_observation(self, camera_id: str = "head_rgb") -> Dict[str, Any]:
        return {
            "primaryCameraId": camera_id,
            "primary_camera_id": camera_id,
            "renderPending": True,
            "render_pending": True,
        }

    def _session_target_pose(self, state: Mapping[str, Any], site_index_path: Optional[Path]) -> Any:
        import numpy as np

        raw_pose = state.get("camera_pose_matrix")
        if isinstance(raw_pose, list):
            try:
                arr = np.array(raw_pose, dtype=np.float64)
                if arr.shape == (4, 4):
                    return arr
            except Exception:
                pass
        if site_index_path is not None:
            initial = _pose_from_site_index(site_index_path)
            if initial is not None:
                return initial
        return np.eye(4, dtype=np.float64)

    def _fallback_pose_update(self, pose: Mapping[str, Any], action: Any) -> Dict[str, float]:
        next_pose = {
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "z": float(pose.get("z", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
            "pitch": float(pose.get("pitch", 0.0)),
        }
        if isinstance(action, dict):
            atype = str(action.get("type") or "move_forward").strip()
            mag = float(action.get("magnitude") or (15.0 if atype.startswith("turn_") else 0.5))
            if atype == "move_forward":
                next_pose["z"] -= mag
            elif atype == "move_backward":
                next_pose["z"] += mag
            elif atype == "turn_left":
                next_pose["yaw"] += math.radians(mag)
            elif atype == "turn_right":
                next_pose["yaw"] -= math.radians(mag)
            elif atype == "move_up":
                next_pose["y"] -= mag
            elif atype == "move_down":
                next_pose["y"] += mag
            return next_pose
        if isinstance(action, (list, tuple)):
            if len(action) >= 1:
                next_pose["z"] -= float(action[0])
            if len(action) >= 2:
                next_pose["x"] += float(action[1])
            if len(action) >= 3:
                next_pose["yaw"] += float(action[2])
        return next_pose

    def _latest_render_path(self, state: Mapping[str, Any]) -> Optional[Path]:
        text = str(state.get("latest_render_path") or "").strip()
        if not text:
            return None
        path = Path(text)
        return path if path.is_file() else None

    def _normalize_control_intent(self, control: Mapping[str, Any], rollout: Mapping[str, Any]) -> Dict[str, Any]:
        prior = dict(rollout.get("control_intent") or {})
        duration_ms = max(
            400,
            int(
                control.get("durationMs")
                or control.get("duration_ms")
                or prior.get("durationMs")
                or rollout.get("chunk_duration_ms")
                or 1200
            ),
        )
        return {
            "seq": int(control.get("seq") or prior.get("seq") or 0),
            "tClientMs": control.get("tClientMs") or control.get("t_client_ms") or prior.get("tClientMs"),
            "vx": float(control.get("vx") or 0.0),
            "vy": float(control.get("vy") or 0.0),
            "vz": float(control.get("vz") or 0.0),
            "yawRate": float(control.get("yawRate") or control.get("yaw_rate") or 0.0),
            "pitchRate": float(control.get("pitchRate") or control.get("pitch_rate") or 0.0),
            "durationMs": duration_ms,
            "updatedAt": _utc_now_iso(),
        }

    def _integrate_control_pose(self, T_world_camera: Any, control: Mapping[str, Any], dt_s: float):
        import numpy as np

        T = np.array(T_world_camera, dtype=np.float64)
        delta = np.eye(4, dtype=np.float64)
        delta[2, 3] = -float(control.get("vx") or 0.0) * dt_s
        delta[0, 3] = float(control.get("vy") or 0.0) * dt_s
        delta[1, 3] = -float(control.get("vz") or 0.0) * dt_s
        yaw = float(control.get("yawRate") or 0.0) * dt_s
        if abs(yaw) > 1e-9:
            delta[:3, :3] = _roty(yaw)
        return T @ delta

    def _trajectory_horizon(
        self,
        *,
        state: Mapping[str, Any],
        site_index_path: Optional[Path],
        control: Mapping[str, Any],
    ) -> Tuple[Any, Any, List[Dict[str, float]]]:
        start_T = self._session_target_pose(state, site_index_path)
        duration_ms = max(400, int(control.get("durationMs") or 1200))
        horizon_steps = max(3, min(8, int(os.getenv("NATIVE_WORLD_MODEL_HORIZON_STEPS", "5"))))
        dt_s = duration_ms / 1000.0 / horizon_steps
        poses: List[Dict[str, float]] = []
        current_T = start_T
        for _ in range(horizon_steps):
            current_T = self._integrate_control_pose(current_T, control, dt_s)
            pose_summary = _pose_summary_from_matrix(current_T)
            pose_summary["pitch"] = float(pose_summary.get("pitch", 0.0)) + float(control.get("pitchRate") or 0.0) * dt_s
            poses.append(pose_summary)
        return start_T, current_T, poses

    def _query_references_for_pose(
        self,
        *,
        site_index_path: Optional[Path],
        target_T_world_camera: Any,
        storage_root: Path,
        bucket: str,
        k: int,
    ) -> List[Dict[str, Any]]:
        if site_index_path is None:
            return []
        try:
            from .synthesis.retrieval_query import query_site

            refs = query_site(
                site_index_path=site_index_path,
                target_T_world_camera=target_T_world_camera,
                k=k,
                mode="spatial",
                storage_root=storage_root,
                bucket=bucket,
            )
            return [_compact_reference_record(item) for item in refs]
        except Exception:
            return []

    def _live_synthesis_mode(self) -> str:
        explicit = str(os.getenv("NATIVE_WORLD_MODEL_SYNTHESIS_MODE") or "").strip()
        if explicit:
            return explicit
        return "cosmos_i2w" if _runtime_readiness().get("ready") else "splat_only"

    def _chunk_record(self, rollout: Mapping[str, Any], chunk_id: str) -> Optional[Dict[str, Any]]:
        for chunk in list(rollout.get("chunks") or []):
            if str(chunk.get("chunk_id") or "") == chunk_id:
                return dict(chunk)
        return None

    def _replace_chunk(self, rollout: Dict[str, Any], chunk_payload: Mapping[str, Any]) -> None:
        chunk_id = str(chunk_payload.get("chunk_id") or "")
        chunks = [dict(item) for item in list(rollout.get("chunks") or []) if str(item.get("chunk_id") or "") != chunk_id]
        chunks.append(dict(chunk_payload))
        chunks.sort(key=lambda item: int(item.get("chunk_index") or 0))
        rollout["chunks"] = chunks[-6:]

    def _current_chunk(self, rollout: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        active_chunk_id = str(rollout.get("active_chunk_id") or "").strip()
        if not active_chunk_id:
            return None
        return self._chunk_record(rollout, active_chunk_id)

    def _refresh_rollout_playback(self, state: Dict[str, Any]) -> None:
        rollout = self._ensure_rollout(state)
        now_ms = _utc_now_ms()
        active_chunk = self._current_chunk(rollout)
        if active_chunk:
            activated_at_ms = int(active_chunk.get("activated_at_ms") or 0)
            chunk_duration_ms = int(active_chunk.get("duration_ms") or rollout.get("chunk_duration_ms") or 0)
            if activated_at_ms > 0 and chunk_duration_ms > 0 and now_ms - activated_at_ms >= chunk_duration_ms:
                buffer_ids = [str(item) for item in list(rollout.get("buffered_chunk_ids") or []) if str(item)]
                try:
                    index = buffer_ids.index(str(active_chunk.get("chunk_id") or ""))
                except ValueError:
                    index = -1
                next_chunk_id = buffer_ids[index + 1] if index >= 0 and index + 1 < len(buffer_ids) else None
                if next_chunk_id:
                    next_chunk = self._chunk_record(rollout, next_chunk_id)
                    if next_chunk is not None:
                        next_chunk["activated_at_ms"] = now_ms
                        self._replace_chunk(rollout, next_chunk)
                        rollout["active_chunk_id"] = next_chunk_id
                        rollout["status"] = "playing"
                        rollout["underrun"] = False
                        rollout["current_media_type"] = next_chunk.get("media_type")
                        rollout["current_render_source"] = next_chunk.get("render_source")
                else:
                    rollout["status"] = "underrun"
                    rollout["underrun"] = True
        if not rollout.get("active_chunk_id"):
            buffer_ids = [str(item) for item in list(rollout.get("buffered_chunk_ids") or []) if str(item)]
            if buffer_ids:
                first_chunk = self._chunk_record(rollout, buffer_ids[0])
                if first_chunk is not None:
                    first_chunk["activated_at_ms"] = now_ms
                    self._replace_chunk(rollout, first_chunk)
                    rollout["active_chunk_id"] = str(first_chunk.get("chunk_id") or "")
                    rollout["status"] = "playing"
                    rollout["underrun"] = False
                    rollout["current_media_type"] = first_chunk.get("media_type")
                    rollout["current_render_source"] = first_chunk.get("render_source")
            elif rollout.get("queued_chunk_ids"):
                rollout["status"] = "buffering"
            elif rollout.get("chunk_count", 0) > 0:
                rollout["status"] = "underrun"

    def _synthesize_step_async(
        self,
        *,
        session_id: str,
        step_index: int,
        site_id: str,
        storage_root: Path,
        bucket: str,
        target_T_world_camera: Any,
        target_intrinsics: Dict[str, float],
        target_h: int,
        target_w: int,
    ) -> None:
        from .synthesis.synthesize import synthesize_view

        output_dir = self._session_dir(session_id) / "live_synth"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"step_{step_index:05d}.png"
        try:
            result = dict(
                synthesize_view(
                    site_id=site_id,
                    storage_root=storage_root,
                    bucket=bucket,
                    target_T_world_camera=target_T_world_camera,
                    target_intrinsics=target_intrinsics,
                    target_h=target_h,
                    target_w=target_w,
                    output_path=output_path,
                    mode=self._live_synthesis_mode(),
                )
            )
        except Exception as exc:
            result = {"status": "failed", "reason": str(exc)}

        with _cosmos_session_lock(f"state:{session_id}"):
            state = self._load_session_state(session_id)
            if int(state.get("pending_step_index") or -1) != step_index:
                return
            state["latest_synthesis"] = result
            state["updated_at"] = _utc_now_iso()
            if result.get("status") == "completed" and output_path.is_file():
                state["status"] = "running"
                state["synthesis_status"] = "completed"
                state["pending_step_index"] = None
                state["latest_render_path"] = str(output_path.resolve())
                state["latest_render_source"] = "live_synthesis"
                state["failure_reason"] = None
                state["observation"] = self._make_observation(
                    session_id,
                    step_index,
                    render_source="live_synthesis",
                )
            else:
                state["status"] = "failed"
                state["synthesis_status"] = "failed"
                state["failure_reason"] = str(result.get("reason") or "synthesis_failed")
            self._store_session_state(session_id, state)

    def _copy_video_to_chunk(
        self,
        *,
        session_id: str,
        source_path: Path,
        chunk_id: str,
        chunk_index: int,
        render_source: str,
    ) -> Optional[Dict[str, Any]]:
        if not source_path.is_file():
            return None
        chunk_path = self._chunk_video_path(session_id, chunk_id)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, chunk_path)
        tail_path = self._chunk_tail_path(session_id, chunk_id)
        self._extract_tail_frame(chunk_path, tail_path)
        return {
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "status": "ready",
            "media_path": str(chunk_path.resolve()),
            "media_type": "video/mp4",
            "render_source": render_source,
            "duration_ms": self._rollout_defaults()["chunk_duration_ms"],
            "tail_path": str(tail_path.resolve()) if tail_path.is_file() else None,
        }

    def _extract_tail_frame(self, video_path: Path, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-sseof",
                    "-0.05",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    str(output_path),
                ],
                capture_output=True,
                check=False,
            )
        except OSError:
            return

    def _image_to_mp4(self, image_path: Path, output_path: Path, duration_ms: int) -> bool:
        """Encode a still image into an fMP4 segment suitable for MSE append."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        duration_s = max(0.4, duration_ms / 1000.0)
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    str(image_path),
                    "-t",
                    f"{duration_s:.2f}",
                    "-vf",
                    "format=yuv420p",
                    # frag_keyframe+empty_moov+default_base_moof produces a
                    # fragmented MP4 (fMP4) that can be appended to a browser
                    # MSE SourceBuffer without re-buffering between chunks.
                    "-movflags",
                    "frag_keyframe+empty_moov+default_base_moof",
                    str(output_path),
                ],
                capture_output=True,
                check=False,
            )
            if result.returncode == 0 and output_path.is_file():
                return True
        except OSError:
            pass

        # ffmpeg is unavailable or failed — do not write fake media (e.g. a
        # PNG renamed to .mp4). Return False so the rollout layer can
        # gracefully fall back to image-only delivery.
        return False

    def _convert_to_fmp4(self, input_path: Path, output_path: Path) -> bool:
        """Re-mux an existing MP4 into fMP4 format for MSE compatibility.

        Uses stream-copy (-c:v copy) so this is fast and lossless.
        The output has frag_keyframe+empty_moov+default_base_moof flags,
        making each chunk directly appendable to a browser SourceBuffer.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(input_path),
                    "-c:v",
                    "copy",
                    "-movflags",
                    "frag_keyframe+empty_moov+default_base_moof",
                    str(output_path),
                ],
                capture_output=True,
                check=False,
            )
        except OSError:
            return False
        return result.returncode == 0 and output_path.is_file()

    def _bootstrap_video_chunk(self, session_id: str, site_world_id: str) -> Optional[Dict[str, Any]]:
        prebuilt = self._find_prebuilt_cosmos_video(site_world_id)
        if prebuilt and prebuilt.suffix.lower() == ".mp4":
            return self._copy_video_to_chunk(
                session_id=session_id,
                source_path=prebuilt,
                chunk_id="chunk-0000",
                chunk_index=0,
                render_source="bootstrap_prebuilt_video",
            )

        cosmos_repo = _find_cosmos_repo()
        cond_frame = self._find_conditioning_frame(site_world_id)
        if cond_frame and cond_frame.is_file():
            still_chunk_path = self._chunk_video_path(session_id, "chunk-0000")
            if self._image_to_mp4(
                cond_frame,
                still_chunk_path,
                self._rollout_defaults()["chunk_duration_ms"],
            ):
                tail_path = self._chunk_tail_path(session_id, "chunk-0000")
                self._extract_tail_frame(still_chunk_path, tail_path)
                return {
                    "chunk_id": "chunk-0000",
                    "chunk_index": 0,
                    "status": "ready",
                    "media_path": str(still_chunk_path.resolve()),
                    "media_type": "video/mp4",
                    "render_source": "bootstrap_conditioning_frame",
                    "duration_ms": self._rollout_defaults()["chunk_duration_ms"],
                    "tail_path": str(tail_path.resolve()) if tail_path.is_file() else None,
                }
            # ffmpeg unavailable — deliver the conditioning frame as PNG
            return {
                "chunk_id": "chunk-0000",
                "chunk_index": 0,
                "status": "ready",
                "media_path": str(cond_frame.resolve()),
                "media_type": "image/png",
                "render_source": "bootstrap_conditioning_frame",
                "duration_ms": self._rollout_defaults()["chunk_duration_ms"],
                "tail_path": None,
            }
        if cosmos_repo and cond_frame:
            frames_dir = self._cosmos_frames_dir(session_id)
            _ = self._run_cosmos_inference_sync(
                session_id=session_id,
                cosmos_repo=cosmos_repo,
                cond_frame=cond_frame,
                frames_dir=frames_dir,
                lora_adapter=None,
            )
            status = _json_read(self._cosmos_status_path(session_id)) if self._cosmos_status_path(session_id).is_file() else {}
            video_path = Path(str(status.get("video") or "").strip()) if str(status.get("video") or "").strip() else None
            if video_path and video_path.is_file():
                return self._copy_video_to_chunk(
                    session_id=session_id,
                    source_path=video_path,
                    chunk_id="chunk-0000",
                    chunk_index=0,
                    render_source="bootstrap_runtime_video",
                )
        return None

    def _ensure_initial_rollout_chunk(self, session_id: str, site_world_id: str) -> None:
        lock = _cosmos_session_lock(f"rollout:{session_id}")
        with lock:
            state = self._load_session_state(session_id)
            rollout = self._ensure_rollout(state)
            if rollout.get("buffered_chunk_ids"):
                return
            chunk = self._bootstrap_video_chunk(session_id, site_world_id)
            if chunk is None:
                # Acquire state lock so this write is serialized with step_session
                # and _synthesize_step_async — prevents clobbering pending_step_index.
                with _cosmos_session_lock(f"state:{session_id}"):
                    pre_write = self._load_session_state(session_id)
                    # Only update rollout when session is still in its initial
                    # "ready" state.  Any other status means step_session or
                    # a test harness has advanced the session — don't clobber.
                    if (
                        str(pre_write.get("synthesis_status") or "").strip() in ("pending", "completed")
                        or pre_write.get("pending_step_index") is not None
                        or str(pre_write.get("status") or "").strip() not in ("ready", "")
                    ):
                        return
                    pre_write_rollout = self._ensure_rollout(pre_write)
                    pre_write_rollout["status"] = "buffering"
                    pre_write["updated_at"] = _utc_now_iso()
                    self._store_session_state(session_id, pre_write)
                return
            with _cosmos_session_lock(f"state:{session_id}"):
                latest_state2 = self._load_session_state(session_id)
                latest_rollout2 = self._ensure_rollout(latest_state2)
                # Another writer may have populated rollout state while the
                # bootstrap worker was generating the initial chunk. Preserve
                # the newer state instead of overwriting it with stale bootstrap
                # output.
                if latest_rollout2.get("buffered_chunk_ids") or latest_rollout2.get("active_chunk_id"):
                    return
                self._replace_chunk(latest_rollout2, chunk)
                latest_rollout2["buffered_chunk_ids"] = [str(chunk["chunk_id"])]
                latest_rollout2["active_chunk_id"] = str(chunk["chunk_id"])
                latest_rollout2["status"] = "playing"
                latest_rollout2["current_media_type"] = "video/mp4"
                latest_rollout2["current_render_source"] = str(chunk.get("render_source") or "")
                latest_rollout2["chunk_count"] = max(1, int(latest_rollout2.get("chunk_count") or 0))
                latest_state2["updated_at"] = _utc_now_iso()
                self._store_session_state(session_id, latest_state2)

    def _queue_chunk_generation(self, session_id: str) -> None:
        threading.Thread(
            target=self._generate_next_chunk,
            args=(session_id,),
            daemon=True,
        ).start()

    def _refine_chunk_async(
        self,
        *,
        session_id: str,
        chunk_id: str,
        chunk_index: int,
        site_id: str,
        storage_root: Path,
        bucket: str,
        target_T_world_camera: Any,
        lookahead_T_world_camera: Any,
        target_intrinsics: Dict[str, float],
        target_h: int,
        target_w: int,
        previous_tail_path: str | None,
        lookahead_ref_uris: List[str],
        grounding_refs: List[Dict[str, Any]],
        lookahead_refs: List[Dict[str, Any]],
        horizon: List[Dict[str, float]],
        control: Mapping[str, Any],
    ) -> None:
        from .synthesis.synthesize import synthesize_view
        from .synthesis.cosmos_inference import describe_cosmos_model, load_cosmos_model

        owner_claimed, owner_session_id = self._claim_generation_owner(session_id)
        if not owner_claimed:
            with _cosmos_session_lock(f"state:{session_id}"):
                state = self._load_session_state(session_id)
                rollout = self._ensure_rollout(state)
                chunk = self._chunk_record(rollout, chunk_id)
                if chunk is not None:
                    chunk["refinement_status"] = "skipped"
                    provenance = dict(chunk.get("provenance") or {})
                    provenance["refinement_status"] = "skipped"
                    provenance["refinement_reason"] = f"generation_worker_owned_by:{owner_session_id}"
                    chunk["provenance"] = provenance
                    self._replace_chunk(rollout, chunk)
                rollout["refinement_status"] = "skipped"
                rollout["refinement_inflight_chunk_id"] = None
                rollout["generation_owner_session_id"] = owner_session_id
                state["updated_at"] = _utc_now_iso()
                self._store_session_state(session_id, state)
            _push_media_event(session_id, {
                "event": "chunk_refinement_skipped",
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "reason": f"generation_worker_owned_by:{owner_session_id}",
                "ts": _utc_now_iso(),
            })
            return

        refinement_started_at_ms = _utc_now_ms()
        _push_media_event(session_id, {
            "event": "chunk_refinement_started",
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "ts": _utc_now_iso(),
        })

        try:
            cosmos_model_description = describe_cosmos_model(load_cosmos_model())
            worker_backend = str(
                cosmos_model_description.get("worker_backend")
                or cosmos_model_description.get("backend")
                or ""
            )
            settings = self._cosmos_refinement_settings(self._rollout_defaults())
            refined_png = self._video_chunks_dir(session_id) / f"{chunk_id}_refined.png"
            result = dict(
                synthesize_view(
                    site_id=site_id,
                    storage_root=storage_root,
                    bucket=bucket,
                    target_T_world_camera=target_T_world_camera,
                    target_intrinsics=target_intrinsics,
                    target_h=target_h,
                    target_w=target_w,
                    output_path=refined_png,
                    mode="cosmos_i2w",
                    k=4,
                    num_frames=int(settings["num_frames"]),
                    cosmos_width=int(settings["width"]),
                    cosmos_height=int(settings["height"]),
                    cosmos_guidance_scale=float(settings["guidance_scale"]),
                    cosmos_num_steps=int(settings["num_steps"]),
                    previous_tail_path=Path(previous_tail_path) if previous_tail_path else None,
                    previous_tail_alpha=1.0,
                    lookahead_target_T_world_camera=lookahead_T_world_camera,
                    lookahead_k=1,
                    lookahead_ref_uris=lookahead_ref_uris,
                )
            )
            result_video_path = str(result.get("video_path") or "").strip()
            result_output_path = Path(str(result.get("output_path") or refined_png).strip())
            refined_video_path = (
                Path(result_video_path)
                if result_video_path
                else result_output_path
                if result_output_path.is_file() and result_output_path.suffix.lower() == ".png"
                else refined_png.with_suffix(".mp4")
            )
            refinement_is_png = False
            if refined_video_path.is_file() and refined_video_path.suffix.lower() == ".png":
                refinement_is_png = True
            elif not refined_video_path.is_file():
                if not refined_png.is_file():
                    raise RuntimeError(str(result.get("reason") or "refined_video_chunk_generation_failed"))
                if not self._image_to_mp4(
                    refined_png,
                    refined_video_path,
                    int(self._rollout_defaults().get("chunk_duration_ms") or 1200),
                ):
                    # ffmpeg unavailable — use PNG for the refinement output
                    refinement_is_png = True
                    refined_video_path = refined_png

            if not refinement_is_png:
                refined_fmp4_path = refined_video_path.with_stem(refined_video_path.stem + "_fmp4")
                if self._convert_to_fmp4(refined_video_path, refined_fmp4_path):
                    refined_video_path = refined_fmp4_path
            refinement_duration_ms = max(0, _utc_now_ms() - refinement_started_at_ms)

            promoted = False
            with _cosmos_session_lock(f"state:{session_id}"):
                state = self._load_session_state(session_id)
                rollout = self._ensure_rollout(state)
                chunk = self._chunk_record(rollout, chunk_id)
                if chunk is None:
                    return
                chunk["refinement_status"] = "completed"
                chunk["refinement_render_source"] = "cosmos_async_refinement"
                chunk["refined_media_path"] = str(refined_video_path.resolve())
                chunk["refinement_generation_ms"] = refinement_duration_ms
                chunk["worker_backend"] = worker_backend
                provenance = dict(chunk.get("provenance") or {})
                provenance["refinement_status"] = "completed"
                provenance["refinement_render_source"] = "cosmos_async_refinement"
                chunk["provenance"] = provenance
                active_chunk_id = str(rollout.get("active_chunk_id") or "").strip()
                if active_chunk_id != chunk_id and str(chunk.get("media_path") or "").strip() == str(chunk.get("preview_media_path") or "").strip():
                    chunk["media_path"] = str(refined_video_path.resolve())
                    chunk["render_source"] = "cosmos_async_refinement"
                    promoted = True
                self._replace_chunk(rollout, chunk)
                refined_chunk_ids = [str(item) for item in list(rollout.get("refined_chunk_ids") or []) if str(item)]
                if chunk_id not in refined_chunk_ids:
                    refined_chunk_ids.append(chunk_id)
                rollout["refined_chunk_ids"] = refined_chunk_ids[-6:]
                rollout["refinement_status"] = "completed"
                rollout["refinement_inflight_chunk_id"] = None
                rollout["worker_backend"] = worker_backend
                rollout["generation_owner_session_id"] = session_id
                self._update_rollout_world_state(
                    session_id=session_id,
                    rollout=rollout,
                    pose=_pose_summary_from_matrix(target_T_world_camera),
                    target_pose=_pose_summary_from_matrix(lookahead_T_world_camera),
                    trajectory_horizon=horizon,
                    grounding_reference_set=grounding_refs,
                    lookahead_anchor=(lookahead_refs[:1] or [None])[0],
                    last_committed_chunk_id=chunk_id,
                )
                state["updated_at"] = _utc_now_iso()
                self._store_session_state(session_id, state)
            _push_media_event(session_id, {
                "event": "chunk_refinement_ready",
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "render_source": "cosmos_async_refinement",
                "generation_duration_ms": refinement_duration_ms,
                "worker_backend": worker_backend,
                "promoted": promoted,
                "ts": _utc_now_iso(),
            })
        except Exception as exc:
            with _cosmos_session_lock(f"state:{session_id}"):
                state = self._load_session_state(session_id)
                rollout = self._ensure_rollout(state)
                chunk = self._chunk_record(rollout, chunk_id)
                if chunk is not None:
                    chunk["refinement_status"] = "failed"
                    provenance = dict(chunk.get("provenance") or {})
                    provenance["refinement_status"] = "failed"
                    provenance["refinement_reason"] = str(exc)
                    chunk["provenance"] = provenance
                    self._replace_chunk(rollout, chunk)
                rollout["refinement_status"] = "failed"
                rollout["refinement_inflight_chunk_id"] = None
                state["updated_at"] = _utc_now_iso()
                self._store_session_state(session_id, state)
            _push_media_event(session_id, {
                "event": "chunk_refinement_failed",
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "reason": str(exc),
                "ts": _utc_now_iso(),
            })
        finally:
            self._release_generation_owner(session_id)

    def _generate_next_chunk(self, session_id: str) -> None:
        lock = _cosmos_session_lock(f"rollout:{session_id}")
        with lock:
            with _cosmos_session_lock(f"state:{session_id}"):
                state = self._load_session_state(session_id)
                rollout = self._ensure_rollout(state)
                queued_chunk_ids = [str(item) for item in list(rollout.get("queued_chunk_ids") or []) if str(item)]
                if queued_chunk_ids:
                    return
                chunk_index = int(rollout.get("chunk_count") or 0)
                chunk_id = f"chunk-{chunk_index:04d}"
                rollout["queued_chunk_ids"] = [chunk_id]
                rollout["status"] = "buffering" if not rollout.get("active_chunk_id") else rollout.get("status") or "playing"
                state["updated_at"] = _utc_now_iso()
                self._store_session_state(session_id, state)

        chunk_generation_started_at_ms = _utc_now_ms()
        primary_mode = self._preview_generation_mode()
        refinement_requested = primary_mode == "splat_only" and self._cosmos_refinement_enabled()
        worker_backend = None

        # Emit: generation has started
        _push_media_event(session_id, {
            "event": "chunk_generation_started",
            "chunk_id": chunk_id,
            "chunk_index": chunk_index,
            "render_mode": primary_mode,
            "ts": _utc_now_iso(),
        })

        try:
            from .synthesis.synthesize import synthesize_view
            if primary_mode == "cosmos_i2w":
                from .synthesis.cosmos_inference import describe_cosmos_model, load_cosmos_model

            state = self._load_session_state(session_id)
            rollout = self._ensure_rollout(state)
            site_world_id = str(state.get("site_world_id") or "").strip()
            site_world = self.load_site_world(site_world_id) if site_world_id else {}
            storage_root = _optional_existing_path(state.get("storage_root")) or _default_storage_root()
            bucket = str(state.get("storage_bucket") or _bucket_from_site_world(site_world)).strip() or "vast-local"
            scene_id = str(state.get("scene_id") or site_world.get("scene_id") or "").strip()
            capture_id = str(state.get("capture_id") or site_world.get("capture_id") or "").strip()
            site_id = str(state.get("site_id") or "").strip() or _resolve_site_id(site_world, scene_id, capture_id, storage_root, bucket)
            site_index_path = _optional_existing_path(state.get("site_index_path"))
            if site_index_path is None and site_id:
                site_index_path = _resolve_site_index_path(site_id, scene_id, capture_id, storage_root, bucket)
            control = dict(rollout.get("control_intent") or {})
            start_T, end_T, horizon = self._trajectory_horizon(
                state=state,
                site_index_path=site_index_path,
                control=control,
            )
            grounding_refs = self._query_references_for_pose(
                site_index_path=site_index_path,
                target_T_world_camera=start_T,
                storage_root=storage_root,
                bucket=bucket,
                k=4,
            )
            lookahead_refs = self._query_references_for_pose(
                site_index_path=site_index_path,
                target_T_world_camera=end_T,
                storage_root=storage_root,
                bucket=bucket,
                k=1,
            )
            if site_index_path is None:
                raise RuntimeError("site_index_camera_calibration_missing")
            target_intrinsics, target_h, target_w = _intrinsics_from_site_index(
                site_index_path
            )
            output_png = self._video_chunks_dir(session_id) / f"{chunk_id}.png"
            previous_tail_path = rollout.get("last_chunk_tail_path")
            cosmos_settings = self._cosmos_refinement_settings(rollout)
            if primary_mode == "cosmos_i2w":
                owner_claimed, owner_session_id = self._claim_generation_owner(session_id)
                if not owner_claimed:
                    raise RuntimeError(f"generation_worker_owned_by:{owner_session_id}")
                cosmos_model_description = describe_cosmos_model(load_cosmos_model())
                worker_backend = str(
                    cosmos_model_description.get("worker_backend")
                    or cosmos_model_description.get("backend")
                    or ""
                )
            # Collect lookahead ref frame URIs so synthesize_view can tile them
            # into the conditioning image as a spatial hint for trajectory heading.
            lookahead_ref_uris: List[str] = [
                str(r.get("frame_uri") or "") for r in lookahead_refs if r.get("frame_uri")
            ]
            result = dict(
                synthesize_view(
                    site_id=site_id,
                    storage_root=storage_root,
                    bucket=bucket,
                    target_T_world_camera=start_T,
                    target_intrinsics=target_intrinsics,
                    target_h=target_h,
                    target_w=target_w,
                    output_path=output_png,
                    mode=primary_mode,
                    k=4,
                    num_frames=int(rollout.get("chunk_frames") or 57),
                    cosmos_width=int(cosmos_settings["width"]),
                    cosmos_height=int(cosmos_settings["height"]),
                    cosmos_guidance_scale=float(cosmos_settings["guidance_scale"]),
                    cosmos_num_steps=int(cosmos_settings["num_steps"]),
                    # Use tail frame directly (alpha=1.0 = no blend with warped
                    # splat). Blending averages out site-specific texture; using
                    # the raw tail frame preserves visual continuity.
                    previous_tail_path=Path(previous_tail_path) if previous_tail_path else None,
                    previous_tail_alpha=1.0,
                    lookahead_target_T_world_camera=end_T,
                    lookahead_k=1,
                    lookahead_ref_uris=lookahead_ref_uris,
                )
            )
            video_path = Path(str(result.get("video_path") or "").strip()) if str(result.get("video_path") or "").strip() else output_png.with_suffix(".mp4")
            # When ffmpeg isn't available, fall back to delivering the PNG output
            # directly rather than raising an error. This keeps the runtime
            # functional on machines without video encoding tooling.
            media_is_png = False
            if not video_path.is_file():
                if not output_png.is_file():
                    raise RuntimeError(str(result.get("reason") or "video_chunk_generation_failed"))
                if not self._image_to_mp4(output_png, video_path, int(rollout.get("chunk_duration_ms") or 1200)):
                    # ffmpeg is not available — deliver the still image as PNG
                    media_is_png = True

            if media_is_png:
                video_path = output_png

            if not media_is_png:
                # Convert to fMP4 so the browser can append it to a MSE SourceBuffer
                # without re-buffering. The fMP4 re-mux is lossless (stream copy) and
                # fast (~50ms for a 2s chunk on modern hardware).
                fmp4_path = video_path.with_stem(video_path.stem + "_fmp4")
                if self._convert_to_fmp4(video_path, fmp4_path):
                    video_path = fmp4_path
            generation_duration_ms = max(0, _utc_now_ms() - chunk_generation_started_at_ms)
            tail_path = self._chunk_tail_path(session_id, chunk_id)
            if not media_is_png:
                self._extract_tail_frame(video_path, tail_path)
            render_source = "truthful_preview_splat" if primary_mode == "splat_only" else str(result.get("mode") or primary_mode)
            grounding_refs_final = grounding_refs or list(result.get("retrieved_references") or [])
            lookahead_anchor = (lookahead_refs or list(result.get("lookahead_references") or []))[:1] or None
            ready_chunk = {
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "status": "ready",
                "media_path": str(video_path.resolve()),
                "preview_media_path": str(video_path.resolve()),
                "media_type": "image/png" if media_is_png else "video/mp4",
                "render_source": render_source,
                "preview_render_source": render_source,
                "duration_ms": int(rollout.get("chunk_duration_ms") or 1200),
                "tail_path": str(tail_path.resolve()) if tail_path.is_file() else None,
                "grounding_references": grounding_refs_final,
                "lookahead_anchor": lookahead_anchor,
                "trajectory_horizon": horizon,
                "conditioning": dict(result.get("conditioning") or {}),
                "generation_duration_ms": generation_duration_ms,
                "worker_backend": worker_backend,
                "presentation_mode": str(rollout.get("presentation_mode") or ""),
                "provenance": self._chunk_provenance(
                    rollout=rollout,
                    render_source=render_source,
                    grounding_refs=grounding_refs_final,
                    lookahead_refs=lookahead_refs,
                    previous_tail_path=str(previous_tail_path or ""),
                    refinement_status="pending" if refinement_requested else "disabled",
                ),
                "refinement_status": "pending" if refinement_requested else "disabled",
                "generated_at": _utc_now_iso(),
            }
            with _cosmos_session_lock(f"state:{session_id}"):
                state = self._load_session_state(session_id)
                rollout = self._ensure_rollout(state)
                rollout["queued_chunk_ids"] = []
                buffer_ids = [str(item) for item in list(rollout.get("buffered_chunk_ids") or []) if str(item)]
                buffer_ids.append(chunk_id)
                rollout["buffered_chunk_ids"] = buffer_ids[-3:]
                rollout["grounding_reference_set"] = grounding_refs_final
                rollout["lookahead_anchor"] = ready_chunk.get("lookahead_anchor")
                rollout["trajectory_horizon"] = ready_chunk.get("trajectory_horizon") or []
                rollout["last_chunk_tail_path"] = ready_chunk.get("tail_path")
                rollout["generation_lag_ms"] = max(0, _utc_now_ms() - int(control.get("tClientMs") or _utc_now_ms()))
                rollout["chunk_count"] = chunk_index + 1
                rollout["generation_owner_session_id"] = session_id if primary_mode == "cosmos_i2w" else rollout.get("generation_owner_session_id")
                rollout["worker_backend"] = worker_backend
                rollout["last_chunk_generation_ms"] = generation_duration_ms
                rollout["last_chunk_ready_latency_ms"] = generation_duration_ms
                rollout["refinement_status"] = "pending" if refinement_requested else rollout.get("refinement_status")
                rollout["refinement_inflight_chunk_id"] = chunk_id if refinement_requested else None
                if int(rollout.get("chunk_count") or 0) == 1 and not rollout.get("first_chunk_generation_ms"):
                    rollout["first_chunk_generation_ms"] = generation_duration_ms
                self._replace_chunk(rollout, ready_chunk)
                self._refresh_rollout_playback(state)
                state["status"] = "running"
                state["step_count"] = max(int(state.get("step_count") or 0), chunk_index + 1)
                state["step_index"] = int(state["step_count"])
                state["observation"] = self._make_observation(
                    session_id,
                    int(state["step_count"]),
                    render_source=render_source,
                )
                state["camera_pose_matrix"] = end_T.tolist()
                state["pose"] = _pose_summary_from_matrix(end_T)
                self._update_rollout_world_state(
                    session_id=session_id,
                    rollout=rollout,
                    pose=_pose_summary_from_matrix(start_T),
                    target_pose=_pose_summary_from_matrix(end_T),
                    trajectory_horizon=horizon,
                    grounding_reference_set=grounding_refs_final,
                    lookahead_anchor=lookahead_anchor[0] if isinstance(lookahead_anchor, list) and lookahead_anchor else None,
                    last_committed_chunk_id=chunk_id,
                )
                state["updated_at"] = _utc_now_iso()
                self._store_session_state(session_id, state)
            # Notify WS clients immediately — don't wait for the 250ms poll cycle
            _push_media_event(session_id, {
                "event": "chunk_ready",
                "chunk_id": chunk_id,
                "chunk_index": chunk_index,
                "duration_ms": int(ready_chunk.get("duration_ms") or 0),
                "render_source": str(ready_chunk.get("render_source") or ""),
                "generation_duration_ms": generation_duration_ms,
                "worker_backend": worker_backend,
                "presentation_mode": str(ready_chunk.get("presentation_mode") or ""),
                "refinement_status": str(ready_chunk.get("refinement_status") or ""),
                "is_fmp4": True,
                "ts": _utc_now_iso(),
            })
            if refinement_requested:
                threading.Thread(
                    target=self._refine_chunk_async,
                    kwargs={
                        "session_id": session_id,
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_index,
                        "site_id": site_id,
                        "storage_root": storage_root,
                        "bucket": bucket,
                        "target_T_world_camera": start_T,
                        "lookahead_T_world_camera": end_T,
                        "target_intrinsics": target_intrinsics,
                        "target_h": target_h,
                        "target_w": target_w,
                        "previous_tail_path": str(previous_tail_path or ""),
                        "lookahead_ref_uris": lookahead_ref_uris,
                        "grounding_refs": grounding_refs_final,
                        "lookahead_refs": lookahead_refs,
                        "horizon": horizon,
                        "control": control,
                    },
                    daemon=True,
                ).start()
        except Exception as exc:
            with _cosmos_session_lock(f"state:{session_id}"):
                state = self._load_session_state(session_id)
                rollout = self._ensure_rollout(state)
                rollout["queued_chunk_ids"] = []
                rollout["status"] = "failed"
                rollout["underrun"] = True
                rollout["underrun_count"] = int(rollout.get("underrun_count") or 0) + 1
                rollout["generation_owner_session_id"] = self._generation_owner_session_id
                state["failure_reason"] = str(exc)
                state["updated_at"] = _utc_now_iso()
                self._store_session_state(session_id, state)
            if primary_mode == "cosmos_i2w":
                self._release_generation_owner(session_id)
            _push_media_event(session_id, {
                "event": "chunk_underrun",
                "chunk_id": chunk_id,
                "buffer_ms": 0,
                "reason": str(exc),
                "ts": _utc_now_iso(),
            })

    def create_session(self, site_world_id: str, **kwargs: Any) -> Dict[str, Any]:
        site_world = self.load_site_world(site_world_id)
        health = self.load_site_world_health(site_world_id)
        launchable = bool(health.get("launchable", False))
        if not launchable:
            blockers = ",".join(_runtime_blockers(site_world, health)) or "site_world_not_launchable"
            raise RuntimeError(f"site world is blocked: {blockers}")

        runtime_eligibility = (
            dict(site_world.get("runtime_eligibility") or {})
            if isinstance(site_world.get("runtime_eligibility"), Mapping)
            else {}
        )
        requested_backend = str(kwargs.get("requested_backend") or "").strip()
        selected_backend = requested_backend or str(runtime_eligibility.get("default_backend") or "native_world_model")
        session_id = strict_identifier(
            str(kwargs.get("session_id") or uuid4()),
            field="session_id",
        )
        scene_id = strict_identifier(site_world.get("scene_id"), field="scene_id")
        capture_id = strict_identifier(site_world.get("capture_id"), field="capture_id")
        storage_root = _default_storage_root()
        bucket = _bucket_from_site_world(site_world)
        site_id = _resolve_site_id(site_world, scene_id, capture_id, storage_root, bucket)
        site_index_path = (
            _resolve_site_index_path(site_id, scene_id, capture_id, storage_root, bucket)
            if site_id
            else None
        )
        site_reference_manifest_path = (
            _resolve_site_reference_manifest_path(site_id, storage_root, bucket)
            if site_id
            else None
        )
        runtime_readiness = _runtime_readiness()
        site_reference_runtime_adapter = _site_reference_runtime_adapter(
            site_id=site_id,
            manifest_path=site_reference_manifest_path,
            site_index_path=site_index_path,
            storage_root=storage_root,
            bucket=bucket,
            runtime_readiness=runtime_readiness,
        )
        site_reference_runtime_artifact_path = self._site_reference_runtime_artifact_path(session_id)

        # Initial camera pose — taken from first record of the site reference index
        initial_T: Optional[list] = None
        if site_index_path:
            T_init = _pose_from_site_index(site_index_path)
            if T_init is not None:
                initial_T = T_init.tolist()

        state = {
            "session_id": session_id,
            "tenant_id": str(kwargs.get("tenant_id") or "").strip() or None,
            "site_world_id": site_world_id,
            "scene_id": scene_id,
            "capture_id": capture_id,
            "site_id": site_id,
            "storage_root": str(storage_root),
            "storage_bucket": bucket,
            "site_index_path": str(site_index_path) if site_index_path else None,
            "site_reference_manifest_path": (
                str(site_reference_manifest_path) if site_reference_manifest_path else None
            ),
            "site_reference_runtime_adapter": site_reference_runtime_adapter,
            "site_reference_runtime_artifact_path": str(site_reference_runtime_artifact_path),
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
            "debug_mode": False,
            "created_at": _utc_now_iso(),
            "updated_at": _utc_now_iso(),
            "step_count": 0,
            "step_index": 0,
            "last_action": [],
            "pose": _pose_summary_from_matrix(initial_T),
            "camera_pose_matrix": initial_T,
            "synthesis_status": "idle",
            "pending_step_index": None,
            "latest_render_path": None,
            "latest_render_source": None,
            "latest_synthesis": None,
            "failure_reason": None,
            "observation": self._make_observation(session_id, 0, render_source="bootstrap_prebuilt"),
            "rollout": self._rollout_defaults(),
        }
        with _cosmos_session_lock(f"state:{session_id}"):
            if self._session_state_path(session_id).exists():
                raise FileExistsError(session_id)
            _json_write(site_reference_runtime_artifact_path, site_reference_runtime_adapter)
            stored = self._store_session_state(session_id, state)
        # Kick off background Cosmos prep so frames are ready for first render
        threading.Thread(
            target=self._ensure_cosmos_frames,
            args=(session_id, site_world_id),
            daemon=True,
        ).start()
        threading.Thread(
            target=self._ensure_initial_rollout_chunk,
            args=(session_id, site_world_id),
            daemon=True,
        ).start()
        return stored

    def reset_session(self, session_id: str, **kwargs: Any) -> Dict[str, Any]:
        self._release_generation_owner(session_id)
        state = self._load_session_state(session_id)
        if kwargs.get("task_id"):
            state["task_id"] = kwargs["task_id"]
        if kwargs.get("scenario_id"):
            state["scenario_id"] = kwargs["scenario_id"]
        if kwargs.get("start_state_id"):
            state["start_state_id"] = kwargs["start_state_id"]
        state["status"] = "ready"
        state["step_count"] = 0
        state["step_index"] = 0
        state["last_action"] = []
        state["pose"] = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0, "pitch": 0.0}
        state["updated_at"] = _utc_now_iso()
        state["observation"] = self._make_observation(session_id, 0, render_source="bootstrap_prebuilt")
        state["synthesis_status"] = "idle"
        state["pending_step_index"] = None
        state["latest_render_path"] = None
        state["latest_render_source"] = None
        state["latest_synthesis"] = None
        state["failure_reason"] = None
        state["rollout"] = self._rollout_defaults()
        # Restore initial pose from site index if available.
        site_id = str(state.get("site_id") or "").strip()
        scene_id = str(state.get("scene_id") or "").strip()
        capture_id = str(state.get("capture_id") or "").strip()
        storage_root = _optional_existing_path(state.get("storage_root")) or _default_storage_root()
        bucket = str(state.get("storage_bucket") or _bucket_from_site_world({})).strip() or "vast-local"
        if site_id:
            idx_path = _resolve_site_index_path(site_id, scene_id, capture_id, storage_root, bucket)
            if idx_path:
                T_init = _pose_from_site_index(idx_path)
                if T_init is not None:
                    state["camera_pose_matrix"] = T_init.tolist()
                    state["pose"] = _pose_summary_from_matrix(T_init)
        stored = self._store_session_state(session_id, state)
        site_world_id = str(state.get("site_world_id") or "").strip()
        if site_world_id:
            threading.Thread(
                target=self._ensure_initial_rollout_chunk,
                args=(session_id, site_world_id),
                daemon=True,
            ).start()
        return stored

    def step_session(self, session_id: str, *, action: Any) -> Dict[str, Any]:
        # Resolve site/storage metadata before taking the state lock so the
        # lock isn't held during potentially slow I/O like load_site_world.
        _pre = self._load_session_state(session_id)
        site_world_id = str(_pre.get("site_world_id") or "").strip()
        site_world = self.load_site_world(site_world_id) if site_world_id else {}
        storage_root = _optional_existing_path(_pre.get("storage_root")) or _default_storage_root()
        bucket = str(_pre.get("storage_bucket") or _bucket_from_site_world(site_world)).strip() or "vast-local"
        scene_id = str(_pre.get("scene_id") or site_world.get("scene_id") or "").strip()
        capture_id = str(_pre.get("capture_id") or site_world.get("capture_id") or "").strip()
        site_id = str(_pre.get("site_id") or "").strip() or _resolve_site_id(
            site_world,
            scene_id,
            capture_id,
            storage_root,
            bucket,
        )
        site_index_path = _optional_existing_path(_pre.get("site_index_path"))
        if site_index_path is None and site_id:
            site_index_path = _resolve_site_index_path(site_id, scene_id, capture_id, storage_root, bucket)

        synthesis_kwargs: Optional[Dict[str, Any]] = None
        with _cosmos_session_lock(f"state:{session_id}"):
            state = self._load_session_state(session_id)
            new_step = int(state.get("step_count") or 0) + 1
            state["step_count"] = new_step
            state["step_index"] = new_step
            state["last_action"] = dict(action) if isinstance(action, Mapping) else list(action or [])
            state["updated_at"] = _utc_now_iso()
            state["failure_reason"] = None

            if site_id and site_index_path is not None:
                target_T_world_camera = self._session_target_pose(state, site_index_path)
                target_T_world_camera = _apply_action(target_T_world_camera, action)
                state["pose"] = _pose_summary_from_matrix(target_T_world_camera)
                state["camera_pose_matrix"] = target_T_world_camera.tolist()
                state["site_id"] = site_id
                state["site_index_path"] = str(site_index_path)
                state["storage_root"] = str(storage_root)
                state["storage_bucket"] = bucket
                state["status"] = "synthesizing"
                state["synthesis_status"] = "pending"
                state["pending_step_index"] = new_step
                state["observation"] = self._make_pending_observation()
                intrinsics, target_h, target_w = _intrinsics_from_site_index(site_index_path)
                stored = self._store_session_state(session_id, state)
                synthesis_kwargs = {
                    "session_id": session_id,
                    "step_index": new_step,
                    "site_id": site_id,
                    "storage_root": storage_root,
                    "bucket": bucket,
                    "target_T_world_camera": target_T_world_camera,
                    "target_intrinsics": intrinsics,
                    "target_h": target_h,
                    "target_w": target_w,
                }
            else:
                state["pose"] = self._fallback_pose_update(state.get("pose") or {}, action)
                state["status"] = "running"
                state["synthesis_status"] = "unavailable"
                state["pending_step_index"] = None
                state["observation"] = self._make_observation(session_id, new_step, render_source="fallback_pose_update")
                stored = self._store_session_state(session_id, state)

        if synthesis_kwargs is not None:
            threading.Thread(
                target=self._synthesize_step_async,
                kwargs=synthesis_kwargs,
                daemon=True,
            ).start()
        return stored

    def control_session(self, session_id: str, *, control: Dict[str, Any]) -> Dict[str, Any]:
        state = self._load_session_state(session_id)
        rollout = self._ensure_rollout(state)
        intent = self._normalize_control_intent(control, rollout)
        site_world_id = str(state.get("site_world_id") or "").strip()
        site_world = self.load_site_world(site_world_id) if site_world_id else {}
        storage_root = _optional_existing_path(state.get("storage_root")) or _default_storage_root()
        bucket = str(state.get("storage_bucket") or _bucket_from_site_world(site_world)).strip() or "vast-local"
        scene_id = str(state.get("scene_id") or site_world.get("scene_id") or "").strip()
        capture_id = str(state.get("capture_id") or site_world.get("capture_id") or "").strip()
        site_id = str(state.get("site_id") or "").strip() or _resolve_site_id(site_world, scene_id, capture_id, storage_root, bucket)
        site_index_path = _optional_existing_path(state.get("site_index_path"))
        if site_index_path is None and site_id:
            site_index_path = _resolve_site_index_path(site_id, scene_id, capture_id, storage_root, bucket)

        start_T, end_T, horizon = self._trajectory_horizon(
            state=state,
            site_index_path=site_index_path,
            control=intent,
        )
        rollout["control_intent"] = intent
        rollout["trajectory_horizon"] = horizon
        rollout["grounding_reference_set"] = self._query_references_for_pose(
            site_index_path=site_index_path,
            target_T_world_camera=start_T,
            storage_root=storage_root,
            bucket=bucket,
            k=4,
        )
        lookahead = self._query_references_for_pose(
            site_index_path=site_index_path,
            target_T_world_camera=end_T,
            storage_root=storage_root,
            bucket=bucket,
            k=1,
        )
        rollout["lookahead_anchor"] = lookahead[0] if lookahead else None
        self._update_rollout_world_state(
            session_id=session_id,
            rollout=rollout,
            pose=_pose_summary_from_matrix(start_T),
            target_pose=_pose_summary_from_matrix(end_T),
            trajectory_horizon=horizon,
            grounding_reference_set=list(rollout.get("grounding_reference_set") or []),
            lookahead_anchor=rollout.get("lookahead_anchor"),
            last_committed_chunk_id=str(rollout.get("active_chunk_id") or "").strip() or None,
        )
        self._refresh_rollout_playback(state)
        if self._should_queue_more_chunks(rollout):
            self._queue_chunk_generation(session_id)
            rollout["status"] = "buffering" if not rollout.get("active_chunk_id") else rollout.get("status") or "playing"
        state["updated_at"] = _utc_now_iso()
        return self._store_session_state(session_id, state)

    def session_state(self, session_id: str) -> Dict[str, Any]:
        with _cosmos_session_lock(f"state:{session_id}"):
            state = self._load_session_state(session_id)
            self._refresh_rollout_playback(state)
            rollout = self._ensure_rollout(state)
            if self._should_queue_more_chunks(rollout):
                self._queue_chunk_generation(session_id)
            state["updated_at"] = _utc_now_iso()
            return self._store_session_state(session_id, state)

    # ---------------------------------------------------------------------------
    # Cosmos frame helpers
    # ---------------------------------------------------------------------------

    def _allow_prebuilt_bootstrap_video(self) -> bool:
        """Generic prebuilt videos are opt-in only.

        Smoke-test MP4s can come from unrelated content. The default live path
        must stay site-truthful, so we only use those artifacts when the
        operator explicitly enables them.
        """
        return os.getenv("NATIVE_WORLD_MODEL_ALLOW_PREBUILT_BOOTSTRAP_VIDEO", "").strip() == "1"

    def _find_prebuilt_cosmos_video(self, site_world_id: str) -> Optional[Path]:
        """Find a pre-existing Cosmos video from pipeline runs for this site world."""
        if not self._allow_prebuilt_bootstrap_video():
            return None
        try:
            sw = self.load_site_world(site_world_id)
        except FileNotFoundError:
            return None
        scene_id = str(sw.get("scene_id") or "").strip()
        capture_id = str(sw.get("capture_id") or "").strip()
        gcs_root = Path(os.getenv("GCS_ROOT", "/root/blueprint-storage"))

        if scene_id and capture_id:
            pipeline_base = (
                gcs_root / "vast-local" / "scenes" / scene_id / "captures" / capture_id / "pipeline"
            )
            candidates: List[Path] = [
                pipeline_base / "cosmos_single_capture_smoke" / "renders" / "video_bootstrap_0000.mp4",
                pipeline_base / "cosmos_single_capture_smoke" / "renders" / "video_bootstrap_0000.jpg",
            ]
            for c in candidates:
                try:
                    if c.is_file():
                        return c
                except OSError:
                    continue

        # Global fallback: manual probe output
        fallback = gcs_root / "manual_cosmos_probe_official" / "blueprint_probe.mp4"
        try:
            if fallback.is_file():
                return fallback
        except OSError:
            return None
        return None

    def _find_conditioning_frame(self, site_world_id: str) -> Optional[Path]:
        """Find best input frame for on-demand Cosmos I2W inference."""
        try:
            sw = self.load_site_world(site_world_id)
        except FileNotFoundError:
            return None
        scene_id = str(sw.get("scene_id") or "").strip()
        capture_id = str(sw.get("capture_id") or "").strip()
        gcs_root = Path(os.getenv("GCS_ROOT", "/root/blueprint-storage"))

        if scene_id and capture_id:
            pipeline_base = (
                gcs_root / "vast-local" / "scenes" / scene_id / "captures" / capture_id / "pipeline"
            )
            candidates: List[Path] = [
                pipeline_base / "cosmos_single_capture_smoke" / "video_bootstrap_frames" / "frame_0000.jpg",
                pipeline_base / "cosmos_single_capture_smoke" / "renders" / "video_bootstrap_0000.jpg",
            ]
            for c in candidates:
                try:
                    if c.is_file():
                        return c
                except OSError:
                    continue
        return None

    def _extract_frames_from_video(self, video_path: Path, frames_dir: Path) -> List[Path]:
        """Extract frames from MP4 at 4 fps using ffmpeg."""
        frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.run(
                [
                    "ffmpeg", "-i", str(video_path),
                    "-vf", "fps=4",
                    str(frames_dir / "frame_%04d.png"),
                    "-y",
                ],
                capture_output=True,
                check=False,
            )
        except OSError:
            return []
        return sorted(frames_dir.glob("frame_*.png"))

    def _extract_single_frame(self, image_path: Path, frames_dir: Path) -> List[Path]:
        """Copy a single image (JPG/PNG) into frames_dir as frame_0001.png."""
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_out = frames_dir / "frame_0001.png"
        try:
            img = Image.open(image_path).convert("RGB")
            img.save(frame_out, format="PNG")
        except Exception:
            try:
                shutil.copy2(image_path, frame_out)
            except OSError:
                return []
        return [frame_out] if frame_out.is_file() else []

    def _ensure_cosmos_frames(self, session_id: str, site_world_id: str) -> List[Path]:
        """
        Ensure Cosmos frames exist for this session. Returns the frame list.
        Thread-safe: only one caller will run inference; others wait and use cache.
        Tries explicit pre-built pipeline output first; otherwise uses the
        current site's conditioning frame or on-demand inference.
        """
        frames_dir = self._cosmos_frames_dir(session_id)

        # Fast path: frames already extracted
        if frames_dir.is_dir():
            existing = sorted(frames_dir.glob("frame_*.png"))
            if existing:
                return existing

        lock = _cosmos_session_lock(session_id)
        with lock:
            # Re-check under lock
            if frames_dir.is_dir():
                existing = sorted(frames_dir.glob("frame_*.png"))
                if existing:
                    return existing

            # Try pre-built pipeline output only when explicitly enabled.
            prebuilt = self._find_prebuilt_cosmos_video(site_world_id)
            if prebuilt:
                if prebuilt.suffix.lower() == ".mp4":
                    frames = self._extract_frames_from_video(prebuilt, frames_dir)
                else:
                    frames = self._extract_single_frame(prebuilt, frames_dir)
                if frames:
                    _json_write(
                        self._cosmos_status_path(session_id),
                        {
                            "source": "prebuilt_pipeline",
                            "video": str(prebuilt),
                            "frame_count": len(frames),
                            "extracted_at": _utc_now_iso(),
                        },
                    )
                    return frames

            cosmos_repo = _find_cosmos_repo()
            cond_frame = self._find_conditioning_frame(site_world_id)
            if cond_frame and cond_frame.is_file():
                frames = self._extract_single_frame(cond_frame, frames_dir)
                if frames:
                    _json_write(
                        self._cosmos_status_path(session_id),
                        {
                            "source": "conditioning_frame",
                            "video": str(cond_frame),
                            "frame_count": len(frames),
                            "extracted_at": _utc_now_iso(),
                        },
                    )
                    return frames

            # Fall back to on-demand Cosmos inference
            if cosmos_repo and cond_frame:
                lora_adapter = self._find_lora_adapter(site_world_id)
                return self._run_cosmos_inference_sync(
                    session_id=session_id,
                    cosmos_repo=cosmos_repo,
                    cond_frame=cond_frame,
                    frames_dir=frames_dir,
                    lora_adapter=lora_adapter,
                )

            return []

    def _find_lora_adapter(self, site_world_id: str) -> Optional[Path]:
        """
        Find a LoRA adapter checkpoint for this site world's capture.
        Checks COSMOS_LORA_CHECKPOINT_PATH env var first, then the standard
        cosmos_training_export/checkpoints/ path under the capture's pipeline dir.
        """
        explicit = os.getenv("COSMOS_LORA_CHECKPOINT_PATH", "").strip()
        if explicit:
            p = Path(explicit)
            return p if p.is_file() else None

        try:
            sw = self.load_site_world(site_world_id)
        except FileNotFoundError:
            return None
        scene_id = str(sw.get("scene_id") or "").strip()
        capture_id = str(sw.get("capture_id") or "").strip()
        gcs_root = Path(os.getenv("GCS_ROOT", "/root/blueprint-storage"))
        if scene_id and capture_id:
            adapter = (
                gcs_root / "vast-local" / "scenes" / scene_id / "captures" / capture_id
                / "pipeline" / "cosmos_training_export" / "checkpoints"
                / "adapter_model.safetensors"
            )
            try:
                if adapter.is_file():
                    return adapter
            except OSError:
                return None
        return None

    def _run_cosmos_inference_sync(
        self,
        session_id: str,
        cosmos_repo: Tuple[Path, Path],
        cond_frame: Path,
        frames_dir: Path,
        lora_adapter: Optional[Path] = None,
    ) -> List[Path]:
        """Run Cosmos I2W inference synchronously. Returns extracted frame list."""
        repo_root, python_bin = cosmos_repo
        cosmos_dir = self._cosmos_dir(session_id)
        cosmos_dir.mkdir(parents=True, exist_ok=True)

        sample_name = f"cosmos_{session_id[:8]}"
        asset_path = cosmos_dir / f"{sample_name}.json"
        output_video = cosmos_dir / f"{sample_name}.mp4"
        log_path = cosmos_dir / "inference.log"

        # COSMOS_CHUNK_SIZE / COSMOS_CHUNK_OVERLAP are the canonical knobs.
        # At H100 speeds, 57 frames (~2s) is the target; drop to 33 if generation
        # takes >1.5s so N+1 is always ready before N ends.
        # chunk_overlap=4 gives the strongest pseudo-autoregressive conditioning
        # available at zero FT: the 4 overlap frames become the conditioning head
        # of the next chunk.
        cosmos_chunk_size = max(
            8,
            int(os.getenv("COSMOS_CHUNK_SIZE") or os.getenv("NATIVE_WORLD_MODEL_CHUNK_FRAMES") or "57"),
        )
        cosmos_chunk_overlap = max(
            1,
            int(os.getenv("COSMOS_CHUNK_OVERLAP", "4")),
        )
        asset_path.write_text(
            json.dumps({
                "inference_type": "image2world",
                "name": sample_name,
                "input_path": str(cond_frame.resolve()),
                "prompt": (
                    "First-person camera moving through a real indoor workspace. "
                    "Preserve the existing geometry and continue the scene naturally."
                ),
                "num_output_frames": cosmos_chunk_size,
                "num_steps": 35,
                "seed": 0,
                "guidance": 7.0,
                "enable_autoregressive": False,
                "chunk_size": cosmos_chunk_size,
                "chunk_overlap": cosmos_chunk_overlap,
            }),
            encoding="utf-8",
        )

        env = {k: v for k, v in os.environ.items() if isinstance(v, str)}
        env["PATH"] = str(repo_root / ".venv" / "bin") + ":" + env.get("PATH", "")

        cmd = [
            str(python_bin),
            "examples/inference.py",
            "-i", str(asset_path),
            "-o", str(cosmos_dir),
            "--model=2B/post-trained",
            "--disable-guardrails",
        ]
        if lora_adapter and lora_adapter.is_file():
            cmd += ["--lora-checkpoint", str(lora_adapter)]

        with log_path.open("w", encoding="utf-8") as lf:
            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(repo_root),
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except OSError:
                return []

        if result.returncode != 0 or not output_video.is_file():
            return []

        # Re-mux to fMP4 so the runtime can serve it directly to MSE clients.
        fmp4_video = output_video.with_stem(output_video.stem + "_fmp4")
        if self._convert_to_fmp4(output_video, fmp4_video):
            output_video = fmp4_video

        frames = self._extract_frames_from_video(output_video, frames_dir)
        if frames:
            _json_write(
                self._cosmos_status_path(session_id),
                {
                    "source": "on_demand_inference",
                    "video": str(output_video),
                    "frame_count": len(frames),
                    "inferred_at": _utc_now_iso(),
                },
            )
        return frames

    # ---------------------------------------------------------------------------
    # Render
    # ---------------------------------------------------------------------------

    def _render_png(
        self,
        session_id: str,
        camera_id: str,
        *,
        pose: Mapping[str, Any] | None = None,
        refine_mode: str | None = None,
        size: tuple[int, int] = (960, 540),
    ) -> bytes:
        """Placeholder render (used when Cosmos frames are not yet ready)."""
        state = self._load_session_state(session_id)
        width, height = size
        image = Image.new("RGB", size, color=(238, 245, 247))
        draw = ImageDraw.Draw(image)
        draw.rectangle([(24, 24), (width - 24, height - 24)], outline=(15, 118, 110), width=3)
        lines = [
            "Blueprint Native Runtime — Cosmos loading",
            f"session_id={session_id}",
            f"site_world_id={state.get('site_world_id')}",
            f"camera_id={camera_id}",
            f"backend={state.get('runtime_backend_selected')}",
            f"step_count={state.get('step_count')}",
            "render_source=placeholder_cosmos_pending",
        ]
        render_pose = dict(pose or state.get("pose") or {})
        lines.append(
            "pose="
            f"({float(render_pose.get('x', 0.0)):.2f},"
            f"{float(render_pose.get('y', 0.0)):.2f},"
            f" yaw={float(render_pose.get('yaw', 0.0)):.2f})"
        )
        if refine_mode:
            lines.append(f"refine_mode={refine_mode}")
        y = 48
        for line in lines:
            draw.text((48, y), line, fill=(22, 45, 52))
            y += 32
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def render_bytes(self, session_id: str, camera_id: str) -> bytes:
        """Render a frame. Priority: synthesized splat > Cosmos video > placeholder."""
        with _cosmos_session_lock(f"state:{session_id}"):
            state = self._load_session_state(session_id)
            live_render = self._latest_render_path(state)
            if live_render is not None:
                live_render_bytes = _read_bytes_if_available(live_render)
                if live_render_bytes is not None:
                    return live_render_bytes

        # 2. Pre-built Cosmos video frames (bootstrap path)
        site_world_id = str(state.get("site_world_id") or "")
        step_count = int(state.get("step_count") or 0)
        if site_world_id:
            frames = self._ensure_cosmos_frames(session_id, site_world_id)
            if frames:
                idx = min(step_count, len(frames) - 1)
                selected = frames[idx]
                with _cosmos_session_lock(f"state:{session_id}"):
                    state = self._load_session_state(session_id)
                    if str(state.get("synthesis_status") or "").strip() == "pending":
                        state["status"] = "running"
                        state["synthesis_status"] = "completed"
                        state["pending_step_index"] = None
                        state["latest_render_path"] = str(selected.resolve())
                        state["latest_render_source"] = "cosmos_frames"
                        state["failure_reason"] = None
                        if not isinstance(state.get("latest_synthesis"), Mapping):
                            state["latest_synthesis"] = {
                                "status": "completed",
                                "source": "cosmos_frames",
                                "output_path": str(selected.resolve()),
                                "frame_count": len(frames),
                            }
                        state["observation"] = self._make_observation(
                            session_id,
                            step_count,
                            render_source="cosmos_frames",
                        )
                        self._store_session_state(session_id, state)
                selected_bytes = _read_bytes_if_available(selected)
                if selected_bytes is not None:
                    return selected_bytes

        return self._render_png(session_id, camera_id)

    def media_response(self, session_id: str, *, camera_id: str, chunk_id: str | None) -> Dict[str, Any]:
        with _cosmos_session_lock(f"state:{session_id}"):
            state = self._load_session_state(session_id)
            self._refresh_rollout_playback(state)
            rollout = self._ensure_rollout(state)
            state["updated_at"] = _utc_now_iso()
            self._store_session_state(session_id, state)
            selected_chunk_id = str(chunk_id or rollout.get("active_chunk_id") or "").strip()
            chunk = self._chunk_record(rollout, selected_chunk_id) if selected_chunk_id else None
        media_path = Path(str(chunk.get("media_path") or "").strip()) if chunk else None
        if media_path and media_path.is_file():
            media_bytes = _read_bytes_if_available(media_path)
            if media_bytes is not None:
                return {
                    "content": media_bytes,
                    "media_type": str(chunk.get("media_type") or "video/mp4"),
                    "headers": {
                        "Cache-Control": "no-store",
                        "X-Blueprint-Render-Source": str(chunk.get("render_source") or "runtime-video-chunk"),
                        "X-Blueprint-Media-Status": str(rollout.get("status") or "playing"),
                        "X-Blueprint-Chunk-Id": selected_chunk_id,
                        "X-Blueprint-Presentation-Mode": str(rollout.get("presentation_mode") or ""),
                        "X-Blueprint-Refinement-Status": str(chunk.get("refinement_status") or rollout.get("refinement_status") or ""),
                    },
                }
        placeholder = self._render_png(session_id, camera_id)
        return {
            "content": placeholder,
            "media_type": "image/png",
            "headers": {
                "Cache-Control": "no-store",
                "X-Blueprint-Render-Source": "placeholder_cosmos_pending",
                "X-Blueprint-Media-Status": str(rollout.get("status") or "buffering"),
                "X-Blueprint-Presentation-Mode": str(rollout.get("presentation_mode") or ""),
                "X-Blueprint-Refinement-Status": str(rollout.get("refinement_status") or ""),
            },
        }

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
        safe_camera_id = strict_identifier(camera_id, field="camera_id")
        state = self._load_session_state(session_id)
        state["pose"] = dict(pose)
        state["updated_at"] = _utc_now_iso()
        self._store_session_state(session_id, state)
        frame_dir = self._session_dir(session_id) / "explorer_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = contained_path(
            frame_dir,
            f"{safe_camera_id}.png",
            field="explorer frame path",
        )
        live_render = self._latest_render_path(state)
        if live_render is not None:
            live_render_bytes = _read_bytes_if_available(live_render)
            if live_render_bytes is not None:
                frame_path.write_bytes(live_render_bytes)
                return {
                    "status": "completed",
                    "session_id": session_id,
                    "camera_id": camera_id,
                    "pose": dict(pose),
                    "refine_mode": refine_mode,
                    "frame_path": str(frame_path.resolve()),
                    "runtime_kind": "native_world_model",
                }

        # Try Cosmos frames first
        site_world_id = str(state.get("site_world_id") or "")
        step_count = int(state.get("step_count") or 0)
        cosmos_bytes: Optional[bytes] = None
        if site_world_id:
            frames = self._ensure_cosmos_frames(session_id, site_world_id)
            if frames:
                idx = min(step_count, len(frames) - 1)
                cosmos_bytes = _read_bytes_if_available(frames[idx])

        if cosmos_bytes:
            frame_path.write_bytes(cosmos_bytes)
        else:
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
        safe_camera_id = strict_identifier(camera_id, field="camera_id")
        frame_path = contained_path(
            self._session_dir(session_id) / "explorer_frames",
            f"{safe_camera_id}.png",
            field="explorer frame path",
        )
        if frame_path.is_file():
            frame_bytes = _read_bytes_if_available(frame_path)
            if frame_bytes is not None:
                return frame_bytes
        return self.render_bytes(session_id, camera_id)

    def drain_media_events(self, session_id: str) -> List[Dict[str, Any]]:
        """Drain all pending media events for this session (non-blocking).

        Called by the WS stream handler each loop iteration to flush events
        that were pushed by background chunk-generation threads.
        """
        return _drain_media_events(session_id)


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
