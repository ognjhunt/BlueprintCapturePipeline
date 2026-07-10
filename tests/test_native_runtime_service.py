from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
import pytest


pytestmark = pytest.mark.slow
pytest.importorskip("PIL")
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from blueprint_pipeline.native_runtime_backend import NativeRuntimeConfig, NativeWorldModelRuntimeStore
from blueprint_pipeline.runtime_service_app import create_runtime_app


def _site_world_payload() -> dict:
    return {
        "spec": {
            "schema_version": "v1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "canonical_package_version": "pkg-1",
            "canonical_package_uri": "gs://bucket/site-worlds/site-1/canonical.json",
            "runtime_eligibility": {
                "launchable": True,
                "readiness_state": "launchable",
                "blockers": [],
                "warnings": [],
                "grounding_status": "grounded",
                "default_backend": "native_world_model",
                "launchable_backends": ["native_world_model"],
            },
        },
        "registration": {
            "schema_version": "v1",
            "site_world_id": "siteworld-1",
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "build_id": "build-1",
        },
        "health": {
            "schema_version": "v1",
            "site_world_id": "siteworld-1",
            "launchable": True,
            "status": "healthy",
            "blockers": [],
        },
    }


def _patch_read_bytes_failure(monkeypatch, failing_paths: set[Path]) -> None:
    from blueprint_pipeline import native_runtime_backend

    original_read_bytes_if_available = native_runtime_backend._read_bytes_if_available

    def flaky_read_bytes_if_available(path: Path):
        if path in failing_paths:
            return None
        return original_read_bytes_if_available(path)

    monkeypatch.setattr(
        native_runtime_backend,
        "_read_bytes_if_available",
        flaky_read_bytes_if_available,
    )


def test_native_runtime_service_contract_round_trip(tmp_path: Path) -> None:
    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    app = create_runtime_app(backend=store, title="test-native-runtime")
    client = TestClient(app)

    runtime_info = client.get("/v1/runtime")
    assert runtime_info.status_code == 200
    assert runtime_info.json()["runtime_kind"] == "native_world_model"

    registration = client.post("/v1/site-worlds", json=_site_world_payload())
    assert registration.status_code == 200
    assert registration.json()["site_world_id"] == "siteworld-1"

    session = client.post(
        "/v1/site-worlds/siteworld-1/sessions",
        json={
            "robot_profile_id": "robot-1",
            "task_id": "task-1",
            "scenario_id": "scenario-1",
            "start_state_id": "start-1",
            "requested_backend": "native_world_model",
        },
    )
    assert session.status_code == 200
    session_id = session.json()["session_id"]

    render = client.get(f"/v1/sessions/{session_id}/render/head_rgb")
    assert render.status_code == 200
    assert render.headers["content-type"].startswith("image/png")

    explorer = client.post(
        f"/v1/sessions/{session_id}/explorer/render",
        json={"camera_id": "head_rgb", "pose": {"x": 1.5, "y": -0.2, "yaw": 0.3}},
    )
    assert explorer.status_code == 200
    frame_path = Path(explorer.json()["frame_path"])
    assert frame_path.is_file()

    state = client.get(f"/v1/sessions/{session_id}/state")
    assert state.status_code == 200
    assert state.json()["pose"]["x"] == 1.5

    explorer_frame = client.get(f"/v1/sessions/{session_id}/explorer/frame/head_rgb")
    assert explorer_frame.status_code == 200
    assert explorer_frame.headers["content-type"].startswith("image/png")


def test_runtime_app_startup_prewarms_backend(tmp_path: Path, monkeypatch) -> None:
    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    called: dict[str, bool] = {}

    def fake_prewarm() -> dict:
        called["ok"] = True
        return {"status": "ready"}

    monkeypatch.setattr(store, "prewarm_runtime", fake_prewarm)

    app = create_runtime_app(backend=store, title="test-native-runtime")
    with TestClient(app):
        pass

    assert called == {"ok": True}


def test_native_runtime_step_session_runs_live_synthesis_when_site_index_exists(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "storage"
    capture_root = storage_root / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    site_index_path = storage_root / "bucket" / "sites" / "site-1" / "reference_memory" / "site_reference_index.jsonl"
    site_index_path.parent.mkdir(parents=True, exist_ok=True)
    capture_root.mkdir(parents=True, exist_ok=True)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps({"metadata": {"site_identity": {"site_id": "site-1"}}}),
        encoding="utf-8",
    )
    site_index_path.write_text(
        json.dumps(
            {
                "reference_id": "ref-1",
                "capture_id": "capture-1",
                "scene_id": "scene-1",
                "site_id": "site-1",
                "frame_id": "000001",
                "geometry_source": "local_sfm",
                "frame_uri": "gs://bucket/frames/000001.jpg",
                "depth_uri": "gs://bucket/depth/000001.png",
                "T_world_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsics": {
                    "fx": 1000.0,
                    "fy": 1000.0,
                    "cx": 320.0,
                    "cy": 240.0,
                    "width": 640,
                    "height": 480,
                },
                "reference_frame": "site_world",
                "camera_frame": "head_rgb",
                "translation_unit": "m",
                "reprojection_error_px": 0.5,
            }
        ) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GCS_ROOT", str(storage_root))

    release = threading.Event()

    def fake_synthesize_view(**kwargs):
        release.wait(timeout=5)
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color=(12, 140, 120)).save(output_path, format="PNG")
        return {
            "status": "completed",
            "output_path": str(output_path),
            "coverage_frac": 1.0,
            "reference_used": {"frame_id": "000001", "capture_id": "capture-1"},
        }

    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.synthesize.synthesize_view",
        fake_synthesize_view,
    )

    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    payload = _site_world_payload()
    payload["spec"]["canonical_package_uri"] = "gs://bucket/site-worlds/site-1/canonical.json"
    store.register_site_world_package(**payload)
    session = store.create_session(
        "siteworld-1",
        robot_profile_id="robot-1",
        task_id="task-1",
        scenario_id="scenario-1",
        start_state_id="start-1",
    )

    stepped = store.step_session(session["session_id"], action=[0.45, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0])
    assert stepped["status"] == "synthesizing"
    assert stepped["synthesis_status"] == "pending"
    assert "worldSnapshot" not in stepped["observation"]
    assert stepped["pose"]["z"] == -0.45
    assert stepped["pose"]["yaw"] == 0.35

    release.set()
    deadline = time.time() + 5
    state = store.session_state(session["session_id"])
    while state.get("synthesis_status") != "completed" and time.time() < deadline:
        time.sleep(0.05)
        state = store.session_state(session["session_id"])

    assert state["synthesis_status"] == "completed"
    assert state["observation"]["worldSnapshot"]["step"] == 1
    assert state["latest_render_source"] == "live_synthesis"
    assert store.render_bytes(session["session_id"], "head_rgb").startswith(b"\x89PNG")


def test_native_runtime_session_loads_site_reference_manifest_and_adapter_readiness(tmp_path: Path, monkeypatch) -> None:
    storage_root = tmp_path / "storage"
    capture_root = storage_root / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    site_root = storage_root / "bucket" / "sites" / "site-1" / "reference_memory"
    site_root.mkdir(parents=True, exist_ok=True)
    capture_root.mkdir(parents=True, exist_ok=True)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps({"metadata": {"site_identity": {"site_id": "site-1"}}}),
        encoding="utf-8",
    )
    (site_root / "site_reference_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "site_reference_database.v1",
                "site_id": "site-1",
                "total_reference_frames": 1,
                "capture_count": 1,
                "chunk_count": 1,
                "readiness": {"state": "ready", "blockers": []},
            }
        ),
        encoding="utf-8",
    )
    (site_root / "site_reference_index.jsonl").write_text(
        json.dumps(
            {
                "reference_id": "ref-1",
                "capture_id": "capture-1",
                "scene_id": "scene-1",
                "site_id": "site-1",
                "frame_id": "000001",
                "geometry_source": "local_sfm",
                "frame_uri": "gs://bucket/frames/000001.jpg",
                "depth_uri": "gs://bucket/depth/000001.png",
                "T_world_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
                "reference_frame": "site_world",
                "camera_frame": "head_rgb",
                "translation_unit": "m",
                "reprojection_error_px": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GCS_ROOT", str(storage_root))
    monkeypatch.setattr("blueprint_pipeline.native_runtime_backend._runtime_readiness", lambda: {"ready": False, "notes": ["native_model_not_provisioned"]})

    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    payload = _site_world_payload()
    payload["spec"]["canonical_package_uri"] = "gs://bucket/site-worlds/site-1/canonical.json"
    store.register_site_world_package(**payload)

    session = store.create_session("siteworld-1", robot_profile_id="robot-1")

    readiness = session["site_reference_runtime_adapter"]
    assert readiness["local_contract_ready"] is True
    assert readiness["runtime_adapter_ready"] is True
    assert readiness["non_arkit_geometry_state"] == "blocked"
    assert readiness["world_model_ready"] is False
    assert readiness["selected_runtime_path"] == "splat_only"
    assert "non_arkit_geometry_not_live_video_to_world" in readiness["blockers"]
    assert "native_model_not_provisioned" in readiness["backend_blockers"]
    assert Path(session["site_reference_runtime_artifact_path"]).is_file()


def test_native_runtime_step_endpoint_accepts_dict_actions(tmp_path: Path) -> None:
    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    app = create_runtime_app(backend=store, title="test-native-runtime")
    client = TestClient(app)
    client.post("/v1/site-worlds", json=_site_world_payload())
    session = client.post(
        "/v1/site-worlds/siteworld-1/sessions",
        json={
            "robot_profile_id": "robot-1",
            "task_id": "task-1",
            "scenario_id": "scenario-1",
            "start_state_id": "start-1",
        },
    )
    response = client.post(
        f"/v1/sessions/{session.json()['session_id']}/step",
        json={"action": {"type": "turn_left", "magnitude": 15}},
    )
    assert response.status_code == 200
    assert response.json()["step_count"] == 1


def test_render_bytes_promotes_pending_session_when_cosmos_frames_exist(tmp_path: Path, monkeypatch) -> None:
    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    payload = _site_world_payload()
    store.register_site_world_package(**payload)
    session = store.create_session(
        "siteworld-1",
        robot_profile_id="robot-1",
        task_id="task-1",
        scenario_id="scenario-1",
        start_state_id="start-1",
    )
    session_id = session["session_id"]
    state = store.session_state(session_id)
    state["status"] = "synthesizing"
    state["synthesis_status"] = "pending"
    state["pending_step_index"] = 1
    state["step_count"] = 1
    state["step_index"] = 1
    store._store_session_state(session_id, state)

    cosmos_frame = store._session_dir(session_id) / "cosmos" / "frames" / "frame_0001.png"
    cosmos_frame.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(180, 90, 60)).save(cosmos_frame, format="PNG")

    monkeypatch.setattr(
        store,
        "_ensure_cosmos_frames",
        lambda session_id_arg, site_world_id_arg: [cosmos_frame],
    )

    payload = store.render_bytes(session_id, "head_rgb")
    updated = store.session_state(session_id)

    assert payload.startswith(b"\x89PNG")
    assert updated["status"] == "running"
    assert updated["synthesis_status"] == "completed"
    assert updated["pending_step_index"] is None
    assert updated["latest_render_source"] == "cosmos_frames"
    assert updated["latest_render_path"] == str(cosmos_frame.resolve())


def test_render_bytes_falls_back_when_latest_render_is_unreadable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    store.register_site_world_package(**_site_world_payload())
    session = store.create_session(
        "siteworld-1",
        robot_profile_id="robot-1",
        task_id="task-1",
        scenario_id="scenario-1",
        start_state_id="start-1",
    )
    session_id = session["session_id"]
    render_path = store._session_dir(session_id) / "live_synth" / "step_00001.png"
    render_path.parent.mkdir(parents=True, exist_ok=True)
    render_path.write_bytes(b"not-an-image")
    state = store.session_state(session_id)
    state["latest_render_path"] = str(render_path)
    state["latest_render_source"] = "live_synthesis"
    state["step_count"] = 1
    store._store_session_state(session_id, state)

    monkeypatch.setattr(store, "_ensure_cosmos_frames", lambda *_args, **_kwargs: [])
    _patch_read_bytes_failure(monkeypatch, {render_path})

    payload = store.render_bytes(session_id, "head_rgb")

    assert payload.startswith(b"\x89PNG")


def test_render_bytes_falls_back_when_cosmos_frame_is_unreadable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    store.register_site_world_package(**_site_world_payload())
    session = store.create_session(
        "siteworld-1",
        robot_profile_id="robot-1",
        task_id="task-1",
        scenario_id="scenario-1",
        start_state_id="start-1",
    )
    session_id = session["session_id"]
    state = store.session_state(session_id)
    state["status"] = "synthesizing"
    state["synthesis_status"] = "pending"
    state["pending_step_index"] = 1
    state["step_count"] = 1
    state["step_index"] = 1
    store._store_session_state(session_id, state)

    cosmos_frame = store._session_dir(session_id) / "cosmos" / "frames" / "frame_0001.png"
    cosmos_frame.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(180, 90, 60)).save(cosmos_frame, format="PNG")
    monkeypatch.setattr(
        store,
        "_ensure_cosmos_frames",
        lambda session_id_arg, site_world_id_arg: [cosmos_frame],
    )
    _patch_read_bytes_failure(monkeypatch, {cosmos_frame})

    payload = store.render_bytes(session_id, "head_rgb")
    updated = store.session_state(session_id)

    assert payload.startswith(b"\x89PNG")
    assert updated["status"] == "running"
    assert updated["synthesis_status"] == "completed"
    assert updated["pending_step_index"] is None
    assert updated["latest_render_source"] == "cosmos_frames"
    assert updated["latest_render_path"] == str(cosmos_frame.resolve())


def test_runtime_control_and_media_endpoints_expose_chunked_rollout(tmp_path: Path) -> None:
    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    app = create_runtime_app(backend=store, title="test-native-runtime")
    client = TestClient(app)
    client.post("/v1/site-worlds", json=_site_world_payload())
    session = client.post(
        "/v1/site-worlds/siteworld-1/sessions",
        json={
            "robot_profile_id": "robot-1",
            "task_id": "task-1",
            "scenario_id": "scenario-1",
            "start_state_id": "start-1",
        },
    ).json()
    session_id = str(session["session_id"])

    chunk_path = store._chunk_video_path(session_id, "chunk-0000")
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(b"fake-mp4-chunk")
    state = store.session_state(session_id)
    state["rollout"] = {
        **store._rollout_defaults(),
        "status": "playing",
        "active_chunk_id": "chunk-0000",
        "buffered_chunk_ids": ["chunk-0000"],
        "chunks": [
            {
                "chunk_id": "chunk-0000",
                "chunk_index": 0,
                "status": "ready",
                "media_path": str(chunk_path.resolve()),
                "media_type": "video/mp4",
                "render_source": "bootstrap_prebuilt_video",
                "duration_ms": 1200,
            }
        ],
        "chunk_count": 1,
    }
    store._store_session_state(session_id, state)

    control = client.post(
        f"/v2/sessions/{session_id}/control",
        json={"seq": 1, "vx": 0.5, "yawRate": 0.25, "durationMs": 900},
    )
    assert control.status_code == 200
    assert control.json()["rollout"]["control_intent"]["seq"] == 1
    assert len(control.json()["rollout"]["trajectory_horizon"]) >= 3

    rollout = client.get(f"/v2/sessions/{session_id}/rollout")
    assert rollout.status_code == 200
    assert rollout.json()["control_intent"]["seq"] == 1
    assert rollout.json()["active_chunk_id"] == "chunk-0000"

    media = client.get(f"/v2/sessions/{session_id}/media")
    assert media.status_code == 200
    assert media.headers["content-type"].startswith("video/mp4")
    assert media.headers["x-blueprint-chunk-id"] == "chunk-0000"
    assert media.content == b"fake-mp4-chunk"

    # WS stream should emit typed messages (Blocker 2)
    with client.websocket_connect(f"/v1/sessions/{session_id}/stream") as ws:
        msg = ws.receive_json()
        # New format: {type: "state", payload: <session_state>}
        assert msg.get("type") == "state", f"expected type=state, got {msg.get('type')}"
        assert "payload" in msg
        assert "rollout" in msg["payload"]

    # drain_media_events should return empty list initially (no chunk worker ran)
    events = store.drain_media_events(session_id)
    assert isinstance(events, list)


def test_media_response_falls_back_when_chunk_file_is_unreadable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    store.register_site_world_package(**_site_world_payload())
    session = store.create_session(
        "siteworld-1",
        robot_profile_id="robot-1",
        task_id="task-1",
        scenario_id="scenario-1",
        start_state_id="start-1",
    )
    session_id = str(session["session_id"])

    chunk_path = store._chunk_video_path(session_id, "chunk-0000")
    chunk_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_path.write_bytes(b"fake-mp4-chunk")
    state = store.session_state(session_id)
    state["rollout"] = {
        **store._rollout_defaults(),
        "status": "playing",
        "active_chunk_id": "chunk-0000",
        "buffered_chunk_ids": ["chunk-0000"],
        "chunks": [
            {
                "chunk_id": "chunk-0000",
                "chunk_index": 0,
                "status": "ready",
                "media_path": str(chunk_path.resolve()),
                "media_type": "video/mp4",
                "render_source": "bootstrap_prebuilt_video",
                "duration_ms": 1200,
            }
        ],
        "chunk_count": 1,
    }
    store._store_session_state(session_id, state)
    _patch_read_bytes_failure(monkeypatch, {chunk_path})

    media = store.media_response(session_id, camera_id="head_rgb", chunk_id="chunk-0000")

    assert media["media_type"] == "image/png"
    assert media["content"].startswith(b"\x89PNG")
    assert media["headers"]["X-Blueprint-Render-Source"] == "placeholder_cosmos_pending"


def test_explorer_frame_bytes_falls_back_when_cached_frame_is_unreadable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    store.register_site_world_package(**_site_world_payload())
    session = store.create_session(
        "siteworld-1",
        robot_profile_id="robot-1",
        task_id="task-1",
        scenario_id="scenario-1",
        start_state_id="start-1",
    )
    session_id = str(session["session_id"])

    frame_path = store._session_dir(session_id) / "explorer_frames" / "head_rgb.png"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.write_bytes(b"not-an-image")
    monkeypatch.setattr(store, "_ensure_cosmos_frames", lambda *_args, **_kwargs: [])
    _patch_read_bytes_failure(monkeypatch, {frame_path})

    payload = store.explorer_frame_bytes(session_id, "head_rgb")

    assert payload.startswith(b"\x89PNG")


class _BranchRuntimeBackend:
    base_url = "http://runtime.test"
    ws_base_url = "ws://runtime.test"

    def __init__(self) -> None:
        self.failures: dict[str, Exception] = {}

    def _maybe_fail(self, method_name: str) -> None:
        exc = self.failures.get(method_name)
        if exc is not None:
            raise exc

    def runtime_info(self, *, service_version: str) -> dict[str, object]:
        self._maybe_fail("runtime_info")
        return {
            "service": "test-runtime",
            "runtime_kind": "native_world_model",
            "production_grade": False,
            "readiness": {"model_ready": True, "checkpoint_ready": True},
            "service_version": service_version,
        }

    def register_site_world_package(self, **_kwargs: object) -> dict[str, object]:
        self._maybe_fail("register_site_world_package")
        return {"site_world_id": "siteworld-1", "status": "registered"}

    def load_site_world(self, site_world_id: str) -> dict[str, object]:
        self._maybe_fail("load_site_world")
        return {"site_world_id": site_world_id, "status": "loaded"}

    def load_site_world_health(self, site_world_id: str) -> dict[str, object]:
        self._maybe_fail("load_site_world_health")
        return {"site_world_id": site_world_id, "status": "healthy"}

    def create_session(self, site_world_id: str, **kwargs: object) -> dict[str, object]:
        self._maybe_fail("create_session")
        return {
            "session_id": str(kwargs.get("session_id") or "session-1"),
            "site_world_id": site_world_id,
            "robot_profile_id": kwargs.get("robot_profile_id"),
        }

    def reset_session(self, session_id: str, **kwargs: object) -> dict[str, object]:
        self._maybe_fail("reset_session")
        return {"session_id": session_id, "status": "reset", "task_id": kwargs.get("task_id")}

    def step_session(self, session_id: str, *, action: object) -> dict[str, object]:
        self._maybe_fail("step_session")
        return {"session_id": session_id, "status": "stepped", "action": action}

    def session_state(self, session_id: str) -> dict[str, object]:
        self._maybe_fail("session_state")
        return {"session_id": session_id, "status": "active", "rollout": {"status": "idle"}}

    def control_session(self, session_id: str, *, control: dict[str, object]) -> dict[str, object]:
        self._maybe_fail("control_session")
        return {"session_id": session_id, "status": "controlled", "control": control}

    def render_bytes(self, _session_id: str, _camera_id: str) -> bytes:
        self._maybe_fail("render_bytes")
        return b"\x89PNG\r\n\x1a\nrender"

    def media_response(
        self,
        _session_id: str,
        *,
        camera_id: str,
        chunk_id: str | None,
    ) -> dict[str, object]:
        self._maybe_fail("media_response")
        return {
            "content": b"media",
            "media_type": "application/octet-stream",
            "headers": {"x-camera-id": camera_id, "x-chunk-id": chunk_id or ""},
        }

    def drain_media_events(self, _session_id: str) -> list[dict[str, object]]:
        self._maybe_fail("drain_media_events")
        return [{"event": "chunk_ready", "chunk_id": "chunk-0000"}]

    def explorer_render(self, session_id: str, **_kwargs: object) -> dict[str, object]:
        self._maybe_fail("explorer_render")
        return {"session_id": session_id, "frame_path": "/tmp/frame.png"}

    def explorer_frame_bytes(self, _session_id: str, _camera_id: str) -> bytes:
        self._maybe_fail("explorer_frame_bytes")
        return b"\x89PNG\r\n\x1a\nexplorer"


class _PrewarmFailRuntimeBackend(_BranchRuntimeBackend):
    def prewarm_runtime(self) -> dict[str, object]:
        raise RuntimeError("prewarm failed")


def _session_create_payload() -> dict[str, object]:
    return {
        "robot_profile_id": "robot-1",
        "task_id": "task-1",
        "scenario_id": "scenario-1",
        "start_state_id": "start-1",
    }


def _assert_failure_pair(
    client: TestClient,
    backend: _BranchRuntimeBackend,
    method_name: str,
    request_method: str,
    path: str,
    *,
    json_payload: dict[str, object] | None = None,
) -> None:
    request = getattr(client, request_method)
    backend.failures[method_name] = FileNotFoundError("missing")
    missing = request(path, json=json_payload) if json_payload is not None else request(path)
    assert missing.status_code == 404

    backend.failures[method_name] = ValueError("invalid request")
    invalid = request(path, json=json_payload) if json_payload is not None else request(path)
    assert invalid.status_code == 400

    backend.failures.clear()


def test_runtime_service_app_branches_use_backend_errors_and_success_paths() -> None:
    backend = _BranchRuntimeBackend()
    app = create_runtime_app(backend=backend, title="branch-runtime")
    client = TestClient(app)

    healthz = client.get("/healthz")
    assert healthz.status_code == 200
    assert healthz.json()["status"] == "ok"
    assert healthz.json()["model_ready"] is True

    invalid_registration = client.post("/v1/site-worlds", json={})
    assert invalid_registration.status_code == 400
    assert "spec + registration + health" in invalid_registration.json()["detail"]

    registration = client.post(
        "/v1/site-worlds",
        json={"spec": {}, "registration": {"site_world_id": "siteworld-1"}, "health": {}},
    )
    assert registration.status_code == 200
    assert registration.json()["health"]["status"] == "healthy"

    assert client.get("/v1/site-worlds/siteworld-1").json()["status"] == "loaded"
    assert client.get("/v1/site-worlds/siteworld-1/health").json()["status"] == "healthy"
    backend.failures["load_site_world_health"] = FileNotFoundError("missing")
    assert client.get("/v1/site-worlds/missing/health").status_code == 404
    backend.failures.clear()

    _assert_failure_pair(
        client,
        backend,
        "create_session",
        "post",
        "/v1/site-worlds/siteworld-1/sessions",
        json_payload=_session_create_payload(),
    )

    session = client.post(
        "/v1/site-worlds/siteworld-1/sessions",
        json=_session_create_payload(),
    )
    assert session.status_code == 200
    assert session.json()["session_id"] == "session-1"

    reset = client.post("/v1/sessions/session-1/reset", json={"task_id": "task-2"})
    assert reset.status_code == 200
    assert reset.json()["task_id"] == "task-2"

    control = client.post("/v2/sessions/session-1/control", json={"seq": 7})
    assert control.status_code == 200
    assert control.json()["control"]["seq"] == 7

    media = client.get("/v2/sessions/session-1/media/head_rgb?chunk_id=chunk-1")
    assert media.status_code == 200
    assert media.headers["x-camera-id"] == "head_rgb"
    assert media.headers["x-chunk-id"] == "chunk-1"

    rollout = client.get("/v2/sessions/session-1/rollout")
    assert rollout.status_code == 200
    assert rollout.json()["status"] == "idle"

    explorer = client.post(
        "/v1/sessions/session-1/explorer/render",
        json={"camera_id": "wrist_rgb", "pose": {"x": 1.0}, "refine_mode": "fast"},
    )
    assert explorer.status_code == 200
    assert explorer.json()["frame_path"] == "/tmp/frame.png"

    assert client.get("/v1/sessions/session-1/explorer/frame/wrist_rgb").status_code == 200

    _assert_failure_pair(
        client,
        backend,
        "reset_session",
        "post",
        "/v1/sessions/missing/reset",
        json_payload={},
    )
    _assert_failure_pair(
        client,
        backend,
        "step_session",
        "post",
        "/v1/sessions/missing/step",
        json_payload={"action": [0.0]},
    )
    _assert_failure_pair(client, backend, "session_state", "get", "/v1/sessions/missing/state")
    _assert_failure_pair(client, backend, "render_bytes", "get", "/v1/sessions/missing/render")
    _assert_failure_pair(
        client,
        backend,
        "control_session",
        "post",
        "/v2/sessions/missing/control",
        json_payload={},
    )
    _assert_failure_pair(client, backend, "media_response", "get", "/v2/sessions/missing/media")
    _assert_failure_pair(client, backend, "session_state", "get", "/v2/sessions/missing/rollout")
    _assert_failure_pair(
        client,
        backend,
        "explorer_render",
        "post",
        "/v1/sessions/missing/explorer/render",
        json_payload={},
    )
    _assert_failure_pair(
        client,
        backend,
        "explorer_frame_bytes",
        "get",
        "/v1/sessions/missing/explorer/frame/head_rgb",
    )


def test_runtime_service_app_prewarm_failure_is_not_swallowed() -> None:
    app = create_runtime_app(backend=_PrewarmFailRuntimeBackend(), title="branch-runtime")

    try:
        with TestClient(app):
            pass
    except RuntimeError as exc:
        assert str(exc) == "prewarm failed"
    else:  # pragma: no cover - startup errors must remain visible to operators.
        raise AssertionError("startup prewarm failure was swallowed")


class _FakeWebSocket:
    def __init__(self, *, disconnect_on_send: bool = False, close_error: bool = False) -> None:
        self.accepted = False
        self.closed = False
        self.disconnect_on_send = disconnect_on_send
        self.close_error = close_error
        self.messages: list[dict[str, object]] = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload: dict[str, object]) -> None:
        self.messages.append(payload)
        if self.disconnect_on_send:
            raise WebSocketDisconnect(code=1000)

    async def close(self) -> None:
        if self.close_error:
            raise RuntimeError("already closed")
        self.closed = True


def _stream_endpoint(app):
    return next(
        route.endpoint
        for route in app.routes
        if getattr(route, "path", "") == "/v1/sessions/{session_id}/stream"
    )


def test_runtime_service_app_websocket_error_and_disconnect_cleanup() -> None:
    missing_backend = _BranchRuntimeBackend()
    missing_backend.failures["session_state"] = FileNotFoundError("missing")
    missing_app = create_runtime_app(backend=missing_backend, title="branch-runtime")
    missing_socket = _FakeWebSocket()

    asyncio.run(_stream_endpoint(missing_app)(session_id="missing", websocket=missing_socket))

    assert missing_socket.accepted is True
    assert missing_socket.closed is True
    assert missing_socket.messages == [{"error": "session not found: missing"}]

    disconnect_backend = _BranchRuntimeBackend()
    disconnect_app = create_runtime_app(backend=disconnect_backend, title="branch-runtime")
    disconnect_socket = _FakeWebSocket(disconnect_on_send=True, close_error=True)

    asyncio.run(_stream_endpoint(disconnect_app)(session_id="session-1", websocket=disconnect_socket))

    assert disconnect_socket.accepted is True
    assert disconnect_socket.messages[0]["type"] == "state"


def test_runtime_control_prefers_truthful_preview_even_when_generation_owner_is_busy(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "storage"
    capture_root = storage_root / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    site_index_path = storage_root / "bucket" / "sites" / "site-1" / "reference_memory" / "site_reference_index.jsonl"
    site_index_path.parent.mkdir(parents=True, exist_ok=True)
    capture_root.mkdir(parents=True, exist_ok=True)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps({"metadata": {"site_identity": {"site_id": "site-1"}}}),
        encoding="utf-8",
    )
    site_index_path.write_text(
        json.dumps(
            {
                "reference_id": "ref-1",
                "capture_id": "capture-1",
                "scene_id": "scene-1",
                "site_id": "site-1",
                "frame_id": "000001",
                "frame_uri": "gs://bucket/frames/000001.jpg",
                "depth_uri": "gs://bucket/depth/000001.png",
                "T_world_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsics": {
                    "fx": 1000.0,
                    "fy": 1000.0,
                    "cx": 320.0,
                    "cy": 240.0,
                    "width": 640,
                    "height": 480,
                },
                "reference_frame": "site_world",
                "camera_frame": "head_rgb",
                "translation_unit": "m",
                "reprojection_error_px": 0.5,
            }
        ) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GCS_ROOT", str(storage_root))
    monkeypatch.setenv("NATIVE_WORLD_MODEL_OUTPUT_PROFILE", "swm_preview_refine")
    monkeypatch.setenv("NATIVE_WORLD_MODEL_ENABLE_COSMOS_REFINEMENT", "0")
    monkeypatch.setenv("NATIVE_WORLD_MODEL_TARGET_READY_CHUNKS", "1")

    modes: list[str] = []

    def fake_synthesize_view(**kwargs):
        modes.append(str(kwargs["mode"]))
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color=(40, 120, 180)).save(output_path, format="PNG")
        return {
            "status": "completed",
            "output_path": str(output_path),
            "mode": kwargs["mode"],
            "conditioning": {"previous_tail_path": None, "lookahead_ref_uris": []},
            "retrieved_references": [{"frame_id": "000001", "capture_id": "capture-1"}],
            "lookahead_references": [{"frame_id": "000001", "capture_id": "capture-1"}],
        }

    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.synthesize.synthesize_view",
        fake_synthesize_view,
    )

    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    payload = _site_world_payload()
    payload["spec"]["canonical_package_uri"] = "gs://bucket/site-worlds/site-1/canonical.json"
    store.register_site_world_package(**payload)
    session = store.create_session(
        "siteworld-1",
        robot_profile_id="robot-1",
        task_id="task-1",
        scenario_id="scenario-1",
        start_state_id="start-1",
    )
    session_id = session["session_id"]
    store._generation_owner_session_id = "other-session"
    monkeypatch.setattr(store, "_queue_chunk_generation", lambda queued_session_id: None)

    store.control_session(
        session_id,
        control={"seq": 1, "vx": 0.5, "yawRate": 0.25, "durationMs": 900},
    )
    store._generate_next_chunk(session_id)

    deadline = time.time() + 5
    state = store._load_session_state(session_id)
    while int(((state.get("rollout") or {}).get("chunk_count") or 0)) < 1 and time.time() < deadline:
        time.sleep(0.05)
        state = store._load_session_state(session_id)

    rollout = dict(state.get("rollout") or {})
    chunk = dict((rollout.get("chunks") or [])[0])

    assert modes == ["splat_only"]
    assert store._generation_owner_session_id == "other-session"
    assert rollout["presentation_mode"] == "truthful_preview"
    assert rollout["interactive_path_kind"] == "retrieval_depth_splat_preview"
    assert chunk["render_source"] == "truthful_preview_splat"
    assert chunk["refinement_status"] == "disabled"
    assert chunk["provenance"]["grounded"] is True
    assert "generation_worker_owned_by" not in str(state.get("failure_reason") or "")


def test_runtime_control_runs_async_cosmos_refinement_after_preview_chunk(
    tmp_path: Path,
    monkeypatch,
) -> None:
    storage_root = tmp_path / "storage"
    capture_root = storage_root / "bucket" / "scenes" / "scene-1" / "captures" / "capture-1"
    site_index_path = storage_root / "bucket" / "sites" / "site-1" / "reference_memory" / "site_reference_index.jsonl"
    site_index_path.parent.mkdir(parents=True, exist_ok=True)
    capture_root.mkdir(parents=True, exist_ok=True)
    (capture_root / "capture_descriptor.json").write_text(
        json.dumps({"metadata": {"site_identity": {"site_id": "site-1"}}}),
        encoding="utf-8",
    )
    site_index_path.write_text(
        json.dumps(
            {
                "reference_id": "ref-1",
                "capture_id": "capture-1",
                "scene_id": "scene-1",
                "site_id": "site-1",
                "frame_id": "000001",
                "frame_uri": "gs://bucket/frames/000001.jpg",
                "depth_uri": "gs://bucket/depth/000001.png",
                "T_world_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "intrinsics": {
                    "fx": 1000.0,
                    "fy": 1000.0,
                    "cx": 320.0,
                    "cy": 240.0,
                    "width": 640,
                    "height": 480,
                },
                "reference_frame": "site_world",
                "camera_frame": "head_rgb",
                "translation_unit": "m",
                "reprojection_error_px": 0.5,
            }
        ) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GCS_ROOT", str(storage_root))
    monkeypatch.setenv("NATIVE_WORLD_MODEL_OUTPUT_PROFILE", "swm_preview_refine")
    monkeypatch.setenv("NATIVE_WORLD_MODEL_ENABLE_COSMOS_REFINEMENT", "1")
    monkeypatch.setenv("NATIVE_WORLD_MODEL_TARGET_READY_CHUNKS", "1")
    monkeypatch.setattr("blueprint_pipeline.native_runtime_backend._runtime_readiness", lambda: {"ready": True})

    calls: list[str] = []

    def fake_synthesize_view(**kwargs):
        calls.append(str(kwargs["mode"]))
        output_path = Path(kwargs["output_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), color=(120, 80, 200)).save(output_path, format="PNG")
        return {
            "status": "completed",
            "output_path": str(output_path),
            "mode": kwargs["mode"],
            "conditioning": {"previous_tail_path": None, "lookahead_ref_uris": []},
            "retrieved_references": [{"frame_id": "000001", "capture_id": "capture-1"}],
            "lookahead_references": [{"frame_id": "000001", "capture_id": "capture-1"}],
        }

    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.synthesize.synthesize_view",
        fake_synthesize_view,
    )
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_inference.load_cosmos_model",
        lambda *args, **kwargs: {"backend": "fake_cosmos"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_inference.describe_cosmos_model",
        lambda model: {"backend": "fake_cosmos"},
    )

    store = NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://127.0.0.1:8791",
            ws_base_url="ws://127.0.0.1:8791",
        )
    )
    payload = _site_world_payload()
    payload["spec"]["canonical_package_uri"] = "gs://bucket/site-worlds/site-1/canonical.json"
    store.register_site_world_package(**payload)
    session = store.create_session(
        "siteworld-1",
        robot_profile_id="robot-1",
        task_id="task-1",
        scenario_id="scenario-1",
        start_state_id="start-1",
    )
    session_id = session["session_id"]

    store.control_session(
        session_id,
        control={"seq": 1, "vx": 0.5, "yawRate": 0.25, "durationMs": 900},
    )

    deadline = time.time() + 5
    state = store._load_session_state(session_id)
    rollout = dict(state.get("rollout") or {})
    chunk = dict((rollout.get("chunks") or [])[0]) if rollout.get("chunks") else {}
    events: list[dict[str, object]] = []
    while chunk.get("refinement_status") != "completed" and time.time() < deadline:
        time.sleep(0.05)
        events.extend(store.drain_media_events(session_id))
        state = store._load_session_state(session_id)
        rollout = dict(state.get("rollout") or {})
        chunk = dict((rollout.get("chunks") or [])[0]) if rollout.get("chunks") else {}

    events.extend(store.drain_media_events(session_id))
    event_names = [str(event.get("event") or "") for event in events]
    event_deadline = time.time() + 5
    while "chunk_refinement_ready" not in event_names and time.time() < event_deadline:
        time.sleep(0.05)
        events.extend(store.drain_media_events(session_id))
        event_names = [str(event.get("event") or "") for event in events]
    ready_event = next(event for event in events if event.get("event") == "chunk_refinement_ready")
    state = store._load_session_state(session_id)
    rollout = dict(state.get("rollout") or {})
    chunks = [dict(item) for item in rollout.get("chunks") or []]
    chunk = next(item for item in chunks if item.get("chunk_id") == ready_event.get("chunk_id"))

    assert calls[:2] == ["splat_only", "cosmos_i2w"]
    assert chunk["refinement_status"] == "completed"
    assert Path(chunk["refined_media_path"]).is_file()
    assert chunk["provenance"]["refinement_status"] == "completed"
    assert "chunk_refinement_ready" in event_names
