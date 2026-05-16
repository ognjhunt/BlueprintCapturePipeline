from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

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
    assert readiness["non_arkit_geometry_state"] == "degraded"
    assert readiness["world_model_ready"] is False
    assert readiness["selected_runtime_path"] == "splat_only"
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

    store.control_session(
        session_id,
        control={"seq": 1, "vx": 0.5, "yawRate": 0.25, "durationMs": 900},
    )

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
    while chunk.get("refinement_status") != "completed" and time.time() < deadline:
        time.sleep(0.05)
        state = store._load_session_state(session_id)
        rollout = dict(state.get("rollout") or {})
        chunk = dict((rollout.get("chunks") or [])[0]) if rollout.get("chunks") else {}

    events = store.drain_media_events(session_id)
    event_names = [str(event.get("event") or "") for event in events]

    assert calls[:2] == ["splat_only", "cosmos_i2w"]
    assert chunk["refinement_status"] == "completed"
    assert Path(chunk["refined_media_path"]).is_file()
    assert chunk["provenance"]["refinement_status"] == "completed"
    assert "chunk_refinement_ready" in event_names
