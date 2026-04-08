from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from blueprint_pipeline.native_runtime_backend import (
    NativeRuntimeConfig,
    NativeWorldModelRuntimeStore,
    _find_cosmos_repo,
    _optional_existing_path,
)
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


def test_optional_existing_path_ignores_permission_denied(monkeypatch) -> None:
    inaccessible = Path("/tmp/inaccessible-path").resolve()

    def fake_exists(self: Path) -> bool:
        if self == inaccessible:
            raise PermissionError("denied")
        return True

    monkeypatch.setattr(Path, "exists", fake_exists)

    assert _optional_existing_path(str(inaccessible)) is None


def test_find_cosmos_repo_ignores_permission_denied_candidates(monkeypatch, tmp_path: Path) -> None:
    blocked = tmp_path / "blocked-cosmos"
    ready = tmp_path / "ready-cosmos"
    (ready / "examples").mkdir(parents=True)
    (ready / ".venv" / "bin").mkdir(parents=True)
    (ready / "examples" / "inference.py").write_text("# test\n", encoding="utf-8")
    (ready / ".venv" / "bin" / "python").write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    original_is_file = Path.is_file

    def fake_is_file(self: Path) -> bool:
        if str(self).startswith(str(blocked)):
            raise PermissionError("denied")
        return original_is_file(self)

    monkeypatch.setattr("blueprint_pipeline.native_runtime_backend._COSMOS_REPO_CANDIDATE_PATHS", [str(blocked), str(ready)])
    monkeypatch.setattr(Path, "is_file", fake_is_file)

    repo = _find_cosmos_repo()

    assert repo is not None
    assert repo[0] == ready
    assert repo[1] == ready / ".venv" / "bin" / "python"


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

    stepped = store.step_session(session["session_id"], action=[0.45, 0.0, 0.35, 0.0, 0.0, 0.0, 1.0])
    assert stepped["status"] == "synthesizing"
    assert stepped["synthesis_status"] == "pending"
    assert "worldSnapshot" not in stepped["observation"]
    assert stepped["pose"]["z"] == -0.45
    assert stepped["pose"]["yaw"] == 0.35

    release.set()
    deadline = time.time() + 10
    state = store.session_state(session["session_id"])
    while state.get("synthesis_status") != "completed" and time.time() < deadline:
        time.sleep(0.05)
        state = store.session_state(session["session_id"])

    assert state["synthesis_status"] == "completed"
    assert state["observation"]["worldSnapshot"]["step"] == 1
    assert state["latest_render_source"] == "live_synthesis"
    assert store.render_bytes(session["session_id"], "head_rgb").startswith(b"\x89PNG")


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

    deadline = time.time() + 10
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

    deadline = time.time() + 10
    state = store._load_session_state(session_id)
    rollout = dict(state.get("rollout") or {})
    chunk = dict((rollout.get("chunks") or [])[0]) if rollout.get("chunks") else {}
    while chunk.get("refinement_status") != "completed" and time.time() < deadline:
        time.sleep(0.05)
        state = store._load_session_state(session_id)
        rollout = dict(state.get("rollout") or {})
        chunk = dict((rollout.get("chunks") or [])[0]) if rollout.get("chunks") else {}

    event_deadline = time.time() + 10
    events = store.drain_media_events(session_id)
    event_names = [str(event.get("event") or "") for event in events]
    while "chunk_refinement_ready" not in event_names and time.time() < event_deadline:
        time.sleep(0.05)
        events = store.drain_media_events(session_id)
        event_names = [str(event.get("event") or "") for event in events]

    assert {"splat_only", "cosmos_i2w"}.issubset(set(calls))
    assert chunk["refinement_status"] == "completed"
    assert Path(chunk["refined_media_path"]).is_file()
    assert chunk["provenance"]["refinement_status"] == "completed"
    assert "chunk_refinement_ready" in event_names


def test_runtime_control_runs_async_cosmos_refinement_without_ffmpeg_uses_png_media(
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
    original_which = shutil.which

    def fake_which(command: str) -> str | None:
        if command == "ffmpeg":
            return None
        return original_which(command)

    monkeypatch.setattr("blueprint_pipeline.native_runtime_backend.shutil.which", fake_which)
    monkeypatch.setattr("blueprint_pipeline.native_runtime_backend._runtime_readiness", lambda: {"ready": True})
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_inference.load_cosmos_model",
        lambda *args, **kwargs: {"backend": "fake_cosmos"},
    )
    monkeypatch.setattr(
        "blueprint_pipeline.synthesis.cosmos_inference.describe_cosmos_model",
        lambda model: {"backend": "fake_cosmos"},
    )

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

    deadline = time.time() + 10
    state = store._load_session_state(session_id)
    rollout = dict(state.get("rollout") or {})
    chunk = dict((rollout.get("chunks") or [])[0]) if rollout.get("chunks") else {}
    while chunk.get("refinement_status") != "completed" and time.time() < deadline:
        time.sleep(0.05)
        state = store._load_session_state(session_id)
        rollout = dict(state.get("rollout") or {})
        chunk = dict((rollout.get("chunks") or [])[0]) if rollout.get("chunks") else {}

    event_deadline = time.time() + 10
    events = store.drain_media_events(session_id)
    event_names = [str(event.get("event") or "") for event in events]
    while "chunk_refinement_ready" not in event_names and time.time() < event_deadline:
        time.sleep(0.05)
        events = store.drain_media_events(session_id)
        event_names = [str(event.get("event") or "") for event in events]
    media = store.media_response(session_id, camera_id="head_rgb", chunk_id=None)

    assert {"splat_only", "cosmos_i2w"}.issubset(set(calls))
    assert chunk["media_type"] == "image/png"
    assert chunk["media_path"].endswith(".png")
    assert Path(chunk["media_path"]).is_file()
    assert Path(chunk["refined_media_path"]).is_file()
    assert Path(chunk["refined_media_path"]).suffix == ".png"
    assert chunk["refinement_status"] == "completed"
    assert chunk["provenance"]["refinement_status"] == "completed"
    assert chunk["provenance"]["refinement_render_source"] == "cosmos_async_refinement"
    assert media["media_type"] == "image/png"
    assert media["content"].startswith(b"\x89PNG")
    assert "chunk_refinement_ready" in event_names
