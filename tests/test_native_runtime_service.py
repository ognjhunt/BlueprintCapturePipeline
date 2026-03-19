from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

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
