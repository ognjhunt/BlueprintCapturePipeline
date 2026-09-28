from __future__ import annotations

import base64
import io
import json
from types import SimpleNamespace

import pytest
from PIL import Image, PngImagePlugin

from blueprint_pipeline import controlled_policy_observations as controlled
from blueprint_pipeline import robot_eval_execution as execution
from tests.test_company_policy_container_contract_v2 import _contract


def inputs():
    contract = _contract()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private_scene_path", "/private/site/scene.usd")
    image = io.BytesIO()
    Image.new("RGB", (320, 180)).save(image, format="PNG", pnginfo=metadata)
    cameras = {camera["name"]: image.getvalue() for camera in contract["observation_schema"]["cameras"]}
    return dict(
        contract=contract, request_id="a" * 32, prompt="Pick up the cup",
        camera_pngs={**cameras, "secret_mesh": b"mesh"},
        robot_state={"joint_position": [0.0] * 7, "gripper_position": [0.5],
                     "scene_uri": "gs://private-site/scene.usd", "scoring_harness": "private"},
    )


def test_constructs_minimum_wire_and_strips_image_metadata():
    wire = controlled.project_controlled_observation(**inputs())
    encoded = json.dumps(wire)
    assert set(wire) == {"schema_version", "request_id", "synthetic", "prompt", "cameras", "state"}
    assert "scene_uri" not in encoded and "scoring_harness" not in encoded
    assert "calibration" not in encoded and "secret_mesh" not in encoded
    for camera in wire["cameras"].values():
        frame = base64.b64decode(camera["data_base64"])
        assert b"private_scene_path" not in frame and b"scene.usd" not in frame
        with Image.open(io.BytesIO(frame)) as image:
            assert image.getpixel((0, 0)) == (0, 0, 0)


@pytest.mark.parametrize("state", [[], [0.0] * 6, [float("nan")] * 7, [True] * 7])
def test_rejects_missing_or_malformed_robot_state(state):
    values = inputs()
    values["robot_state"]["joint_position"] = state
    with pytest.raises(ValueError, match="controlled_policy_state"):
        controlled.project_controlled_observation(**values)


def test_remote_call_has_no_redirects_and_refuses_policy_outcome_claim(monkeypatch):
    calls = []
    def fetch(endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        return SimpleNamespace(status=200, body=json.dumps({"actions": [[0.0] * 8] * 15}).encode())
    monkeypatch.setattr(controlled, "fetch_bounded_https", fetch)
    actions, wire, receipt = controlled.call_customer_hosted_policy(
        endpoint="https://policy.example/v1/actions", allowed_origins=("https://policy.example",), **inputs(),
    )
    assert actions["actions"] == [[0.0] * 8] * 15
    assert calls[0][1]["max_redirects"] == 0
    assert calls[0][1]["allowed_origins"] == ("https://policy.example",)
    assert json.loads(calls[0][1]["data"]) == wire
    assert receipt["task_success_proven"] is False
    monkeypatch.setattr(controlled, "fetch_bounded_https", lambda *a, **k: SimpleNamespace(
        status=200, body=b'{"actions": [], "success": true}',
    ))
    with pytest.raises(ValueError, match="response_shape_invalid"):
        controlled.call_customer_hosted_policy(
            endpoint="https://policy.example/v1/actions", allowed_origins=("https://policy.example",), **inputs(),
        )


def test_skill_sequence_is_intent_with_unknown_success(tmp_path):
    sequence = [{"skill_id": "find_cup"}, {"skill_id": "pick_cup"}, {"skill_id": "place_cup"}]
    replay = execution._replay_reference_payload(
        modality="high_level_skill_trace", payload={"ordered_skill_sequence": sequence},
        capture_root=tmp_path, job_dir=tmp_path,
    )
    attempts = execution._normalize_policy_attempts(
        payload=replay, modality="high_level_skill_trace", observations=[{"observation_id": "obs"}],
        generated_at="2026-09-28T00:00:00Z",
    )
    assert attempts[0]["skills"] == sequence
    assert attempts[0]["actions"] == []
    assert attempts[0]["status"] == "submitted_unexecuted"
    assert attempts[0]["success"] is None
    assert attempts[0]["metrics"]["execution_evidence_available"] is False
    assert execution._policy_run_coverage(attempts, ["run-1"])["covered_scenario_eval_run_count"] == 0
