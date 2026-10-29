"""Beta L5 Dream contract: wire, controlled observations and no live inference."""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from blueprint_pipeline.dreamlabs_policy_client import (
    CAMERA_ROLES,
    ENDPOINT_URL,
    GATEWAY_ORIGIN,
    INTERFACE_ID,
    METADATA_URL,
    SDK_VERSION,
    SDK_WHEEL_SHA256,
    DreamLabsPolicyClient,
    DreamLabsPolicySpec,
    discover_model_metadata,
    provider_metadata_digest,
)
from blueprint_pipeline.policy_request_evidence import persist_request_evidence, restore_request, validate_request_evidence

MODEL = {"id": "dreamlabs/dream-1-beta", "owner": "public", "base_model": "dreamlabs/dream-1-beta",
         "embodiment": "yam", "servable": True, "created_at": None, "cameras": list(CAMERA_ROLES),
         "embodiment_config": None, "parts_withheld": False}
SPEC = DreamLabsPolicySpec(policy_id="team-dream-exact", endpoint_url=ENDPOINT_URL, metadata_url=METADATA_URL,
                           allowed_origin=GATEWAY_ORIGIN, model_identity_digest=provider_metadata_digest(MODEL),
                           embodiment_id="piper_bimanual", action_chunk_rows=48, open_loop_horizon=30,
                           model_id=MODEL["id"], camera_names=["overhead", "left_wrist", "right_wrist"],
                           camera_roles={"overhead": "overhead", "wrist_l": "left_wrist", "wrist_r": "right_wrist"})
# A real, committed development-only review image exercises JPEG conversion;
# the warehouse's .jpg is intentionally an ASCII placeholder, not image data.
FIXTURE = (Path(__file__).parents[1] / "docs/experiments/policy_ranking_successor_experiment_20260727/"
           "exact_main_attempt_v11/first_mid_last.png")


def observation() -> dict[str, Any]:
    with Image.open(FIXTURE) as source:
        image = np.asarray(source.convert("RGB").crop((0, 0, 640, 528))).copy()
    return {"images": {name: image.copy() for name in SPEC.camera_names},
            "prompt": "Move the box to the marked area.",
            "state": {"arm_names": ["left", "right"],
                      "joint_names": ["r1", "r2", "l1", "l2"], "joint_position": [0.1, 0.2, 0.3, 0.4],
                      "joint_velocity": [0.0] * 4,
                      "arm_joint_names": {"left": ["l1", "l2"], "right": ["r1", "r2"]},
                      "arm_eef_pose": {"left": {"position_m": [0.1, 0.2, 0.3], "orientation_xyzw": [0, 0, 0, 1]},
                                       "right": {"position_m": [0.4, 0.5, 0.6], "orientation_xyzw": [0, 0, 1, 0]}},
                      # Reverse tool order ensures the named association, not
                      # an accidental left/right packed-vector convention.
                      "gripper_names": ["right_gripper", "left_gripper"], "gripper_position": [0.2, 0.8],
                      "arm_gripper_names": {"left": "left_gripper", "right": "right_gripper"}}}


def response(*, session: str = "session-one", code: str = "DA-0001", rows: int = 2) -> dict[str, Any]:
    per_key = {}
    for arm in ("left", "right"):
        per_key.update({f"action.{arm}_eef_xyz": [[0.1, 0.2, 0.3]] * rows,
                        f"action.{arm}_eef_rot6": [[1, 0, 0, 0, 1, 0]] * rows,
                        f"action.{arm}_grip": [[0.25 if arm == "left" else 0.75]] * rows,
                        f"action.{arm}_arm": None})
    return {"session_id": session, "status": {"code": code, "reason": "new"},
            "actions": {"per_key": per_key}, "timing": {"inference_ms": 25}}


class Service:
    def __init__(self, *, models=None, replies=None, release=None):
        self.calls: list[dict[str, Any]] = []
        self.models = [copy.deepcopy(MODEL)] if models is None else models
        self.replies = list(replies) if replies is not None else [response()]
        self.release = {"status": {"code": "DA-0000"}} if release is None else release
        self.before_post = None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["url"] == METADATA_URL:
            value = self.models
        elif kwargs["url"].endswith("/end_session"):
            value = self.release
        else:
            if self.before_post:
                self.before_post(kwargs)
            value = self.replies.pop(0)
        if isinstance(value, Exception):
            raise value
        return value if isinstance(value, bytes) else json.dumps(value).encode()


def client(service=None, *, spec=SPEC):
    return DreamLabsPolicyClient(spec=spec, bearer_token="test-token-never-retain", transport=service or Service())


def test_discovery_uses_real_models_endpoint_exact_id_and_metadata_not_weights():
    service = Service()
    found = discover_model_metadata(model_id=MODEL["id"], bearer_token="test-token", transport=service)
    assert found["provider_model_record"] == MODEL
    assert found["model_identity_digest"] == SPEC.model_identity_digest
    assert found["identity_basis"] == "provider_metadata" and found["weights_identity_verified"] is False
    assert found["candidate_inference_performed"] is False
    assert [(call["url"], call["method"], call["body"]) for call in service.calls] == [(METADATA_URL, "GET", None)]
    with pytest.raises(ValueError, match="exact_account_model_unavailable"):
        discover_model_metadata(model_id="dreamlabs/dream-1", bearer_token="test-token", transport=service)


@pytest.mark.parametrize("models", [
    {"models": [MODEL]}, [MODEL, MODEL], [{**MODEL, "servable": False}], [{**MODEL, "embodiment": "custom"}],
    [{**MODEL, "cameras": ["front"]}], [{**MODEL, "parts_withheld": True}],
    [{**MODEL, "embodiment_config": {"state": {}}}], [{**MODEL, "unreviewed_extension": "value"}],
])
def test_discovery_refuses_undisclosed_or_unavailable_contracts(models):
    with pytest.raises(ValueError):
        client(Service(models=models)).preflight_readiness()


def test_readiness_is_no_inference_and_rechecks_retained_provider_metadata():
    service = Service()
    policy = client(service)
    ready = policy.preflight_readiness()
    assert ready["identity_verified"] and ready["candidate_inference_performed"] is False
    assert ready["server_metadata"]["sdk_recording_enabled"] is False
    assert ready["server_metadata"]["sdk_wheel_sha256"] == SDK_WHEEL_SHA256
    service.models[0]["base_model"] = "different-release"
    with pytest.raises(ValueError, match="metadata_drift"):
        policy.preflight_readiness()
    assert all(call["method"] == "GET" for call in service.calls)


@pytest.mark.parametrize("change", [
    {"endpoint_url": "https://api.dreamlabs.ai/v1/get_action?key=secret"},
    {"endpoint_url": "https://user:secret@api.dreamlabs.ai/v1/get_action"},
    {"endpoint_url": "https://127.0.0.1/v1/get_action"}, {"allowed_origin": "https://attacker.example"},
    {"metadata_url": "https://api.dreamlabs.ai/v1/metadata"}, {"model_id": ""}, {"model_identity_digest": "unknown"},
    {"record": True}, {"record": 0}, {"sdk_version": "1.0.2"}, {"action_hz": 60},
    {"robot": "yam"}, {"observation_schema": "station-guessed"}, {"action_space": "joint_position"},
    {"action_chunk_rows": 49}, {"open_loop_horizon": 49}, {"camera_roles": {}},
    {"camera_roles": {"overhead": "overhead", "wrist_l": "left_wrist", "wrist_r": "left_wrist"}},
])
def test_protocol_binding_rejects_unapproved_urls_recording_and_schema_changes(change):
    with pytest.raises(ValueError):
        client(spec=replace(SPEC, **change))


def test_exact_published_wire_captured_before_post_and_native_pose_conversion(tmp_path):
    service = Service()
    policy = client(service)
    seen = []

    def retain(evidence):
        assert not any(call["method"] == "POST" for call in service.calls)
        seen.append(validate_request_evidence(evidence))
        persist_request_evidence(evidence, root=tmp_path, episode_id="development-episode", query_index=0)

    policy.bind_request_evidence_sink(retain)
    obs = observation()
    assert policy.observe_frame(obs, 0.0) is True
    actions = policy.infer(obs)
    post = next(call for call in service.calls if call["method"] == "POST")
    wire = json.loads(post["body"])
    assert list(wire) == ["checkpoint", "session_id", "reset", "observation"]
    assert wire["checkpoint"] == MODEL["id"] and wire["session_id"] is None and wire["reset"] is False
    assert post["body"] == json.dumps(wire, separators=(",", ":")).encode()
    assert post["headers"]["Dream-Client-Version"] == SDK_VERSION
    assert post["headers"]["Authorization"] == "Bearer test-token-never-retain"
    assert post["headers"]["Content-Length"] == str(len(post["body"]))
    assert post["headers"]["User-Agent"].startswith("dream-client/1.0.3 ")
    assert len(post["headers"]["User-Agent"].split()[1]) == 8
    sent = wire["observation"]
    assert sent["embodiment"] == "other_embodiment"
    assert sent["state.left_arm"] == [0.0] * 6 and sent["state.right_arm"] == [0.0] * 6
    assert sent["state.left_eef_xyz"] == [0.1, 0.2, 0.3]
    assert sent["state.right_eef_xyz"] == [0.4, 0.5, 0.6]
    assert sent["state.left_eef_rot6"] == [1, 0, 0, 0, 1, 0]
    assert sent["state.right_eef_rot6"] == [-1, 0, 0, 0, -1, 0]
    assert sent["state.left_grip"] == pytest.approx([0.2])
    assert sent["state.right_grip"] == pytest.approx([0.8])
    assert "_record" not in sent and "_record" not in wire  # Local-only SDK control; no recorder exists.
    assert all(len(sent["video." + role]) == 1 for role in CAMERA_ROLES)
    assert base64.b64decode(seen[0]["scientific_wire_bytes_base64"]) == post["body"]
    assert restore_request(seen[0]["request"]) == wire
    assert seen[0]["serialization_verified"] is True
    assert actions["schema_version"] == "blueprint.named_arm_pose_chunk.v1"
    assert actions["arm_names"] == ["left", "right"] and actions["frame"] == "arm_base"
    assert actions["positions_m"] == [[[0.1, 0.2, 0.3]] * 2] * 2
    assert actions["orientations_xyzw"] == [[[0, 0, 0, 1]] * 2] * 2
    assert actions["gripper_closures"] == [[0.75, 0.25]] * 2
    assert policy.action_hz == 20 and policy.observation_hz == 5  # Native owns hold3, no client resampling.
    assert policy.interface_id == INTERFACE_ID
    evidence = policy.last_inference_evidence()
    assert evidence["action_payload_returned"] and evidence["actions_extracted"]
    assert evidence["raw_vendor_action_response"]["per_key"]["action.left_grip"] == [[0.25]] * 2
    assert "test-token-never-retain" not in json.dumps([policy.evidence_summary(), seen])


def test_exact_decoded_jpeg_frames_and_hashes_are_available_for_lossless_media():
    policy = client()
    obs = observation()
    policy.observe_frame(obs, 0.0)
    policy.infer(obs)
    wire = restore_request(policy.last_request_evidence()["request"])
    frames = policy.last_policy_input_frames()
    assert len(frames) == 3
    for frame in frames:
        jpeg = base64.b64decode(wire["observation"]["video." + frame["provider_role"]][0])
        with Image.open(io.BytesIO(jpeg)) as decoded:
            actual_rgb = np.asarray(decoded.convert("RGB"))
        assert frame["simulation_time_s"] == 0.0 and frame["rgb"].shape == (176, 320, 3)
        assert np.array_equal(frame["rgb"], actual_rgb)
        assert frame["image_sha256"] == "sha256:" + hashlib.sha256(jpeg).hexdigest()
        assert frame["decoded_rgb_sha256"] == "sha256:" + hashlib.sha256(actual_rgb.tobytes()).hexdigest()
        frame["rgb"][:] = 0
    assert any(policy.last_policy_input_frames()[0]["rgb"].ravel())


def test_nine_frame_history_uses_real_timestamps_and_no_query_padding():
    service = Service(replies=[response(), response(code="DA-0000")])
    policy = client(service)
    obs = observation()
    accepted = []
    for tick in range(121):
        if policy.observe_frame(obs, tick / 60):
            accepted.append(tick)
    assert accepted == list(range(0, 121, 12))
    assert policy.observe_frame(obs, 2.0) is False
    obs["state"]["arm_eef_pose"]["left"]["position_m"][0] = 0.123
    policy.infer(obs)
    first = service.calls[-1]["body"]
    assert len(json.loads(first)["observation"]["video.overhead"]) == 9
    assert json.loads(first)["observation"]["state.left_eef_xyz"][0] == 0.123
    assert policy.last_inference_evidence()["camera_history_timestamps_s"] == pytest.approx([tick / 60 for tick in accepted[-9:]])
    policy.infer(obs)
    second = json.loads(service.calls[-1]["body"])
    assert second["session_id"] == "session-one"
    assert second["observation"]["video.overhead"] == json.loads(first)["observation"]["video.overhead"]
    with pytest.raises(ValueError, match="timestamp_invalid"):
        policy.observe_frame(obs, 1.0)


def test_no_camera_history_or_retention_failure_prevents_inference_transport():
    service = Service()
    policy = client(service)
    with pytest.raises(ValueError, match="history_missing"):
        policy.infer(observation())
    assert not service.calls
    policy.observe_frame(observation(), 0.0)

    def refuse(_evidence):
        raise OSError("local retention refused")

    policy.bind_request_evidence_sink(refuse)
    with pytest.raises(OSError, match="retention refused"):
        policy.infer(observation())
    assert all(call["method"] == "GET" for call in service.calls)
    assert policy.candidate_policy_queried is False
    assert policy.last_policy_input_frames() == []


@pytest.mark.parametrize("mutate", [
    lambda obs: obs["state"].pop("arm_gripper_names"),
    lambda obs: obs["state"]["arm_gripper_names"].update(left="right_gripper"),
    lambda obs: obs["state"]["arm_joint_names"].update(left=["r1", "r2"]),
    lambda obs: obs["state"]["arm_eef_pose"]["left"].update(orientation_xyzw=[0, 0, 0, 2]),
    lambda obs: obs["state"]["arm_eef_pose"]["right"].update(position_m=[float("nan"), 0, 0]),
    lambda obs: obs["state"].update(gripper_position=[-1, 0]),
    lambda obs: obs["state"].update(joint_position=[float("inf")] * 4),
    lambda obs: obs["images"].pop("right_wrist"),
    lambda obs: obs.update(prompt=""),
])
def test_bad_named_proprio_camera_binding_or_prompt_fails_before_transport(mutate):
    service = Service()
    policy = client(service)
    obs = observation()
    mutate(obs)
    with pytest.raises(ValueError):
        policy.observe_frame(obs, 0.0)
    assert not service.calls


@pytest.mark.parametrize("bad", [
    {"status": {"code": "DA-0000"}, "actions": {}},
    {"session_id": "a", "status": {"code": "DA-0000"}, "actions": {"q_r_chunk": [[0] * 7]}},
    b'{"status":{"code":"DA-0000"},"status":{"code":"DA-0000"}}',
    b'{"status":{"code":"DA-0000"},"actions":NaN}',
    b'{"status":{"code":"DA-0000"},"actions":1e999}',
])
def test_malformed_or_joint_only_response_is_not_executed(bad):
    policy = client(Service(replies=[bad]))
    policy.observe_frame(observation(), 0.0)
    with pytest.raises(ValueError):
        policy.infer(observation())
    assert policy.last_inference_evidence()["actions_extracted"] is False
    with pytest.raises(ValueError, match="faulted"):
        policy.infer(observation())


@pytest.mark.parametrize("part,bad", [
    ("eef_xyz", [[0, 0]]), ("eef_rot6", [[0] * 6]),
    ("eef_rot6", [[1, 0, 0, 2, 0, 0]]), ("grip", [[1.01], [0.2]]),
    ("grip", [[float("nan")]]), ("eef_xyz", [[0, 0, 0]] * 49),
])
def test_action_shapes_limits_rotation_and_crossarm_lengths_are_strict(part, bad):
    reply = response()
    reply["actions"]["per_key"]["action.left_" + part] = bad
    policy = client(Service(replies=[reply]))
    policy.observe_frame(observation(), 0.0)
    with pytest.raises(ValueError):
        policy.infer(observation())


@pytest.mark.parametrize("code", ["DA-0000", "DA-0001"])
def test_session_replacement_never_returns_new_episode_actions(code):
    service = Service(replies=[response(), response(session="replacement", code=code)])
    policy = client(service)
    policy.observe_frame(observation(), 0.0)
    policy.infer(observation())
    with pytest.raises(ValueError, match="session_replaced_mid_episode"):
        policy.infer(observation())
    assert policy.last_inference_evidence()["actions_extracted"] is False
    policy.reset()
    released = {json.loads(call["body"])["session_id"] for call in service.calls if call["url"].endswith("/end_session")}
    assert released == {"session-one", "replacement"}
    with pytest.raises(ValueError, match="history_missing"):
        policy.infer(observation())


def test_reset_and_close_release_exact_session_with_bounded_timeout_and_no_retry():
    service = Service(replies=[response(), response(session="next-session")])
    policy = client(service)
    policy.observe_frame(observation(), 0.0)
    policy.infer(observation())
    policy.reset_episode()
    release = service.calls[-1]
    assert release["url"] == GATEWAY_ORIGIN + "/v1/end_session" and release["timeout_seconds"] == 2
    assert json.loads(release["body"]) == {"checkpoint": MODEL["id"], "session_id": "session-one"}
    assert policy.evidence_summary()["session_release_status"] == "confirmed"
    assert policy.observe_frame(observation(), 0.0)
    policy.infer(observation())
    assert json.loads(service.calls[-1]["body"])["session_id"] is None
    policy.close()
    assert policy.evidence_summary()["last_inference_evidence"]["actions_extracted"] is True
    calls = len(service.calls)
    policy.close()
    assert len(service.calls) == calls
    with pytest.raises(ValueError, match="closed"):
        policy.observe_frame(observation(), 0.0)


def test_transport_failure_is_not_retried_and_error_text_never_discloses_secret():
    service = Service(replies=[OSError("https://token:secret@example/api?api_key=test-token-never-retain")])
    policy = client(service)
    policy.observe_frame(observation(), 0.0)
    with pytest.raises(ValueError, match="^dreamlabs_transport_failed$") as failure:
        policy.infer(observation())
    assert "secret" not in str(failure.value)
    assert len([call for call in service.calls if call["method"] == "POST"]) == 1
    assert policy.last_request_evidence()["serialization_verified"] is True
    assert policy.last_inference_evidence()["server_response_received"] is False
    assert policy.candidate_policy_queried is False
    assert all(frame["transmission_attempted"] is True and frame["server_response_received"] is False
               for frame in policy.last_policy_input_frames())
    with pytest.raises(ValueError, match="session_outcome_unknown"):
        policy.reset()
    with pytest.raises(ValueError, match="session_outcome_unknown"):
        policy.close()


def test_provider_refusal_and_release_refusal_keep_episode_faulted_without_error_body():
    reply = {"status": {"code": "DA-1000", "detail": "secret=test-token-never-retain"}}
    service = Service(replies=[reply])
    policy = client(service)
    policy.observe_frame(observation(), 0.0)
    with pytest.raises(ValueError, match="^dreamlabs_provider_refused$"):
        policy.infer(observation())
    assert "test-token-never-retain" not in json.dumps(policy.evidence_summary())
    service = Service(release={"status": {"code": "DA-5000"}})
    policy = client(service)
    policy.observe_frame(observation(), 0.0)
    policy.infer(observation())
    with pytest.raises(ValueError, match="release_unconfirmed"):
        policy.reset()
    with pytest.raises(ValueError, match="faulted"):
        policy.infer(observation())
    assert policy.evidence_summary()["session_release_status"] == "unconfirmed"


@pytest.mark.parametrize("code", ["DA-0000", "DA-0001", "DA-5000"])
@pytest.mark.parametrize("operation", ["reset", "close"])
def test_every_known_hosted_session_is_released_before_cleanup_can_confirm(code, operation):
    service = Service(replies=[response(), response(session="replacement", code=code)])
    active = set()

    def transport(**kwargs):
        if kwargs["url"].endswith("/end_session"):
            service.calls.append(kwargs)
            active.remove(json.loads(kwargs["body"])["session_id"])
            return b'{"status":{"code":"DA-0000"}}'
        raw = service(**kwargs)
        if kwargs["method"] == "POST":
            active.add(json.loads(raw)["session_id"])
        return raw

    policy = client(transport)
    policy.observe_frame(observation(), 0.0)
    policy.infer(observation())
    assert active == {"session-one"}
    with pytest.raises(ValueError, match="session_replaced_mid_episode|provider_refused"):
        policy.infer(observation())
    assert active == {"session-one", "replacement"}
    assert policy.evidence_summary()["known_sessions_pending_release"] == 2
    getattr(policy, operation)()
    assert active == set()
    assert policy.evidence_summary()["session_release_status"] == "confirmed"
    assert policy.evidence_summary()["known_sessions_pending_release"] == 0
    releases = [call for call in service.calls if call["url"].endswith("/end_session")]
    assert len(releases) == 2 and all(call["timeout_seconds"] == 2 for call in releases)


def test_a_failed_old_session_release_is_not_hidden_by_a_successful_new_session_release():
    service = Service(replies=[response(), response(session="replacement", code="DA-0000")])
    active = set()
    refuse_old = True

    def transport(**kwargs):
        if kwargs["url"].endswith("/end_session"):
            service.calls.append(kwargs)
            session = json.loads(kwargs["body"])["session_id"]
            if session == "session-one" and refuse_old:
                return b'{"status":{"code":"DA-5000"}}'
            active.remove(session)
            return b'{"status":{"code":"DA-0000"}}'
        raw = service(**kwargs)
        if kwargs["method"] == "POST":
            active.add(json.loads(raw)["session_id"])
        return raw

    policy = client(transport)
    policy.observe_frame(observation(), 0.0)
    policy.infer(observation())
    with pytest.raises(ValueError, match="replaced"):
        policy.infer(observation())
    with pytest.raises(ValueError, match="release_unconfirmed"):
        policy.reset()
    assert active == {"session-one"}
    assert policy.evidence_summary()["known_sessions_pending_release"] == 1
    assert policy.evidence_summary()["session_release_status"] == "unconfirmed"
    with pytest.raises(ValueError, match="faulted"):
        policy.infer(observation())
    refuse_old = False
    policy.close()
    assert not active and policy.evidence_summary()["session_release_status"] == "confirmed"


def test_repeated_close_cannot_turn_a_failed_release_into_a_confirmed_close():
    service = Service(release={"status": {"code": "DA-5000"}})
    policy = client(service)
    policy.observe_frame(observation(), 0.0)
    policy.infer(observation())
    with pytest.raises(ValueError, match="release_unconfirmed"):
        policy.close()
    count = len(service.calls)
    with pytest.raises(ValueError, match="release_unconfirmed"):
        policy.close()
    assert len(service.calls) == count  # Credential was discarded; no unauthenticated retry.
    assert policy.evidence_summary()["known_sessions_pending_release"] == 1
    assert policy.evidence_summary()["session_release_status"] == "unconfirmed"


def test_an_unbound_server_failure_does_not_prove_that_no_hosted_session_exists():
    service = Service(replies=[{"status": {"code": "DA-5000"}}])
    policy = client(service)
    policy.observe_frame(observation(), 0.0)
    with pytest.raises(ValueError, match="provider_refused"):
        policy.infer(observation())
    assert policy.evidence_summary()["session_outcome_unknown"] is True
    with pytest.raises(ValueError, match="session_outcome_unknown"):
        policy.close()
    assert policy.evidence_summary()["session_release_status"] == "outcome_unknown"


def test_builtin_transport_uses_shared_proxy_aware_bounds_and_zero_redirects(monkeypatch):
    import blueprint_pipeline.dreamlabs_policy_client as module
    calls = []

    def bounded(url, **kwargs):
        calls.append((url, kwargs))
        return type("Response", (), {"body": json.dumps([MODEL]).encode(),
                                      "content_type": "application/json; charset=utf-8"})()

    monkeypatch.setattr(module, "request", bounded)
    ready = DreamLabsPolicyClient(spec=SPEC, bearer_token="token").preflight_readiness()
    assert ready["identity_verified"]
    assert len(calls) == 1 and calls[0][0] == METADATA_URL
    assert calls[0][1]["policy"].allowed_hosts == frozenset({"platform.dreamlabs.ai"})
    assert calls[0][1]["policy"].follow_same_origin_redirects is False
    assert calls[0][1]["max_response_bytes"] == calls[0][1]["policy"].max_response_bytes == 1024 * 1024
    assert calls[0][1]["timeout_seconds"] == 15


def test_builtin_discovery_uses_environment_https_proxy_without_live_io(monkeypatch):
    import urllib.request
    from blueprint_pipeline import safe_outbound_http as http
    from tests.test_safe_outbound_http import _FakeResponse

    monkeypatch.setenv("https_proxy", "http://proxy.example:3128")
    calls = []

    def open_fake(opener, request, timeout):
        calls.append(request)
        proxies = [handler for handler in opener.handlers if isinstance(handler, urllib.request.ProxyHandler)]
        assert any(handler.proxies.get("https") == "http://proxy.example:3128" for handler in proxies)
        assert any(isinstance(handler, http._PolicyRedirectHandler) for handler in opener.handlers)
        response = _FakeResponse(body=json.dumps([MODEL]).encode(), final_url=request.full_url)
        response.headers = {"Content-Type": "Application/JSON; charset=utf-8"}
        return response

    monkeypatch.setattr(urllib.request.OpenerDirector, "open", open_fake)
    result = DreamLabsPolicyClient(spec=SPEC, bearer_token="test-token-never-retain").preflight_readiness()
    assert result["identity_verified"] and len(calls) == 1
    assert calls[0].full_url == METADATA_URL and calls[0].get_method() == "GET"


@pytest.mark.parametrize("content_type", [None, "", "text/html", "text/plain", "application/jsonp"])
def test_builtin_transport_rejects_non_json_media_type_without_leaking_headers(monkeypatch, content_type):
    import blueprint_pipeline.dreamlabs_policy_client as module
    from blueprint_pipeline.safe_outbound_http import SafeHttpResponse
    calls = []

    def respond(url, **kwargs):
        calls.append(url)
        return SafeHttpResponse(200, json.dumps([MODEL]).encode(), url, url, content_type)

    monkeypatch.setattr(module, "request", respond)
    with pytest.raises(ValueError, match="^dreamlabs_transport_failed$"):
        DreamLabsPolicyClient(spec=SPEC, bearer_token="test-token-never-retain").preflight_readiness()
    assert calls == [METADATA_URL]


@pytest.mark.parametrize("url", [ENDPOINT_URL + "?key=secret", ENDPOINT_URL.replace("api.", "evil."),
                               "http://api.dreamlabs.ai/v1/get_action",
                               "https://token:secret@api.dreamlabs.ai/v1/get_action"])
def test_builtin_transport_rejects_unapproved_url_before_token_can_leave(monkeypatch, url):
    import blueprint_pipeline.dreamlabs_policy_client as module

    def unexpected(*args, **kwargs):
        pytest.fail("unapproved endpoint reached transport")

    monkeypatch.setattr(module, "request", unexpected)
    with pytest.raises(ValueError, match="^dreamlabs_endpoint_not_approved$"):
        module._bounded_https(url=url, method="POST", body=b"{}", headers={"Authorization": "Bearer test-token"})


@pytest.mark.parametrize("fault", ["redirect", "oversize", "truncated"])
def test_builtin_post_transport_fails_once_and_keeps_credentials_out_of_errors(monkeypatch, fault):
    from http.client import IncompleteRead
    from blueprint_pipeline import safe_outbound_http as http
    import blueprint_pipeline.dreamlabs_policy_client as module
    from tests.test_safe_outbound_http import _FakeResponse

    calls = []

    def open_wire(request, timeout, policy):
        calls.append(request)
        assert policy.allowed_hosts == frozenset({"api.dreamlabs.ai"})
        assert not policy.follow_same_origin_redirects
        if fault == "truncated":
            raise IncompleteRead(b"test-token-never-retain", 100)
        if fault == "redirect":
            return http._PolicyRedirectHandler(policy).redirect_request(
                request, None, 307, "redirect", {}, "https://credential-sink.example/collect")
        response = _FakeResponse(body=b"x" * (module.MAX_RESPONSE_BYTES + 1), final_url=request.full_url)
        response.headers = {"Content-Type": "application/json"}
        return response

    monkeypatch.setattr(http, "_open_with_policy", open_wire)
    with pytest.raises(ValueError, match="^dreamlabs_transport_failed$") as failure:
        module._call(module._bounded_https, url=ENDPOINT_URL, method="POST", body=b"{}",
                     bearer_token="test-token-never-retain", user_agent="transport-test")
    assert len(calls) == 1
    assert calls[0].full_url == ENDPOINT_URL
    assert calls[0].get_method() == "POST" and calls[0].data == b"{}"
    assert calls[0].get_header("Authorization") == "Bearer test-token-never-retain"
    assert "test-token-never-retain" not in str(failure.value)
