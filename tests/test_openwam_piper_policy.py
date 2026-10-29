"""Development-only adapter contracts; no GPU, checkpoint execution or paid calls."""

from __future__ import annotations

import base64
import copy
from dataclasses import replace
import hashlib
import io
import json
import math
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from blueprint_pipeline import openwam_piper_assets as assets
from blueprint_pipeline.openwam_piper_policy import (
    ARMS,
    PROMPT_PREFIX,
    OpenWAMPiperBinding,
    OpenWAMPiperPolicyClient,
    decode_action,
    encode_observation,
)
from blueprint_pipeline.openwam_piper_policy_server import (
    OpenWAMPiperServer,
    validate_wire_observation,
)
from blueprint_pipeline.policy_request_evidence import restore_request, validate_request_evidence
from blueprint_pipeline.team_arm_kinematics import rotation_xyzw, validate_pose_chunk

IDENTITY = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}


def binding(**overrides):
    return OpenWAMPiperBinding(
        camera_names=("head", "left_wrist", "right_wrist"),
        control_hz=30,
        configuration_digest="sha256:" + "a" * 64,
        frame_binding_reference="development_only_explicit_identity_fixture",
        policy_base_from_native_base={arm: copy.deepcopy(IDENTITY) for arm in ARMS},
        native_tcp_from_policy_tcp={arm: copy.deepcopy(IDENTITY) for arm in ARMS},
        **overrides,
    )


def observation():
    return {
        "images": {
            name: np.full((12, 20, 3), color, dtype=np.uint8)
            for name, color in zip(("head", "left_wrist", "right_wrist"), (11, 29, 47), strict=True)
        },
        "prompt": "Move the object",
        "state": {
            "joint_names": [f"{arm}_{i}" for arm in ARMS for i in range(6)],
            "joint_position": [0.0] * 12,
            "joint_velocity": [0.0] * 12,
            "arm_names": list(ARMS),
            "arm_joint_names": {arm: [f"{arm}_{i}" for i in range(6)] for arm in ARMS},
            "gripper_names": ["right_gripper", "left_gripper"],
            "gripper_position": [0.8, 0.2],
            "arm_gripper_names": {arm: arm + "_gripper" for arm in ARMS},
            "arm_eef_pose": {arm: copy.deepcopy(IDENTITY) for arm in ARMS},
            "base": {"not_sent": True},
        },
    }


class Backend:
    def __init__(self, action=None):
        self.action = action or [
            0.1,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            -0.1,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            1.0,
        ]
        self.calls, self.resets, self.step = [], 0, 0

    def predict(self, obs):
        self.calls.append(copy.deepcopy(obs))
        self.step += 1
        return {"action": self.action, "step": self.step, "latency_ms": 0.1}

    def reset(self):
        self.step = 0
        self.resets += 1


class Socket:
    def __init__(self, server, fault=None):
        self.server, self.fault, self.closed, self.requests = server, fault, False, []

    def send(self, text):
        self.requests.append(json.loads(text))

    def recv(self, timeout):
        request = self.requests[-1]
        if request["type"] == "obs" and self.fault is not None:
            if isinstance(self.fault, Exception):
                raise self.fault
            return self.fault
        return json.dumps(self.server.handle(request))

    def close(self, **_):
        self.closed = True


def client_and_backend(spec=None, fault=None):
    spec = spec or binding()
    backend = Backend()
    server = OpenWAMPiperServer(
        backend=backend, binding=spec, checkpoint_receipt={"checkpoint_bytes_verified": True}
    )
    sockets = []

    def connect(*_, **__):
        socket = Socket(server, fault=fault)
        sockets.append(socket)
        return socket

    return OpenWAMPiperPolicyClient(binding=spec, socket_factory=connect), backend, sockets


def test_publisher_eef20_columns_and_independent_gripper_order():
    obs = observation()
    obs["state"]["arm_eef_pose"]["left"] = {
        "position_m": [0.1, 0.2, 0.3],
        "orientation_xyzw": [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
    }
    wire = encode_observation(obs, binding())
    assert wire["state"][:10] == pytest.approx([0.1, 0.2, 0.3, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.8])
    assert wire["state"][19] == pytest.approx(0.2)
    assert wire["prompt"] == PROMPT_PREFIX + "Move the object"
    assert set(wire) == {"type", "images", "state", "prompt"}
    for name, local in zip(
        ("head_camera", "left_wrist_camera", "right_wrist_camera"),
        binding().camera_names,
        strict=True,
    ):
        with Image.open(io.BytesIO(base64.b64decode(wire["images"][name]))) as image:
            assert image.format == "PNG"
            np.testing.assert_array_equal(np.asarray(image), obs["images"][local])


def test_nonidentity_base_and_tcp_calibration_is_inverted_for_native_targets():
    spec = binding()
    spec.policy_base_from_native_base["left"].update(
        position_m=[1.0, 2.0, 0.0], orientation_xyzw=[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
    )
    spec.native_tcp_from_policy_tcp["left"]["position_m"] = [0.2, 0.0, 0.0]
    obs = observation()
    obs["state"]["arm_eef_pose"]["left"]["position_m"] = [1.0, 0.0, 0.0]
    wire = encode_observation(obs, spec)
    assert wire["state"][:3] == pytest.approx([1.0, 3.2, 0.0])
    returned = decode_action(wire["state"], spec)
    validated = validate_pose_chunk(returned, arm_names=list(ARMS), maximum_rows=1)
    assert validated["positions_m"][0, 0] == pytest.approx([1.0, 0.0, 0.0])
    assert rotation_xyzw(validated["orientations_xyzw"][0, 0]) == pytest.approx(np.eye(3))
    assert validated["gripper_closures"][0] == pytest.approx([0.2, 0.8])


@pytest.mark.parametrize(
    "fault",
    [
        "camera",
        "private_channel",
        "arm_order",
        "joint_overlap",
        "one_tool",
        "nan_pose",
        "gripper",
        "image_dtype",
    ],
)
def test_invalid_or_excess_observations_cannot_reach_model(fault):
    obs = observation()
    if fault == "camera":
        obs["images"].pop("right_wrist")
    elif fault == "private_channel":
        obs["task_object_pose"] = [1, 2, 3]
    elif fault == "arm_order":
        obs["state"]["arm_names"].reverse()
    elif fault == "joint_overlap":
        obs["state"]["arm_joint_names"]["right"] = obs["state"]["arm_joint_names"]["left"]
    elif fault == "one_tool":
        obs["state"]["arm_gripper_names"]["right"] = "left_gripper"
    elif fault == "nan_pose":
        obs["state"]["arm_eef_pose"]["left"]["position_m"][0] = math.nan
    elif fault == "gripper":
        obs["state"]["gripper_position"][0] = 2
    else:
        obs["images"]["head"] = obs["images"]["head"].astype(float)
    client, backend, sockets = client_and_backend()
    with pytest.raises(ValueError):
        client.infer(obs)
    assert not backend.calls and not sockets


def test_exact_request_persisted_before_send_and_reset_clears_action_buffer():
    client, backend, sockets = client_and_backend()
    retained = []

    def sink(value):
        validate_request_evidence(value)
        assert not any(request["type"] == "obs" for request in sockets[-1].requests)
        retained.append(value)

    client.bind_request_evidence_sink(sink)
    result = client.infer(observation())
    assert result["gripper_closures"] == [[1.0, 0.0]]
    assert client.last_inference_evidence()["actions_extracted"]
    assert restore_request(retained[0]["request"]) == sockets[0].requests[-1]
    assert retained[0]["serialization_verified"] is True
    client.bind_request_evidence_sink(None)
    client.infer(observation())
    assert backend.step == 2
    client.reset()
    client.infer(observation())
    assert backend.step == 1 and backend.resets == 2
    assert client.evidence_summary()["catalog_registered"] is None
    assert client.evidence_summary()["execution_binding_digest"] is None
    client.close()


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        '{"type":"action","action":[NaN],"step":1}',
        '{"type":"action","action":[],"step":1}',
        '{"type":"action","action":[],"step":2}',
        '{"type":"error","code":"model_failed"}',
    ],
)
def test_completed_malformed_query_is_truthful_and_never_implicitly_retried(response):
    client, _, sockets = client_and_backend(fault=response)
    with pytest.raises(ValueError):
        client.infer(observation())
    assert client.candidate_policy_queried is True
    assert client.last_inference_evidence()["server_response_received"] is True
    assert client.last_inference_evidence()["actions_extracted"] is False
    with pytest.raises(ValueError, match="episode_reset_required"):
        client.infer(observation())
    assert len([x for x in sockets[0].requests if x["type"] == "obs"]) == 1


def test_uncertain_transport_requires_explicit_reset_without_claiming_response():
    client, _, sockets = client_and_backend(fault=TimeoutError("fixture"))
    with pytest.raises(TimeoutError):
        client.infer(observation())
    assert client.candidate_policy_queried is False
    assert client.last_inference_evidence()["server_response_received"] is False
    with pytest.raises(ValueError, match="episode_reset_required"):
        client.infer(observation())
    assert sockets[0].closed and len(sockets) == 1


def test_sink_failure_prevents_any_model_request():
    client, backend, sockets = client_and_backend()
    client.bind_request_evidence_sink(lambda _: (_ for _ in ()).throw(OSError("fixture full disk")))
    with pytest.raises(OSError):
        client.infer(observation())
    assert not backend.calls and all(x["type"] != "obs" for x in sockets[0].requests)


@pytest.mark.parametrize(
    "field,value",
    [
        ("control_hz", 0),
        ("execution_horizon", 33),
        ("configuration_digest", "missing"),
        ("frame_binding_reference", ""),
    ],
)
def test_candidate_requires_explicit_frame_and_rate_binding(field, value):
    with pytest.raises(ValueError):
        replace(binding(), **{field: value}).validate()


def test_mismatched_worker_identity_does_not_query():
    spec = binding()
    backend = Backend()
    server = OpenWAMPiperServer(
        backend=backend,
        binding=replace(spec, control_hz=20),
        checkpoint_receipt={"checkpoint_bytes_verified": True},
    )
    client = OpenWAMPiperPolicyClient(binding=spec, socket_factory=lambda *_, **__: Socket(server))
    with pytest.raises(ValueError, match="identity_mismatch"):
        client.infer(observation())
    assert not backend.calls


@pytest.mark.parametrize("mutation", ["short", "nan", "parallel_rotation", "nonfinite_gripper"])
def test_invalid_physical_actions_cannot_become_native_commands(mutation):
    action = Backend().action.copy()
    if mutation == "short":
        action.pop()
    elif mutation == "nan":
        action[0] = math.nan
    elif mutation == "parallel_rotation":
        action[6:9] = action[3:6]
    else:
        action[9] = math.inf
    with pytest.raises(ValueError):
        decode_action(action, binding())


def test_gripper_overshoot_saturates_without_losing_raw_response_or_episode():
    spec = binding()
    backend = Backend()
    backend.action[9], backend.action[19] = -0.01, 1.02
    server = OpenWAMPiperServer(
        backend=backend, binding=spec, checkpoint_receipt={"checkpoint_bytes_verified": True}
    )
    client = OpenWAMPiperPolicyClient(binding=spec, socket_factory=lambda *_, **__: Socket(server))
    result = client.infer(observation())
    assert result["gripper_closures"] == [[1.0, 0.0]]
    evidence = client.last_inference_evidence()
    assert evidence["raw_vendor_action_response"]["action"][9] == -0.01
    assert evidence["raw_vendor_action_response"]["action"][19] == 1.02
    assert evidence["gripper_projection"] == {
        "rule": "saturate_finite_opening_to_[0,1]_at_native_interface",
        "arm_names": ["left", "right"],
        "raw_openings": [-0.01, 1.02],
        "applied_openings": [0.0, 1.0],
        "clipped": [True, True],
    }
    backend.action[9], backend.action[19] = 0.25, 0.75
    assert client.infer(observation())["gripper_closures"] == [[0.75, 0.25]]
    assert client.last_inference_evidence()["gripper_projection"]["clipped"] == [False, False]
    assert len(backend.calls) == 2


def test_server_accepts_only_png_rgb_and_declared_scientific_payload():
    wire = encode_observation(observation(), binding())
    assert validate_wire_observation(wire)["state"] == wire["state"]
    wire["scorer"] = {"success": True}
    with pytest.raises(ValueError):
        validate_wire_observation(wire)
    wire.pop("scorer")
    image = io.BytesIO()
    Image.fromarray(observation()["images"]["head"]).save(image, format="JPEG")
    wire["images"]["head_camera"] = base64.b64encode(image.getvalue()).decode()
    with pytest.raises(ValueError):
        validate_wire_observation(wire)


def test_checkpoint_verification_hashes_bytes_and_refuses_ambiguous_or_tampered_weights(
    tmp_path, monkeypatch
):
    payload = b"development-only-fixture-no-weights"
    name = "checkpoint_step_30000.safetensors"
    monkeypatch.setattr(
        assets, "EXPECTED_FILES", {name: (len(payload), hashlib.sha256(payload).hexdigest())}
    )
    (tmp_path / name).write_bytes(payload)
    assert assets.verify_checkpoint_directory(tmp_path)["inference_performed"] is False
    (tmp_path / "checkpoint_step_99999.safetensors").write_bytes(payload)
    with pytest.raises(ValueError, match="ambiguous"):
        assets.verify_checkpoint_directory(tmp_path)
    (tmp_path / "checkpoint_step_99999.safetensors").unlink()
    (tmp_path / name).write_bytes(b"x" * len(payload))
    with pytest.raises(ValueError, match="digest_mismatch"):
        assets.verify_checkpoint_directory(tmp_path)


def test_actual_team_episode_consumes_pose_response_and_retains_request(tmp_path, monkeypatch):
    from blueprint_pipeline import (
        adp_task_scoring,
        episode_visual_evidence,
        native_task_episode_environment,
    )
    from blueprint_pipeline.team_embodiment import (
        apply_team_embodiment,
        read_model_structure,
        team_embodiment_binding,
        validate_embodiment_manifest,
        verify_model_file,
    )
    from blueprint_pipeline.team_embodiment_runtime import run_team_episode
    from tests.test_team_arm_kinematics import CartesianRobot, named_inputs
    from tests.test_team_embodiment import _Camera, _Scoring, station_plan

    model, value = named_inputs(tmp_path)
    declared = validate_embodiment_manifest(value)
    model_binding = team_embodiment_binding(
        declared,
        model_record=verify_model_file(declared, model),
        structure=read_model_structure(model),
        scene_plan=station_plan(),
    )
    plan = {**apply_team_embodiment(station_plan(), model_binding), "scenario": {"seed": 7}}
    plan["cadence"].update(maximum_action_steps=4, settle_window_samples=0)
    row = plan["robot"]

    class Env:
        def __init__(self):
            self.robot, self.steps = CartesianRobot(row), []
            self.unwrapped = SimpleNamespace(
                scene={
                    "robot": self.robot,
                    **{
                        name: _Camera()
                        for name in (
                            "team_camera_0",
                            "team_camera_1",
                            "team_camera_2",
                            "external_camera_2",
                        )
                    },
                },
                device="cpu",
            )

        def reset(self, seed):
            self.robot.q = np.array(
                [row["joint_reset_positions_rad"][n] for n in self.robot.joint_names]
            )

        def step(self, action):
            order = [*row["joint_names"], *(j["name"] for j in row["gripper_joints"])]
            self.steps.append(list(action[0]))
            self.robot.q = np.array([action[0][order.index(n)] for n in self.robot.joint_names])

    env = Env()
    cameras = [camera["name"] for camera in declared["cameras"]]
    spec = replace(binding(), camera_names=(cameras[2], cameras[0], cameras[1]))
    client, backend, _ = client_and_backend(spec)
    built = SimpleNamespace(
        env=env,
        camera_scene_names={
            "team_camera_0": "team_camera_0",
            "team_camera_1": "team_camera_1",
            "team_camera_2": "team_camera_2",
            "overview": "external_camera_2",
        },
    )
    monkeypatch.setattr(native_task_episode_environment, "NativeRigidScoringEnvironment", _Scoring)
    monkeypatch.setattr(
        episode_visual_evidence,
        "finalize_multicamera_visual_evidence",
        lambda **_: ({"status": "complete"}, []),
    )
    monkeypatch.setattr(
        adp_task_scoring, "score_task_episode_from_spec", lambda **_: {"task_succeeded": False}
    )
    result = run_team_episode(
        built=built,
        task_readback=None,
        policy=client,
        plan=plan,
        prompt="Move the object",
        max_policy_queries=2,
        open_loop_horizon=1,
        media_root=tmp_path / "episodes",
        episode_id="development-only-piper-adapter",
        progress={},
        to_action=lambda targets: [targets],
    )
    assert len(backend.calls) == 2 and len(env.steps) == 4
    assert result["policy_request_evidence_complete"] is True
    assert result["policy_cadence"]["native_hold_ticks_per_action"] == 2
    assert env.steps[-1][-2:] == [0.0, 0.03]
    assert backend.calls[1]["state"][0] > backend.calls[0]["state"][0]
    assert backend.calls[1]["state"][10] < backend.calls[0]["state"][10]
    assert result["score"]["task_succeeded"] is False
    assert client.evidence_summary()["native_execution_validated"] is False


def test_real_local_websocket_transport_preserves_episode_isolation():
    import threading
    from websockets.sync.server import serve

    spec = binding()
    backend = Backend()
    server = OpenWAMPiperServer(
        backend=backend, binding=spec, checkpoint_receipt={"checkpoint_bytes_verified": True}
    )
    with serve(server.serve_connection, "127.0.0.1", 0, compression=None) as listener:
        thread = threading.Thread(target=listener.serve_forever, daemon=True)
        thread.start()
        port = listener.socket.getsockname()[1]
        client = OpenWAMPiperPolicyClient(binding=spec, port=port)
        try:
            result = client.infer(observation())
            assert result["gripper_closures"] == [[1.0, 0.0]]
            assert backend.step == 1
            second = OpenWAMPiperPolicyClient(binding=spec, port=port)
            try:
                from websockets.exceptions import ConnectionClosedError

                with pytest.raises(ConnectionClosedError):
                    second.infer(observation())
            finally:
                second.close()
            assert backend.step == 1
            client.reset()
            client.infer(observation())
            assert backend.step == 1
        finally:
            client.close()
            listener.shutdown()
            thread.join(timeout=3)
            assert not thread.is_alive()


def test_disconnect_resets_model_session_without_observation_or_extra_inference():
    backend = Backend()
    server = OpenWAMPiperServer(
        backend=backend, binding=binding(), checkpoint_receipt={"checkpoint_bytes_verified": True}
    )

    class Connection:
        closed = False
        sent = []

        def __iter__(self):
            return iter([json.dumps(encode_observation(observation(), binding()))])

        def send(self, value):
            self.sent.append(json.loads(value))

        def close(self, **_):
            self.closed = True

    connection = Connection()
    server.serve_connection(connection)
    assert len(backend.calls) == 1 and backend.step == 0 and backend.resets == 2
    assert connection.closed and connection.sent[0]["step"] == 1


def test_factory_uses_exact_upstream_builder_and_no_second_normalizer(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    from blueprint_pipeline import openwam_piper_policy_server as module

    source = tmp_path / "source"
    source.mkdir()
    calls = []
    backend = Backend()
    upstream = ModuleType("openwam.deploy.server")
    upstream.__file__ = str(source / "openwam/deploy/server.py")

    def builder(cfg, directory, *, device, ckpt_name):
        calls.append((cfg, directory, device, ckpt_name))
        return backend

    upstream.build_server_from_config = builder
    package = ModuleType("openwam")
    deploy = ModuleType("openwam.deploy")
    deploy.server = upstream
    monkeypatch.setitem(sys.modules, "openwam", package)
    monkeypatch.setitem(sys.modules, "openwam.deploy", deploy)
    monkeypatch.setitem(sys.modules, "openwam.deploy.server", upstream)
    omega = ModuleType("omegaconf")
    omega.OmegaConf = SimpleNamespace(create=lambda value: value)
    monkeypatch.setitem(sys.modules, "omegaconf", omega)
    monkeypatch.setattr(module, "verify_source_checkout", lambda _: source)
    monkeypatch.setattr(
        module, "verify_checkpoint_directory", lambda _: {"checkpoint_bytes_verified": True}
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "fixture")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "fixture")
    previous_path = list(sys.path)
    try:
        runtime = module.build_server(checkpoint=tmp_path, source_root=source, binding=binding())
    finally:
        sys.path[:] = previous_path
    cfg, directory, device, name = calls[0]
    assert len(calls) == 1 and runtime.backend is backend
    assert (
        directory == str(tmp_path)
        and device == "cuda"
        and name == "checkpoint_step_30000.safetensors"
    )
    assert (
        cfg["inference"]["inference_mode"] == "sync" and cfg["inference"]["inference_horizon"] == 8
    )
    assert (
        cfg["optimization"]["decode_video"] is False
        and cfg["optimization"]["compile"]["enabled"] is False
    )
    assert "dataloader" not in cfg and "model" not in cfg


def test_declared_websocket_runtime_minimum_is_explicit_before_connect(monkeypatch):
    import websockets
    from websockets.sync import client as socket_module

    monkeypatch.setattr(websockets, "__version__", "12.0")
    attempted = []
    monkeypatch.setattr(socket_module, "connect", lambda *a, **k: attempted.append((a, k)))
    client = OpenWAMPiperPolicyClient(binding=binding())
    with pytest.raises(ValueError, match="requires_websockets_16"):
        client.infer(observation())
    assert not attempted
