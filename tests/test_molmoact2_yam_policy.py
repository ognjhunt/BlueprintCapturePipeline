"""Development-only contracts; no checkpoints, GPU, paid service or robot runs."""
from __future__ import annotations

from contextlib import nullcontext
import copy
from dataclasses import replace
import hashlib
from http.server import ThreadingHTTPServer
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from blueprint_pipeline import molmoact2_yam_assets as assets
from blueprint_pipeline.molmoact2_yam_policy import (
    ARMS, GRIPPERS, JOINTS, WIRE_CAMERAS, MolmoAct2YAMBinding, MolmoAct2YAMPolicyClient,
    _json, decode_actions, decode_array, encode_array, encode_observation,
)
from blueprint_pipeline.molmoact2_yam_policy_server import (
    MolmoAct2YAMBackend, MolmoAct2YAMServer, build_server, decode_observation, handler_for,
)
from blueprint_pipeline.policy_request_evidence import restore_request, validate_request_evidence


def binding():
    return MolmoAct2YAMBinding(("overhead", "left_wrist", "right_wrist"), "sha256:" + "a" * 64,
                              "development_only_publisher_simulation_camera_assumption")


def observation():
    # Deliberately permuted native arrays. The policy must gather by exact names.
    names = list(reversed(JOINTS))
    return {"prompt": "Move the cup.",
            "images": {name: np.full((360, 640, 3), i * 30 + 7, dtype=np.uint8)
                       for i, name in enumerate(binding().camera_names)},
            "state": {"joint_names": names, "joint_position": [JOINTS.index(name) / 10 for name in names],
                      "joint_velocity": [0.] * 12, "arm_names": list(ARMS),
                      "arm_joint_names": {side: list(JOINTS[i * 6:(i + 1) * 6]) for i, side in enumerate(ARMS)},
                      "gripper_names": ["right_gripper", "left_gripper"], "gripper_position": [.8, .2],
                      "arm_gripper_names": dict(zip(ARMS, GRIPPERS, strict=True)),
                      "arm_eef_pose": {"private_extra_ignored": True}, "base": {"not_sent": True}}}


class Backend:
    def __init__(self):
        self.calls, self.resets = [], 0
        self.action = np.tile(np.array([.1, .2, .3, .4, .5, .6, .8, .7, .8, .9, 1., 1.1, 1.2, .2], dtype=np.float32), (30, 1))

    def predict(self, obs):
        self.calls.append(copy.deepcopy(obs))
        return self.action.copy()

    def reset(self):
        self.resets += 1


class Connection:
    def __init__(self, runtime, requests, fault=None):
        self.runtime, self.requests, self.fault = runtime, requests, fault
        self.closed = False

    def request(self, method, path, body, headers):
        self.requests.append((method, path, body, headers))
        self.path, self.body, self.headers = path, body, headers

    def getresponse(self):
        if self.path == "/act" and self.fault is not None:
            if isinstance(self.fault, Exception):
                raise self.fault
            value = self.fault
        else:
            value = self.runtime.handle(self.path, json.loads(self.body),
                                        session=self.headers.get("X-Blueprint-Session"),
                                        sequence=self.headers.get("X-Blueprint-Sequence"))
        raw = _json(value)
        return SimpleNamespace(status=200, read=lambda limit: raw[:limit])

    def close(self):
        self.closed = True


def client_runtime(fault=None):
    backend = Backend()
    runtime = MolmoAct2YAMServer(backend=backend, binding=binding(),
                               checkpoint_receipt={"checkpoint_bytes_verified": True})
    requests, connections = [], []

    def connect(*_, **__):
        connection = Connection(runtime, requests, fault)
        connections.append(connection)
        return connection

    return MolmoAct2YAMPolicyClient(binding=binding(), connection_factory=connect), runtime, backend, requests, connections


def test_exact_state_order_rgb_and_absolute_action_no_double_normalization():
    obs = observation()
    wire = encode_observation(obs, binding())
    values = decode_observation(wire)
    assert values["state"] == pytest.approx([0, .1, .2, .3, .4, .5, .8, .6, .7, .8, .9, 1., 1.1, .2])
    assert values["instruction"] == "Move the cup."
    assert set(wire) == {*WIRE_CAMERAS, "instruction", "state", "num_steps", "enable_cuda_graph"}
    for local, remote in zip(binding().camera_names, WIRE_CAMERAS, strict=True):
        np.testing.assert_array_equal(values[remote], obs["images"][local])
    action = Backend().action
    decoded, projection = decode_actions(action, joint_names=obs["state"]["joint_names"],
                                         gripper_names=obs["state"]["gripper_names"])
    assert decoded.shape == (30, 14)
    assert decoded[0, :12] == pytest.approx([1.2, 1.1, 1., .9, .8, .7, .6, .5, .4, .3, .2, .1])
    assert decoded[0, 12:] == pytest.approx([.8, .2])
    assert projection["clipped_count"] == 0
    # An arm is never offset by the observed state or converted to a delta.
    assert decoded[0, 0] == pytest.approx(1.2)


def test_independent_gripper_projection_is_visible_and_arms_are_not_clipped():
    actions = Backend().action
    actions[0, 0], actions[0, 6], actions[0, 13] = 100, -0.2, 1.1
    native, projection = decode_actions(actions, joint_names=list(JOINTS), gripper_names=list(GRIPPERS))
    assert native[0, 0] == 100  # Native per-robot bounds remain authoritative.
    assert native[0, 12:].tolist() == [1., 0.]
    assert projection["clipped_count"] == 2
    assert projection["raw_openings"][0] == pytest.approx([-.2, 1.1])


@pytest.mark.parametrize("fault", ["camera", "dtype", "size", "joint", "arm_order", "tool", "nonfinite", "private"])
def test_unsupported_observations_never_reach_transport(fault):
    client, _, backend, requests, _ = client_runtime()
    obs = observation()
    if fault == "camera":
        obs["images"].pop("left_wrist")
    elif fault == "dtype":
        obs["images"]["overhead"] = obs["images"]["overhead"].astype(float)
    elif fault == "size":
        obs["images"]["overhead"] = obs["images"]["overhead"][:176, :320]
    elif fault == "joint":
        obs["state"]["joint_names"][0] = "different_robot_joint"
    elif fault == "arm_order":
        obs["state"]["arm_names"].reverse()
    elif fault == "tool":
        obs["state"]["arm_gripper_names"]["right"] = "left_gripper"
    elif fault == "nonfinite":
        obs["state"]["joint_position"][0] = float("nan")
    else:
        obs["success_state"] = True
    with pytest.raises(ValueError):
        client.infer(obs)
    assert not requests and not backend.calls


def test_lifecycle_retains_exact_request_before_transmission_and_resets_sequence():
    client, _, backend, requests, connections = client_runtime()
    captured = []

    def sink(receipt):
        assert [path for _, path, _, _ in requests] == ["/reset"]
        captured.append(validate_request_evidence(receipt))

    client.bind_request_evidence_sink(sink)
    result = client.infer(observation())
    assert result.shape == (30, 14) and backend.resets == 1 and len(backend.calls) == 1
    wire = restore_request(captured[0]["request"])
    assert _json(wire) == requests[1][2]
    assert all(connection.closed for connection in connections)
    assert client.last_inference_evidence()["sequence"] == 1
    assert client.last_inference_evidence()["actions_extracted"] is True
    first_session = client.last_inference_evidence()["session"]
    client.bind_request_evidence_sink(None)
    client.reset()
    assert client.candidate_policy_queried is False
    client.infer(observation())
    assert client.last_inference_evidence()["sequence"] == 1
    assert client.last_inference_evidence()["session"] != first_session
    assert client.open_loop_horizon == 25 and client.action_hz == 30
    assert client.evidence_summary()["native_execution_validated"] is False


@pytest.mark.parametrize("fault", [TimeoutError("development_only"), {"actions": [], "session": "old", "sequence": 1}])
def test_failure_has_no_retry_or_stale_episode_reuse(fault):
    client, _, _, requests, connections = client_runtime(fault)
    with pytest.raises((ValueError, TimeoutError)):
        client.infer(observation())
    with pytest.raises(ValueError, match="reset_required"):
        client.infer(observation())
    assert [path for _, path, _, _ in requests].count("/act") == 1
    assert all(connection.closed for connection in connections)
    assert client.last_inference_evidence()["transmission_attempted"] is True
    assert client.last_inference_evidence()["actions_extracted"] is False


def test_identity_mismatch_prevents_model_query():
    client, runtime, backend, requests, _ = client_runtime()
    runtime.identity["model_revision"] = "0" * 40
    with pytest.raises(ValueError, match="identity_mismatch"):
        client.infer(observation())
    assert not backend.calls and len(requests) == 1


@pytest.mark.parametrize("fault", ["nan", "short", "wide", "boolean"])
def test_invalid_action_matrix_is_not_a_partial_success(fault):
    actions = Backend().action
    if fault == "nan":
        actions[-1, -1] = float("nan")
    elif fault == "short":
        actions = actions[:29]
    elif fault == "wide":
        actions = np.zeros((30, 32))
    else:
        actions = np.ones((30, 14), dtype=bool)
    with pytest.raises(ValueError):
        decode_actions(actions, joint_names=list(JOINTS), gripper_names=list(GRIPPERS))


def test_server_refuses_old_sessions_replay_concurrency_and_poisoned_backend():
    _, runtime, backend, _, _ = client_runtime()
    payload = encode_observation(observation(), binding())
    old = runtime.handle("/reset", {})["session"]
    current = runtime.handle("/reset", {})["session"]
    with pytest.raises(ValueError, match="sequence"):
        runtime.handle("/act", payload, session=old, sequence="1")
    runtime._lock.acquire()
    try:
        with pytest.raises(ValueError, match="busy"):
            runtime.handle("/reset", {})
    finally:
        runtime._lock.release()
    runtime.handle("/act", payload, session=current, sequence="1")
    with pytest.raises(ValueError, match="sequence"):
        runtime.handle("/act", payload, session=current, sequence="1")
    backend.action[0, 0] = float("nan")
    with pytest.raises(ValueError, match="backend_action"):
        runtime.handle("/act", payload, session=current, sequence="2")
    assert runtime._session is None


def test_exact_publisher_model_call_owns_normalization(monkeypatch):
    calls = []
    model = SimpleNamespace(predict_action=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(actions=Backend().action))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(inference_mode=nullcontext, is_tensor=lambda _: False))
    backend = MolmoAct2YAMBackend(model=model, processor="development_only_processor")
    obs = decode_observation(encode_observation(observation(), binding()))
    np.testing.assert_array_equal(backend.predict(obs), Backend().action)
    call = calls[0]
    assert call["norm_tag"] == "yam_dual_molmoact2"
    assert call["inference_action_mode"] == "continuous"
    assert call["n_action_steps"] == 30 and call["num_steps"] == 10
    assert call["enable_depth_reasoning"] is False and call["enable_cuda_graph"] is False
    assert call["normalize_language"] is True
    np.testing.assert_array_equal(call["state"], obs["state"])
    assert [np.asarray(image)[0, 0, 0] for image in call["images"]] == [7, 37, 67]


def test_release_inventory_is_complete_and_rights_basis_is_explicit():
    inventory = assets.release_inventory()
    files = inventory["files"]
    assert len(files) == 25 and sum(row["bytes"] for row in files) == 21782347229
    assert sum(row["bytes"] for row in files if row["path"].endswith(".safetensors")) == 21768955912
    assert sum(row["required_for_inference"] for row in files) == 20
    assert inventory["license"] == "Apache-2.0"
    assert inventory["hf_model_card_license_field"] is None
    assert inventory["weight_files_downloaded"] is False
    assert "README.md#7-license" in inventory["license_evidence_url"]
    assert all(row["publisher_lfs_sha256"] or row["locally_verified_sha256"]
               for row in files if row["required_for_inference"])
    cameras = assets.publisher_camera_reference()
    assert cameras["top_cam"]["horizontal_fov_deg"] == 69.4
    assert cameras["left_cam"]["horizontal_fov_deg"] == 87
    assert cameras["current_cad_discrepancy"] == "overhead_D405_body_with_simulated_D435i_optics"


@pytest.fixture
def small_checkpoint(tmp_path, monkeypatch):
    files = {"config.json": b"{}", "modeling_molmoact2.py": b"# development_only\n",
             "model-00001.safetensors": b"development_only_not_real_weights",
             "model.safetensors.index.json": b'{"weight_map":{"fixture":"model-00001.safetensors"}}'}
    manifest = assets.release_inventory()
    manifest["files"] = [{"path": name, "bytes": len(raw), "locally_verified_sha256": hashlib.sha256(raw).hexdigest(),
                          "publisher_lfs_sha256": None, "required_for_inference": True} for name, raw in files.items()]
    monkeypatch.setattr(assets, "release_inventory", lambda: manifest)
    root = tmp_path / "model"
    root.mkdir()
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    return root, manifest


def test_snapshot_verification_hashes_bytes(small_checkpoint):
    root, _ = small_checkpoint
    assert assets.verify_checkpoint_directory(root)["files_verified"] == 4
    (root / "config.json").write_bytes(b"[]")
    with pytest.raises(ValueError, match="digest_mismatch"):
        assets.verify_checkpoint_directory(root)


@pytest.mark.parametrize("name", ["unlisted.py", "special_tokens_map.json", "added_tokens.json", "adapter_config.json"])
def test_snapshot_refuses_every_unlisted_loader_sidecar(small_checkpoint, name):
    root, _ = small_checkpoint
    (root / name).write_bytes(b"{}")
    with pytest.raises(ValueError, match="unlisted_file"):
        assets.verify_checkpoint_directory(root)


def test_snapshot_verifies_optional_publisher_files_and_keeps_runtime_state_outside(small_checkpoint):
    root, manifest = small_checkpoint
    raw = b"development_only_optional_publisher_asset"
    manifest["files"].append({"path": "assets/example.svg", "bytes": len(raw), "required_for_inference": False,
                              "publisher_lfs_sha256": None, "locally_verified_sha256": None,
                              "git_blob_id": hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()})
    # Omitted optional files are permitted; normal runtime state is not in the snapshot.
    (root.parent / "runtime-state.json").write_bytes(b"{}")
    assert assets.verify_checkpoint_directory(root)["files_verified"] == 4
    (root / "assets").mkdir()
    (root / "assets/example.svg").write_bytes(raw)
    assert assets.verify_checkpoint_directory(root)["files_verified"] == 5
    (root / "assets/example.svg").write_bytes(b"x" * len(raw))
    with pytest.raises(ValueError, match="digest_mismatch"):
        assets.verify_checkpoint_directory(root)


def test_snapshot_refuses_symlink_directories(small_checkpoint):
    root, _ = small_checkpoint
    outside = root.parent / "cache"
    outside.mkdir()
    (root / "cache").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="file_invalid"):
        assets.verify_checkpoint_directory(root)


def test_loader_verifies_before_importing_any_checkpoint_code(tmp_path, monkeypatch):
    calls = []
    fake = SimpleNamespace(from_pretrained=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoProcessor=fake, AutoModelForImageTextToText=fake))
    with pytest.raises(ValueError, match="checkpoint_file_invalid"):
        build_server(checkpoint=tmp_path, binding=binding())
    assert not calls


@pytest.mark.parametrize("options", [{"host": "example.com"}, {"timeout_seconds": float("inf")}, {"port": 0}])
def test_local_transport_is_bounded(options):
    with pytest.raises(ValueError):
        MolmoAct2YAMPolicyClient(binding=binding(), **options)


def test_binding_identity_and_numpy_wire_reject_ambiguous_shapes():
    with pytest.raises(ValueError):
        replace(binding(), execution_horizon=31).validate()
    wire = encode_array(np.zeros((30, 14), dtype=np.float32))
    wire["shape"] = [14, 30]
    with pytest.raises(ValueError):
        decode_array(wire, (30, 14), "<f4")


def test_real_loopback_http_roundtrip_and_new_episode():
    """Exercise HTTP bytes/header plumbing against a fake, development backend."""
    backend = Backend()
    runtime = MolmoAct2YAMServer(backend=backend, binding=binding(),
                               checkpoint_receipt={"checkpoint_bytes_verified": True})
    with ThreadingHTTPServer(("127.0.0.1", 0), handler_for(runtime)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = MolmoAct2YAMPolicyClient(binding=binding(), port=server.server_port, timeout_seconds=3)
        try:
            result = client.infer(observation())
            assert result.shape == (30, 14) and len(backend.calls) == 1
            validate_request_evidence(client.last_request_evidence())
            client.close()
            with pytest.raises(ValueError, match="reset_required"):
                client.infer(observation())
            client.reset()
            client.infer(observation())
            assert backend.resets == 2 and len(backend.calls) == 2
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=3)
