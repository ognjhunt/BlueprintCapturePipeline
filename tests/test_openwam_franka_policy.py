"""Development-only EEF10 wire/frame contracts; no model/GPU/provider execution."""

import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline import openwam_franka_assets as assets
from blueprint_pipeline.openwam_franka_policy import (
    OpenWAMFrankaBinding,
    OpenWAMFrankaPolicyClient,
    encode_observation,
    decode_action,
)
from blueprint_pipeline.openwam_franka_policy_server import (
    OpenWAMFrankaServer,
    validate_wire_observation,
)
from blueprint_pipeline.team_arm_kinematics import pose_matrix, validate_pose_chunk
from tests.test_openwam_piper_policy import IDENTITY, Backend, Socket


def binding():
    return OpenWAMFrankaBinding(
        camera_names=("overhead", "arm_wrist"),
        control_hz=15,
        configuration_digest="sha256:" + "a" * 64,
        frame_binding_reference="development_only_explicit_frames",
        policy_base_from_native_base={"arm": copy.deepcopy(IDENTITY)},
        native_tcp_from_policy_tcp={"arm": copy.deepcopy(IDENTITY)},
    )


def observation():
    names = ["fr3_joint" + str(index) for index in range(1, 8)]
    return {
        "images": {
            "overhead": np.full((480, 640, 3), 37, dtype=np.uint8),
            "arm_wrist": np.full((480, 640, 3), 99, dtype=np.uint8),
        },
        "prompt": "Place the item on the tray",
        "state": {
            "arm_names": ["arm"],
            "joint_names": names,
            "joint_position": [0.0] * 7,
            "arm_joint_names": {"arm": names},
            "gripper_names": ["arm_gripper"],
            "gripper_position": [0.2],
            "arm_gripper_names": {"arm": "arm_gripper"},
            "arm_eef_pose": {"arm": copy.deepcopy(IDENTITY)},
        },
    }


def connected(action=None, fault=None):
    spec = binding()
    backend = Backend(action=action or [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.6])
    server = OpenWAMFrankaServer(
        backend=backend, binding=spec, checkpoint_receipt={"checkpoint_bytes_verified": True}
    )
    sockets = []

    def factory(*_args, **kwargs):
        assert kwargs["proxy"] is None
        sock = Socket(server, fault=fault)
        sockets.append(sock)
        return sock

    return OpenWAMFrankaPolicyClient(binding=spec, socket_factory=factory), backend, sockets


@pytest.mark.parametrize("closure,signed", [(0, 1), (0.2, 0.6), (0.5, 0), (1, -1)])
def test_signed_gripper_raw_prompt_two_real_views_and_exact_inverse(closure, signed):
    obs = observation()
    obs["state"]["gripper_position"] = [closure]
    wire = encode_observation(obs, binding())
    assert wire["prompt"] == obs["prompt"]
    assert set(wire["images"]) == {"head_camera", "left_wrist_camera"}
    assert len(wire["state"]) == 10 and wire["state"][9] == pytest.approx(signed)
    assert validate_wire_observation(wire)["state"] == wire["state"]
    native = validate_pose_chunk(
        decode_action(wire["state"], binding()), arm_names=["arm"], maximum_rows=1
    )
    assert native["gripper_closures"][0, 0] == pytest.approx(closure)


def test_nonidentity_frames_use_column_rotations_and_invert_full_tool_transform():
    spec = binding()
    spec.policy_base_from_native_base["arm"].update(
        position_m=[1.0, 2.0, 3.0], orientation_xyzw=[0, 0, math.sqrt(0.5), math.sqrt(0.5)]
    )
    spec.native_tcp_from_policy_tcp["arm"].update(
        position_m=[0.1, 0.2, 0.3], orientation_xyzw=[0.5, 0.5, 0.5, 0.5]
    )
    obs = observation()
    obs["state"]["arm_eef_pose"]["arm"]["position_m"] = [0.3, 0.4, 0.5]
    wire = encode_observation(obs, spec)
    expected = (
        pose_matrix(
            **{
                "position": spec.policy_base_from_native_base["arm"]["position_m"],
                "orientation_xyzw": spec.policy_base_from_native_base["arm"]["orientation_xyzw"],
            }
        )
        @ pose_matrix([0.3, 0.4, 0.5], [0, 0, 0, 1])
        @ pose_matrix([0.1, 0.2, 0.3], [0.5, 0.5, 0.5, 0.5])
    )
    assert wire["state"][:3] == pytest.approx(expected[:3, 3])
    assert wire["state"][3:9] == pytest.approx(expected[:3, :2].T.reshape(-1))
    native = decode_action(wire["state"], spec)
    assert native["positions_m"][0][0] == pytest.approx([0.3, 0.4, 0.5])
    assert native["orientations_xyzw"][0][0] == pytest.approx([0, 0, 0, 1])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_wrist",
        "extra_camera",
        "wrong_dimensions",
        "joint_contract",
        "private_target",
        "wrong_arm",
        "bad_closure",
    ],
)
def test_missing_real_view_or_wrong_robot_never_silently_fills_a_policy_input(fault):
    obs = observation()
    if fault == "missing_wrist":
        obs["images"].pop("arm_wrist")
    elif fault == "extra_camera":
        obs["images"]["fabricated_right"] = obs["images"]["arm_wrist"]
    elif fault == "wrong_dimensions":
        obs["images"]["arm_wrist"] = np.zeros((176, 320, 3), dtype=np.uint8)
    elif fault == "joint_contract":
        obs["state"]["joint_names"].pop()
    elif fault == "private_target":
        obs["object_pose"] = [1, 2, 3]
    elif fault == "wrong_arm":
        obs["state"]["arm_names"] = ["left", "right"]
    else:
        obs["state"]["gripper_position"] = [2]
    client, backend, sockets = connected()
    with pytest.raises(ValueError):
        client.infer(obs)
    assert not backend.calls and not sockets


@pytest.mark.parametrize("hz", [30, 60, True, float("nan"), "15"])
def test_source_dataset_policy_clock_cannot_be_relabelled(hz):
    with pytest.raises(ValueError):
        replace(binding(), control_hz=hz).validate()


@pytest.mark.parametrize("signed,closure", [(-1.04, 1), (1.02, 0), (0.6, 0.2)])
def test_local_transport_reset_identity_raw_evidence_and_projection(signed, closure):
    action = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, signed]
    client, backend, sockets = connected(action)
    retained = []
    client.bind_request_evidence_sink(retained.append)
    result = client.infer(observation())
    assert result["gripper_closures"][0][0] == pytest.approx(closure)
    assert backend.calls[0]["prompt"] == observation()["prompt"]
    evidence = client.last_inference_evidence()
    assert evidence["raw_vendor_action_response"]["action"] == action
    assert evidence["gripper_projection"]["clipped"] == [not -1 <= signed <= 1]
    assert retained[0]["serialization_verified"] is True
    assert client.evidence_summary()["policy_interface_id"] == "openwam_franka_eef10.v1"
    client.infer(observation())
    assert backend.step == 2
    client.reset()
    client.infer(observation())
    assert backend.step == 1
    assert sockets[0].requests[0] == {"type": "ping"}


def test_returned_invalid_response_retains_query_truth_and_requires_reset():
    client, _, _ = connected(fault=json.dumps({"type": "action", "action": [], "step": 1}))
    with pytest.raises(ValueError, match="action_shape"):
        client.infer(observation())
    assert (
        client.candidate_policy_queried
        and client.last_inference_evidence()["server_response_received"]
    )
    with pytest.raises(ValueError, match="reset_required"):
        client.infer(observation())


@pytest.mark.parametrize(
    "sidecar",
    ["added_tokens.json", "tokenizer/google/umt5-xxl/added_tokens.json", "generation_config.json"],
)
def test_complete_checkpoint_closure_rejects_unverified_loader_sidecars(
    tmp_path, monkeypatch, sidecar
):
    files = {
        "config.yaml": b"development_only",
        "checkpoint_step_9860.safetensors": b"not real weights",
    }
    expected = {name: (len(data), hashlib.sha256(data).hexdigest()) for name, data in files.items()}
    monkeypatch.setattr(assets, "EXPECTED_FILES", expected)
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    assert assets.verify_checkpoint_directory(tmp_path)["checkpoint_bytes_verified"]
    path = tmp_path / sidecar
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    with pytest.raises(ValueError, match="inventory_changed"):
        assets.verify_checkpoint_directory(tmp_path)


def test_exact_checkpoint_card_license_inventory_and_stats_identity():
    root = Path(__file__).parents[1]
    card = root / "config/policy_runtimes/openwam_franka_model_card.md"
    assert assets.EXPECTED_FILES["README.md"] == (
        len(card.read_bytes()),
        hashlib.sha256(card.read_bytes()).hexdigest(),
    )
    assert b"license: apache-2.0" in card.read_bytes()
    assert assets.EXPECTED_FILES["checkpoint_step_9860.safetensors"][0] == 24813767464
    assert assets.EXPECTED_FILES["normalization_stats.npy"] == (
        1368,
        "d54b7b411f545de021a8ffd04f35637db6588d36bbcbd4463755271b91859d9a",
    )


def test_pinned_stats_retain_signed_opening_and_next_achieved_state_contract():
    import yaml

    root = Path(__file__).parent / "fixtures/openwam_franka"
    for name in ("config.yaml", "normalization_stats.npy"):
        data = (root / name).read_bytes()
        assert (len(data), hashlib.sha256(data).hexdigest()) == assets.EXPECTED_FILES[name]
    # Only the exact hash-pinned publisher object is deserialized, never caller
    # input. The runtime performs the same full byte verification before load.
    stats = np.load(root / "normalization_stats.npy", allow_pickle=True).item()["eef"]
    assert stats["gripper_convention"] == "minus1_closed_plus1_open"
    assert stats["gripper_transform"] == "1_minus_2_times_closedness"
    assert stats["action_alignment"] == "action[t]=state[t+1]; repeated terminal action excluded"
    assert len(stats["min"]) == 10 and stats["min"][9] == -1
    assert stats["max"][9] == pytest.approx(0.7535127401351929)
    config = yaml.safe_load((root / "config.yaml").read_text())["dataloader"]
    assert config["unify_action_map"] == ["0-9"] and config["normalize_mode"] == "min-max"
    assert config["camera_layout"] == [
        "observation.images.left",
        "observation.images.left_wrist",
        None,
    ]
    assert config["num_frames"] - 1 == 32 and config["video_stride"] == 4


def test_server_loads_exact_franka_weight_name_with_checkpoint_owned_normalization(
    tmp_path, monkeypatch
):
    import sys
    from types import ModuleType, SimpleNamespace
    from blueprint_pipeline import openwam_franka_policy_server as module
    from blueprint_pipeline import openwam_piper_policy_server as shared

    source = tmp_path / "source"
    source.mkdir()
    calls, verified = [], []
    backend = Backend(action=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    upstream = ModuleType("openwam.deploy.server")
    upstream.__file__ = str(source / "openwam/deploy/server.py")
    upstream.build_server_from_config = lambda cfg, directory, **kwargs: (
        calls.append((cfg, directory, kwargs)) or backend
    )
    deploy = ModuleType("openwam.deploy")
    deploy.server = upstream
    for name, value in (
        ("openwam", ModuleType("openwam")),
        ("openwam.deploy", deploy),
        ("openwam.deploy.server", upstream),
    ):
        monkeypatch.setitem(sys.modules, name, value)
    omega = ModuleType("omegaconf")
    omega.OmegaConf = SimpleNamespace(create=lambda value: value)
    monkeypatch.setitem(sys.modules, "omegaconf", omega)
    monkeypatch.setattr(shared, "verify_source_checkout", lambda _: source)
    monkeypatch.setattr(
        module,
        "verify_checkpoint_directory",
        lambda path, **kwargs: verified.append(path) or {"checkpoint_bytes_verified": True},
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "fixture")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "fixture")
    monkeypatch.setenv("DIFFSYNTH_SKIP_DOWNLOAD", "fixture")
    monkeypatch.setattr(sys, "path", list(sys.path))
    result = module.build_server(checkpoint=tmp_path, source_root=source, binding=binding())
    assert verified == [tmp_path] and result.backend is backend and len(calls) == 1
    config, directory, options = calls[0]
    assert options == {"device": "cuda", "ckpt_name": "checkpoint_step_9860.safetensors"}
    assert directory == str(tmp_path) and "model" not in config and "dataloader" not in config
    assert config["inference"]["inference_horizon"] == 8
