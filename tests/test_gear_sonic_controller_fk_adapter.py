from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import gear_sonic_controller_fk_adapter as adapter
from blueprint_pipeline import gear_sonic_joint_order_contract as contract
from blueprint_pipeline.oscar_isaac_closed_loop_eval import (
    SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    make_controller_fk_skeleton_projector,
)

FIXTURE_MODEL = (
    Path(__file__).parent / "fixtures" / "gear_sonic_g1_min" / "g1_29dof_with_hand_min.xml"
)

FAKE_EXECUTOR_TEMPLATE = """
import json, os
from blueprint_pipeline import gear_sonic_joint_order_contract as contract
request = json.load(open(os.environ['BLUEPRINT_GEAR_SONIC_INPUT']))
action = request['action']['action_chunk']
names = list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
positions = [round(0.01 * index, 4) for index in range(43)]
result = {
  'status': 'completed',
  'runtime_result_id': 'gear-sonic-runtime-1',
  'source_action_sha256': request['source_action_sha256'],
  'landmarks': [
    {'name': 'right_wrist_yaw_link', 'x': action[0], 'y': action[1], 'z': 1.0},
    {'name': 'right_hand_palm_link', 'x': action[1], 'y': action[0], 'z': 0.9},
  ],
  'joint_positions': positions,
  'joint_names': names,
  'joint_order_schema_version': contract.JOINT_ORDER_SCHEMA_VERSION,
  'mapping_digest': contract.PROTOCOL_V4_MAPPING_DIGEST,
  'controller_revision': 'wbc-deploy-2026-07',
  'applied_dof_mapping': [
    {'joint_name': name, 'protocol_index': index, 'model_qpos_address': index + 7,
     'applied_value': positions[index]}
    for index, name in enumerate(names)
  ],
  'proprioceptive_state': {'base_height_m': 0.79, 'official_controller_protocol': 4},
  'state_timestamp': '2026-07-10T12:00:00Z',
}
MUTATE
json.dump(result, open(os.environ['BLUEPRINT_GEAR_SONIC_OUTPUT'], 'w'))
""".strip()


def _write_signing_key(tmp_path, monkeypatch) -> None:
    key = Ed25519PrivateKey.from_private_bytes(b"\x09" * 32)
    key_path = tmp_path / "key.pem"
    key_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public = key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    monkeypatch.setenv(adapter.SIGNING_KEY_ENV, str(key_path))
    monkeypatch.setenv(
        SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(public).hexdigest(),
    )


def _install_wbc(tmp_path, monkeypatch, *, mutate: str = "") -> Path:
    root = tmp_path / "wbc"
    deploy = root / "gear_sonic_deploy" / "g1"
    deploy.mkdir(parents=True)
    model = deploy / "g1_29dof_with_hand.xml"
    shutil.copyfile(FIXTURE_MODEL, model)
    runner = root / "executor.py"
    runner.write_text(
        FAKE_EXECUTOR_TEMPLATE.replace("MUTATE", mutate), encoding="utf-8"
    )
    monkeypatch.setenv(adapter.ROOT_ENV, str(root))
    monkeypatch.setenv(adapter.ROBOT_MODEL_ENV, str(model))
    monkeypatch.setenv(adapter.EXECUTOR_COMMAND_ENV, f"{sys.executable} {runner}")
    return root


def _projector(tmp_path):
    return make_controller_fk_skeleton_projector(
        command=f"{sys.executable} -m blueprint_pipeline.gear_sonic_controller_fk_adapter",
        work_dir=tmp_path / "projector",
    )


def _action() -> dict:
    return {
        "policy_action": "UNITREE_G1_SONIC",
        "action_chunk": [0.25, -0.5],
        "action_units": ["latent", "latent"],
        "action_timing": {"control_hz": 50.0, "sample_index": 0},
    }


def _run_adapter_directly(tmp_path) -> dict:
    action = _action()
    request = {
        "schema_version": "controller_fk_skeleton_request.v1",
        "step_index": 1,
        "source_action_sha256": hashlib.sha256(
            json.dumps(action, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest(),
        "action": action,
    }
    input_path = tmp_path / "controller_fk_input.json"
    input_path.write_text(json.dumps(request), encoding="utf-8")
    return adapter.run_adapter(
        input_path=input_path, output_path=tmp_path / "controller_fk_output.json"
    )


def test_official_gear_sonic_adapter_binds_action_controller_fk_and_state(
    tmp_path, monkeypatch
) -> None:
    _install_wbc(tmp_path, monkeypatch)
    _write_signing_key(tmp_path, monkeypatch)
    result = _projector(tmp_path)(_action(), 1)
    assert result["schema_version"] == "gear_sonic_controller_fk_execution.v1"
    assert result["controller_id"] == "nvidia/GEAR-SONIC:/opt/wbc@protocol-v4"
    assert result["source_action_sha256"]
    assert result["joint_order_schema_version"] == contract.JOINT_ORDER_SCHEMA_VERSION
    assert result["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    state = result["generated_robot_state"]
    assert state["proxy_or_surrogate"] is False
    assert state["joint_names"] == list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    assert len(state["joint_positions"]) == 43
    assert state["joint_order_schema_version"] == contract.JOINT_ORDER_SCHEMA_VERSION
    assert state["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    assert len(state["applied_dof_mapping"]) == 43
    assert result["action_contract"] == {
        "command": "UNITREE_G1_SONIC",
        "dimension": 2,
        "values_sha256": result["action_contract"]["values_sha256"],
        "timing": {"control_hz": 50.0, "sample_index": 0},
        "units": ["latent", "latent"],
    }


def test_adapter_rejects_executor_output_without_joint_order_contract(
    tmp_path, monkeypatch
) -> None:
    _install_wbc(
        tmp_path,
        monkeypatch,
        mutate=(
            "del result['joint_order_schema_version']\n"
            "del result['mapping_digest']\n"
            "del result['applied_dof_mapping']"
        ),
    )
    _write_signing_key(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="joint_order_schema_missing_or_unsupported"):
        _run_adapter_directly(tmp_path)


def test_adapter_rejects_executor_output_with_wrong_mapping_digest(
    tmp_path, monkeypatch
) -> None:
    _install_wbc(tmp_path, monkeypatch, mutate="result['mapping_digest'] = 'f' * 64")
    _write_signing_key(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="mapping_digest_missing_or_mismatch"):
        _run_adapter_directly(tmp_path)


def test_adapter_rejects_executor_output_with_permuted_joint_names(
    tmp_path, monkeypatch
) -> None:
    _install_wbc(
        tmp_path,
        monkeypatch,
        mutate=(
            "left = names.index('left_elbow_joint')\n"
            "right = names.index('right_elbow_joint')\n"
            "names[left], names[right] = names[right], names[left]"
        ),
    )
    _write_signing_key(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="executor_joint_names_permuted"):
        _run_adapter_directly(tmp_path)


def test_adapter_rejects_executor_output_without_applied_dof_mapping(
    tmp_path, monkeypatch
) -> None:
    _install_wbc(tmp_path, monkeypatch, mutate="result['applied_dof_mapping'] = []")
    _write_signing_key(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="applied_dof_mapping_missing_or_invalid"):
        _run_adapter_directly(tmp_path)


REAL_EXECUTOR_RUNNER = """
import json, os
from blueprint_pipeline import gear_sonic_joint_order_contract as contract
from blueprint_pipeline import gear_sonic_official_zmq_executor as executor


def transport(**kwargs):
    return {
        "token_state": kwargs["motion_token"],
        "body_q_target": [0.05] * 29,
        "body_q_measured": [0.0] * 29,
        "last_left_hand_action": kwargs["left_hand"],
        "last_right_hand_action": kwargs["right_hand"],
        "base_quat_measured": [1.0, 0.0, 0.0, 0.0],
        "ros_timestamp": 123,
        "controller_revision": "wbc-deploy-2026-07",
        "joint_order_schema_version": contract.JOINT_ORDER_SCHEMA_VERSION,
        "body_joint_names": list(contract.PROTOCOL_V4_BODY_JOINT_NAMES),
        "left_hand_joint_names": list(contract.PROTOCOL_V4_LEFT_HAND_JOINT_NAMES),
        "right_hand_joint_names": list(contract.PROTOCOL_V4_RIGHT_HAND_JOINT_NAMES),
        "mapping_digest": contract.PROTOCOL_V4_MAPPING_DIGEST,
    }


request = json.load(open(os.environ["BLUEPRINT_GEAR_SONIC_INPUT"]))
result = executor.execute(
    request,
    transport=transport,
    controller_revision_resolver=lambda _root: contract.PINNED_WBC_SOURCE_REVISION,
)
json.dump(result, open(os.environ["BLUEPRINT_GEAR_SONIC_OUTPUT"], "w"))
""".strip()


def test_protocol_v4_result_traverses_policy_zmq_controller_mujoco_fk_path(
    tmp_path, monkeypatch
) -> None:
    """Acceptance: one action SHA and one mapping digest carried through
    policy action -> controller transport boundary -> real MuJoCo FK ->
    attested adapter payload."""
    pytest.importorskip("mujoco", reason="mujoco_not_installed_in_venv")
    root = _install_wbc(tmp_path, monkeypatch)
    runner = root / "real_executor_runner.py"
    runner.write_text(REAL_EXECUTOR_RUNNER, encoding="utf-8")
    monkeypatch.setenv(adapter.EXECUTOR_COMMAND_ENV, f"{sys.executable} {runner}")
    _write_signing_key(tmp_path, monkeypatch)
    action = {
        "policy_action": "UNITREE_G1_SONIC",
        "sonic_action_chunk": [0.01] * 78,
        "action_units": ["latent"] * 78,
        "action_timing": {"control_hz": 50.0, "sample_index": 0},
    }
    result = _projector(tmp_path)(action, 2)
    expected_sha = hashlib.sha256(
        json.dumps(action, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert result["source_action_sha256"] == expected_sha
    assert result["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    state = result["generated_robot_state"]
    assert state["source_action_sha256"] == expected_sha
    assert state["mapping_digest"] == contract.PROTOCOL_V4_MAPPING_DIGEST
    assert state["joint_names"] == list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    assert state["joint_positions"][:29] == [0.05] * 29
    assert state["joint_positions"][29:] == [0.01] * 14
    assert result["landmarks"]
    assert result["derived_via_controller_fk"] is True
