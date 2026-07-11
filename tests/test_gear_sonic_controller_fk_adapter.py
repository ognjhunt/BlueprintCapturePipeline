from __future__ import annotations

import hashlib
import sys

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import gear_sonic_controller_fk_adapter as adapter
from blueprint_pipeline.oscar_isaac_closed_loop_eval import (
    SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    make_controller_fk_skeleton_projector,
)


def test_official_gear_sonic_adapter_binds_action_controller_fk_and_state(tmp_path, monkeypatch) -> None:
    root = tmp_path / "wbc"
    deploy = root / "gear_sonic_deploy"
    deploy.mkdir(parents=True)
    model = deploy / "g1_29dof_with_hand.xml"
    model.write_text("<robot/>", encoding="utf-8")
    runner = root / "executor.py"
    runner.write_text(
        """
import json, os
request = json.load(open(os.environ['BLUEPRINT_GEAR_SONIC_INPUT']))
action = request['action']['action_chunk']
json.dump({
  'status': 'completed',
  'runtime_result_id': 'gear-sonic-runtime-1',
  'source_action_sha256': request['source_action_sha256'],
  'landmarks': [
    {'name': 'right_wrist', 'x': action[0], 'y': action[1], 'z': 1.0},
    {'name': 'right_hand', 'x': action[1], 'y': action[0], 'z': 0.9},
  ],
  'joint_positions': action,
  'joint_names': ['right_shoulder_pitch_joint', 'right_elbow_joint'],
  'proprioceptive_state': {'base_height_m': 0.79, 'official_controller_protocol': 4},
  'state_timestamp': '2026-07-10T12:00:00Z',
}, open(os.environ['BLUEPRINT_GEAR_SONIC_OUTPUT'], 'w'))
""".strip(),
        encoding="utf-8",
    )
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
    monkeypatch.setenv(adapter.ROOT_ENV, str(root))
    monkeypatch.setenv(adapter.ROBOT_MODEL_ENV, str(model))
    monkeypatch.setenv(adapter.EXECUTOR_COMMAND_ENV, f"{sys.executable} {runner}")
    monkeypatch.setenv(adapter.SIGNING_KEY_ENV, str(key_path))
    monkeypatch.setenv(
        SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
        hashlib.sha256(public).hexdigest(),
    )

    projector = make_controller_fk_skeleton_projector(
        command=f"{sys.executable} -m blueprint_pipeline.gear_sonic_controller_fk_adapter",
        work_dir=tmp_path / "projector",
    )
    action = {
        "policy_action": "UNITREE_G1_SONIC",
        "action_chunk": [0.25, -0.5],
        "action_units": ["latent", "latent"],
        "action_timing": {"control_hz": 50.0, "sample_index": 0},
    }
    result = projector(action, 1)
    assert result["schema_version"] == "gear_sonic_controller_fk_execution.v1"
    assert result["controller_id"] == "nvidia/GEAR-SONIC:/opt/wbc@protocol-v4"
    assert result["source_action_sha256"]
    assert result["generated_robot_state"]["joint_positions"] == [0.25, -0.5]
    assert result["generated_robot_state"]["proxy_or_surrogate"] is False
    assert result["action_contract"] == {
        "command": "UNITREE_G1_SONIC",
        "dimension": 2,
        "values_sha256": result["action_contract"]["values_sha256"],
        "timing": {"control_hz": 50.0, "sample_index": 0},
        "units": ["latent", "latent"],
    }
