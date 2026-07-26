from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from blueprint_pipeline import gear_sonic_controller_fk_adapter as adapter
from blueprint_pipeline import gear_sonic_joint_order_contract as contract
from blueprint_pipeline import gear_sonic_official_zmq_executor as official_executor
from blueprint_pipeline import isaac_runtime_task_backend as isaac_backend
from blueprint_pipeline.oscar_isaac_closed_loop_eval import (
    CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_ENV,
    SC3_FK_EXECUTOR_TRUSTED_PUBLIC_KEY_SHA256_ENV,
    make_controller_fk_skeleton_projector,
)

FIXTURE_MODEL = (
    Path(__file__).parent / "fixtures" / "gear_sonic_g1_min" / "g1_29dof_with_hand_min.xml"
)

FAKE_EXECUTOR_TEMPLATE = """
import hashlib, json, os
from blueprint_pipeline import gear_sonic_joint_order_contract as contract
request = json.load(open(os.environ['BLUEPRINT_GEAR_SONIC_INPUT']))
action = request['action']['action_chunk']
camera_context = request['camera_projection_context']
camera_context_sha = hashlib.sha256(
    json.dumps(camera_context, sort_keys=True, separators=(',', ':')).encode()
).hexdigest()
source_frame_sha = camera_context['source_frame_artifact']['sha256']
names = list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
positions = [round(0.01 * index, 4) for index in range(43)]
result = {
  'status': 'completed',
  'runtime_result_id': 'gear-sonic-runtime-1',
  'source_action_sha256': request['source_action_sha256'],
  'landmarks': [
    {'name': 'right_wrist_yaw_link', 'landmark_id': 'right_wrist_yaw_link',
     'x': action[0], 'y': action[1], 'z': 1.0,
     'image_projection': {'available': True, 'u_px': 310.0, 'v_px': 230.0,
      'projection_context_sha256': camera_context_sha,
      'source_frame_sha256': source_frame_sha}},
    {'name': 'right_hand_palm_link', 'landmark_id': 'right_hand_palm_link',
     'x': action[1], 'y': action[0], 'z': 0.9,
     'image_projection': {'available': True, 'u_px': 330.0, 'v_px': 250.0,
      'projection_context_sha256': camera_context_sha,
      'source_frame_sha256': source_frame_sha}},
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
  'proprioceptive_state': {
    'base_height_m': 0.79, 'official_controller_protocol': 4,
    'base_quat_measured': [1.0, 0.0, 0.0, 0.0],
  },
  'state_timestamp': '2026-07-10T12:00:00Z',
  'camera_projection_context_sha256': camera_context_sha,
  'camera_source_frame_sha256': source_frame_sha,
  'cross_simulator_registration': {
    'schema_version': 'gear_sonic_isaac_named_link_registration.v1',
    'status': 'passed', 'surrogate': False,
  },
}
controller_action = request['action'].get('controller_action')
if controller_action:
  frames = controller_action['frames']
  rows = []
  for frame_index, frame in enumerate(frames):
    frame_positions = [round(0.01 * index + 0.1 * frame_index, 4) for index in range(43)]
    frame_landmarks = [
      {'name': 'right_wrist_yaw_link', 'landmark_id': 'right_wrist_yaw_link',
       'x': frame[0], 'y': frame[1], 'z': 1.0,
       'image_projection': {'available': True, 'u_px': 310.0, 'v_px': 230.0,
        'projection_context_sha256': camera_context_sha,
        'source_frame_sha256': source_frame_sha}},
      {'name': 'right_hand_palm_link', 'landmark_id': 'right_hand_palm_link',
       'x': frame[1], 'y': frame[0], 'z': 0.9,
       'image_projection': {'available': True, 'u_px': 330.0, 'v_px': 250.0,
        'projection_context_sha256': camera_context_sha,
        'source_frame_sha256': source_frame_sha}},
    ]
    rows.append({
      'horizon_frame_index': frame_index,
      'controller_frame_index': (
        (request['step_index'] - 1) * controller_action['source_horizon_frame_count']
        + 1 + frame_index
      ),
      'source_action_frame_sha256': hashlib.sha256(
        json.dumps(frame, sort_keys=True, separators=(',', ':')).encode()
      ).hexdigest(),
      'controller_state_sha256': hashlib.sha256(
        json.dumps({'frame_index': frame_index, 'frame': frame},
                   sort_keys=True, separators=(',', ':')).encode()
      ).hexdigest(),
      'command_send_offset_seconds': frame_index / controller_action['control_hz'],
      'joint_positions': frame_positions,
      'joint_names': names,
      'applied_dof_mapping': [
        {'joint_name': name, 'protocol_index': index,
         'model_qpos_address': index + 7, 'applied_value': frame_positions[index]}
        for index, name in enumerate(names)
      ],
      'landmarks': frame_landmarks,
      'proprioceptive_state': {
        'base_height_m': 0.79, 'official_controller_protocol': 4,
        'base_quat_measured': [1.0, 0.0, 0.0, 0.0],
      },
      'state_timestamp': str(1000 + frame_index),
    })
  sequence_sha = hashlib.sha256(
    json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()
  ).hexdigest()
  result.update({
    'landmarks': rows[-1]['landmarks'],
    'joint_positions': rows[-1]['joint_positions'],
    'joint_names': rows[-1]['joint_names'],
    'applied_dof_mapping': rows[-1]['applied_dof_mapping'],
    'proprioceptive_state': rows[-1]['proprioceptive_state'],
    'state_timestamp': rows[-1]['state_timestamp'],
    'controller_fk_sequence': rows,
    'controller_fk_sequence_sha256': sequence_sha,
    'execution_contract': {
      'schema_version': 'gear_sonic_controller_horizon_execution.v1',
      'execution_mode': controller_action['execution_mode'],
      'controller_session_count': 1,
      'execution_frame_count': len(frames),
      'source_horizon_frame_count': controller_action['source_horizon_frame_count'],
      'frame_dimension': controller_action['frame_dimension'],
      'control_hz': controller_action['control_hz'],
      'sample_period_seconds': controller_action['sample_period_seconds'],
      'declared_execution_duration_seconds': controller_action['execution_duration_seconds'],
      'input_action_frames_sha256': controller_action['frames_sha256'],
      'source_action_frames_sha256': controller_action['source_frames_sha256'],
      'controller_state_sequence_sha256': 'c' * 64,
      'controller_fk_sequence_sha256': sequence_sha,
      'final_controller_fk_frame_sha256': hashlib.sha256(
        json.dumps(rows[-1], sort_keys=True, separators=(',', ':')).encode()
      ).hexdigest(),
    },
  })
state_seed = request['controller_fk_state_seed']
initial_fk_frame = {
  'frame_kind': 'initial_observation_live_isaac_state',
  'seed_authority': 'initial_observation_live_isaac_state',
  'source_state_seed_sha256': request['controller_fk_state_seed_sha256'],
  'joint_positions': (
    state_seed['body_q'] + state_seed['left_hand_q'] + state_seed['right_hand_q']
  ),
  'joint_names': names,
  'applied_dof_mapping': [
    {'joint_name': name, 'protocol_index': index,
     'model_qpos_address': index + 7, 'applied_value': 0.0}
    for index, name in enumerate(names)
  ],
  'base_quaternion_wxyz': state_seed['base_quaternion_wxyz'],
  'landmarks': result['landmarks'],
}
result['initial_fk_frame'] = initial_fk_frame
result['initial_fk_frame_sha256'] = hashlib.sha256(
  json.dumps(initial_fk_frame, sort_keys=True, separators=(',', ':')).encode()
).hexdigest()
result['controller_fk_state_seed_sha256'] = request['controller_fk_state_seed_sha256']
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
    for relative_path in adapter.CONTROLLER_RUNTIME_ARTIFACT_RELATIVE_PATHS:
        artifact = root / relative_path
        if artifact == model:
            continue
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(f"fixture:{relative_path}".encode("utf-8"))
    runner = root / "executor.py"
    runner.write_text(
        FAKE_EXECUTOR_TEMPLATE.replace("MUTATE", mutate), encoding="utf-8"
    )
    for relative_path in adapter.CONTROLLER_RUNTIME_ARTIFACT_RELATIVE_PATHS:
        runtime_artifact = root / relative_path
        runtime_artifact.parent.mkdir(parents=True, exist_ok=True)
        if not runtime_artifact.exists():
            runtime_artifact.write_bytes(f"fixture:{relative_path}".encode())
    monkeypatch.setenv(adapter.ROOT_ENV, str(root))
    monkeypatch.setenv(adapter.ROBOT_MODEL_ENV, str(model))
    monkeypatch.setenv(adapter.EXECUTOR_COMMAND_ENV, f"{sys.executable} {runner}")
    camera_context = tmp_path / "controller_fk_camera_projection_context.json"
    seed_frame = tmp_path / "seed.png"
    seed_frame.write_bytes(b"live-isaac-seed")
    seed_sha256 = hashlib.sha256(seed_frame.read_bytes()).hexdigest()
    camera_context.write_text(
        json.dumps(
            {
                "schema_version": "controller_fk_camera_projection_context.v1",
                "status": "captured_from_live_persistent_isaac_session",
                "attempt_id": "attempt-1",
                "launch_nonce": "nonce-1",
                "simulator_session_id": "isaac-session-1",
                "stage_id": "stage-1",
                "source_frame_artifact": {
                    "path": str(seed_frame),
                    "sha256": seed_sha256,
                    "width": 640,
                    "height": 480,
                },
                "camera_contract": {
                    "available": True,
                    "camera_id": "robot_pov",
                    "camera_path": "/World/Cameras/robot_pov",
                    "projection_token": "perspective",
                    "viewpoint_mode": "robot_head_mounted_egocentric",
                    "mount_motion_model": "rigid_head_local_transform",
                    "gaze_motion_model": "inherits_head_orientation_no_task_reaim",
                    "intrinsics": {
                        "available": True,
                        "fx": 100.0,
                        "fy": 100.0,
                        "cx": 320.0,
                        "cy": 240.0,
                        "image_width": 640,
                        "image_height": 480,
                    },
                    "camera_world_xyz_m": [0.0, 0.0, 2.0],
                    "camera_xmat_row_major": [
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ],
                    "clipping_range_m": [0.01, 1000.0],
                    "resolution": [640, 480],
                },
                "live_isaac_pelvis_world_pose": {
                    "prim_path": "/World/G1/pelvis",
                    "position_xyz": [0.0, 0.0, 0.8],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "standing_cross_simulator_registration": {
                    "status": (
                        "pending_official_mujoco_named_link_residual_verification"
                    ),
                    "required_landmark_names": [
                        "left_shoulder_pitch_link",
                        "left_elbow_link",
                        "left_wrist_yaw_link",
                        "right_shoulder_pitch_link",
                        "right_elbow_link",
                        "right_wrist_yaw_link",
                    ],
                    "isaac_named_link_world_poses": [],
                    "standing_joint_names": list(contract.PROTOCOL_V4_FULL_JOINT_ORDER),
                    "standing_joint_positions": [0.0] * 43,
                    "maximum_residual_tolerance_m": 0.025,
                    "surrogate": False,
                },
                "coordinate_transform": (
                    "mujoco_pelvis_relative_to_live_isaac_pelvis_wxyz"
                ),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(
        CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_ENV,
        str(camera_context),
    )
    return root


def _projector(tmp_path):
    source_root = Path(__file__).parents[1] / "src"
    bootstrap = (
        "import runpy,sys;"
        f"sys.path.insert(0,{str(source_root)!r});"
        "runpy.run_module('blueprint_pipeline.gear_sonic_controller_fk_adapter',"
        "run_name='__main__')"
    )
    return make_controller_fk_skeleton_projector(
        command=[sys.executable, "-c", bootstrap],
        work_dir=tmp_path / "projector",
    )


def _live_seed() -> dict:
    names = list(contract.PROTOCOL_V4_FULL_JOINT_ORDER)
    return isaac_backend.build_gear_sonic_isaac_state_snapshot(
        live_joint_names=names,
        live_joint_positions=[0.0] * len(names),
        live_joint_velocities=[0.0] * len(names),
        base_quaternion_wxyz=[1.0, 0.0, 0.0, 0.0],
        base_angular_velocity_xyz=[0.0, 0.0, 0.0],
        simulator_session_id="isaac-session-1",
        stage_id="stage-1",
        heartbeat_sequence=1,
        captured_at_ns=time.time_ns(),
        source="test_live_seed",
    )


def _action() -> dict:
    return {
        "policy_action": "UNITREE_G1_SONIC",
        "action_chunk": [0.25, -0.5],
        "action_units": ["latent", "latent"],
        "action_timing": {"control_hz": 50.0, "sample_index": 0},
        "controller_fk_state_seed": _live_seed(),
    }


def _horizon_action(frame_count: int = 3) -> dict:
    frames = [
        [float(frame_index * 1000 + value_index) for value_index in range(78)]
        for frame_index in range(frame_count)
    ]
    frames_sha256 = hashlib.sha256(
        json.dumps(frames, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "policy_action": "UNITREE_G1_SONIC",
        "action_chunk": frames[0],
        "action_units": ["latent"] * 64 + ["rad"] * 14,
        "action_timing": {
            "control_hz": 50.0,
            "sample_period_seconds": 0.02,
        },
        "controller_action": {
            "schema_version": "gear_sonic_controller_action_sequence.v1",
            "execution_mode": "bounded_model_horizon_prefix",
            "execution_frame_count": frame_count,
            "source_horizon_frame_count": 40,
            "frame_dimension": 78,
            "control_hz": 50.0,
            "sample_period_seconds": 0.02,
            "execution_duration_seconds": frame_count / 50.0,
            "frames": frames,
            "frames_sha256": frames_sha256,
            "source_frames_sha256": "f" * 64,
        },
        "controller_fk_state_seed": _live_seed(),
    }


def _run_adapter_directly(tmp_path) -> dict:
    action = _action()
    request = {
        "schema_version": "controller_fk_skeleton_request.v3",
        "step_index": 1,
        "source_action_sha256": hashlib.sha256(
            json.dumps(action, sort_keys=True, separators=(",", ":"), default=str).encode(
                "utf-8"
            )
        ).hexdigest(),
        "action": action,
        "controller_fk_state_seed": action["controller_fk_state_seed"],
        "controller_fk_state_seed_sha256": action["controller_fk_state_seed"][
            "payload_sha256"
        ],
        "camera_projection_context": json.loads(
            Path(
                os.environ[CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_ENV]
            ).read_text(encoding="utf-8")
        ),
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
    controller_manifest = json.loads(
        Path(result["controller_code_artifact"]["path"]).read_text(encoding="utf-8")
    )
    assert controller_manifest["schema_version"] == (
        "gear_sonic_controller_runtime_manifest.v1"
    )
    assert [row["relative_path"] for row in controller_manifest["files"]] == list(
        adapter.CONTROLLER_RUNTIME_ARTIFACT_RELATIVE_PATHS
    )
    assert all("executor.py" not in row["relative_path"] for row in controller_manifest["files"])
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


def test_adapter_attests_every_explicit_controller_fk_horizon_frame(
    tmp_path, monkeypatch
) -> None:
    _install_wbc(tmp_path, monkeypatch)
    _write_signing_key(tmp_path, monkeypatch)
    action = _horizon_action()

    result = _projector(tmp_path)(action, 7)

    rows = result["controller_fk_sequence"]
    assert len(rows) == 3
    assert result["controller_fk_sequence_sha256"] == adapter._canonical(rows)
    assert [row["horizon_frame_index"] for row in rows] == [0, 1, 2]
    assert [row["controller_frame_index"] for row in rows] == [241, 242, 243]
    assert result["generated_robot_state"]["controller_fk_sequence"] == rows
    assert result["generated_robot_state"]["executed_control_frame_count"] == 3
    assert result["generated_robot_state"]["joint_positions"] == rows[-1][
        "joint_positions"
    ]
    execution = result["controller_execution_contract"]
    assert execution["controller_session_count"] == 1
    assert execution["execution_frame_count"] == 3
    assert execution["source_horizon_frame_count"] == 40
    assert result["action_contract"]["execution_frame_count"] == 3
    assert result["action_contract"]["execution_frames_sha256"] == action[
        "controller_action"
    ]["frames_sha256"]


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


def test_adapter_persists_nested_executor_diagnostics_on_nonzero(
    tmp_path, monkeypatch
) -> None:
    root = _install_wbc(tmp_path, monkeypatch)
    runner = root / "failing_executor.py"
    runner.write_text(
        "import sys\n"
        "print('nested executor stdout https://example.invalid/a?X-Amz-Signature=secret')\n"
        "print('HF_TOKEN=super-secret-env-value')\n"
        "print('Authorization: Bearer do-not-record', file=sys.stderr)\n"
        "print('RuntimeError: exact_nested_failure', file=sys.stderr)\n"
        "raise SystemExit(9)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(adapter.EXECUTOR_COMMAND_ENV, f"{sys.executable} {runner}")
    monkeypatch.setenv("HF_TOKEN", "super-secret-env-value")

    with pytest.raises(
        RuntimeError,
        match="official_gear_sonic_executor_returncode_9:RuntimeError: exact_nested_failure",
    ):
        _run_adapter_directly(tmp_path)

    assert (tmp_path / "gear_sonic_executor_stdout.log").read_text() == (
        "nested executor stdout https://example.invalid/a?[REDACTED_QUERY]\n"
        "HF_TOKEN=[REDACTED]\n"
    )
    assert (tmp_path / "gear_sonic_executor_stderr.log").read_text() == (
        "Authorization: Bearer [REDACTED]\nRuntimeError: exact_nested_failure\n"
    )
    result = json.loads(
        (tmp_path / "gear_sonic_executor_command_result.json").read_text()
    )
    assert result["schema_version"] == "gear_sonic_executor_command_result.v1"
    assert result["status"] == "failed"
    assert result["returncode"] == 9
    assert result["failure_summary"] == "RuntimeError: exact_nested_failure"
    assert result["diagnostics_redacted"] is True
    assert result["stdout_truncated"] is False
    assert result["stderr_truncated"] is False
    assert result["output_present"] is False


def test_executor_diagnostics_are_bounded_and_keep_exception_tail() -> None:
    diagnostic, truncated = adapter._bounded_redacted_executor_log(
        "x" * (adapter.EXECUTOR_LOG_MAX_CHARS + 1000)
        + "\nRuntimeError: controller_revision_mismatch\n"
    )
    assert truncated is True
    assert len(diagnostic) == adapter.EXECUTOR_LOG_MAX_CHARS
    assert "[executor diagnostic truncated]" in diagnostic
    assert diagnostic.endswith("RuntimeError: controller_revision_mismatch\n")


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
    source_root = Path(__file__).parents[1] / "src"
    runner.write_text(
        f"import sys\nsys.path.insert(0, {str(source_root)!r})\n"
        + REAL_EXECUTOR_RUNNER,
        encoding="utf-8",
    )
    monkeypatch.setenv(adapter.EXECUTOR_COMMAND_ENV, f"{sys.executable} {runner}")
    context_path = Path(
        os.environ[CONTROLLER_FK_CAMERA_PROJECTION_CONTEXT_ENV]
    )
    context_payload = json.loads(context_path.read_text(encoding="utf-8"))
    registration = context_payload["standing_cross_simulator_registration"]
    standing_positions = registration["standing_joint_positions"]
    _, _, standing_landmarks, _ = official_executor._official_mujoco_fk(
        model_path=FIXTURE_MODEL,
        body_positions=standing_positions[:29],
        left_hand=standing_positions[29:36],
        right_hand=standing_positions[36:43],
    )
    by_name = {row["name"]: row for row in standing_landmarks}
    pelvis_origin = context_payload["live_isaac_pelvis_world_pose"]["position_xyz"]
    registration["isaac_named_link_world_poses"] = [
        {
            "landmark_id": name,
            "prim_path": f"/World/G1/{name}",
            "world_position_xyz": [
                pelvis_origin[index]
                + by_name[name]["model_root_relative_xyz"][index]
                for index in range(3)
            ],
        }
        for name in registration["required_landmark_names"]
    ]
    context_path.write_text(json.dumps(context_payload), encoding="utf-8")
    _write_signing_key(tmp_path, monkeypatch)
    action = {
        "policy_action": "UNITREE_G1_SONIC",
        "sonic_action_chunk": [0.01] * 78,
        "action_units": ["latent"] * 78,
        "action_timing": {"control_hz": 50.0, "sample_index": 0},
        "controller_fk_state_seed": _live_seed(),
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
