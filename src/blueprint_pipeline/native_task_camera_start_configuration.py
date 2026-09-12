"""Bind the final robot reset and policy cameras before native execution.

USD joint-frame kinematics and camera frusta are provisional placement evidence.
Rendered semantic visibility and native collision checks remain mandatory.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .franka_kinematics import FRANKA_JOINT_LIMITS_RAD

SCHEMA = "policy_canary_camera_start_configuration.v1"
BODY = "/panda/Gripper/Robotiq_2F_85/base_link"
WRIST_POSITION = [0.011, -0.031, -0.074]
WRIST_QUATERNION_XYZW = [0.570, 0.576, -0.409, -0.420]
EXTERNAL_POSITION = [0.05, 0.57, 0.66]
EXTERNAL_QUATERNION_XYZW = [-0.195, 0.399, 0.805, -0.393]


def pose_matrix(position, quaternion) -> np.ndarray:
    p = np.asarray(position, dtype=float)
    q = np.asarray(quaternion, dtype=float)
    if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("policy_camera_pose_invalid")
    norm = float(np.linalg.norm(q))
    if not math.isclose(norm, 1.0, abs_tol=0.002):
        raise ValueError("policy_camera_quaternion_invalid")
    x, y, z, w = q / norm
    value = np.eye(4)
    value[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    value[:3, 3] = p
    return value


def joint_body_poses(chain, joints, base_pose) -> dict[str, np.ndarray]:
    poses = {"/panda/panda_link0": pose_matrix(base_pose["position_world_m"], base_pose["orientation_xyzw"])}
    pending = list(chain)
    while pending:
        advanced = False
        for row in pending[:]:
            if row["body0"] not in poses:
                continue
            if row["body1"] in poses:
                raise ValueError("policy_camera_joint_chain_duplicate_body")
            turn = np.eye(4)
            if row["axis"] is not None:
                axis = np.eye(3)["XYZ".index(row["axis"])]
                angle = float(joints.get(row["joint"], 0.0))
                turn = pose_matrix([0, 0, 0], [*(axis*math.sin(angle/2)), math.cos(angle/2)])
            local0, local1 = np.asarray(row["local0"], dtype=float), np.asarray(row["local1"], dtype=float)
            if any(m.shape != (4, 4) or not np.isfinite(m).all() for m in (local0, local1)):
                raise ValueError("policy_camera_joint_frame_invalid")
            poses[row["body1"]] = poses[row["body0"]] @ local0 @ turn @ np.linalg.inv(local1)
            pending.remove(row)
            advanced = True
        if not advanced:
            raise ValueError("policy_camera_joint_chain_unresolved")
    if BODY not in poses:
        raise ValueError("policy_camera_wrist_body_missing")
    return poses


def external_camera_offset_position(plan) -> list[float]:
    """Apply the preregistered world-X shift in the DROID camera's parent frame."""
    delta = float(((plan.get("scenario") or {}).get("parameters") or {}).get("external_camera_x_delta_m", 0.0))
    if not math.isfinite(delta) or abs(delta) > 0.1:
        raise ValueError("policy_camera_external_variation_out_of_bounds")
    base = plan["robot"]["base_pose_world"]
    rotation = pose_matrix(base["position_world_m"], base["orientation_xyzw"])[:3, :3]
    return (np.asarray(EXTERNAL_POSITION) + rotation.T @ np.asarray([delta, 0., 0.])).tolist()


def resolved_camera_matrices(plan, chain, joints) -> dict[str, tuple[np.ndarray, list[list[float]]]]:
    base = plan["robot"]["base_pose_world"]
    poses = joint_body_poses(chain, joints, base)
    overview = next(c for c in plan["cameras"] if c["role"] == "overview")
    if overview["pose_frame"] != "world" or overview["optical_convention"] != "opencv":
        raise ValueError("policy_camera_overview_frame_invalid")
    k = overview["intrinsics"]
    return {
        "external": (pose_matrix(base["position_world_m"], base["orientation_xyzw"])
                     @ pose_matrix(external_camera_offset_position(plan), EXTERNAL_QUATERNION_XYZW),
                     [[500., 0., 640.], [0., 500., 360.], [0., 0., 1.]]),
        "wrist": (poses[BODY] @ pose_matrix(WRIST_POSITION, WRIST_QUATERNION_XYZW),
                  [[2000/3, 0., 640.], [0., 2000/3, 360.], [0., 0., 1.]]),
        "overview": (np.asarray(overview["frame_from_camera_matrix"]).reshape(4, 4) @ np.diag([1., -1., -1., 1.]),
                     [[k["fx"], 0., k["cx"]], [0., k["fy"], k["cy"]], [0., 0., 1.]]),
    }


def camera_framing_report(plan, chain, joints) -> dict[str, Any]:
    points = {"subject_start": plan["task_spec"]["start_pose_world"][:3],
              "subject_destination": plan["task_spec"]["target_position_world_m"]}
    rows = []
    for role, (matrix, intrinsics) in resolved_camera_matrices(plan, chain, joints).items():
        camera = next(c for c in plan["cameras"] if c["role"] == role)
        width, height = (1280, 720) if role != "overview" else (camera["intrinsics"]["width"], camera["intrinsics"]["height"])
        for name, point in points.items():
            if role == "wrist" and name == "subject_destination":
                continue
            local = matrix[:3, :3].T @ (np.asarray(point)-matrix[:3, 3])
            depth = float(-local[2])
            uv = [float(intrinsics[0][2]+intrinsics[0][0]*local[0]/depth),
                  float(intrinsics[1][2]-intrinsics[1][1]*local[1]/depth)] if depth > 0 else None
            passed = bool(uv is not None and 0.05*width <= uv[0] <= 0.95*width and 0.05*height <= uv[1] <= 0.95*height)
            rows.append({"camera_role": role, "subject": name, "position_world_m": matrix[:3, 3].tolist(),
                         "world_from_camera_opengl": matrix.tolist(), "intrinsic_matrix": intrinsics,
                         "projected_center_uv": uv, "depth_m": depth, "passed": passed})
    return {"status": "passed" if all(row["passed"] for row in rows) else "blocked", "views": rows,
            "native_visibility_claimed": False, "occlusion_qualification_claimed": False}


def validate_camera_start_configuration(plan, value) -> dict[str, Any]:
    binding = deepcopy(dict(value))
    if (binding.get("schema_version") != SCHEMA
        or binding.get("configuration_digest") != canonical_digest(binding, digest_field="configuration_digest")
        or binding.get("task_success_contract_digest") != plan["task_spec"]["task_success_contract"]["contract_digest"]
        or binding.get("robot_base_pose_world") != plan["robot"]["base_pose_world"]
        or binding.get("native_qualification_claimed") is not False
        or binding.get("official_camera_calibration_preserved") is not True
        or binding.get("camera_plan_digest") != canonical_digest({"cameras": plan["cameras"]})):
        raise ValueError("policy_camera_start_configuration_binding_invalid")
    joints = binding.get("joint_reset_positions_rad")
    if not isinstance(joints, Mapping) or set(joints) != {f"panda_joint{i}" for i in range(1, 8)}:
        raise ValueError("policy_camera_start_joint_inventory_invalid")
    for index, limits in enumerate(FRANKA_JOINT_LIMITS_RAD, 1):
        angle = joints[f"panda_joint{index}"]
        if isinstance(angle, bool) or not isinstance(angle, (int, float)) or not math.isfinite(angle) or not limits[0]+0.05 <= angle <= limits[1]-0.05:
            raise ValueError("policy_camera_start_joint_margin_invalid")
    reference = binding["native_reference"]
    baseline = joint_body_poses(binding["source_joint_chain"], reference["joint_reset_positions_rad"], reference["robot_base_pose_world"])[BODY]
    baseline = baseline @ pose_matrix(WRIST_POSITION, WRIST_QUATERNION_XYZW)
    observed = np.asarray(reference["world_from_wrist_camera_opengl"], dtype=float)
    if observed.shape != (4, 4) or not np.isfinite(observed).all() or np.max(np.abs(baseline-observed)) > 0.001:
        raise ValueError("policy_camera_source_kinematics_native_readback_mismatch")
    report = camera_framing_report(plan, binding["source_joint_chain"], joints)
    if report["status"] != "passed":
        raise ValueError("policy_camera_final_reset_does_not_frame_task")
    return binding


def materialize_camera_start_from_construction(*, plan, construction, source_binding,
        native_reference_gate, robot_asset_sha256, runtime_digest, calibration_digest):
    """Reuse robot kinematics, binding this scene only to its actual native reset."""
    if (source_binding.get('schema_version') != SCHEMA
            or source_binding.get('configuration_digest') != canonical_digest(source_binding, digest_field='configuration_digest')
            or source_binding.get('source_robot_asset_sha256') != robot_asset_sha256
            or native_reference_gate.get('schema_version') != 'policy_canary_runtime_observation_integrity_gate.v1'
            or native_reference_gate.get('gate_digest') != canonical_digest(native_reference_gate, digest_field='gate_digest')
            or source_binding.get('native_reference', {}).get('gate_digest') != native_reference_gate.get('gate_digest')
            or native_reference_gate.get('candidate_policy_queried') is not False
            or construction.get('schema_version') != 'native_task_arena_construction_result.v1'
            or construction.get('result_digest') != canonical_digest(construction, digest_field='result_digest')
            or construction.get('status') != 'completed' or construction.get('construction_gate_qualified') is not True
            or construction.get('candidate_policy_queried') is not False or construction.get('blockers') not in ([], ())
            or construction.get('reset_replay', {}).get('passed') is not True):
        raise ValueError('policy_camera_construction_calibration_identity_invalid')
    wrists = [c for c in native_reference_gate.get('snapshot', {}).get('cameras', []) if c.get('role') == 'wrist']
    if len(wrists) != 1:
        raise ValueError('policy_camera_calibration_native_wrist_missing')
    observed = pose_matrix(wrists[0]['position_world_m'], wrists[0]['quaternion_world_opengl_xyzw'])
    reference = source_binding['native_reference']
    if not np.allclose(observed, reference['world_from_wrist_camera_opengl'], atol=1e-6, rtol=0):
        raise ValueError('policy_camera_calibration_native_wrist_changed')
    initial = construction.get('initial_readback') or {}
    names, positions = initial.get('robot_joint_names'), initial.get('robot_joint_positions_rad')
    pose = initial.get('robot_root_pose_world')
    if (not isinstance(names, list) or not isinstance(positions, list) or len(names) != len(positions)
            or len(set(names)) != len(names) or not isinstance(pose, list) or len(pose) != 7):
        raise ValueError('policy_camera_construction_reset_readback_missing')
    measured = dict(zip(names, positions, strict=True))
    joints = {f'panda_joint{i}': measured.get(f'panda_joint{i}') for i in range(1, 8)}
    base = plan['robot']['base_pose_world']
    if not np.allclose(pose_matrix(pose[:3], pose[3:]), pose_matrix(base['position_world_m'], base['orientation_xyzw']),
                       atol=1e-4, rtol=0):
        raise ValueError('policy_camera_construction_base_readback_changed')
    value = {'schema_version': SCHEMA, 'source_robot_asset_sha256': robot_asset_sha256,
        'source_joint_chain': deepcopy(source_binding['source_joint_chain']), 'native_reference': deepcopy(reference),
        'robot_base_pose_world': deepcopy(base), 'joint_reset_positions_rad': joints,
        'task_success_contract_digest': plan['task_spec']['task_success_contract']['contract_digest'],
        'camera_plan_digest': canonical_digest({'cameras':plan['cameras']}),
        'native_qualification_claimed': False, 'official_camera_calibration_preserved': True,
        'construction_result_digest': construction['result_digest'], 'runtime_digest': runtime_digest,
        'source_calibration_digest': calibration_digest, 'historical_scene_visibility_adopted': False}
    value['configuration_digest'] = canonical_digest(value, digest_field='configuration_digest')
    return validate_camera_start_configuration(plan, value)
