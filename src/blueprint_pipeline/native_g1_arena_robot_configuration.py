"""Author the exact G1/Dex3 robot row for an existing task/site packet.

The official Arena USD owns joint limits in degrees. The pinned Arena G1_CFG
owns reset posture and PD gains. This builder converts those facts into the
shared native task packet's radian and named-joint contract.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from .gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from .native_g1_embodiment import ACTION_INTERFACE, validate_g1_robot_plan
from .native_task_robot_contact_topology import _asset_rigid_body_paths


ARENA_G1_SOURCE_REVISION = "8b82dca224f2b5af08f339f987613c59ce9cdbaa"
ARENA_G1_SOURCE_SHA256 = "38130595846dc0eb960cb45395ee4427135884251c176151593581ecaa9cc7d3"
G1_USD_SHA256 = "a7a2bab76981d19a1d76adecdfffec9b52afa34df9ba8e288ccedf410d3ce6bd"
G1_USD_SIZE_BYTES = 38195671

_RESET = {
    "left_hip_pitch_joint": -0.1,
    "left_knee_joint": 0.3,
    "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.1,
    "right_knee_joint": 0.3,
    "right_ankle_pitch_joint": -0.2,
}


def _actuator_parameters(name: str) -> dict[str, float]:
    if "_hand_" in name:
        return dict(stiffness=4.0, damping=0.5, effort_limit=5.0, velocity_limit=10.0)
    if "_ankle_" in name:
        return dict(stiffness=40.0, damping=2.0, effort_limit=50.0, velocity_limit=37.0)
    if "_hip_" in name or "_knee_" in name:
        knee = "_knee_" in name
        return dict(
            stiffness=300.0 if knee else 150.0,
            damping=4.0 if knee else 2.0,
            effort_limit=139.0 if knee else 88.0,
            velocity_limit=20.0 if knee else 32.0,
        )
    if name.startswith("waist_"):
        yaw = name == "waist_yaw_joint"
        return dict(
            stiffness=250.0,
            damping=5.0,
            effort_limit=88.0 if yaw else 50.0,
            velocity_limit=32.0 if yaw else 37.0,
        )
    if "_shoulder_" in name or "_elbow_" in name or "_wrist_" in name:
        shoulder_pitch_or_roll = name.endswith(("shoulder_pitch_joint", "shoulder_roll_joint"))
        shoulder_yaw_or_elbow = name.endswith(("shoulder_yaw_joint", "elbow_joint"))
        wrist_roll = name.endswith("wrist_roll_joint")
        return dict(
            stiffness=100.0 if shoulder_pitch_or_roll else 40.0 if shoulder_yaw_or_elbow else 20.0,
            damping=5.0 if shoulder_pitch_or_roll else 2.0,
            effort_limit=25.0 if shoulder_pitch_or_roll or shoulder_yaw_or_elbow or wrist_roll else 5.0,
            velocity_limit=37.0 if shoulder_pitch_or_roll or shoulder_yaw_or_elbow or wrist_roll else 22.0,
        )
    raise ValueError("native_g1_actuator_name_unmatched:" + name)


def _joint_limits_from_usd(asset: Path) -> dict[str, list[float]]:
    try:
        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(str(asset))
        if stage is None or not stage.GetDefaultPrim().IsValid():
            raise ValueError("default_prim_missing")
        expected = set(PROTOCOL_V4_FULL_JOINT_ORDER)
        limits: dict[str, list[float]] = {}
        for prim in stage.Traverse():
            name = prim.GetName()
            if name not in expected:
                continue
            if not prim.IsA(UsdPhysics.RevoluteJoint) or name in limits:
                raise ValueError("joint_type_or_duplicate")
            joint = UsdPhysics.RevoluteJoint(prim)
            lower = joint.GetLowerLimitAttr().Get()
            upper = joint.GetUpperLimitAttr().Get()
            if (
                isinstance(lower, bool) or isinstance(upper, bool)
                or not isinstance(lower, (int, float))
                or not isinstance(upper, (int, float))
                or not math.isfinite(lower) or not math.isfinite(upper)
                or not lower < upper
            ):
                raise ValueError("joint_limit_invalid")
            limits[name] = [math.radians(float(lower)), math.radians(float(upper))]
        if set(limits) != expected:
            raise ValueError("joint_inventory_incomplete")
        return {name: limits[name] for name in PROTOCOL_V4_FULL_JOINT_ORDER}
    except Exception as exc:  # noqa: BLE001 - malformed USD fails before packet publication
        raise ValueError("native_g1_usd_joint_limits_unavailable") from exc


def build_pinned_g1_arena_robot_configuration(
    *,
    asset: Path,
    evidence_root: Path,
    base_pose_world: Mapping[str, Any],
    task_hand: str,
) -> dict[str, Any]:
    """Return one packet-ready G1 row using the official 29+14 joint asset."""

    if task_hand not in {"left", "right"}:
        raise ValueError("native_g1_task_hand_invalid")
    if asset.is_symlink() or not asset.is_file() or asset.stat().st_size != G1_USD_SIZE_BYTES:
        raise ValueError("native_g1_arena_asset_identity_mismatch")
    digest = hashlib.sha256()
    with asset.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != G1_USD_SHA256:
        raise ValueError("native_g1_arena_asset_identity_mismatch")
    root = evidence_root.resolve()
    source = asset.resolve()
    if root not in source.parents:
        raise ValueError("native_g1_arena_asset_outside_evidence_root")
    relative = source.relative_to(root)
    limits = _joint_limits_from_usd(source)
    reset = {name: _RESET.get(name, 0.0) for name in PROTOCOL_V4_FULL_JOINT_ORDER}
    index = f"{task_hand}_hand_index_1_link"
    thumb = f"{task_hand}_hand_thumb_2_link"
    middle = f"{task_hand}_hand_middle_1_link"
    contacts = [f"{{ENV_REGEX_NS}}/Robot/{name}" for name in (index, thumb, middle)]
    if not set(contacts).issubset(_asset_rigid_body_paths(source)):
        raise ValueError("native_g1_arena_contact_bodies_missing")
    row = {
        "robot_id": "unitree_g1",
        "hand_id": "unitree_dex3_1",
        "action_interface": ACTION_INTERFACE,
        "usd_path": str(source),
        "usd_sha256": "sha256:" + G1_USD_SHA256,
        "asset_source": {
            "root": "evidence",
            "relative_path": relative.as_posix(),
            "size_bytes": G1_USD_SIZE_BYTES,
            "sha256": "sha256:" + G1_USD_SHA256,
        },
        "base_pose_world": dict(base_pose_world),
        "joint_reset_positions_rad": reset,
        "joint_position_limits_rad": limits,
        "actuator_parameters": {
            name: _actuator_parameters(name) for name in PROTOCOL_V4_FULL_JOINT_ORDER
        },
        "task_contact_body_paths": contacts,
        "grasp_frame": {
            "kind": "body_midpoint",
            "body_names": [index, thumb],
        },
        "configuration_authority": {
            "arena_source_revision": ARENA_G1_SOURCE_REVISION,
            "arena_source_sha256": "sha256:" + ARENA_G1_SOURCE_SHA256,
            "usd_sha256": "sha256:" + G1_USD_SHA256,
            "joint_limits_source": "usd_physics_revolute_joint_degrees_converted_to_radians",
        },
    }
    validate_g1_robot_plan(row)
    return row
