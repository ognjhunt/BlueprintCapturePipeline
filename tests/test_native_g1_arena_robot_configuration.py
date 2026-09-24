from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from blueprint_pipeline.gear_sonic_joint_order_contract import PROTOCOL_V4_FULL_JOINT_ORDER
from blueprint_pipeline import native_g1_arena_robot_configuration as config


def _asset(path: Path) -> None:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/G1")
    stage.SetDefaultPrim(root.GetPrim())
    for name in PROTOCOL_V4_FULL_JOINT_ORDER:
        joint = UsdPhysics.RevoluteJoint.Define(stage, "/G1/" + name)
        joint.GetLowerLimitAttr().Set(-180.0)
        joint.GetUpperLimitAttr().Set(180.0)
    for hand in ("left", "right"):
        for suffix in ("index_1_link", "thumb_2_link", "middle_1_link"):
            prim = UsdGeom.Xform.Define(stage, f"/G1/{hand}_hand_{suffix}").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
    stage.GetRootLayer().Save()


def test_pinned_robot_row_uses_usd_limits_and_dex3_grasp(
    tmp_path: Path, monkeypatch
) -> None:
    evidence = tmp_path / "evidence"
    robot_dir = evidence / "robot"
    robot_dir.mkdir(parents=True)
    asset = robot_dir / "g1.usda"
    _asset(asset)
    monkeypatch.setattr(config, "G1_USD_SHA256", hashlib.sha256(asset.read_bytes()).hexdigest())
    monkeypatch.setattr(config, "G1_USD_SIZE_BYTES", asset.stat().st_size)
    row = config.build_pinned_g1_arena_robot_configuration(
        asset=asset,
        evidence_root=evidence,
        base_pose_world={
            "position_world_m": [0.0, 0.0, 0.8],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        task_hand="right",
    )
    assert len(row["joint_position_limits_rad"]) == 43
    assert row["joint_position_limits_rad"]["right_hand_thumb_2_joint"] == pytest.approx(
        [-math.pi, math.pi]
    )
    assert row["joint_reset_positions_rad"]["right_knee_joint"] == 0.3
    assert row["actuator_parameters"]["right_knee_joint"]["stiffness"] == 300.0
    assert row["actuator_parameters"]["right_hand_index_1_joint"]["stiffness"] == 4.0
    assert row["grasp_frame"]["body_names"] == [
        "right_hand_index_1_link", "right_hand_thumb_2_link"
    ]
    assert len(row["task_contact_body_paths"]) == 3
    assert row["asset_source"]["relative_path"] == "robot/g1.usda"


def test_bad_asset_is_rejected_before_robot_configuration(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    asset = evidence / "g1.usda"
    asset.write_text("#usda 1.0\n")
    with pytest.raises(ValueError, match="asset_identity_mismatch"):
        config.build_pinned_g1_arena_robot_configuration(
            asset=asset,
            evidence_root=evidence,
            base_pose_world={
                "position_world_m": [0.0, 0.0, 0.8],
                "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            task_hand="right",
        )
