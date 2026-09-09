"""Final reset/camera checks reproduce the failed wrist view without Isaac."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.droid_policy_canary_embodiment import (
    apply_droid_policy_canary_profile,
    DROID_NATIVE_RESET_JOINTS_RAD,
)
from blueprint_pipeline.native_task_camera_start_configuration import (
    camera_framing_report,
    external_camera_offset_position,
    resolved_camera_matrices,
    validate_camera_start_configuration,
)


def fixture():
    return json.loads(
        (Path(__file__).parent / "fixtures/policy_camera_start_configuration.json").read_text()
    )


def test_source_joint_camera_prediction_preserves_task_aligned_start():
    plan = fixture()
    binding = plan["policy_canary_camera_start_configuration"]
    assert validate_camera_start_configuration(plan, binding) == binding
    resolved = apply_droid_policy_canary_profile(plan)
    assert resolved["robot"]["joint_reset_positions_rad"] == binding["joint_reset_positions_rad"]
    assert resolved["robot"]["joint_reset_positions_rad"] != DROID_NATIVE_RESET_JOINTS_RAD
    assert (
        resolved["policy_canary_embodiment_profile"]["preserve_official_policy_camera_calibration"]
        is True
    )
    assert (
        resolved["policy_canary_embodiment_profile"]["preserve_official_reset_joint_positions"]
        is False
    )
    assert (
        camera_framing_report(
            resolved, binding["source_joint_chain"], binding["joint_reset_positions_rad"]
        )["status"]
        == "passed"
    )
    # The predecessor's later default-reset override loses the task at the wrist.
    failed = camera_framing_report(
        plan, binding["source_joint_chain"], DROID_NATIVE_RESET_JOINTS_RAD
    )
    assert failed["status"] == "blocked"
    assert (
        next(row for row in failed["views"] if row["camera_role"] == "wrist")[
            "projected_center_uv"
        ][1]
        > 720
    )


@pytest.mark.parametrize(
    "fault", ["base", "camera", "joint_limit", "source_chain", "reference", "digest", "task"]
)
def test_start_binding_refuses_configuration_drift(fault):
    plan = fixture()
    binding = deepcopy(plan["policy_canary_camera_start_configuration"])
    if fault == "base":
        plan["robot"]["base_pose_world"]["position_world_m"][0] += 0.1
    if fault == "camera":
        plan["cameras"][-1]["frame_from_camera_matrix"][3] += 0.1
    if fault == "joint_limit":
        binding["joint_reset_positions_rad"]["panda_joint4"] = 0
    if fault == "source_chain":
        binding["source_joint_chain"][0]["local0"][0][3] += 0.1
    if fault == "reference":
        binding["native_reference"]["world_from_wrist_camera_opengl"][2][3] += 0.1
    if fault == "task":
        binding["task_success_contract_digest"] = "sha256:" + "b" * 64
    if fault != "digest":
        binding["configuration_digest"] = canonical_digest(
            binding, digest_field="configuration_digest"
        )
    else:
        binding["configuration_digest"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="policy_camera"):
        validate_camera_start_configuration(plan, binding)


def test_resealed_default_reset_is_refused_before_gpu():
    plan = fixture()
    binding = plan["policy_canary_camera_start_configuration"]
    binding["joint_reset_positions_rad"] = dict(DROID_NATIVE_RESET_JOINTS_RAD)
    binding["configuration_digest"] = canonical_digest(binding, digest_field="configuration_digest")
    with pytest.raises(ValueError, match="final_reset_does_not_frame_task"):
        apply_droid_policy_canary_profile(plan)


def test_preregistered_camera_shift_reaches_the_official_parent_frame():
    import numpy as np
    plan = fixture()
    binding = plan["policy_canary_camera_start_configuration"]
    nominal = resolved_camera_matrices(plan, binding["source_joint_chain"], binding["joint_reset_positions_rad"])["external"][0]
    plan["scenario"] = {"parameters": {"external_camera_x_delta_m": 0.02}}
    shifted = resolved_camera_matrices(plan, binding["source_joint_chain"], binding["joint_reset_positions_rad"])["external"][0]
    np.testing.assert_allclose(shifted[:3, 3]-nominal[:3, 3], [0.02, 0., 0.], atol=1e-12)
    # The base faces 180 degrees, so a world-positive shift is parent-negative.
    assert external_camera_offset_position(plan)[0] == pytest.approx(0.03)
    assert validate_camera_start_configuration(plan, binding) == binding
