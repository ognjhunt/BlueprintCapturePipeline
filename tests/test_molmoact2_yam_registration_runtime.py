"""Development-only parity between service-user and independently installed profiles."""
from copy import deepcopy
import json
from pathlib import Path
import sys

from blueprint_pipeline.molmoact2_yam_runtime import build_policy_spec, configure_prepared_inputs
from blueprint_pipeline.team_embodiment import validate_embodiment_manifest
from tests.test_team_arm_kinematics import named_inputs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy/operator-door"))
from operator_door import public_policy_registration_contract as contract  # noqa: E402
from tests.test_operator_door_public_policy_registration import request  # noqa: E402


def test_root_fixed_profile_and_service_user_runtime_reconstruct_same_native_manifest(tmp_path):
    _, manifest = named_inputs(tmp_path)
    encoded = json.dumps(manifest)
    for side in ("left", "right"):
        for index in range(6):
            encoded = encoded.replace(f"{side}_joint_{index}", f"{side}_joint{index + 1}")
    manifest = json.loads(encoded)
    for arm, gripper in zip(manifest["arms"], manifest["grippers"], strict=True):
        arm["gripper_name"] += "_gripper"
        gripper["name"] += "_gripper"
    manifest["embodiment_id"] = contract.YAM_EMBODIMENT_ID
    for camera, name in zip(manifest["cameras"], ("overhead", "left_wrist", "right_wrist"), strict=True):
        camera["name"] = name
    manifest = validate_embodiment_manifest(manifest)
    before = deepcopy(manifest)
    prepared = configure_prepared_inputs({"schema_version": "embodiment_runtime_binding_inputs.v1",
        "robot_embodiment": manifest})
    profile = contract.load_profile(request(yam=True), ROOT / "config/policy_runtimes")
    expected = contract.expected_manifest(manifest, profile)
    assert expected == prepared["robot_embodiment"]
    assert contract.expected_policy(expected, profile) == build_policy_spec(expected)
    assert manifest == before
    for original, configured in zip(before["cameras"], expected["cameras"], strict=True):
        for key in ("mount_link", "position_m", "orientation_xyzw"):
            assert original.get(key) == configured.get(key)
    assert prepared["simulation_assumptions"][-1] == profile["policy_spec_template"]["binding"]["camera_binding_reference"]
