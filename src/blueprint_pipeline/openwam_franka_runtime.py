"""Fixed public OpenWAM single-FR3 runtime binding; registration never launches compute.

The initial frame/camera/cadence configuration is an explicit simulation
assumption. GPU memory is a conservative requested floor, not measured proof.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .openwam_franka_assets import EXPECTED_FILES
from .openwam_franka_policy import INTERFACE_ID, MODEL_ID, MODEL_REVISION, SOURCE_REVISION, OpenWAMFrankaBinding
from .team_embodiment import camera_names, validate_embodiment_manifest

PROFILE_ID = "openwam_alpha_real_franka.v1"
CANDIDATE_ID = "openwam_alpha_franka"
EMBODIMENT_ID = "franka_research_3_robotiq_2f85_sim"
MINIMUM_GPU_MEMORY_GB = 80
INITIAL_CONTROL_HZ = 15
INITIAL_NATIVE_CONTROL_HZ = 60
MODEL_SOURCE = "https://huggingface.co/" + MODEL_ID
PORT = 8848
ASSUMPTION_REFERENCE = ("initial_simulation_assumption:FR3_Robotiq2F85_zero_offset_mount_at_link8;"
    "nominal_134mm_grasp_TCP_as_recorded_EEF;publisher_SKU_mount_TCP_calibration_unknown;"
    "source_camera_poses_90deg_optics_not_training_calibration;dataset_policy15Hz_native60Hz")


def configure_prepared_inputs(prepared: Mapping[str, Any]) -> dict[str, Any]:
    from copy import deepcopy

    result = deepcopy(dict(prepared))
    manifest = validate_embodiment_manifest(result["robot_embodiment"])
    if manifest["embodiment_id"] != EMBODIMENT_ID or camera_names(manifest) != ["overhead", "arm_wrist"]:
        raise ValueError("openwam_franka_prepared_body_invalid")
    manifest.pop("embodiment_digest")
    manifest.update(control_hz=INITIAL_NATIVE_CONTROL_HZ, action_chunk_rows=1)
    for camera in manifest["cameras"]:
        camera.update(width=640, height=480)
    result["robot_embodiment"] = validate_embodiment_manifest(manifest)
    result["runtime_settings"] = {"control_hz": INITIAL_NATIVE_CONTROL_HZ, "action_chunk_rows": 1}
    result.setdefault("simulation_assumptions", []).append(ASSUMPTION_REFERENCE)
    return result


def checkpoint_identity() -> dict[str, Any]:
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "files": [{"path": name, "size_bytes": size, "sha256": "sha256:" + digest}
                      for name, (size, digest) in sorted(EXPECTED_FILES.items())]}


def build_policy_spec(manifest: Mapping[str, Any], *, binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = validate_embodiment_manifest(manifest)
    identity = {"position_m": [0.0, 0.0, 0.0], "orientation_xyzw": [0.0, 0.0, 0.0, 1.0]}
    declared = dict(binding) if binding is not None else {
        "camera_names": ["overhead", "arm_wrist"],
        "control_hz": INITIAL_CONTROL_HZ, "configuration_digest": manifest["embodiment_digest"],
        "frame_binding_reference": ASSUMPTION_REFERENCE,
        "policy_base_from_native_base": {arm: dict(identity) for arm in ("arm",)},
        "native_tcp_from_policy_tcp": {arm: dict(identity) for arm in ("arm",)},
        "execution_horizon": 8}
    value = {"policy_interface_id": INTERFACE_ID, "policy_id": PROFILE_ID,
             "model_identity_digest": canonical_digest(checkpoint_identity()),
             "embodiment_id": manifest["embodiment_id"], "action_chunk_rows": 1, "open_loop_horizon": 1,
             "action_hz": declared["control_hz"], "camera_names": declared["camera_names"],
             "binding": declared, "source_revision": SOURCE_REVISION}
    validate_policy_spec(value, manifest)
    return value


def policy_binding(policy: Mapping[str, Any]) -> OpenWAMFrankaBinding:
    if (not isinstance(policy, Mapping) or set(policy) != {"policy_interface_id", "policy_id",
            "model_identity_digest", "embodiment_id", "action_chunk_rows", "open_loop_horizon",
            "action_hz", "camera_names", "binding", "source_revision"}
            or policy["policy_interface_id"] != INTERFACE_ID or policy["policy_id"] != PROFILE_ID
            or policy["source_revision"] != SOURCE_REVISION or policy["embodiment_id"] != EMBODIMENT_ID
            or policy["model_identity_digest"] != canonical_digest(checkpoint_identity())
            or type(policy["action_chunk_rows"]) is not int or policy["action_chunk_rows"] != 1
            or type(policy["open_loop_horizon"]) is not int or policy["open_loop_horizon"] != 1):
        raise ValueError("openwam_runtime_policy_identity_invalid")
    raw = dict(policy["binding"])
    raw["camera_names"] = tuple(raw["camera_names"])
    binding = OpenWAMFrankaBinding(**raw)
    binding.validate()
    if (list(binding.camera_names) != policy["camera_names"] or binding.control_hz != policy["action_hz"]):
        raise ValueError("openwam_runtime_policy_binding_changed")
    return binding


def validate_policy_spec(policy: Mapping[str, Any], manifest: Mapping[str, Any]) -> OpenWAMFrankaBinding:
    manifest = validate_embodiment_manifest(manifest)
    binding = policy_binding(policy)
    arms = manifest.get("arms") or []
    if (manifest["embodiment_id"] != EMBODIMENT_ID or manifest["base"]["mode"] != "fixed"
            or len(arms) != 1 or [arm["name"] for arm in arms] != ["arm"]
            or any(len(arm["joint_names"]) != 7 for arm in arms)
            or len({arm["gripper_name"] for arm in arms}) != 1 or len(manifest["grippers"]) != 1
            or arms[0]["joint_names"] != [f"fr3_joint{index}" for index in range(1, 8)]
            or arms[0]["tool_link"] != "robotiq_85_base_link"
            or set(manifest["grippers"][0]["contact_links"]) != {"robotiq_85_left_finger_tip_link", "robotiq_85_right_finger_tip_link"}
            or manifest["action_chunk_rows"] != 1 or binding.control_hz > manifest["control_hz"]
            or not math.isclose(manifest["control_hz"] / binding.control_hz,
                                round(manifest["control_hz"] / binding.control_hz))
            or binding.configuration_digest != manifest["embodiment_digest"]
            or list(binding.camera_names) != ["overhead", "arm_wrist"]
            or list(binding.camera_names) != camera_names(manifest)
            or any((camera["width"], camera["height"]) != (640, 480) for camera in manifest["cameras"])):
        raise ValueError("openwam_runtime_robot_binding_invalid")
    return binding


def interface_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    manifest = spec["team_embodiment"]["manifest"]
    binding = validate_policy_spec(spec["policy_spec"], manifest)
    if spec["checkpoint_digest"] != canonical_digest(checkpoint_identity()) or spec["open_loop_horizon"] != 1:
        raise ValueError("openwam_runtime_spec_identity_invalid")
    return {"policy_interface_id": INTERFACE_ID, "robot_profile_id": EMBODIMENT_ID,
            "observation_profile_id": "openwam_franka_two_rgb_eef10.v1", "action_schema_id": "named_arm_pose_chunk.v1",
            "control_hz": manifest["control_hz"], "action_horizon": 1, "execution_horizon": 1,
            "camera_roles": list(binding.camera_names), "image_height": 384, "image_width": 320,
            "image_encoding": "png", "image_preprocessing": "upstream_two_view_l_shape_structural_blank_third",
            "history_frames": 1, "normalization_digest": "sha256:" + EXPECTED_FILES["normalization_stats.npy"][1]}


def validate_execution_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if (spec.get("policy_interface_id") != INTERFACE_ID or spec.get("evaluation_backend") != "robolab.v1"
            or spec.get("policy_endpoint") != {"host": "127.0.0.1", "port": PORT}
            or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")):
        raise ValueError("openwam_runtime_execution_spec_invalid")
    interface_contract(spec)
    return dict(spec)


def serialized_binding(policy: Mapping[str, Any]) -> dict[str, Any]:
    return asdict(policy_binding(policy))
