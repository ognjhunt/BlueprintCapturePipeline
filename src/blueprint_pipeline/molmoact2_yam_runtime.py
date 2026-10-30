"""Fixed public MolmoAct2 YAM runtime binding; registration launches no compute.

Camera mounts are the prepared source rig, with explicit simulation optics.
The requested memory floor includes simulator headroom and is not measured.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import math
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .molmoact2_yam_assets import INTERFACE_ID, MODEL_ID, MODEL_REVISION, SOURCE_REVISION, NORM_SHA256, release_inventory
from .molmoact2_yam_policy import MolmoAct2YAMBinding, JOINTS, GRIPPERS
from .team_embodiment import camera_names, validate_embodiment_manifest

PROFILE_ID = "molmoact2_bimanual_yam.v1"
CANDIDATE_ID = "molmoact2_bimanual_yam"
EMBODIMENT_ID = "i2rt_yam_bimanual"
MINIMUM_GPU_MEMORY_GB = 48
INITIAL_NATIVE_CONTROL_HZ = 60
MODEL_SOURCE = "https://huggingface.co/" + MODEL_ID
PORT = 8202
ASSUMPTION_REFERENCE = ("initial_simulation_assumption:source_configured_camera_mounts_and_poses;"
    "640x360_overhead69.4deg_wrists87deg_optics_not_publisher_or_site_calibration;"
    "overhead_D405_CAD_with_simulated_D435i_optics;native60Hz_policy30Hz_execution25_of30")


def checkpoint_identity() -> dict[str, Any]:
    return {"model_id": MODEL_ID, "revision": MODEL_REVISION,
            "files": [{"path": row["path"], "size_bytes": row["bytes"],
                       "sha256": "sha256:" + (row["publisher_lfs_sha256"] or row["locally_verified_sha256"])}
                      for row in sorted(release_inventory()["files"], key=lambda row: row["path"])
                      if row["required_for_inference"]]}


def configure_prepared_inputs(prepared: Mapping[str, Any]) -> dict[str, Any]:
    """Reseal only the declared runtime rig after the importer verified the body."""
    if prepared.get("schema_version") != "embodiment_runtime_binding_inputs.v1":
        raise ValueError("molmoact2_prepared_inputs_invalid")
    value = deepcopy(dict(prepared))
    manifest = validate_embodiment_manifest(value["robot_embodiment"])
    if manifest["embodiment_id"] != EMBODIMENT_ID or set(camera_names(manifest)) != {"overhead", "left_wrist", "right_wrist"}:
        raise ValueError("molmoact2_prepared_body_invalid")
    manifest.pop("embodiment_digest", None)
    manifest.update(control_hz=INITIAL_NATIVE_CONTROL_HZ, action_chunk_rows=30)
    for camera in manifest["cameras"]:
        camera.update(width=640, height=360, horizontal_fov_deg=69.4 if camera["name"] == "overhead" else 87.0)
    value["robot_embodiment"] = validate_embodiment_manifest(manifest)
    value["runtime_settings"] = {"control_hz": INITIAL_NATIVE_CONTROL_HZ, "action_chunk_rows": 30}
    value["simulation_assumptions"] = [*value.get("simulation_assumptions", []), ASSUMPTION_REFERENCE]
    return value


def build_policy_spec(manifest: Mapping[str, Any], *, binding: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = validate_embodiment_manifest(manifest)
    declared = dict(binding) if binding is not None else {
        "camera_names": ["overhead", "left_wrist", "right_wrist"],
        "configuration_digest": manifest["embodiment_digest"],
        "camera_binding_reference": ASSUMPTION_REFERENCE, "execution_horizon": 25}
    value = {"policy_interface_id": INTERFACE_ID, "policy_id": PROFILE_ID,
             "model_identity_digest": canonical_digest(checkpoint_identity()),
             "embodiment_id": manifest["embodiment_id"], "action_chunk_rows": 30,
             "open_loop_horizon": declared["execution_horizon"], "action_hz": 30,
             "camera_names": declared["camera_names"], "binding": declared, "source_revision": SOURCE_REVISION}
    validate_policy_spec(value, manifest)
    return value


def policy_binding(policy: Mapping[str, Any]) -> MolmoAct2YAMBinding:
    if (not isinstance(policy, Mapping) or set(policy) != {"policy_interface_id", "policy_id",
            "model_identity_digest", "embodiment_id", "action_chunk_rows", "open_loop_horizon",
            "action_hz", "camera_names", "binding", "source_revision"}
            or policy["policy_interface_id"] != INTERFACE_ID or policy["policy_id"] != PROFILE_ID
            or policy["source_revision"] != SOURCE_REVISION or policy["embodiment_id"] != EMBODIMENT_ID
            or policy["model_identity_digest"] != canonical_digest(checkpoint_identity())
            or type(policy["action_chunk_rows"]) is not int or policy["action_chunk_rows"] != 30
            or type(policy["action_hz"]) not in (int, float) or policy["action_hz"] != 30):
        raise ValueError("molmoact2_runtime_policy_identity_invalid")
    raw = dict(policy["binding"])
    raw["camera_names"] = tuple(raw["camera_names"])
    binding = MolmoAct2YAMBinding(**raw)
    binding.validate()
    if (list(binding.camera_names) != policy["camera_names"] or type(policy["open_loop_horizon"]) is not int
            or policy["open_loop_horizon"] != binding.execution_horizon):
        raise ValueError("molmoact2_runtime_policy_binding_changed")
    return binding


def validate_policy_spec(policy: Mapping[str, Any], manifest: Mapping[str, Any]) -> MolmoAct2YAMBinding:
    manifest = validate_embodiment_manifest(manifest)
    binding = policy_binding(policy)
    arms = manifest.get("arms") or []
    if (manifest["embodiment_id"] != EMBODIMENT_ID or manifest["base"]["mode"] != "fixed"
            or len(arms) != 2 or [arm["name"] for arm in arms] != ["left", "right"]
            or [arm["joint_names"] for arm in arms] != [list(JOINTS[:6]), list(JOINTS[6:])]
            or [arm["gripper_name"] for arm in arms] != list(GRIPPERS)
            or {row["name"] for row in manifest["grippers"]} != set(GRIPPERS)
            or {joint["name"] for joint in manifest["joints"] if joint["control"] == "position"} != set(JOINTS)
            or manifest["action_chunk_rows"] != 30 or manifest["control_hz"] < 30
            or not math.isclose(manifest["control_hz"] / 30, round(manifest["control_hz"] / 30))
            or binding.configuration_digest != manifest["embodiment_digest"]
            or set(binding.camera_names) != set(camera_names(manifest))
            or any((camera["width"], camera["height"]) != (640, 360) for camera in manifest["cameras"])):
        raise ValueError("molmoact2_runtime_robot_binding_invalid")
    return binding


def interface_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    manifest = spec["team_embodiment"]["manifest"]
    binding = validate_policy_spec(spec["policy_spec"], manifest)
    if (spec["checkpoint_digest"] != canonical_digest(checkpoint_identity())
            or spec["open_loop_horizon"] != binding.execution_horizon):
        raise ValueError("molmoact2_runtime_spec_identity_invalid")
    return {"policy_interface_id": INTERFACE_ID, "robot_profile_id": EMBODIMENT_ID,
            "observation_profile_id": "molmoact2_named_rgb_joint14.v1", "action_schema_id": "blueprint_joint_position_gripper_closure.v1",
            "control_hz": manifest["control_hz"], "action_horizon": 30, "execution_horizon": binding.execution_horizon,
            "camera_roles": list(binding.camera_names), "image_height": 360, "image_width": 640,
            "image_encoding": "rgb_uint8", "image_preprocessing": "identity", "history_frames": 1,
            "normalization_digest": "sha256:" + NORM_SHA256}


def validate_execution_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if (spec.get("policy_interface_id") != INTERFACE_ID or spec.get("evaluation_backend") != "robolab.v1"
            or spec.get("policy_endpoint") != {"host": "127.0.0.1", "port": PORT}
            or spec.get("execution_spec_digest") != canonical_digest(spec, digest_field="execution_spec_digest")):
        raise ValueError("molmoact2_runtime_execution_spec_invalid")
    interface_contract(spec)
    return dict(spec)


def serialized_binding(policy: Mapping[str, Any]) -> dict[str, Any]:
    return asdict(policy_binding(policy))
