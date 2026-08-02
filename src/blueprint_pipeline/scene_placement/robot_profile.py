"""Robot embodiment profiles: robot-specific placement data, not code.

A :class:`RobotProfile` collects every robot-specific number the placement and
spawn pipeline needs (footprint, pelvis height, reach geometry, USD prim
metadata) into one swappable object, so that a task string + a scene + a
profile produce a stance for ANY robot — not just the Unitree G1. New robots
register a profile (in code via :func:`register_robot_profile`, or as data via
:func:`robot_profile_from_json_file`); nothing downstream should hardcode G1
dimensions.

Dependency-light on purpose: stdlib only, safe to ship inside provider bundles.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


Vec3 = Tuple[float, float, float]


@dataclass(frozen=True)
class RobotProfile:
    """Embodiment parameters that drive placement, validation, and spawn."""

    robot_id: str
    embodiment_type: str = "generic_robot"

    # Standing / footprint geometry. Footprint half extent is (front-back
    # depth, lateral width, vertical half height) in the robot's local frame.
    pelvis_height_m: float = 0.79
    footprint_half_extent_xyz: Vec3 = (0.28, 0.28, 0.62)

    # Placement solver tuning.
    standing_distance_m: float = 0.55
    probe_step_m: float = 0.10
    probe_max_out_m: float = 2.5
    probe_clearance_m: float = 0.10
    openable_standoff_extra_m: float = 0.25

    # Validation tolerances.
    max_facing_error_deg: float = 30.0
    standoff_range_m: Tuple[float, float] = (0.4, 1.2)
    floor_tol_m: float = 0.08
    foot_clearance_m: float = 0.40
    min_obstacle_clearance_m: float = 0.08

    # Manipulation reach geometry (single fully-extended arm from shoulder).
    arm_span_m: float = 0.45
    shoulder_forward_offset_m: float = 0.0
    shoulder_lateral_offset_m: float = 0.16
    shoulder_above_root_m: float = 0.29
    max_effector_to_affordance_m: float = 0.35

    # Spawn / USD metadata.
    usd_prim_path: str = "/World/Robot"
    articulation_name: str = "robot"
    head_link_candidates: Tuple[str, ...] = ("camera_link", "head_link", "neck_link")

    # Nominal rest-pose link offsets relative to the pelvis/root (robot frame).
    # Used for the no-GPU dry-render skeleton preview; approximate by design.
    link_rest_offsets: Tuple[Tuple[str, Vec3], ...] = ()

    # Arm joint deltas that put the robot in a manipulation-ready pose, keyed
    # by side ("left"/"right") then joint name. Empty means "no known posing".
    manipulation_ready_arm_joint_deltas: Dict[str, Dict[str, float]] = field(
        default_factory=dict
    )

    # Robot Embodiment Pack contract fields. These are optional for placement,
    # but first-class for customer-owned robot/policy review surfaces.
    kinematics: Dict[str, object] = field(default_factory=dict)
    action_interface: Dict[str, object] = field(default_factory=dict)
    camera_rigs: Tuple[Dict[str, object], ...] = ()
    observation_schema: Dict[str, object] = field(default_factory=dict)
    simulator_asset_refs: Dict[str, object] = field(default_factory=dict)
    controller_constraints: Dict[str, object] = field(default_factory=dict)
    calibration_requirements: Dict[str, object] = field(default_factory=dict)
    claim_boundaries: Dict[str, object] = field(
        default_factory=lambda: {
            "robot_profile_is_configuration_not_execution_proof": True,
            "robot_profile_does_not_prove_physical_readiness": True,
            "robot_profile_does_not_prove_safety_validation": True,
            "default_robot_profile_is_not_customer_requirement": True,
            "explicit_customer_or_task_robot_overrides_default": True,
        }
    )

    def max_shoulder_to_affordance_m(self, margin_m: float = 0.0) -> float:
        """Farthest shoulder→affordance distance a seed pose may claim reachable."""
        return self.arm_span_m + self.max_effector_to_affordance_m + margin_m

    def to_dict(self) -> Dict[str, object]:
        """JSON-safe dict; round-trips through :func:`robot_profile_from_dict`."""
        d = dataclasses.asdict(self)
        d["footprint_half_extent_xyz"] = list(self.footprint_half_extent_xyz)
        d["standoff_range_m"] = list(self.standoff_range_m)
        d["head_link_candidates"] = list(self.head_link_candidates)
        d["link_rest_offsets"] = [[name, list(off)] for name, off in self.link_rest_offsets]
        d["camera_rigs"] = [dict(rig) for rig in self.camera_rigs]
        return d


_G1_LINK_REST_OFFSETS: Tuple[Tuple[str, Vec3], ...] = (
    ("head_link", (0.06, 0.0, 0.45)),
    ("right_shoulder_link", (0.0, -0.16, 0.34)),
    ("right_elbow_link", (0.06, -0.20, 0.10)),
    ("right_wrist_link", (0.12, -0.22, -0.06)),
    ("right_hand_link", (0.16, -0.23, -0.14)),
    ("left_shoulder_link", (0.0, 0.16, 0.34)),
    ("left_elbow_link", (0.06, 0.20, 0.10)),
    ("left_wrist_link", (0.12, 0.22, -0.06)),
    ("left_hand_link", (0.16, 0.23, -0.14)),
)

UNITREE_G1_PROFILE = RobotProfile(
    robot_id="unitree_g1",
    embodiment_type="humanoid",
    pelvis_height_m=0.79,
    footprint_half_extent_xyz=(0.12, 0.23, 0.62),
    arm_span_m=0.45,
    shoulder_forward_offset_m=0.0,
    shoulder_lateral_offset_m=0.16,
    shoulder_above_root_m=0.29,
    max_effector_to_affordance_m=0.35,
    usd_prim_path="/World/G1",
    articulation_name="g1",
    link_rest_offsets=_G1_LINK_REST_OFFSETS,
    kinematics={
        "base": "floating_or_locomotion_controller",
        "manipulators": ["left_arm", "right_arm"],
        "end_effectors": ["left_hand", "right_hand"],
        "action_space": "whole_body_or_arm_hand_chunks",
    },
    action_interface={
        "schema_ref": "blueprint://schemas/robot_eval_action_trace.v1",
        "preferred_action_chunk": "unitree_g1_normalized_action_chunk",
        "lerobot_export": {
            "schema_version": "robot_profile_lerobot_action_layout.v1",
            "layout_id": "unitree_g1_whole_body_arm_hand_chunks_v1",
            "action_dim": 78,
            "vector_keys": [
                "unitree_g1_normalized_action_chunk",
                "unitree_g1_sonic_action_chunk",
                "unitree_g1_sonic_latent_action_chunk",
                "whole_body_action_chunk",
                "whole_body_action_vector",
                "action_chunk",
            ],
            "segments": [
                {
                    "name": "base_velocity_xy_yaw",
                    "start": 0,
                    "end": 3,
                    "source_keys": ["base_velocity_xy_yaw", "base_velocity"],
                },
                {
                    "name": "left_arm_joint_delta",
                    "start": 3,
                    "end": 10,
                    "source_keys": ["left_arm_joint_delta", "left_arm_joints"],
                },
                {
                    "name": "left_hand_joint_delta",
                    "start": 10,
                    "end": 17,
                    "source_keys": ["left_hand_joint_delta", "left_hand_joints"],
                },
                {
                    "name": "right_arm_joint_delta",
                    "start": 17,
                    "end": 24,
                    "source_keys": ["right_arm_joint_delta", "right_arm_joints"],
                },
                {
                    "name": "right_hand_joint_delta",
                    "start": 24,
                    "end": 31,
                    "source_keys": ["right_hand_joint_delta", "right_hand_joints"],
                },
                {
                    "name": "whole_body_residual_or_policy_latent",
                    "start": 31,
                    "end": 78,
                    "source_keys": ["whole_body_residual_or_policy_latent"],
                },
            ],
            "legacy_supported_layouts": ["sc3_7d_delta_end_effector_pose"],
            "claim_boundary": (
                "Layout declares profile-compatible export vector shape; it does not "
                "prove Unitree policy execution, safety validation, or physical readiness."
            ),
        },
        "claim_boundary": "Unitree-native policy required for G1 policy-execution claims.",
    },
    camera_rigs=(
        {
            "camera_id": "head_rgbd",
            "mount": "head",
            "modalities": ["rgb", "depth"],
            "calibration_status": "owner_or_profile_calibration_required_for_launch",
        },
        {
            "camera_id": "wrist_or_hand_rgb",
            "mount": "wrist_or_hand",
            "modalities": ["rgb"],
            "calibration_status": "optional_support_until_owner_calibrated",
        },
    ),
    observation_schema={
        "schema_ref": "blueprint://schemas/robot_eval_observation.v1",
        "required_fields": [
            "observation_id",
            "scenario_eval_run_id",
            "camera",
            "visual_observation",
        ],
    },
    simulator_asset_refs={
        "usd_prim_path": "/World/G1",
        "mjcf_root_env": "BLUEPRINT_MUJOCO_G1_MODEL_ROOT",
        "isaac_asset_family": "Isaac/Robots/Unitree/G1",
    },
    controller_constraints={
        "requires_unitree_native_policy_for_g1_policy_claims": True,
        "wam_or_openvla_may_only_supply_evaluator_support": True,
    },
    calibration_requirements={
        "owner_camera_intrinsics_required_for_launch_ready_profile": True,
        "owner_camera_extrinsics_required_for_launch_ready_profile": True,
        "default_profile_launch_mode": "smoke_only_until_owner_calibration",
    },
    claim_boundaries={
        "robot_profile_is_configuration_not_execution_proof": True,
        "unitree_g1_is_default_humanoid_not_customer_requirement": True,
        "profile_does_not_prove_unitree_policy_execution": True,
        "profile_does_not_prove_physical_readiness": True,
        "profile_does_not_prove_safety_validation": True,
    },
)

# Blueprint's general-purpose default embodiment is the Franka Panda.  This is
# a selection default only: an explicit customer/task robot always wins, and a
# task that explicitly requires a humanoid selects the Unitree G1 profile below
# via ``default_robot_id_for_embodiment``.
FRANKA_PANDA_PROFILE = RobotProfile(
    robot_id="franka_panda",
    embodiment_type="fixed_base_single_arm_manipulator",
    pelvis_height_m=0.0,
    footprint_half_extent_xyz=(0.18, 0.18, 0.36),
    standing_distance_m=0.0,
    probe_step_m=0.05,
    probe_max_out_m=0.9,
    probe_clearance_m=0.05,
    openable_standoff_extra_m=0.10,
    max_facing_error_deg=45.0,
    standoff_range_m=(0.15, 0.85),
    floor_tol_m=0.02,
    foot_clearance_m=0.0,
    min_obstacle_clearance_m=0.05,
    arm_span_m=0.855,
    shoulder_forward_offset_m=0.0,
    shoulder_lateral_offset_m=0.0,
    shoulder_above_root_m=0.333,
    max_effector_to_affordance_m=0.30,
    usd_prim_path="/World/Franka",
    articulation_name="franka",
    head_link_candidates=("camera_link", "panda_hand"),
    kinematics={
        "base": "fixed",
        "arm_count": 1,
        "degrees_of_freedom": 7,
        "simulator_dof_count_with_gripper": 9,
        "gripper": "parallel_jaw",
        "has_legs": False,
        "has_torso": False,
    },
    action_interface={
        "action_schema_id": "sc3_7d_delta_end_effector.v1",
        "dim": 7,
        "representation": "7d_delta_end_effector_pose",
        "claim_boundary": (
            "Action interface declares the profile-compatible command layout; "
            "it does not prove Franka policy execution or physical readiness."
        ),
    },
    camera_rigs=(
        {
            "camera_id": "fixed_external_rgbd",
            "mount": "external_static",
            "modalities": ["rgb", "depth"],
            "calibration_status": "owner_or_profile_calibration_required_for_launch",
        },
        {
            "camera_id": "wrist_rgb",
            "mount": "wrist",
            "modalities": ["rgb"],
            "calibration_status": "owner_or_profile_calibration_required_for_launch",
        },
    ),
    observation_schema={
        "schema_ref": "blueprint://schemas/robot_eval_observation.v1",
        "required_fields": [
            "observation_id",
            "scenario_eval_run_id",
            "camera",
            "visual_observation",
        ],
    },
    simulator_asset_refs={
        "usd_prim_path": "/World/Franka",
        "isaac_asset": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd",
        "isaac_asset_family": "Isaac/Robots/FrankaRobotics/FrankaPanda",
        "mjcf_model": "mjx_panda.xml",
    },
    controller_constraints={
        "requires_franka_compatible_action_space_for_policy_claims": True,
        "wam_or_openvla_may_only_supply_evaluator_support": True,
    },
    calibration_requirements={
        "owner_camera_intrinsics_required_for_launch_ready_profile": True,
        "owner_camera_extrinsics_required_for_launch_ready_profile": True,
        "default_profile_launch_mode": "smoke_only_until_owner_calibration",
    },
    claim_boundaries={
        "robot_profile_is_configuration_not_execution_proof": True,
        "franka_panda_is_default_robot_not_customer_requirement": True,
        "explicit_customer_or_task_robot_overrides_default": True,
        "profile_does_not_prove_franka_policy_execution": True,
        "profile_does_not_prove_physical_readiness": True,
        "profile_does_not_prove_safety_validation": True,
    },
)

# A second registered embodiment that differs from the G1 on the axes that
# actually matter for evaluation -- action interface (7-D delta end-effector,
# not 78-D whole-body), observation interface (fixed external + wrist, no head
# camera), and kinematics (fixed base, no pelvis or legs). It deliberately
# matches the DROID single-arm family that public real-world leaderboards
# evaluate, so the harness can be exercised end to end against an embodiment
# whose reference outcomes already exist.
#
# Registering it is a conformance fixture, not a product claim: it proves the
# registry, export and placement paths are not silently hardcoded to one robot.
FIXED_BASE_SINGLE_ARM_PROFILE = RobotProfile(
    robot_id="fixed_base_single_arm_reference",
    embodiment_type="fixed_base_single_arm_manipulator",
    # A table-mounted arm has no pelvis; the root sits at the mount plate.
    pelvis_height_m=0.0,
    footprint_half_extent_xyz=(0.18, 0.18, 0.36),
    standing_distance_m=0.0,
    probe_step_m=0.05,
    probe_max_out_m=0.9,
    probe_clearance_m=0.05,
    openable_standoff_extra_m=0.10,
    max_facing_error_deg=45.0,
    standoff_range_m=(0.15, 0.85),
    floor_tol_m=0.02,
    foot_clearance_m=0.0,
    min_obstacle_clearance_m=0.05,
    arm_span_m=0.855,
    shoulder_forward_offset_m=0.0,
    shoulder_lateral_offset_m=0.0,
    shoulder_above_root_m=0.333,
    max_effector_to_affordance_m=0.30,
    usd_prim_path="/World/SingleArm",
    articulation_name="single_arm",
    head_link_candidates=("camera_link",),
    kinematics={
        "base": "fixed",
        "arm_count": 1,
        "degrees_of_freedom": 7,
        "gripper": "parallel_jaw",
        "has_legs": False,
        "has_torso": False,
        "claim_boundary": (
            "Kinematics describe a reference fixed-base arm; they do not "
            "identify a specific vendor unit or prove calibration."
        ),
    },
    action_interface={
        "action_schema_id": "sc3_7d_delta_end_effector.v1",
        "dim": 7,
        "representation": "7d_delta_end_effector_pose",
        "claim_boundary": (
            "Action interface declares the profile-compatible command layout; "
            "it does not prove policy execution or physical readiness."
        ),
    },
    camera_rigs=(
        {
            "camera_id": "fixed_external_left",
            "mount": "external_static",
            "modalities": ["rgb"],
            "calibration_status": "owner_or_profile_calibration_required_for_launch",
        },
        {
            "camera_id": "fixed_external_right",
            "mount": "external_static",
            "modalities": ["rgb"],
            "calibration_status": "owner_or_profile_calibration_required_for_launch",
        },
        {
            "camera_id": "wrist_rgb",
            "mount": "wrist",
            "modalities": ["rgb"],
            "calibration_status": "owner_or_profile_calibration_required_for_launch",
        },
    ),
    observation_schema={
        "schema_ref": "blueprint://schemas/robot_eval_observation.v1",
        "required_fields": [
            "observation_id",
            "scenario_eval_run_id",
            "camera",
            "visual_observation",
        ],
    },
    simulator_asset_refs={
        "usd_prim_path": "/World/SingleArm",
        "isaac_asset_family": "Isaac/Robots/Franka",
    },
    controller_constraints={
        "requires_registered_action_space_for_policy_claims": True,
        "wam_or_openvla_may_only_supply_evaluator_support": True,
    },
    calibration_requirements={
        "owner_camera_intrinsics_required_for_launch_ready_profile": True,
        "owner_camera_extrinsics_required_for_launch_ready_profile": True,
        "default_profile_launch_mode": "smoke_only_until_owner_calibration",
    },
    claim_boundaries={
        "robot_profile_is_configuration_not_execution_proof": True,
        "profile_is_a_conformance_fixture_not_a_supported_product_embodiment": True,
        "profile_does_not_prove_policy_execution": True,
        "profile_does_not_prove_physical_readiness": True,
        "profile_does_not_prove_safety_validation": True,
    },
)

_REGISTRY: Dict[str, RobotProfile] = {
    FRANKA_PANDA_PROFILE.robot_id: FRANKA_PANDA_PROFILE,
    UNITREE_G1_PROFILE.robot_id: UNITREE_G1_PROFILE,
    FIXED_BASE_SINGLE_ARM_PROFILE.robot_id: FIXED_BASE_SINGLE_ARM_PROFILE,
}

DEFAULT_ROBOT_ID = FRANKA_PANDA_PROFILE.robot_id
DEFAULT_HUMANOID_ROBOT_ID = UNITREE_G1_PROFILE.robot_id
ROBOT_EMBODIMENT_PACK_SCHEMA_VERSION = "robot_embodiment_pack.v1"
RobotEmbodimentPack = RobotProfile


def default_robot_id_for_embodiment(embodiment_type: str | None = None) -> str:
    """Return the configured default without overriding an explicit robot ID.

    The general default is Franka Panda.  Humanoid morphology is the only
    special case and resolves to Unitree G1.  Callers with an explicit robot ID
    should use that ID directly rather than calling this selector.
    """

    normalized = str(embodiment_type or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "humanoid",
        "bipedal_humanoid",
        "legged_humanoid",
        "humanoid_robot",
    }:
        return DEFAULT_HUMANOID_ROBOT_ID
    return DEFAULT_ROBOT_ID


def register_robot_profile(profile: RobotProfile) -> RobotProfile:
    """Register (or replace) a profile so ``get_robot_profile`` can find it."""
    _REGISTRY[profile.robot_id] = profile
    return profile


def known_robot_ids() -> List[str]:
    return sorted(_REGISTRY)


class UnknownRobotProfileError(KeyError):
    """Raised when a caller names an unregistered robot profile.

    Subclasses ``KeyError`` so existing handlers keep working, while giving
    call sites that accept a job-request-supplied ``robot_id`` something
    specific to catch instead of a bare lookup failure.
    """


def get_robot_profile(robot_id: str) -> RobotProfile:
    try:
        return _REGISTRY[robot_id]
    except KeyError:
        raise UnknownRobotProfileError(
            f"unknown robot_id {robot_id!r}; known: {', '.join(known_robot_ids())}. "
            "Register one via register_robot_profile() or load a JSON profile via "
            "robot_profile_from_json_file()."
        ) from None


_FIELD_NAMES = {f.name for f in dataclasses.fields(RobotProfile)}
_TUPLE3_FIELDS = ("footprint_half_extent_xyz",)
_TUPLE2_FIELDS = ("standoff_range_m",)


def robot_profile_from_dict(data: Dict[str, object]) -> RobotProfile:
    """Build a profile from JSON-shaped data. Unknown keys are an error so a
    typo (``pelvis_heigth_m``) fails loudly instead of silently keeping the
    default."""
    if not isinstance(data, dict):
        raise ValueError(f"robot profile must be a JSON object, got {type(data).__name__}")
    unknown = sorted(set(data) - _FIELD_NAMES)
    if unknown:
        raise ValueError(
            f"unknown robot profile key(s): {', '.join(unknown)}; "
            f"valid keys: {', '.join(sorted(_FIELD_NAMES))}"
        )
    if "robot_id" not in data or not str(data["robot_id"]).strip():
        raise ValueError("robot profile requires a non-empty 'robot_id'")
    kwargs = dict(data)
    for key in _TUPLE3_FIELDS + _TUPLE2_FIELDS:
        if key in kwargs:
            kwargs[key] = tuple(float(v) for v in kwargs[key])  # type: ignore[arg-type]
    if "head_link_candidates" in kwargs:
        kwargs["head_link_candidates"] = tuple(str(v) for v in kwargs["head_link_candidates"])  # type: ignore[arg-type]
    if "link_rest_offsets" in kwargs:
        kwargs["link_rest_offsets"] = tuple(
            (str(name), tuple(float(v) for v in off)) for name, off in kwargs["link_rest_offsets"]  # type: ignore[misc]
        )
    if "camera_rigs" in kwargs:
        kwargs["camera_rigs"] = tuple(
            dict(rig) for rig in kwargs["camera_rigs"] if isinstance(rig, dict)  # type: ignore[union-attr]
        )
    return RobotProfile(**kwargs)  # type: ignore[arg-type]


def robot_profile_from_json_file(path: str | Path) -> RobotProfile:
    with open(path, "r", encoding="utf-8") as fh:
        return robot_profile_from_dict(json.load(fh))


def robot_embodiment_pack_contract(profile: RobotProfile) -> Dict[str, object]:
    """Return the review contract for a robot profile / embodiment pack."""

    return {
        "schema_version": ROBOT_EMBODIMENT_PACK_SCHEMA_VERSION,
        "robot_id": profile.robot_id,
        "embodiment_type": profile.embodiment_type,
        "kinematics": dict(profile.kinematics),
        "action_interface": dict(profile.action_interface),
        "camera_rigs": [dict(rig) for rig in profile.camera_rigs],
        "observation_schema": dict(profile.observation_schema),
        "simulator_asset_refs": dict(profile.simulator_asset_refs),
        "controller_constraints": dict(profile.controller_constraints),
        "calibration_requirements": dict(profile.calibration_requirements),
        "claim_boundaries": dict(profile.claim_boundaries),
        "data_driven_profile": True,
        "g1_only_contract": False,
    }
