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
)

_REGISTRY: Dict[str, RobotProfile] = {UNITREE_G1_PROFILE.robot_id: UNITREE_G1_PROFILE}

DEFAULT_ROBOT_ID = UNITREE_G1_PROFILE.robot_id


def register_robot_profile(profile: RobotProfile) -> RobotProfile:
    """Register (or replace) a profile so ``get_robot_profile`` can find it."""
    _REGISTRY[profile.robot_id] = profile
    return profile


def known_robot_ids() -> List[str]:
    return sorted(_REGISTRY)


def get_robot_profile(robot_id: str) -> RobotProfile:
    try:
        return _REGISTRY[robot_id]
    except KeyError:
        raise KeyError(
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
    return RobotProfile(**kwargs)  # type: ignore[arg-type]


def robot_profile_from_json_file(path: str | Path) -> RobotProfile:
    with open(path, "r", encoding="utf-8") as fh:
        return robot_profile_from_dict(json.load(fh))
