"""Robot bodies a policy could run on, and why most cannot run yet.

An embodiment profile is a complete robot setup: arm(s), tool, camera rig,
controller and simulator runtime. Its declared observation and action schemas
are what a policy must match. Only a profile whose asset, controller, camera rig
and reset are integrated and have executed an integration canary is
``runnable``. Others are listed so routing can say which body would unlock which
policy and what work remains. Listing a robot here makes no claim about its
capability or about who deploys it.

Adding a robot: declare its interface and ``missing`` work here. Promoting it
needs a verified asset, controller adapter, camera rig, reset pose and an
integration canary receipt; editing ``status`` alone must not make it run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

from .droid_policy_canary_embodiment import (
    DROID_ARENA_SOURCE_PATH,
    DROID_ARENA_SOURCE_REVISION,
    DROID_EMBODIMENT_ID,
)
from .task_evaluation_g1_catalog import G1_EMBODIMENT_ID
from .task_evaluation_policy_candidate_registry import (
    ABSOLUTE_JOINT_POSITION_ACTION,
    BIMANUAL_EEF_POSE_ACTION,
    DREAM1_NATIVE_JOINT_ACTION,
    HEAD_AND_WRIST_CAMERAS_OBSERVATION,
    TWO_CAMERA_OBSERVATION,
)
from .task_evaluation_rigid_relocation_native_adapter import NATIVE_PHYSICS_FREQUENCY_HZ

EmbodimentStatus = Literal["runnable", "not_integrated", "unavailable"]
Mobility = Literal["fixed", "mobile_base", "humanoid"]

FIXED_ARM_TASK_PROFILES = ("rigid_pick_place", "articulated_open_close")
_INTEGRATION_WORK = (
    "verified_robot_asset", "controller_adapter", "camera_rig", "reset_pose",
    "integration_canary_receipt",
)
_DEPLOYMENT_CATALOG_PATH = (
    Path(__file__).parent / "_catalog_data" / "deployment_embodiment_support.v1.json"
)


def deployment_embodiment_support_catalog() -> dict[str, Any]:
    """Read the planning catalog without adding runtime or admission claims.

    Return a fresh object so a response consumer cannot mutate the canonical
    metadata or change subsequent routing calls. Catalog validation and its
    response digest belong to the deployment-support boundary.
    """

    value = json.loads(_DEPLOYMENT_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("deployment_embodiment_support_catalog_not_object")
    return value


@dataclass(frozen=True)
class EmbodimentProfile:
    embodiment_id: str
    robot: str
    tool: str | None
    arm_count: int
    mobility: Mobility
    observation_schema_ids: tuple[str, ...]
    action_schema_ids: tuple[str, ...]
    controller: str | None
    simulator_runtime_id: str | None
    sensor_reference: str | None
    task_profile_ids: tuple[str, ...]
    status: EmbodimentStatus
    missing: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "runnable"


def _arm(embodiment_id: str, robot: str, arm_count: int = 1, **interface) -> EmbodimentProfile:
    """A fixed arm we can name but have not integrated into the simulator."""

    return EmbodimentProfile(
        embodiment_id=embodiment_id, robot=robot, tool=interface.get("tool"), arm_count=arm_count,
        mobility="fixed",
        observation_schema_ids=interface.get("observation_schema_ids", ()),
        action_schema_ids=interface.get("action_schema_ids", ()),
        controller=None, simulator_runtime_id=None, sensor_reference=None,
        task_profile_ids=FIXED_ARM_TASK_PROFILES, status="not_integrated",
        missing=interface.get("missing", _INTEGRATION_WORK),
        reason=f"{robot} is not integrated: its simulator asset, controller adapter, camera rig and reset pose are not built yet.",
    )


def _catalog_arm(row: dict[str, Any]) -> EmbodimentProfile:
    """List an evidenced asset option while leaving every runtime gap explicit."""

    # Only the publisher's actual paired Dream demonstrations declare its
    # interface here. Neither a robot asset nor a customer-supplied route is an
    # inferred policy match. These remain non-runnable even when schemas match.
    dream_demonstrated = row["arm_count"] == 2 and any(
        route["route_id"] == "dreamlabs_dream1"
        and route["evidence"] == "publisher_demonstrated"
        for route in row["policy_routes"]
    )
    actions = (BIMANUAL_EEF_POSE_ACTION,) if dream_demonstrated else ()
    if row["embodiment_id"] == "i2rt_yam_bimanual":
        actions = (DREAM1_NATIVE_JOINT_ACTION, BIMANUAL_EEF_POSE_ACTION)
    missing = [*_INTEGRATION_WORK, *row["required_artifacts"]]
    if dream_demonstrated:
        missing.append("eef_pose_ik_adapter")
    return _arm(
        row["embodiment_id"], row["display_name"], arm_count=row["arm_count"],
        tool=row["tool_requirement"],
        observation_schema_ids=(HEAD_AND_WRIST_CAMERAS_OBSERVATION,) if dream_demonstrated else (),
        action_schema_ids=actions,
        missing=tuple(dict.fromkeys(missing)),
    )


EMBODIMENT_REGISTRY: tuple[EmbodimentProfile, ...] = (
    EmbodimentProfile(
        embodiment_id=DROID_EMBODIMENT_ID,
        robot="Franka Emika Panda", tool="Robotiq 2F-85", arm_count=1, mobility="fixed",
        observation_schema_ids=(TWO_CAMERA_OBSERVATION,),
        action_schema_ids=(ABSOLUTE_JOINT_POSITION_ACTION,),
        controller=f"absolute joint position at 15 Hz over {NATIVE_PHYSICS_FREQUENCY_HZ} Hz physics",
        simulator_runtime_id="isaac_native_arena_v1",
        sensor_reference=f"{DROID_ARENA_SOURCE_PATH}@{DROID_ARENA_SOURCE_REVISION}",
        task_profile_ids=FIXED_ARM_TASK_PROFILES, status="runnable",
    ),
    EmbodimentProfile(
        embodiment_id=G1_EMBODIMENT_ID,
        robot="Unitree G1", tool="Dex3 hand", arm_count=2, mobility="humanoid",
        observation_schema_ids=("humanoidarena_head_rgb_state64_v1",),
        action_schema_ids=("humanoidarena_semantic_v3",),
        controller="SONIC whole-body controller", simulator_runtime_id="isaac_native_arena_g1_v1",
        sensor_reference="configs/g1_humanoidarena_checkpoint_inventory.v1.json",
        task_profile_ids=("rigid_pick_place",), status="unavailable",
        missing=("verified_site_episode", "reviewed_checkpoint_terms"),
        reason="No production policy-canary profile has executed a verified G1 episode on a site scene, and its checkpoint terms are pending review.",
    ),
    EmbodimentProfile(
        embodiment_id="boston_dynamics_atlas",
        robot="Boston Dynamics Atlas", tool=None, arm_count=2, mobility="humanoid",
        observation_schema_ids=(), action_schema_ids=(), controller=None,
        simulator_runtime_id=None, sensor_reference=None, task_profile_ids=(), status="unavailable",
        missing=("accessible_runtime", "simulator_embodiment", "policy_checkpoint_or_api", "delivery_channel"),
        reason="No Atlas runtime, simulator embodiment, policy checkpoint or delivery channel is accessible to Blueprint.",
    ),
    *(_catalog_arm(row) for row in deployment_embodiment_support_catalog()["embodiments"]),
    # Compatibility rows remain distinct from the exact variants above.
    _arm("universal_robots_ur10e", "Universal Robots UR10e"),
    _arm("fanuc_crx_10ia", "FANUC CRX-10iA"),
    _arm("abb_gofa_crb15000", "ABB GoFa CRB 15000"),
    _arm("kinova_gen3", "Kinova Gen3"),
    _arm("ufactory_xarm7", "UFACTORY xArm 7"),
)

_BY_ID = {row.embodiment_id: row for row in EMBODIMENT_REGISTRY}


def embodiment(embodiment_id: str) -> EmbodimentProfile | None:
    return _BY_ID.get(embodiment_id)


def embodiment_registry_violations() -> list[str]:
    """Why the registry would overstate which robots can run; empty when it does not."""

    violations = []
    if len({row.embodiment_id for row in EMBODIMENT_REGISTRY}) != len(EMBODIMENT_REGISTRY):
        violations.append("embodiment_id_duplicated")
    for row in EMBODIMENT_REGISTRY:
        if row.available:
            if row.missing or not (row.observation_schema_ids and row.action_schema_ids
                                   and row.controller and row.simulator_runtime_id):
                violations.append(f"runnable_embodiment_incomplete:{row.embodiment_id}")
        elif not row.reason or not row.missing:
            violations.append(f"unavailable_embodiment_without_reason:{row.embodiment_id}")
    return violations


__all__ = [
    "EMBODIMENT_REGISTRY", "EmbodimentProfile", "deployment_embodiment_support_catalog",
    "embodiment", "embodiment_registry_violations",
]
