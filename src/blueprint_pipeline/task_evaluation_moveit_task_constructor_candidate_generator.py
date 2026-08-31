"""ROS 2 MoveIt Task Constructor process adapter.

MoveIt Task Constructor is deliberately not imported into the Isaac image.
The adapter talks to an exact external ROS 2 process through immutable JSON
request/result files and refuses use unless that process proves the pinned
Jazzy/MTC identity.  This keeps MTC replaceable behind the same inventory
contract while avoiding a false claim that ROS exists in the current worker.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .task_evaluation_collision_aware_candidate_generation import (
    CandidateGeneratorContext,
    CollisionAwareCandidateGenerationError,
    CommandRunner,
    JsonProcessCandidateGenerator,
)


MOVEIT_TASK_CONSTRUCTOR_BACKEND_IDENTITY: dict[str, Any] = {
    "backend_id": "moveit_task_constructor_ros2_jazzy",
    "ros_distro": "jazzy",
    "package_name": "moveit_task_constructor_core",
    "package_version": "0.1.4-2",
    "source_url": "https://github.com/moveit/moveit_task_constructor",
    "source_revision": "3cdf7f60fae2ca813de7b6616a654fef1bdb26cf",
    "source_tree": "8ca41a51ac13243256115ad4c0dfe8d06a409bee",
    "source_tag": "ros2-0.1.4",
    "release_repository": (
        "https://github.com/ros2-gbp/moveit_task_constructor-release"
    ),
    "release_revision": "2adc2fd14ba5282b7e2b4c509b505361d522fc4e",
    "release_tag": "release/jazzy/moveit_task_constructor_core/0.1.4-2",
    "license_expression": "BSD-3-Clause",
    "license_url": (
        "https://github.com/moveit/moveit_task_constructor/blob/"
        "ros2-0.1.4/LICENSE.txt"
    ),
    "runtime_kind": "separate_ros2_service_process",
}


class MoveItTaskConstructorCandidateGenerator(JsonProcessCandidateGenerator):
    """Generate typed MTC stage solutions through a separately provisioned node."""

    def __init__(
        self,
        *,
        context: CandidateGeneratorContext,
        command: Sequence[str] | None,
        runner: CommandRunner | None = None,
    ) -> None:
        if not command:
            raise CollisionAwareCandidateGenerationError(
                "moveit_task_constructor_runtime_unavailable"
            )
        kwargs: dict[str, Any] = {}
        if runner is not None:
            kwargs["runner"] = runner
        super().__init__(
            context=context,
            backend_identity=MOVEIT_TASK_CONSTRUCTOR_BACKEND_IDENTITY,
            command=command,
            require_cuda=False,
            environment={"ROS_DISTRO": "jazzy"},
            **kwargs,
        )


def moveit_task_constructor_runtime_capability_contract() -> dict[str, Any]:
    """Return the separately provisioned ROS service contract."""

    return {
        "schema_version": "task_evaluation_candidate_generator_capability.v1",
        "backend_identity": dict(MOVEIT_TASK_CONSTRUCTOR_BACKEND_IDENTITY),
        "required_capabilities": {
            "operating_system": "ubuntu-24.04",
            "ros_distro": "jazzy",
            "moveit_task_constructor_core_debian_version": "0.1.4-2",
            "planning_scene_service": True,
            "robot_model_service": True,
            "json_request_result_bridge": True,
            "sealed_robot_configuration_required": True,
            "sealed_world_configuration_required": True,
            "sealed_task_trajectory_required": True,
            "sealed_analytic_inventory_required": True,
        },
        "stage_solution_contract": {
            "ordered_stage_kinds": [
                "entry",
                "approach",
                "contact",
                "release",
                "retreat",
            ],
            "every_stage_has_joint_waypoints": True,
            "world_and_self_clearance_evidence_required": True,
            "joint_limit_compliance_evidence_required": True,
        },
        "provisioning": {
            "coinstallation_in_current_isaac_image_claimed": False,
            "separate_ros2_process_required": True,
            "runtime_probe_required_before_generation": True,
            "fail_closed_when_process_or_identity_unavailable": True,
        },
        "claim_boundary": {
            "emits_collision_aware_candidate_evidence": True,
            "native_orientation_execution_unresolved": True,
            "native_collision_and_contact_readback_unresolved": True,
            "native_camera_observability_unresolved": True,
            "native_task_execution_unresolved": True,
        },
    }


__all__ = [
    "MOVEIT_TASK_CONSTRUCTOR_BACKEND_IDENTITY",
    "MoveItTaskConstructorCandidateGenerator",
    "moveit_task_constructor_runtime_capability_contract",
]
