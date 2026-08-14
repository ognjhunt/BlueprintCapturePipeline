#!/usr/bin/env python3
"""Materialize the active paired-target/native construction input chain.

This command reads retained, digest-bound inputs and writes receipts only. It
does not allocate a provider or execute a simulator.
"""

from __future__ import annotations

from collections.abc import Sequence

from blueprint_pipeline.materializer_cli import Param, Step, run
from blueprint_pipeline.paired_target_articulated_kinematic_path import (
    materialize_paired_target_articulated_kinematic_path,
)
from blueprint_pipeline.paired_target_interaction_affordance_candidate import (
    materialize_paired_target_interaction_affordance_candidate,
)
from blueprint_pipeline.paired_target_native_arena_request import (
    materialize_paired_target_native_arena_requests,
)
from blueprint_pipeline.paired_target_native_camera_rig_candidate import (
    materialize_paired_target_native_camera_rig_candidate,
)
from blueprint_pipeline.paired_target_native_construction_bindings import (
    materialize_paired_target_native_construction_bindings,
)


STEPS: dict[str, Step] = {
    "interaction-affordance": Step(
        "Bind one registered replacement to its robot interaction affordance.",
        materialize_paired_target_interaction_affordance_candidate,
        {
            "task_freeze_path": Param("--task-freeze", required=True),
            "registered_asset_receipt_path": Param(
                "--registered-asset-receipt", required=True
            ),
            "robot_base_position_world_m": Param(
                "--robot-base-position",
                "JSON file containing the three-element world position.",
                required=True,
                json_file=True,
            ),
            "output_path": Param("--output", required=True),
            "parallel_jaw_stroke_m": Param(
                "--parallel-jaw-stroke-m", type=float, default=0.085
            ),
        },
    ),
    "articulated-kinematic-path": Step(
        "Derive the articulated waypoint path for one paired target.",
        materialize_paired_target_articulated_kinematic_path,
        {
            "task_freeze_path": Param("--task-freeze", required=True),
            "interaction_affordance_path": Param(
                "--interaction-affordance", required=True
            ),
            "output_path": Param("--output", required=True),
            "waypoint_count": Param("--waypoint-count", type=int, default=5),
        },
    ),
    "camera-rig": Step(
        "Bind calibrated native cameras to the paired-target interaction.",
        materialize_paired_target_native_camera_rig_candidate,
        {
            "interaction_affordance_candidate_path": Param(
                "--interaction-affordance", required=True
            ),
            "franka_placement_packet_path": Param(
                "--franka-placement-packet", required=True
            ),
            "droid_native_profile_request_path": Param(
                "--droid-native-profile-request", required=True
            ),
            "output_path": Param("--output", required=True),
            "external_height_m": Param(
                "--external-height-m", type=float, default=1.35
            ),
            "overview_lateral_distance_m": Param(
                "--overview-lateral-distance-m", type=float, default=0.9
            ),
            "overview_height_m": Param(
                "--overview-height-m", type=float, default=1.45
            ),
        },
    ),
    "construction-bindings": Step(
        "Seal the admitted 1-5 object native construction bindings.",
        materialize_paired_target_native_construction_bindings,
        {
            "manipulation_preflight_path": Param(
                "--manipulation-preflight", required=True
            ),
            "output_path": Param("--output", required=True),
        },
    ),
    "arena-requests": Step(
        "Materialize native arena requests for all admitted task inputs.",
        materialize_paired_target_native_arena_requests,
        {
            "construction_bindings_path": Param(
                "--construction-bindings", required=True
            ),
            "task_inputs": Param(
                "--task-inputs",
                "JSON file containing the task input array.",
                required=True,
                json_file=True,
            ),
            "evidence_root": Param("--evidence-root", required=True),
            "output_root": Param("--output-root", required=True),
        },
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    return run(STEPS, argv, description=__doc__)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
