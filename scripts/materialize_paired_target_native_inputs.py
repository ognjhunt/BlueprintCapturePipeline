#!/usr/bin/env python3
"""Materialize the active replacement/native construction input chain.

This command reads retained, digest-bound inputs and writes receipts only. It
does not allocate a provider or execute a simulator.
"""

from __future__ import annotations

from collections.abc import Sequence

from blueprint_pipeline.materializer_cli import Param, Step, run
from blueprint_pipeline.native_task_arena_packet import (
    materialize_native_task_arena_appearance_variant_request,
    materialize_native_task_arena_packet,
)
from blueprint_pipeline.native_droid_grasp_swept_volume import (
    materialize_native_droid_grasp_swept_volume,
)
from blueprint_pipeline.native_task_arena_policy_bundle import (
    materialize_native_task_policy_execution_spec,
)
from blueprint_pipeline.native_task_policy_cadence_amendment import (
    materialize_native_task_policy_cadence_amendment,
    materialize_native_task_policy_cadence_amendment_request,
)
from blueprint_pipeline.native_task_arena_stance_variant import (
    materialize_native_task_arena_stance_variant_request,
)
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
from blueprint_pipeline.paired_target_native_manipulation_preflight import (
    materialize_paired_target_native_manipulation_preflight,
)
from blueprint_pipeline.source_collider_subtree_removal import (
    materialize_source_collider_batch_removal,
)


STEPS: dict[str, Step] = {
    "droid-grasp-swept-volume": Step(
        "Conservatively qualify DROID open-gripper clearance at a grasp patch.",
        materialize_native_droid_grasp_swept_volume,
        {
            "robot_usd_path": Param("--robot-usd", required=True),
            "expected_robot_sha256": Param(
                "--expected-robot-sha256", required=True
            ),
            "robot_asset_uri": Param("--robot-asset-uri", required=True),
            "registered_asset_receipt_path": Param(
                "--registered-asset-receipt", required=True
            ),
            "interaction_affordance_path": Param(
                "--interaction-affordance", required=True
            ),
            "output_path": Param("--output", required=True),
            "search_step_m": Param(
                "--search-step-m", type=float, default=0.001
            ),
            "clearance_margin_m": Param(
                "--clearance-margin-m", type=float, default=0.004
            ),
            "maximum_standoff_m": Param(
                "--maximum-standoff-m", type=float, default=0.03
            ),
        },
    ),
    "arena-stance-variant-request": Step(
        "Derive a front-facing Franka stance from the task affordance.",
        materialize_native_task_arena_stance_variant_request,
        {
            "base_request_path": Param("--base-request", required=True),
            "output_path": Param("--output", required=True),
            "door_standoff_m": Param(
                "--door-standoff-m", type=float, default=0.55
            ),
        },
    ),
    "arena-appearance-variant-request": Step(
        "Derive a native Arena request with a sealed ParticleField appearance.",
        materialize_native_task_arena_appearance_variant_request,
        {
            "base_request_path": Param("--base-request", required=True),
            "appearance_authoring_receipt_path": Param(
                "--appearance-authoring-receipt", required=True
            ),
            "evidence_root": Param("--evidence-root", required=True),
            "output_path": Param("--output", required=True),
            "filename": Param(
                "--filename", default="scene_appearance.usdc"
            ),
        },
    ),
    "source-collider-batch-removal": Step(
        "Remove every selected source collider into one shared Arena scene.",
        materialize_source_collider_batch_removal,
        {
            "source_usd_path": Param("--source-usd", required=True),
            "targets": Param(
                "--targets",
                "JSON file containing 1-5 removal-id and target-prim records.",
                required=True,
                json_file=True,
            ),
            "output_root": Param("--output-root", required=True),
            "expected_source_sha256": Param("--expected-source-sha256"),
        },
    ),
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
            "source_collider_batch_removal_path": Param(
                "--source-collider-batch-removal", required=True
            ),
            "output_path": Param("--output", required=True),
        },
    ),
    "manipulation-preflight": Step(
        "Bind pre-arena or complete native manipulation inputs.",
        materialize_paired_target_native_manipulation_preflight,
        {
            "paired_target_preflight_path": Param(
                "--paired-target-preflight", required=True
            ),
            "native_import_result_path": Param(
                "--native-import-result", required=True
            ),
            "task_records": Param(
                "--task-records",
                "JSON file containing the task record array.",
                required=True,
                json_file=True,
            ),
            "freeze_carry_forward": Param("--freeze-carry-forward", json_file=True),
            "output_path": Param("--output", required=True),
            "phase": Param("--phase", type=str, default="arena_packet"),
        },
    ),
    "arena-requests": Step(
        "Materialize native arena requests for all admitted task inputs.",
        materialize_paired_target_native_arena_requests,
        {
            "construction_bindings_path": Param(
                "--construction-bindings", required=True
            ),
            "freeze_carry_forward": Param(
                "--freeze-carry-forward", json_file=True
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
    "arena-packet": Step(
        "Copy verified scene bytes into one immutable native Arena packet.",
        materialize_native_task_arena_packet,
        {
            "request": Param("--request", required=True, json_file=True),
            "evidence_root": Param("--evidence-root", required=True),
            "output_dir": Param("--output-dir", required=True),
            "link_sources_within": Param("--link-sources-within"),
        },
    ),
    "policy-execution-spec": Step(
        "Seal one frozen candidate request without contacting its endpoint.",
        materialize_native_task_policy_execution_spec,
        {
            "request": Param("--request", required=True, json_file=True),
            "output_path": Param("--output", required=True),
        },
    ),
    "policy-cadence-amendment-request": Step(
        "Bind one exact 20-to-15 Hz policy-cadence amendment request.",
        materialize_native_task_policy_cadence_amendment_request,
        {
            "task_freeze_path": Param("--task-freeze", required=True),
            "packet_request_path": Param("--packet-request", required=True),
            "authorized_by": Param("--authorized-by", required=True),
            "authorized_at": Param("--authorized-at", required=True),
            "output_path": Param("--output", required=True),
        },
    ),
    "policy-cadence-amendment": Step(
        "Materialize the amended freeze and complete 15 Hz Arena packet.",
        materialize_native_task_policy_cadence_amendment,
        {
            "amendment_request_path": Param("--amendment-request", required=True),
            "task_freeze_path": Param("--task-freeze", required=True),
            "packet_request_path": Param("--packet-request", required=True),
            "evidence_root": Param("--evidence-root", required=True),
            "output_dir": Param("--output-dir", required=True),
        },
    ),
}


def main(argv: Sequence[str] | None = None) -> int:
    return run(STEPS, argv, description=__doc__)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
