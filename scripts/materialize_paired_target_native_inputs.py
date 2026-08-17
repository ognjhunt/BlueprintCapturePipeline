#!/usr/bin/env python3
"""Materialize the active replacement/native construction input chain.

This command reads retained, digest-bound inputs and writes receipts only. It
does not allocate a provider or execute a simulator.
"""

from __future__ import annotations

from collections.abc import Sequence

from blueprint_pipeline.agent_cad_graph_visual_composition import (
    materialize_agent_cad_visual_composition,
    seal_agent_cad_visual_binding,
)
from blueprint_pipeline.materializer_cli import Param, Step, run
from blueprint_pipeline.native_task_arena_packet import (
    materialize_native_task_arena_packet,
)
from blueprint_pipeline.native_task_arena_policy_bundle import (
    materialize_native_task_policy_execution_spec,
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
from blueprint_pipeline.registered_replacement_asset import (
    materialize_registered_replacement_asset,
)
from blueprint_pipeline.simready_graph_asset import author_simready_graph_asset
from blueprint_pipeline.simready_graph_asset_static_qualification import (
    qualify_simready_graph_asset_static,
)


STEPS: dict[str, Step] = {
    "simready-graph-asset": Step(
        "Author one task-neutral SimReady graph asset from sealed inputs.",
        author_simready_graph_asset,
        {
            "spec": Param("--spec", required=True, json_file=True),
            "task_freeze_receipt_path": Param("--task-freeze", required=True),
            "source_asset_receipt_path": Param(
                "--source-asset-receipt", required=True
            ),
            "destination": Param("--output-usd", required=True),
            "receipt_path": Param("--output-receipt", required=True),
        },
    ),
    "simready-static-qualification": Step(
        "Reopen and statically qualify authored or registered SimReady bytes.",
        qualify_simready_graph_asset_static,
        {
            "spec": Param("--spec", required=True, json_file=True),
            "authoring_receipt_path": Param(
                "--authoring-receipt", required=True
            ),
            "registered_replacement_asset_receipt_path": Param(
                "--registered-asset-receipt"
            ),
            "output_path": Param("--output", required=True),
        },
    ),
    "visual-binding": Step(
        "Seal the selected CAD visual meshes onto the graph asset links.",
        seal_agent_cad_visual_binding,
        {
            "graph_authoring_receipt_path": Param(
                "--graph-authoring-receipt", required=True
            ),
            "cad_agent_output_receipt_path": Param("--cad-agent-output-receipt"),
            "cad_agent_matrix_path": Param("--cad-agent-matrix"),
            "cad_agent_backend_id": Param("--cad-agent-backend-id"),
            "cad_agent_visual_review_path": Param(
                "--cad-agent-visual-review", required=True
            ),
            "mesh_projection_receipt_path": Param(
                "--mesh-projection-receipt", required=True
            ),
            "link_bindings": Param("--link-bindings", required=True, json_file=True),
            "unmapped_graph_link_reasons": Param(
                "--unmapped-graph-link-reasons", required=True, json_file=True
            ),
            "output_path": Param("--output", required=True),
        },
    ),
    "visual-composition": Step(
        "Compose the exact selected CAD visuals into the graph asset USD.",
        materialize_agent_cad_visual_composition,
        {
            "binding_path": Param("--binding", required=True),
            "destination_usd_path": Param("--output-usd", required=True),
            "receipt_path": Param("--receipt"),
        },
    ),
    "registered-asset": Step(
        "Apply the reviewed frame registration to one composed replacement.",
        materialize_registered_replacement_asset,
        {
            "visual_composition_receipt_path": Param(
                "--visual-composition-receipt", required=True
            ),
            "frame_registration_path": Param("--frame-registration", required=True),
            "output_usd_path": Param("--output-usd", required=True),
            "output_receipt_path": Param("--output-receipt", required=True),
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
    "arena-packet": Step(
        "Copy verified scene bytes into one immutable native Arena packet.",
        materialize_native_task_arena_packet,
        {
            "request": Param("--request", required=True, json_file=True),
            "evidence_root": Param("--evidence-root", required=True),
            "output_dir": Param("--output-dir", required=True),
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
}


def main(argv: Sequence[str] | None = None) -> int:
    return run(STEPS, argv, description=__doc__)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
