"""Ground a robot-visualization proposal in verified provider NuRec evidence.

This module bridges the live provider-package Isaac compatibility result to the
existing ``robot_placement_result.v1`` contract.  It deliberately keeps two
different artifacts:

* a runtime visualization proposal, which may be rendered to diagnose the robot
  asset and pose in the exact scene; and
* the formal placement result, which remains an abstention until footprint,
  access, reset, human-clearance, and captured-coverage evidence exists.

Rendering the proposal therefore cannot silently become deployment approval.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .external_provider_nurec import (
    ExternalProviderNuRecError,
    build_provider_nurec_isaac_request,
    build_provider_nurec_isaac_runtime_result,
    sha256_file,
)
from .reconstruction_capability import score_robot_placements
from .scene_placement.robot_profile import DEFAULT_ROBOT_ID, get_robot_profile


PROPOSAL_SCHEMA_VERSION = "provider_nurec_robot_placement_proposal.v1"
TASK_SCHEMA_VERSION = "provider_nurec_site_task_definition.v1"
PACKET_SCHEMA_VERSION = "provider_nurec_robot_placement_packet.v1"
FRANKA_POLICY_TRACE_REQUEST_SCHEMA_VERSION = "franka_articulated_policy_trace_request.v1"


class ProviderNuRecRobotPlacementError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        result = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ProviderNuRecRobotPlacementError(["placement_input_not_json_serializable"]) from exc
    return result


def build_default_franka_policy_trace_request(*, robot_prim_path: str) -> dict[str, Any]:
    """Build the frozen two-candidate articulated trace request used by site evals.

    The request is intentionally target-agnostic: it proves that the exact
    official Franka can execute two controlled, distinguishable joint behaviors
    from an identical reset. Target binding and policy ranking remain separate
    qualification gates.
    """

    if (
        not isinstance(robot_prim_path, str)
        or not robot_prim_path.startswith("/")
        or ".." in robot_prim_path.split("/")
    ):
        raise ProviderNuRecRobotPlacementError(["franka_policy_trace_robot_prim_path_invalid"])
    start = [0.0, -0.55, 0.0, -2.6, 0.0, 2.05, 0.75]
    request = {
        "schema_version": FRANKA_POLICY_TRACE_REQUEST_SCHEMA_VERSION,
        "robot_id": "franka_panda",
        "robot_prim_path": robot_prim_path,
        "controller_id": "deterministic_franka_joint_position_pair.v1",
        "joint_names": [f"panda_joint{index}" for index in range(1, 8)],
        "start_joint_positions_rad": start,
        "physics_dt_seconds": 1.0 / 60.0,
        "reset_settle_steps": 30,
        "sample_interval_steps": 10,
        "reset_position_error_threshold_rad": 0.1,
        "reset_velocity_threshold_rad_s": 2.0,
        "distinctness_threshold_rad": 0.1,
        "identical_start_tolerance_rad": 0.02,
        "candidates": [
            {
                "policy_id": "franka-fixed-hold-v1",
                "duration_steps": 120,
                "final_joint_positions_rad": start,
            },
            {
                "policy_id": "franka-inspection-sweep-v1",
                "duration_steps": 120,
                "final_joint_positions_rad": [
                    0.35,
                    -0.55,
                    0.0,
                    -2.6,
                    0.0,
                    2.05,
                    0.75,
                ],
            },
        ],
        "egocentric_camera": {
            "parent_link_name": "panda_hand",
            "local_position_m": [0.05, 0.0, 0.04],
            "local_target_m": [0.3, 0.0, 0.04],
            "local_up": [0.0, 0.0, 1.0],
            "fov_degrees": 70.0,
            "width": 320,
            "height": 240,
        },
        "physical_success_claimed": False,
    }
    request["policy_trace_request_digest"] = canonical_digest(
        request, digest_field="policy_trace_request_digest"
    )
    return request


def _validated_independent_qualification(
    value: Mapping[str, Any], *, request: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    qualification = _clone(value)
    errors: list[str] = []
    if qualification.get("schema_version") != "reconstruction_isaac_independent_qualification.v1":
        errors.append("independent_qualification_schema_invalid")
    if qualification.get("status") != "verified_compatibility_only":
        errors.append("independent_qualification_not_verified")
    if qualification.get("blockers") not in ([], ()):
        errors.append("independent_qualification_has_blockers")
    if qualification.get("isaac_verification_request_digest") != request.get(
        "isaac_verification_request_digest"
    ):
        errors.append("independent_qualification_request_mismatch")
    if qualification.get("runtime_result_digest") != runtime.get("isaac_runtime_result_digest"):
        errors.append("independent_qualification_runtime_mismatch")
    if qualification.get("qualification_digest") != canonical_digest(
        qualification, digest_field="qualification_digest"
    ):
        errors.append("independent_qualification_digest_mismatch")
    if qualification.get("claim_ceiling") != "isaac_load_render_compatibility":
        errors.append("independent_qualification_claim_ceiling_invalid")
    if errors:
        raise ProviderNuRecRobotPlacementError(errors)
    return qualification


def build_provider_nurec_robot_placement_packet(
    *,
    verification_request: Mapping[str, Any],
    runtime_result: Mapping[str, Any],
    independent_qualification: Mapping[str, Any],
    site_id: str,
    task_id: str,
    robot_id: str = DEFAULT_ROBOT_ID,
    include_articulated_policy_trace_pair: bool = False,
) -> dict[str, Any]:
    """Build a proposal, formal placement abstention, and render options.

    The exact physics-probe contact location is reused as the visualization
    anchor.  It proves a surface contact at one point; it does *not* prove the
    complete Franka footprint or an access/reset envelope, so those formal gates
    remain false and ``score_robot_placements`` abstains.
    """

    try:
        request = build_provider_nurec_isaac_request(verification_request)
        runtime = build_provider_nurec_isaac_runtime_result(
            runtime_result, verification_request=request
        )
    except ExternalProviderNuRecError as exc:
        raise ProviderNuRecRobotPlacementError(
            [f"provider_nurec_input_invalid:{code}" for code in exc.codes]
        ) from exc
    qualification = _validated_independent_qualification(
        independent_qualification, request=request, runtime=runtime
    )
    errors: list[str] = []
    if runtime.get("status") != "completed":
        errors.append("isaac_runtime_not_completed")
    if not isinstance(site_id, str) or not site_id.strip():
        errors.append("site_id_missing")
    if not isinstance(task_id, str) or not task_id.strip():
        errors.append("task_id_missing")
    physics = runtime.get("physics_probe")
    surface = physics.get("ground_surface") if isinstance(physics, Mapping) else None
    requested_probe = request.get("physics_probe_request")
    xy = requested_probe.get("probe_xy_m") if isinstance(requested_probe, Mapping) else None
    ground_z = surface.get("probe_height_m") if isinstance(surface, Mapping) else None
    if (
        not isinstance(xy, list)
        or len(xy) != 2
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in (xy or [])
        )
    ):
        errors.append("verified_ground_probe_xy_missing")
    if (
        isinstance(ground_z, bool)
        or not isinstance(ground_z, (int, float))
        or not math.isfinite(float(ground_z))
    ):
        errors.append("verified_ground_probe_height_missing")
    if (
        not isinstance(physics, Mapping)
        or physics.get("ground_contact_surface_present") is not True
        or physics.get("live_rigid_body_pose_observed") is not True
        or physics.get("test_body_fell_through_floor") is not False
        or int(physics.get("contact_event_count") or 0) < 1
    ):
        errors.append("verified_ground_contact_missing")
    if errors:
        raise ProviderNuRecRobotPlacementError(errors)

    profile = get_robot_profile(robot_id)
    profile_value = profile.to_dict()
    profile_digest = canonical_digest(profile_value)
    package_digest = request["package_digest"]
    x, y, z = float(xy[0]), float(xy[1]), float(ground_z)
    task_definition = {
        "schema_version": TASK_SCHEMA_VERSION,
        "site_id": site_id.strip(),
        "task_id": task_id.strip(),
        "task_description": (
            "Inspect the exact Isaac-verified ground-probe waypoint from a fixed Franka base "
            "and compare deterministic arm/camera behaviors from the frozen start."
        ),
        "source_asset": {
            "source_class": "public_provider_sample",
            "asset_digest": package_digest,
            "blueprint_raw_capture_truth": False,
            "external_derived_support_asset": True,
        },
        "task_object_id": "verified_ground_contact_surface",
        "target_region_id": "isaac_ground_probe_waypoint",
        "target_position_site_m": [x, y, round(z + 0.5, 9)],
        "requested_claim_types": [
            "robot_placement",
            "perception_visibility",
            "deterministic_candidate_distinguishability",
        ],
        "proof_boundary": {
            "task_is_provider_sample_harness_probe": True,
            "task_is_customer_approved_physical_work": False,
            "semantic_object_identity_proven": False,
            "physical_success_proven": False,
        },
    }
    task_definition["approved_task_digest"] = canonical_digest(
        task_definition, digest_field="approved_task_digest"
    )

    robot_binding = {
        "robot_id": profile.robot_id,
        "embodiment_version": "blueprint_robot_profile.v1",
        "robot_profile_digest": profile_digest,
        "base_footprint": {
            "shape": "axis_aligned_box_half_extents",
            "half_extents_xyz_m": list(profile.footprint_half_extent_xyz),
        },
        "sensors": {row["camera_id"]: dict(row) for row in profile.camera_rigs},
        "controller_id": "deterministic_franka_joint_position_pair.v1",
        "end_effector_id": "panda_hand_parallel_jaw",
    }
    evidence_digests = sorted(
        {
            package_digest,
            runtime["isaac_runtime_result_digest"],
            qualification["qualification_digest"],
        }
    )
    proposal = {
        "schema_version": PROPOSAL_SCHEMA_VERSION,
        "status": "runtime_visualization_candidate_only",
        "site_id": site_id.strip(),
        "task_id": task_id.strip(),
        "approved_task_digest": task_definition["approved_task_digest"],
        "package_digest": package_digest,
        "isaac_verification_request_digest": request["isaac_verification_request_digest"],
        "isaac_runtime_result_digest": runtime["isaac_runtime_result_digest"],
        "independent_qualification_digest": qualification["qualification_digest"],
        "robot_binding": robot_binding,
        "robot_binding_digest": canonical_digest(robot_binding),
        "candidate_id": "franka-at-verified-ground-probe",
        "robot_pose_xyzyaw_site": [x, y, z, 0.0],
        "site_from_robot_base": [
            1.0,
            0.0,
            0.0,
            x,
            0.0,
            1.0,
            0.0,
            y,
            0.0,
            0.0,
            1.0,
            z,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "ground_surface_prim": surface.get("prim_path"),
        "evidence_digests": evidence_digests,
        "known_support": {
            "metric_stage_semantics_declared": (
                runtime.get("stage", {}).get("meters_per_unit") == 1.0
            ),
            "independent_known_distance_scale_anchor": False,
            "z_up_verified": runtime.get("stage", {}).get("up_axis") == "Z",
            "point_contact_verified": True,
            "nonblank_scene_view_verified": True,
        },
        "formal_gaps": [
            "complete_robot_footprint_not_collision_probed",
            "access_path_not_qualified",
            "reset_envelope_not_qualified",
            "human_clearance_not_qualified",
            "captured_placement_coverage_not_measured",
            "task_target_semantics_not_observed",
            "independent_known_distance_scale_anchor_missing",
        ],
        "proof_boundary": {
            "runtime_visualization_is_formal_placement_approval": False,
            "robot_spawn_or_render_proves_task_execution": False,
            "physical_success_proven": False,
        },
    }
    proposal["placement_proposal_digest"] = canonical_digest(
        proposal, digest_field="placement_proposal_digest"
    )

    candidate = {
        "candidate_id": proposal["candidate_id"],
        "site_from_robot_base": proposal["site_from_robot_base"],
        "floor_support_valid": True,
        "footprint_clear": False,
        "access_path_clear": False,
        "collision_free": False,
        "reset_feasible": False,
        "human_clearance_valid": False,
        "captured_coverage": 0.0,
        "reachability_score": 0.5,
        "manipulability_score": 0.5,
        "sensor_visibility_score": 0.5,
        "approach_direction_score": 0.5,
        "cable_controller_score": 0.5,
        "stability_score": 0.5,
        "calibration_uncertainty_m": 0.1,
        "method_qualification_status": "analytic_only",
        "evidence_digests": sorted(evidence_digests + [proposal["placement_proposal_digest"]]),
    }
    placement_result = score_robot_placements(
        robot_binding=robot_binding,
        approved_task_digest=task_definition["approved_task_digest"],
        capture_digest=package_digest,
        task_object_id=task_definition["task_object_id"],
        target_region_id=task_definition["target_region_id"],
        candidates=[candidate],
    )
    render_options = {
        "robot_id": profile.robot_id,
        "robot_usd": str(profile.simulator_asset_refs["isaac_asset"]).lstrip("/"),
        "robot_prim_path": profile.usd_prim_path,
        "robot_pose": proposal["robot_pose_xyzyaw_site"],
        "robot_ground_z": z,
        "robot_only_pass": True,
        "robot_placement_digest": placement_result["robot_placement_digest"],
        "placement_proposal_digest": proposal["placement_proposal_digest"],
        "lights_path": "/World/Lights",
    }
    if include_articulated_policy_trace_pair:
        if profile.robot_id != "franka_panda":
            raise ProviderNuRecRobotPlacementError(
                ["articulated_policy_trace_pair_requires_franka_panda"]
            )
        render_options["articulated_policy_trace_request"] = (
            build_default_franka_policy_trace_request(robot_prim_path=profile.usd_prim_path)
        )
    return {
        "task_definition": task_definition,
        "placement_proposal": proposal,
        "robot_placement_result": placement_result,
        "render_options": render_options,
    }


def write_provider_nurec_robot_placement_packet(
    *, output_dir: str | Path, packet: Mapping[str, Any]
) -> dict[str, Any]:
    root = Path(output_dir)
    if root.is_symlink():
        raise ProviderNuRecRobotPlacementError(["placement_packet_output_symlink_forbidden"])
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "task_definition": root / "site_task_definition.json",
        "placement_proposal": root / "robot_placement_proposal.json",
        "robot_placement_result": root / "robot_placement_result.json",
        "render_options": root / "render_options.json",
    }
    for key, path in files.items():
        write_json(path, packet[key])
    manifest = {
        "schema_version": PACKET_SCHEMA_VERSION,
        "status": "formal_placement_abstained_runtime_visualization_ready",
        "artifact_digests": {key: sha256_file(path) for key, path in files.items()},
        "task_digest": packet["task_definition"]["approved_task_digest"],
        "placement_proposal_digest": packet["placement_proposal"]["placement_proposal_digest"],
        "robot_placement_digest": packet["robot_placement_result"]["robot_placement_digest"],
        "formal_placement_status": packet["robot_placement_result"]["status"],
        "runtime_visualization_authorized": True,
        "physical_robot_execution_authorized": False,
        "proof_boundary": {
            "packet_performs_provider_allocation": False,
            "packet_proves_robot_render": False,
            "packet_proves_task_success": False,
            "packet_proves_physical_success": False,
        },
    }
    manifest["packet_digest"] = canonical_digest(manifest, digest_field="packet_digest")
    write_json(root / "robot_placement_packet_manifest.json", manifest)
    return {**manifest, "output_dir": str(root.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a formal placement abstention plus exact robot render options"
    )
    parser.add_argument("--verification-request", required=True)
    parser.add_argument("--runtime-result", required=True)
    parser.add_argument("--independent-qualification", required=True)
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--robot-id", default=DEFAULT_ROBOT_ID)
    parser.add_argument("--include-articulated-policy-traces", action="store_true")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    try:

        def load(path: str) -> Any:
            return json.loads(Path(path).read_text(encoding="utf-8"))

        packet = build_provider_nurec_robot_placement_packet(
            verification_request=load(args.verification_request),
            runtime_result=load(args.runtime_result),
            independent_qualification=load(args.independent_qualification),
            site_id=args.site_id,
            task_id=args.task_id,
            robot_id=args.robot_id,
            include_articulated_policy_trace_pair=args.include_articulated_policy_traces,
        )
        receipt = write_provider_nurec_robot_placement_packet(
            output_dir=args.output_dir, packet=packet
        )
    except (OSError, json.JSONDecodeError, ProviderNuRecRobotPlacementError) as exc:
        codes = (
            list(exc.codes)
            if isinstance(exc, ProviderNuRecRobotPlacementError)
            else [f"placement_packet_input_error:{type(exc).__name__}"]
        )
        print(json.dumps({"status": "abstention", "blockers": sorted(codes)}, sort_keys=True))
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FRANKA_POLICY_TRACE_REQUEST_SCHEMA_VERSION",
    "PACKET_SCHEMA_VERSION",
    "PROPOSAL_SCHEMA_VERSION",
    "TASK_SCHEMA_VERSION",
    "ProviderNuRecRobotPlacementError",
    "build_default_franka_policy_trace_request",
    "build_provider_nurec_robot_placement_packet",
    "write_provider_nurec_robot_placement_packet",
]
