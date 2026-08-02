"""Explicitly authorized, hermetic adapters for bounded local evidence claims."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .decision_evidence_execution import EvidenceMethodAdapterRegistry


ANALYTIC_REACHABILITY_ADAPTER = "local://analytic-reachability-v1"
CAPTURED_VISIBILITY_ADAPTER = "local://captured-visibility-v1"
PROCESSED_OBSERVATION_VISIBILITY_ADAPTER = "local://processed-observation-visibility-v1"
SWEPT_AABB_COLLISION_SIMULATION_ADAPTER = "local://swept-aabb-collision-simulation-v1"
SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER = "local://signed-isaac-visual-placement-replay-v1"
SIGNED_ISAAC_POINT_CONTACT_ADAPTER = "local://signed-isaac-point-contact-replay-v1"
SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER = "local://signed-isaac-policy-trace-pair-replay-v1"


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return (
        [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []
    )


def _vector3(value: Any) -> tuple[float, float, float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value)
    ):
        return None
    result = tuple(float(item) for item in value)
    return result if all(math.isfinite(item) for item in result) else None


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _unavailable(*blockers: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "supports_claim": None,
        "uncertainty": 1.0,
        "coverage": 0.0,
        "blockers": sorted(set(blockers)),
        "invalid_rollout_reasons": [],
        "raw_artifact_references": [],
        "provenance": {
            "execution_mode": "hermetic_local_read_only",
            "physical_robot_run_initiated": False,
        },
        "claim_ceiling": {
            "physical_success": False,
            "deployment_readiness": False,
            "safety_certification": False,
        },
    }


@dataclass(frozen=True)
class AnalyticReachabilityAdapter:
    adapter_reference: str = ANALYTIC_REACHABILITY_ADAPTER

    def execute(self, **kwargs: Any) -> Mapping[str, Any]:
        claim = _mapping(kwargs.get("claim"))
        testbed = _mapping(kwargs.get("testbed"))
        if str(claim.get("claim_type") or "") not in {"reachability", "analytic_reachability"}:
            return _unavailable("analytic_reachability_claim_type_not_supported")
        subject_value = claim.get("subject")
        subject = _mapping(subject_value)
        bindings = _mapping(testbed.get("robot_sensor_controller_bindings"))
        embodiment = _mapping(bindings.get("embodiment"))
        placement = _mapping(bindings.get("selected_robot_placement"))
        base = _vector3(placement.get("base_position_site_m"))
        target_region_id = str(
            subject.get("target_region_id")
            or (subject_value if isinstance(subject_value, str) else "")
        ).strip()
        target_regions = [
            row
            for row in _rows(testbed.get("target_regions"))
            if str(row.get("region_id") or "").strip() == target_region_id
        ]
        target = _vector3(subject.get("target_position_site_m"))
        if target is None and len(target_regions) == 1:
            target = _vector3(target_regions[0].get("position_site_m"))
        reach = _mapping(embodiment.get("reach_envelope"))
        minimum = _number(reach.get("minimum_m"))
        maximum = _number(reach.get("maximum_m"))
        blockers: list[str] = []
        if base is None:
            blockers.append("robot_base_metric_position_missing")
        if target is None:
            blockers.append("claim_target_metric_position_missing")
        if minimum is None or maximum is None or minimum < 0 or maximum <= minimum:
            blockers.append("robot_reach_envelope_missing_or_invalid")
        if placement.get("method_qualification_status") not in {"qualified", "analytic_only"}:
            blockers.append("robot_placement_not_qualified_for_analytic_use")
        if blockers:
            return _unavailable(*blockers)
        assert (
            base is not None and target is not None and minimum is not None and maximum is not None
        )
        distance = math.dist(base, target)
        uncertainty = _number(placement.get("calibration_uncertainty_m"))
        if uncertainty is None or uncertainty < 0:
            return _unavailable("robot_placement_calibration_uncertainty_missing")
        fully_inside = distance - uncertainty >= minimum and distance + uncertainty <= maximum
        fully_outside = distance + uncertainty < minimum or distance - uncertainty > maximum
        if not fully_inside and not fully_outside:
            return {
                **_unavailable("reach_boundary_intersects_calibration_uncertainty"),
                "status": "uncertain",
                "observed_value": distance,
                "categorical_finding": "reach_boundary_uncertain",
                "uncertainty": min(1.0, uncertainty / max(maximum, 1e-9)),
                "coverage": float(placement.get("captured_coverage") or 0.0),
            }
        return {
            "status": "valid",
            "supports_claim": fully_inside,
            "observed_value": distance,
            "categorical_finding": "within_reach_envelope"
            if fully_inside
            else "outside_reach_envelope",
            "uncertainty": min(1.0, uncertainty / max(maximum, 1e-9)),
            "coverage": float(placement.get("captured_coverage") or 0.0),
            "blockers": [],
            "invalid_rollout_reasons": [],
            "raw_artifact_references": [],
            "provenance": {
                "execution_mode": "hermetic_local_read_only",
                "calculation": "euclidean_base_to_target_distance_with_uncertainty_interval",
                "robot_placement_digest": _mapping(testbed.get("validation_envelope")).get(
                    "robot_placement_digest"
                ),
                "physical_robot_run_initiated": False,
            },
            "claim_ceiling": {
                "analytic_reachability": True,
                "physical_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
        }


@dataclass(frozen=True)
class CapturedVisibilityAdapter:
    adapter_reference: str = CAPTURED_VISIBILITY_ADAPTER

    def execute(self, **kwargs: Any) -> Mapping[str, Any]:
        claim = _mapping(kwargs.get("claim"))
        testbed = _mapping(kwargs.get("testbed"))
        if str(claim.get("claim_type") or "") not in {
            "visibility",
            "captured_visibility",
            "perception_visibility",
        }:
            return _unavailable("captured_visibility_claim_type_not_supported")
        subject = claim.get("subject")
        target_region_id = str(
            _mapping(subject).get("target_region_id")
            or (subject if isinstance(subject, str) else "")
        ).strip()
        regions = [
            row
            for row in _rows(testbed.get("target_regions"))
            if str(row.get("region_id") or "").strip() == target_region_id
        ]
        if len(regions) != 1:
            return _unavailable("captured_visibility_target_region_not_found")
        region = regions[0]
        frames = sorted(
            {str(item).strip() for item in region.get("supporting_frames", []) if str(item).strip()}
        )
        coverage = _number(region.get("captured_coverage"))
        if not frames:
            return _unavailable("captured_visibility_supporting_frames_missing")
        if coverage is None or not 0 <= coverage <= 1:
            return _unavailable("captured_visibility_coverage_missing_or_invalid")
        return {
            "status": "valid",
            "supports_claim": True,
            "observed_value": coverage,
            "categorical_finding": "target_region_visible_in_retained_capture",
            "uncertainty": 1.0 - coverage,
            "coverage": coverage,
            "blockers": [],
            "invalid_rollout_reasons": [],
            "raw_artifact_references": [
                {"uri": f"capture-frame://{frame}", "frame_id": frame} for frame in frames
            ],
            "provenance": {
                "execution_mode": "hermetic_local_read_only",
                "source": "retained_capture_supporting_frames",
                "physical_robot_run_initiated": False,
            },
            "claim_ceiling": {
                "captured_observation_visibility": True,
                "metric_geometry": False,
                "physical_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
        }


@dataclass(frozen=True)
class ProcessedObservationVisibilityAdapter:
    """Review visibility in a hash-bound processed public observation set.

    This adapter deliberately does not reuse retained-capture wording: dataset
    frames can support an exact-view visibility finding, but they cannot prove
    Blueprint encoder retention, decoded PTS, raw capture, or metric authority.
    """

    adapter_reference: str = PROCESSED_OBSERVATION_VISIBILITY_ADAPTER

    def execute(self, **kwargs: Any) -> Mapping[str, Any]:
        claim = _mapping(kwargs.get("claim"))
        testbed = _mapping(kwargs.get("testbed"))
        if str(claim.get("claim_type") or "") not in {
            "visibility",
            "captured_visibility",
            "perception_visibility",
        }:
            return _unavailable("processed_observation_visibility_claim_type_not_supported")
        validation = _mapping(testbed.get("validation_envelope"))
        if validation.get("capture_authority_profile") != ("public_processed_rgbd_pose_sequence"):
            return _unavailable("processed_observation_profile_required")
        source_rows = [
            row
            for row in _rows(testbed.get("evidence_inventory"))
            if row.get("evidence_id") == "processed_capture_observations"
        ]
        if len(source_rows) != 1 or source_rows[0].get("raw_capture_authority") is not False:
            return _unavailable("processed_observation_source_binding_missing")
        subject = claim.get("subject")
        target_region_id = str(
            _mapping(subject).get("target_region_id")
            or (subject if isinstance(subject, str) else "")
        ).strip()
        regions = [
            row
            for row in _rows(testbed.get("target_regions"))
            if str(row.get("region_id") or "").strip() == target_region_id
        ]
        if len(regions) != 1:
            return _unavailable("processed_observation_target_region_not_found")
        region = regions[0]
        frames = sorted(
            {str(item).strip() for item in region.get("supporting_frames", []) if str(item).strip()}
        )
        coverage = _number(region.get("captured_coverage"))
        if not frames:
            return _unavailable("processed_observation_supporting_frames_missing")
        if coverage is None or not 0 <= coverage <= 1:
            return _unavailable("processed_observation_coverage_missing_or_invalid")
        return {
            "status": "valid",
            "supports_claim": True,
            "observed_value": coverage,
            "categorical_finding": "target_region_visible_in_processed_dataset_views",
            "uncertainty": 1.0 - coverage,
            "coverage": coverage,
            "blockers": [],
            "invalid_rollout_reasons": [],
            "raw_artifact_references": [
                {"uri": f"processed-observation://{frame}", "frame_id": frame} for frame in frames
            ],
            "provenance": {
                "execution_mode": "hermetic_local_read_only",
                "source": "processed_public_dataset_supporting_frames",
                "source_capture_digest": source_rows[0].get("digest"),
                "raw_capture_authority": False,
                "decoded_video_timing_verified": False,
                "physical_robot_run_initiated": False,
            },
            "claim_ceiling": {
                "processed_captured_observation_visibility": True,
                "raw_capture_authority": False,
                "metric_geometry": False,
                "physical_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
        }


def _segment_intersects_aabb(
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
) -> bool:
    lower = 0.0
    upper = 1.0
    for axis in range(3):
        delta = end[axis] - start[axis]
        if abs(delta) < 1e-12:
            if start[axis] < minimum[axis] or start[axis] > maximum[axis]:
                return False
            continue
        first = (minimum[axis] - start[axis]) / delta
        second = (maximum[axis] - start[axis]) / delta
        entry, exit_ = sorted((first, second))
        lower = max(lower, entry)
        upper = min(upper, exit_)
        if lower > upper:
            return False
    return True


@dataclass(frozen=True)
class SweptAabbCollisionSimulationAdapter:
    """Deterministic sim-only swept-volume collision check over qualified AABBs."""

    adapter_reference: str = SWEPT_AABB_COLLISION_SIMULATION_ADAPTER

    def execute(self, **kwargs: Any) -> Mapping[str, Any]:
        claim = _mapping(kwargs.get("claim"))
        testbed = _mapping(kwargs.get("testbed"))
        if str(claim.get("claim_type") or "") not in {
            "collision_contact",
            "modeled_collision_clearance",
        }:
            return _unavailable("collision_simulation_claim_type_not_supported")
        validation = _mapping(testbed.get("validation_envelope"))
        layers = _mapping(validation.get("reconstruction_layers"))
        physics_rows = [
            row
            for row in _rows(layers.get("physics_layer"))
            if row.get("output") == "collision_geometry"
        ]
        if len(physics_rows) != 1:
            return _unavailable("qualified_collision_geometry_not_unique")
        physics = physics_rows[0]
        if _rows(physics.get("generated_regions")):
            return _unavailable("generated_region_cannot_supply_collision_geometry")
        ceiling = _mapping(physics.get("claim_ceiling"))
        if ceiling.get("collision_geometry") is not True:
            return _unavailable("collision_geometry_claim_ceiling_missing")
        scene = _mapping(_mapping(physics.get("asset_references")).get("collision_scene"))
        supplied_digest = str(scene.get("collision_scene_digest") or "")
        if scene.get(
            "schema_version"
        ) != "collision_scene_aabb.v1" or supplied_digest != canonical_digest(
            scene, digest_field="collision_scene_digest"
        ):
            return _unavailable("collision_scene_digest_invalid")
        scene_validation = _mapping(scene.get("validation"))
        blockers: list[str] = []
        if scene.get("scale_status") != "metric_verified":
            blockers.append("collision_scene_metric_scale_unverified")
        if scene.get("coordinate_frame") != "site":
            blockers.append("collision_scene_not_in_site_frame")
        if scene.get("generated_geometry") is not False:
            blockers.append("generated_collision_geometry_forbidden")
        if (
            scene_validation.get("status") != "qualified"
            or scene_validation.get("independent_validation") is not True
        ):
            blockers.append("collision_scene_independent_validation_missing")
        source_digests = {
            str(row.get("digest") or "") for row in _rows(testbed.get("source_capture_bundles"))
        }
        if scene.get("source_capture_digest") not in source_digests:
            blockers.append("collision_scene_source_capture_mismatch")
        subject = _mapping(claim.get("subject"))
        raw_points = subject.get("trajectory_points_site_m")
        points = [_vector3(point) for point in raw_points] if isinstance(raw_points, list) else []
        if len(points) < 2 or any(point is None for point in points):
            blockers.append("collision_trajectory_missing_or_invalid")
        radius = _number(subject.get("swept_radius_m"))
        if radius is None or radius < 0:
            blockers.append("collision_swept_radius_missing_or_invalid")
        excluded = (
            {
                str(item).strip()
                for item in subject.get("excluded_collision_object_ids", [])
                if str(item).strip()
            }
            if isinstance(subject.get("excluded_collision_object_ids"), list)
            else set()
        )
        primitives: list[tuple[str, tuple[float, float, float], tuple[float, float, float]]] = []
        for row in _rows(scene.get("primitives")):
            object_id = str(row.get("object_id") or row.get("primitive_id") or "").strip()
            minimum = _vector3(row.get("minimum_site_m"))
            maximum = _vector3(row.get("maximum_site_m"))
            if (
                not object_id
                or minimum is None
                or maximum is None
                or any(minimum[axis] >= maximum[axis] for axis in range(3))
            ):
                blockers.append("collision_scene_primitive_invalid")
                continue
            if object_id not in excluded:
                primitives.append((object_id, minimum, maximum))
        if not primitives:
            blockers.append("collision_scene_primitives_missing")
        coverage = _number(scene_validation.get("coverage"))
        spatial_uncertainty = _number(scene_validation.get("maximum_spatial_uncertainty_m"))
        if coverage is None or not 0 <= coverage <= 1:
            blockers.append("collision_scene_coverage_missing_or_invalid")
        if spatial_uncertainty is None or spatial_uncertainty < 0:
            blockers.append("collision_scene_spatial_uncertainty_missing")
        if blockers:
            return _unavailable(*blockers)
        assert radius is not None and coverage is not None and spatial_uncertainty is not None
        trajectory = [point for point in points if point is not None]
        expanded_by = radius + spatial_uncertainty
        contacts: set[str] = set()
        for start, end in zip(trajectory, trajectory[1:]):
            for object_id, minimum, maximum in primitives:
                expanded_minimum = tuple(value - expanded_by for value in minimum)
                expanded_maximum = tuple(value + expanded_by for value in maximum)
                if _segment_intersects_aabb(start, end, expanded_minimum, expanded_maximum):
                    contacts.add(object_id)
        collision_free = not contacts
        return {
            "status": "valid",
            "supports_claim": collision_free,
            "observed_value": len(contacts),
            "categorical_finding": (
                "modeled_trajectory_collision_free"
                if collision_free
                else "modeled_trajectory_collision_detected"
            ),
            "uncertainty": min(1.0, spatial_uncertainty / max(radius, 0.01)),
            "coverage": coverage,
            "blockers": [],
            "invalid_rollout_reasons": [],
            "raw_artifact_references": [
                {
                    "uri": f"collision-scene://{supplied_digest[7:]}",
                    "digest": supplied_digest,
                },
                {
                    "uri": f"reconstruction-result://{physics['result_digest'][7:]}",
                    "digest": physics["result_digest"],
                },
            ],
            "provenance": {
                "execution_mode": "hermetic_local_deterministic_simulation",
                "simulation_model": "piecewise_linear_swept_sphere_against_expanded_aabb",
                "collision_scene_digest": supplied_digest,
                "contact_object_ids": sorted(contacts),
                "physical_robot_run_initiated": False,
            },
            "claim_ceiling": {
                "sim_only_modeled_collision_clearance": True,
                "contact_dynamics": False,
                "physical_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
        }


def _evidence_row(testbed: Mapping[str, Any], evidence_id: str) -> dict[str, Any] | None:
    matches = [
        row
        for row in _rows(testbed.get("evidence_inventory"))
        if str(row.get("evidence_id") or "") == evidence_id
    ]
    return matches[0] if len(matches) == 1 else None


def _sha256_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


@dataclass(frozen=True)
class SignedIsaacVisualPlacementReplayAdapter:
    """Replay an exact, independently rehashed Isaac robot-depth observation.

    This answers only whether the named robot was visible at the exact
    digest-bound pose in the exact retained simulated views.
    """

    adapter_reference: str = SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER

    def execute(self, **kwargs: Any) -> Mapping[str, Any]:
        claim = _mapping(kwargs.get("claim"))
        testbed = _mapping(kwargs.get("testbed"))
        if str(claim.get("claim_type") or "") != "perception_visibility":
            return _unavailable("signed_isaac_visual_claim_type_not_supported")
        evidence = _evidence_row(testbed, "signed_isaac_visual_placement")
        if evidence is None or evidence.get("independently_rehashed") is not True:
            return _unavailable("signed_isaac_visual_evidence_missing")
        camera_rows = _rows(evidence.get("camera_evidence"))
        if not camera_rows or any(
            row.get("visual_geometry_observed") is not True
            or not str(row.get("rgb_artifact_reference") or "")
            or not str(row.get("distance_artifact_reference") or "")
            or not _sha256_digest(row.get("rgb_digest"))
            or not _sha256_digest(row.get("distance_digest"))
            for row in camera_rows
        ):
            return _unavailable("signed_isaac_visual_camera_evidence_invalid")
        coverage = _number(evidence.get("exact_view_coverage"))
        if coverage is None or not 0 <= coverage <= 1:
            return _unavailable("signed_isaac_visual_coverage_invalid")
        return {
            "status": "valid",
            "supports_claim": True,
            "observed_value": len(camera_rows),
            "categorical_finding": "robot_visible_at_exact_simulated_pose",
            "uncertainty": 1.0 - coverage,
            "coverage": coverage,
            "blockers": [],
            "invalid_rollout_reasons": [],
            "raw_artifact_references": [
                {
                    "uri": f"artifact://{row['rgb_artifact_reference']}",
                    "digest": row["rgb_digest"],
                }
                for row in camera_rows
            ]
            + [
                {
                    "uri": f"artifact://{row['distance_artifact_reference']}",
                    "digest": row["distance_digest"],
                }
                for row in camera_rows
            ],
            "provenance": {
                "execution_mode": "hermetic_local_signed_evidence_replay",
                "source_runtime": "isaac_sim",
                "provider_robot_placement_evidence_digest": evidence.get(
                    "provider_robot_placement_evidence_digest"
                ),
                "physical_robot_run_initiated": False,
            },
            "claim_ceiling": {
                "isaac_visual_robot_placement": True,
                "formal_robot_placement": False,
                "kinematic_reachability": False,
                "task_success": False,
                "physical_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
        }


@dataclass(frozen=True)
class SignedIsaacPointContactReplayAdapter:
    """Replay exact point-contact evidence from an independently qualified run."""

    adapter_reference: str = SIGNED_ISAAC_POINT_CONTACT_ADAPTER

    def execute(self, **kwargs: Any) -> Mapping[str, Any]:
        claim = _mapping(kwargs.get("claim"))
        testbed = _mapping(kwargs.get("testbed"))
        if str(claim.get("claim_type") or "") != "collision_contact":
            return _unavailable("signed_isaac_contact_claim_type_not_supported")
        evidence = _evidence_row(testbed, "signed_isaac_point_contact")
        count = _number(_mapping(evidence).get("contact_event_count"))
        if (
            evidence is None
            or evidence.get("independently_qualified") is not True
            or evidence.get("test_body_fell_through_floor") is not False
            or not _sha256_digest(evidence.get("isaac_runtime_result_digest"))
            or not _sha256_digest(evidence.get("independent_qualification_digest"))
            or count is None
            or count < 1
        ):
            return _unavailable("signed_isaac_point_contact_evidence_missing")
        return {
            "status": "valid",
            "supports_claim": True,
            "observed_value": int(count),
            "categorical_finding": "point_contact_observed_at_committed_probe",
            "uncertainty": 0.0,
            "coverage": 1.0,
            "blockers": [],
            "invalid_rollout_reasons": [],
            "raw_artifact_references": [
                {
                    "uri": "artifact://provider_nurec_isaac_runtime_result",
                    "digest": evidence["isaac_runtime_result_digest"],
                },
                {
                    "uri": "artifact://independent_isaac_qualification",
                    "digest": evidence["independent_qualification_digest"],
                },
            ],
            "provenance": {
                "execution_mode": "hermetic_local_signed_evidence_replay",
                "source_runtime": "isaac_sim",
                "probe_scope": "single_precommitted_point",
                "visual_robot_excluded_from_environment_probe": True,
                "physical_robot_run_initiated": False,
            },
            "claim_ceiling": {
                "isaac_point_contact_presence": True,
                "complete_robot_collision_clearance": False,
                "task_success": False,
                "physical_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
        }


@dataclass(frozen=True)
class SignedIsaacPolicyTracePairReplayAdapter:
    """Replay two independently validated, exact Isaac Franka traces.

    This supports only the boolean claim that the two frozen candidates were
    executed from an identical start and produced distinct simulated traces.
    It does not rank them or turn stage-unit motion into metric task success.
    """

    adapter_reference: str = SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER

    def execute(self, **kwargs: Any) -> Mapping[str, Any]:
        claim = _mapping(kwargs.get("claim"))
        testbed = _mapping(kwargs.get("testbed"))
        if str(claim.get("claim_type") or "") != "simulated_policy_trace_distinguishability":
            return _unavailable("signed_isaac_policy_trace_claim_type_not_supported")
        evidence = _evidence_row(testbed, "signed_isaac_articulated_policy_trace_pair")
        candidates = _rows(_mapping(evidence).get("candidate_traces"))
        expected_ids = ["franka-fixed-hold-v1", "franka-inspection-sweep-v1"]
        if (
            evidence is None
            or evidence.get("independently_validated") is not True
            or evidence.get("identical_frozen_start_observed") is not True
            or evidence.get("distinct") is not True
            or not _sha256_digest(evidence.get("articulated_policy_trace_pair_digest"))
            or [row.get("policy_id") for row in candidates] != expected_ids
            or any(
                row.get("status") != "completed"
                or not _sha256_digest(row.get("policy_trace_digest"))
                or not _sha256_digest(row.get("egocentric_observation_digest"))
                for row in candidates
            )
        ):
            return _unavailable("signed_isaac_policy_trace_pair_evidence_missing")
        return {
            "status": "valid",
            "supports_claim": True,
            "observed_value": evidence.get("maximum_end_joint_delta_rad"),
            "categorical_finding": "frozen_franka_candidates_produced_distinct_simulated_traces",
            "uncertainty": 0.0,
            "coverage": 1.0,
            "blockers": [],
            "invalid_rollout_reasons": [],
            "raw_artifact_references": [
                {
                    "uri": f"artifact://policy-trace/{row['policy_id']}",
                    "digest": row["policy_trace_digest"],
                }
                for row in candidates
            ],
            "provenance": {
                "execution_mode": "hermetic_local_signed_evidence_replay",
                "source_runtime": "isaac_sim",
                "controller_id": evidence.get("controller_id"),
                "identical_frozen_start_observed": True,
                "robot_relative_egocentric_camera": True,
                "physical_robot_run_initiated": False,
            },
            "claim_ceiling": {
                "simulated_articulated_policy_execution": True,
                "simulated_policy_trace_distinguishability": True,
                "comparative_policy_ranking": False,
                "metric_task_success": False,
                "physical_success": False,
                "deployment_readiness": False,
                "safety_certification": False,
            },
        }


def authorized_local_evidence_adapter_registry(
    authorized_references: Sequence[str],
) -> EvidenceMethodAdapterRegistry:
    available = {
        ANALYTIC_REACHABILITY_ADAPTER: AnalyticReachabilityAdapter(),
        CAPTURED_VISIBILITY_ADAPTER: CapturedVisibilityAdapter(),
        PROCESSED_OBSERVATION_VISIBILITY_ADAPTER: ProcessedObservationVisibilityAdapter(),
        SWEPT_AABB_COLLISION_SIMULATION_ADAPTER: SweptAabbCollisionSimulationAdapter(),
        SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER: SignedIsaacVisualPlacementReplayAdapter(),
        SIGNED_ISAAC_POINT_CONTACT_ADAPTER: SignedIsaacPointContactReplayAdapter(),
        SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER: SignedIsaacPolicyTracePairReplayAdapter(),
    }
    requested = sorted(
        {str(item or "").strip() for item in authorized_references if str(item or "").strip()}
    )
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"local_evidence_adapter_not_registered:{','.join(unknown)}")
    return EvidenceMethodAdapterRegistry([available[reference] for reference in requested])


__all__ = [
    "ANALYTIC_REACHABILITY_ADAPTER",
    "CAPTURED_VISIBILITY_ADAPTER",
    "PROCESSED_OBSERVATION_VISIBILITY_ADAPTER",
    "SIGNED_ISAAC_POINT_CONTACT_ADAPTER",
    "SIGNED_ISAAC_POLICY_TRACE_PAIR_ADAPTER",
    "SIGNED_ISAAC_VISUAL_PLACEMENT_ADAPTER",
    "SWEPT_AABB_COLLISION_SIMULATION_ADAPTER",
    "AnalyticReachabilityAdapter",
    "CapturedVisibilityAdapter",
    "ProcessedObservationVisibilityAdapter",
    "SignedIsaacPointContactReplayAdapter",
    "SignedIsaacPolicyTracePairReplayAdapter",
    "SignedIsaacVisualPlacementReplayAdapter",
    "SweptAabbCollisionSimulationAdapter",
    "authorized_local_evidence_adapter_registry",
]
