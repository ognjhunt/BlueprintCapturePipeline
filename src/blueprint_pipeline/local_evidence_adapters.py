"""Explicitly authorized, hermetic adapters for bounded local evidence claims."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .decision_evidence_execution import EvidenceMethodAdapterRegistry


ANALYTIC_REACHABILITY_ADAPTER = "local://analytic-reachability-v1"
CAPTURED_VISIBILITY_ADAPTER = "local://captured-visibility-v1"
SWEPT_AABB_COLLISION_SIMULATION_ADAPTER = (
    "local://swept-aabb-collision-simulation-v1"
)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    return [dict(row) for row in value if isinstance(row, Mapping)] if isinstance(value, list) else []


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
            subject.get("target_region_id") or (
                subject_value if isinstance(subject_value, str) else ""
            )
        ).strip()
        target_regions = [
            row for row in _rows(testbed.get("target_regions"))
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
        assert base is not None and target is not None and minimum is not None and maximum is not None
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
            "categorical_finding": "within_reach_envelope" if fully_inside else "outside_reach_envelope",
            "uncertainty": min(1.0, uncertainty / max(maximum, 1e-9)),
            "coverage": float(placement.get("captured_coverage") or 0.0),
            "blockers": [],
            "invalid_rollout_reasons": [],
            "raw_artifact_references": [],
            "provenance": {
                "execution_mode": "hermetic_local_read_only",
                "calculation": "euclidean_base_to_target_distance_with_uncertainty_interval",
                "robot_placement_digest": _mapping(testbed.get("validation_envelope")).get("robot_placement_digest"),
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
        if str(claim.get("claim_type") or "") not in {"visibility", "captured_visibility"}:
            return _unavailable("captured_visibility_claim_type_not_supported")
        subject = claim.get("subject")
        target_region_id = str(
            _mapping(subject).get("target_region_id") or (
                subject if isinstance(subject, str) else ""
            )
        ).strip()
        regions = [
            row for row in _rows(testbed.get("target_regions"))
            if str(row.get("region_id") or "").strip() == target_region_id
        ]
        if len(regions) != 1:
            return _unavailable("captured_visibility_target_region_not_found")
        region = regions[0]
        frames = sorted({str(item).strip() for item in region.get("supporting_frames", []) if str(item).strip()})
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
        if (
            scene.get("schema_version") != "collision_scene_aabb.v1"
            or supplied_digest
            != canonical_digest(scene, digest_field="collision_scene_digest")
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
            str(row.get("digest") or "")
            for row in _rows(testbed.get("source_capture_bundles"))
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
        excluded = {
            str(item).strip()
            for item in subject.get("excluded_collision_object_ids", [])
            if str(item).strip()
        } if isinstance(subject.get("excluded_collision_object_ids"), list) else set()
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


def authorized_local_evidence_adapter_registry(
    authorized_references: Sequence[str],
) -> EvidenceMethodAdapterRegistry:
    available = {
        ANALYTIC_REACHABILITY_ADAPTER: AnalyticReachabilityAdapter(),
        CAPTURED_VISIBILITY_ADAPTER: CapturedVisibilityAdapter(),
        SWEPT_AABB_COLLISION_SIMULATION_ADAPTER: SweptAabbCollisionSimulationAdapter(),
    }
    requested = sorted({str(item or "").strip() for item in authorized_references if str(item or "").strip()})
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"local_evidence_adapter_not_registered:{','.join(unknown)}")
    return EvidenceMethodAdapterRegistry([available[reference] for reference in requested])


__all__ = [
    "ANALYTIC_REACHABILITY_ADAPTER",
    "CAPTURED_VISIBILITY_ADAPTER",
    "SWEPT_AABB_COLLISION_SIMULATION_ADAPTER",
    "AnalyticReachabilityAdapter",
    "CapturedVisibilityAdapter",
    "SweptAabbCollisionSimulationAdapter",
    "authorized_local_evidence_adapter_registry",
]
