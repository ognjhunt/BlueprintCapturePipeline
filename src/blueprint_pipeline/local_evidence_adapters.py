"""Explicitly authorized, hermetic adapters for bounded local evidence claims."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .decision_evidence_execution import EvidenceMethodAdapterRegistry


ANALYTIC_REACHABILITY_ADAPTER = "local://analytic-reachability-v1"
CAPTURED_VISIBILITY_ADAPTER = "local://captured-visibility-v1"


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
        subject = _mapping(claim.get("subject"))
        bindings = _mapping(testbed.get("robot_sensor_controller_bindings"))
        embodiment = _mapping(bindings.get("embodiment"))
        placement = _mapping(bindings.get("selected_robot_placement"))
        base = _vector3(placement.get("base_position_site_m"))
        target = _vector3(subject.get("target_position_site_m"))
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
        target_region_id = str(_mapping(claim.get("subject")).get("target_region_id") or "").strip()
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


def authorized_local_evidence_adapter_registry(
    authorized_references: Sequence[str],
) -> EvidenceMethodAdapterRegistry:
    available = {
        ANALYTIC_REACHABILITY_ADAPTER: AnalyticReachabilityAdapter(),
        CAPTURED_VISIBILITY_ADAPTER: CapturedVisibilityAdapter(),
    }
    requested = sorted({str(item or "").strip() for item in authorized_references if str(item or "").strip()})
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"local_evidence_adapter_not_registered:{','.join(unknown)}")
    return EvidenceMethodAdapterRegistry([available[reference] for reference in requested])


__all__ = [
    "ANALYTIC_REACHABILITY_ADAPTER",
    "CAPTURED_VISIBILITY_ADAPTER",
    "AnalyticReachabilityAdapter",
    "CapturedVisibilityAdapter",
    "authorized_local_evidence_adapter_registry",
]
