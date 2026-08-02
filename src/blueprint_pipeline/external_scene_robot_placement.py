"""Propose a Franka runtime pose around a registered external-scene target.

The dynamic solver reuses Blueprint's general ring-scan placement engine with a
collision-mesh vertex-overlap probe. The deterministic result remains a runtime
visualization candidate until metric scale, live contact, full footprint, and
reach checks are independently qualified.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .decision_evidence_contracts import canonical_digest
from .external_scene_collision_candidate import _flatten_glb, _sha256
from .provider_nurec_robot_placement import build_default_franka_policy_trace_request
from .scene_placement.placement import ring_scan_stand_pose
from .scene_placement.robot_profile import get_robot_profile
from .scene_placement.types import SceneObject


REQUEST_SCHEMA = "external_scene_robot_placement_request.v1"
RESULT_SCHEMA = "external_scene_robot_placement_candidate.v1"


class ExternalSceneRobotPlacementError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _finite3(value: Any) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result).all() else None


def build_external_scene_robot_placement_request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExternalSceneRobotPlacementError(["external_placement_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("external_placement_request_schema_invalid")
    if request.get("robot_id") != "franka_panda":
        errors.append("external_placement_robot_must_be_default_franka")
    for key in (
        "source_scene_digest",
        "target_analysis_digest",
        "target_binding_digest",
        "scene_frame_binding_digest",
        "collision_candidate_digest",
        "collision_source_digest",
    ):
        if not _digest(request.get(key)):
            errors.append(f"external_placement_{key}_invalid")
    if _finite3(request.get("target_position_collision_stage")) is None:
        errors.append("external_placement_target_position_invalid")
    uncertainty = request.get("target_spatial_uncertainty_stage_units")
    if (
        isinstance(uncertainty, bool)
        or not isinstance(uncertainty, (int, float))
        or not math.isfinite(float(uncertainty))
        or float(uncertainty) <= 0
    ):
        errors.append("external_placement_target_uncertainty_invalid")
    if request.get("metric_scale_status") not in {
        "validated",
        "provider_declared_not_independently_validated",
        "unverified",
    }:
        errors.append("external_placement_metric_status_invalid")
    if request.get("collision_status") not in {"candidate_compiled", "qualified"}:
        errors.append("external_placement_collision_status_invalid")
    if request.get("candidate_may_self_authorize") is not False:
        errors.append("external_placement_self_authorization_forbidden")
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        errors.append("external_placement_request_digest_mismatch")
    if errors:
        raise ExternalSceneRobotPlacementError(errors)
    request["request_digest"] = expected
    return request


def propose_external_scene_robot_placement(
    *, collision_glb_path: str | Path, request: Mapping[str, Any]
) -> dict[str, Any]:
    admitted = build_external_scene_robot_placement_request(request)
    glb = Path(collision_glb_path).resolve(strict=True)
    if glb.suffix.lower() != ".glb" or _sha256(glb) != admitted["collision_source_digest"]:
        raise ExternalSceneRobotPlacementError(["external_placement_collision_source_mismatch"])
    vertices, _, _ = _flatten_glb(glb)
    # Same exact GLB Y-up -> collision-stage Z-up transform used by the collider compiler.
    stage_vertices = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    floor_z = float(stage_vertices[:, 2].min())
    target = _finite3(admitted["target_position_collision_stage"])
    assert target is not None
    uncertainty = float(admitted["target_spatial_uncertainty_stage_units"])
    horizontal_half_extent = max(0.05, uncertainty)
    target_object = SceneObject(
        id="registered_scene_task_target",
        label=str(admitted.get("target_label") or "registered task target"),
        bbox_min=tuple(target - [horizontal_half_extent, horizontal_half_extent, 0.15]),
        bbox_max=tuple(target + [horizontal_half_extent, horizontal_half_extent, 0.15]),
        centroid=tuple(target),
        category="fixture",
        source="registered_external_reconstruction",
        confidence=float(admitted.get("visual_confidence") or 0.0),
    )
    profile = get_robot_profile("franka_panda")
    half_x, half_y, _ = profile.footprint_half_extent_xyz

    def probe(pose, yaw) -> int:
        x, y, _ = pose
        cosine, sine = math.cos(yaw), math.sin(yaw)
        delta = stage_vertices[:, :2] - [x, y]
        local_x = cosine * delta[:, 0] + sine * delta[:, 1]
        local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
        occupied = (
            (np.abs(local_x) <= half_x + profile.probe_clearance_m)
            & (np.abs(local_y) <= half_y + profile.probe_clearance_m)
            & (stage_vertices[:, 2] >= floor_z + 0.04)
            & (stage_vertices[:, 2] <= floor_z + 2.0 * profile.footprint_half_extent_xyz[2])
        )
        return int(occupied.sum())

    stance = ring_scan_stand_pose(
        target_object,
        probe=probe,
        floor_z=floor_z,
        standing_distance=profile.standoff_range_m[0],
        max_standing_distance=profile.standoff_range_m[1],
        radial_step=profile.probe_step_m,
        n_azimuths=144,
        robot_profile=profile,
    )
    reach_limit = float(profile.max_shoulder_to_affordance_m())

    def shoulder_distance(candidate) -> float:
        shoulder = np.asarray(candidate.position) + [
            0.0,
            0.0,
            profile.shoulder_above_root_m,
        ]
        return float(np.linalg.norm(target - shoulder))

    placement_selection_strategy = "nearest_nominal_standoff_collision_clear_candidate"
    initial_shoulder_distance = shoulder_distance(stance)
    # A collision-clear pose at the nominal standoff can still put a high or deep
    # target just outside the arm envelope. Search the narrow gap down to the
    # profile's own base standoff, while retaining the footprint-clearance probe.
    # This is an analytic rescue candidate only; it cannot qualify metric reach.
    if stance.clear and initial_shoulder_distance > reach_limit:
        rescue_minimum_standoff = max(
            float(profile.standing_distance_m),
            float(profile.probe_clearance_m),
        )
        rescue = ring_scan_stand_pose(
            target_object,
            probe=probe,
            floor_z=floor_z,
            standing_distance=rescue_minimum_standoff,
            max_standing_distance=profile.standoff_range_m[0],
            radial_step=profile.probe_step_m,
            n_azimuths=144,
            robot_profile=profile,
        )
        if rescue.clear and shoulder_distance(rescue) <= reach_limit:
            stance = rescue
            placement_selection_strategy = "collision_clear_analytic_reach_rescue_candidate"

    hits = probe(stance.position, stance.yaw)
    shoulder_distance_value = shoulder_distance(stance)
    analytic_reach_candidate = bool(shoulder_distance_value <= reach_limit)
    placement = {
        "schema_version": RESULT_SCHEMA,
        "status": "runtime_visualization_candidate_only" if stance.clear else "abstained",
        "request_digest": admitted["request_digest"],
        "robot_id": "franka_panda",
        "official_isaac_asset": profile.simulator_asset_refs["isaac_asset"],
        "robot_prim_path": profile.usd_prim_path,
        "target_position_collision_stage": list(admitted["target_position_collision_stage"]),
        "target_spatial_uncertainty_stage_units": uncertainty,
        "robot_pose_xyzyaw_collision_stage": [
            round(float(stance.position[0]), 9),
            round(float(stance.position[1]), 9),
            round(float(floor_z), 9),
            round(float(stance.yaw), 12),
        ],
        "floor_height_collision_stage": round(floor_z, 9),
        "mesh_vertex_overlap_probe_hits": hits,
        "mesh_vertex_overlap_probe_clear": bool(stance.clear and hits == 0),
        "standoff_stage_units": round(float(stance.standoff_m), 9),
        "placement_selection_strategy": placement_selection_strategy,
        "analytic_shoulder_to_target_distance_stage_units": round(shoulder_distance_value, 9),
        "analytic_profile_reach_limit_stage_units": round(reach_limit, 9),
        "analytic_reach_candidate": analytic_reach_candidate,
        "metric_scale_status": admitted["metric_scale_status"],
        "metric_reach_qualified": bool(
            analytic_reach_candidate
            and admitted["metric_scale_status"] == "validated"
            and admitted["collision_status"] == "qualified"
        ),
        "collision_status": admitted["collision_status"],
        "scene_frame_binding_digest": admitted["scene_frame_binding_digest"],
        "collision_candidate_digest": admitted["collision_candidate_digest"],
        "target_analysis_digest": admitted["target_analysis_digest"],
        "target_binding_digest": admitted["target_binding_digest"],
        "formal_gaps": [
            *(
                []
                if admitted["metric_scale_status"] == "validated"
                else ["independent_metric_scale_missing"]
            ),
            *(
                []
                if admitted["collision_status"] == "qualified"
                else ["live_collision_contact_and_full_footprint_not_qualified"]
            ),
            *(
                ["placement_below_nominal_standoff_range"]
                if stance.standoff_m < profile.standoff_range_m[0]
                else []
            ),
            "access_reset_and_human_clearance_not_qualified",
        ],
        "candidate_may_self_authorize": False,
        "physical_execution_authorized": False,
        "proof_effect": "external_scene_runtime_robot_visualization_candidate",
        "claim_ceiling": "analytic_robot_placement_candidate",
    }
    placement["placement_proposal_digest"] = canonical_digest(
        placement, digest_field="placement_proposal_digest"
    )
    render_options = {
        "robot_id": "franka_panda",
        "robot_usd": str(profile.simulator_asset_refs["isaac_asset"]).lstrip("/"),
        "robot_prim_path": profile.usd_prim_path,
        "robot_pose": placement["robot_pose_xyzyaw_collision_stage"],
        "robot_ground_z": placement["floor_height_collision_stage"],
        "robot_only_pass": True,
        "robot_placement_digest": placement["placement_proposal_digest"],
        "placement_proposal_digest": placement["placement_proposal_digest"],
        "lights_path": "/World/Lights",
        "articulated_policy_trace_request": build_default_franka_policy_trace_request(
            robot_prim_path=profile.usd_prim_path
        ),
    }
    render_options["render_options_digest"] = canonical_digest(
        render_options, digest_field="render_options_digest"
    )
    return {"placement": placement, "render_options": render_options}


__all__ = [
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "ExternalSceneRobotPlacementError",
    "build_external_scene_robot_placement_request",
    "propose_external_scene_robot_placement",
]
