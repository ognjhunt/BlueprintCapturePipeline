"""Turn scene-analysis proposals into bounded, site-specific task targets.

Proposal generation may be dynamic (vision model, semantic detector, or agent),
but target authorization is deterministic and fail-closed. External
reconstruction renders are valid derived observations when raw source video is
unavailable; they carry a lower claim ceiling rather than blocking the flow.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest, canonical_json


REQUEST_SCHEMA = "scene_task_target_analysis_request.v1"
RESULT_SCHEMA = "scene_task_target_analysis_result.v1"
_DIGEST_PREFIX = "sha256:"


class SceneTaskTargetPipelineError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith(_DIGEST_PREFIX)
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _finite3(value: Any) -> list[float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        )
    ):
        return None
    return [float(item) for item in value]


def build_scene_task_target_request(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        request = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise SceneTaskTargetPipelineError(["target_request_not_json"]) from exc
    supplied = request.pop("request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != REQUEST_SCHEMA:
        errors.append("target_request_schema_invalid")
    if not str(request.get("scene_id") or "").strip() or not _digest(
        request.get("source_scene_digest")
    ):
        errors.append("target_request_scene_binding_invalid")
    if request.get("source_video_available") not in {True, False}:
        errors.append("target_request_source_video_status_invalid")
    if request.get("source_observation_profile") not in {
        "raw_capture_and_reconstruction_views",
        "authorized_external_reconstruction_views",
    }:
        errors.append("target_request_observation_profile_invalid")
    views = request.get("rendered_views")
    if not isinstance(views, list) or not views:
        errors.append("target_request_rendered_views_missing")
        views = []
    view_ids: list[str] = []
    for row in views:
        if (
            not isinstance(row, Mapping)
            or not str(row.get("view_id") or "").strip()
            or not _digest(row.get("rgb_digest"))
            or row.get("observation_source")
            not in {
                "raw_capture",
                "reconstruction_render",
            }
        ):
            errors.append("target_request_rendered_view_invalid")
            continue
        view_ids.append(str(row["view_id"]))
    if len(view_ids) != len(set(view_ids)):
        errors.append("target_request_rendered_view_ids_duplicate")
    proposals = request.get("object_affordance_proposals")
    if not isinstance(proposals, list):
        errors.append("target_request_proposals_invalid")
    threshold = request.get("minimum_visual_confidence")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        errors.append("target_request_confidence_threshold_invalid")
    if request.get("threshold_frozen_before_analysis") is not True:
        errors.append("target_request_threshold_not_frozen")
    if request.get("candidate_may_self_authorize") is not False:
        errors.append("target_request_self_authorization_forbidden")
    analyzer = request.get("analyzer_provenance")
    if not isinstance(analyzer, Mapping):
        errors.append("target_request_analyzer_provenance_missing")
    else:
        for key in ("analyzer_id", "implementation_version"):
            if not str(analyzer.get(key) or "").strip():
                errors.append(f"target_request_analyzer_{key}_missing")
        if not _digest(analyzer.get("analyzer_contract_digest")):
            errors.append("target_request_analyzer_contract_digest_invalid")
        if analyzer.get("proposal_generation_is_dynamic") is not True:
            errors.append("target_request_dynamic_proposal_declaration_missing")
        if analyzer.get("candidate_may_self_authorize") is not False:
            errors.append("target_request_analyzer_self_authorization_forbidden")
    if request.get("robot_id") not in {"franka_panda", "unitree_g1"}:
        errors.append("target_request_robot_invalid")
    if request.get("metric_scale_status") not in {
        "validated",
        "provider_declared_not_independently_validated",
        "sensor_metric_unvalidated",
        "unverified",
    }:
        errors.append("target_request_metric_scale_status_invalid")
    expected = canonical_digest(request, digest_field="request_digest")
    if supplied is not None and supplied != expected:
        errors.append("target_request_digest_mismatch")
    if errors:
        raise SceneTaskTargetPipelineError(errors)
    request["request_digest"] = expected
    return request


def _proposal_result(
    proposal: Mapping[str, Any],
    *,
    request: Mapping[str, Any],
    view_ids: set[str],
) -> dict[str, Any]:
    blockers: list[str] = []
    proposal_id = str(proposal.get("proposal_id") or "").strip()
    label = str(proposal.get("object_label") or "").strip().lower()
    task_family = str(proposal.get("task_family") or "").strip()
    confidence = proposal.get("visual_confidence")
    confidence_value = (
        float(confidence)
        if not isinstance(confidence, bool)
        and isinstance(confidence, (int, float))
        and math.isfinite(float(confidence))
        else None
    )
    supporting = proposal.get("supporting_view_ids")
    supporting_ids = (
        sorted(set(str(item) for item in supporting if str(item)))
        if isinstance(supporting, list)
        else []
    )
    if not proposal_id or not label or not task_family:
        blockers.append("target_proposal_identity_missing")
    if confidence_value is None or not 0.0 <= confidence_value <= 1.0:
        blockers.append("target_proposal_confidence_invalid")
    elif confidence_value < float(request["minimum_visual_confidence"]):
        blockers.append("target_proposal_below_visual_threshold")
    if not supporting_ids or any(item not in view_ids for item in supporting_ids):
        blockers.append("target_proposal_supporting_views_invalid")
    evidence_digests = proposal.get("visual_evidence_digests")
    if (
        not isinstance(evidence_digests, list)
        or not evidence_digests
        or any(not _digest(item) for item in evidence_digests)
    ):
        blockers.append("target_proposal_visual_evidence_invalid")
    binding = proposal.get("target_binding")
    if not isinstance(binding, Mapping):
        binding = {}
        blockers.append("target_proposal_3d_binding_missing")
    method = binding.get("method")
    if method not in {
        "rendered_depth_backprojection",
        "scene_object_aabb_centroid",
        "multi_view_ray_intersection",
    }:
        blockers.append("target_proposal_3d_binding_method_invalid")
    position = _finite3(binding.get("position_scene"))
    if position is None:
        blockers.append("target_proposal_3d_position_invalid")
    if not _digest(binding.get("binding_evidence_digest")):
        blockers.append("target_proposal_binding_evidence_invalid")
    uncertainty = binding.get("spatial_uncertainty_scene_units")
    if (
        isinstance(uncertainty, bool)
        or not isinstance(uncertainty, (int, float))
        or not math.isfinite(float(uncertainty))
        or float(uncertainty) < 0.0
    ):
        blockers.append("target_proposal_spatial_uncertainty_invalid")
        uncertainty_value = None
    else:
        uncertainty_value = float(uncertainty)
    metric = request["metric_scale_status"] == "validated"
    collision = request.get("collision_support")
    collision_candidate_unbound = (
        isinstance(collision, Mapping)
        and collision.get("status") in {"candidate_compiled", "qualified"}
        and _digest(collision.get("collision_digest"))
    )
    collision_frame_bound = bool(
        collision_candidate_unbound
        and (
            collision.get("source_scene_digest") == request["source_scene_digest"]
            or (
                collision.get("frame_registration_status") in {"candidate_registered", "qualified"}
                and _digest(collision.get("scene_frame_binding_digest"))
            )
        )
    )
    collision_candidate = bool(collision_candidate_unbound and collision_frame_bound)
    collision_qualified = collision_candidate and collision.get("status") == "qualified"
    reach = request.get("reach_support")
    reach_checked = (
        metric
        and isinstance(reach, Mapping)
        and reach.get("status") in {"reachable", "unreachable"}
        and _digest(reach.get("reach_evidence_digest"))
    )
    bounded_sim_ready = not blockers and collision_candidate
    metric_sim_ready = bounded_sim_ready and metric and collision_qualified and reach_checked
    status = (
        "authorized_metric_sim_target"
        if metric_sim_ready
        else "authorized_derived_sim_target"
        if bounded_sim_ready
        else "abstained"
    )
    qualification_gaps = []
    if not metric:
        qualification_gaps.append("independent_metric_scale_missing")
    if not collision_candidate:
        qualification_gaps.append(
            "collision_candidate_frame_binding_missing"
            if collision_candidate_unbound
            else "collision_candidate_missing"
        )
    elif not collision_qualified:
        qualification_gaps.append("collision_candidate_not_independently_qualified")
    if not reach_checked:
        qualification_gaps.append("metric_reach_not_checked")
    return {
        "proposal_id": proposal_id,
        "object_label": label,
        "task_family": task_family,
        "affordances": sorted(
            set(str(item).strip() for item in proposal.get("affordances", []) if str(item).strip())
        ),
        "visual_confidence": confidence_value,
        "supporting_view_ids": supporting_ids,
        "target_position_scene": position,
        "target_binding_method": method,
        "spatial_uncertainty_scene_units": uncertainty_value,
        "status": status,
        "blockers": sorted(set(blockers)),
        "qualification_gaps": sorted(set(qualification_gaps)),
        "metric_scale_verified": metric,
        "collision_candidate_available": bool(collision_candidate),
        "collision_candidate_unbound_available": bool(collision_candidate_unbound),
        "collision_frame_bound": collision_frame_bound,
        "collision_qualified": bool(collision_qualified),
        "metric_reach_checked": bool(reach_checked),
        "claim_ceiling": (
            "metric_sim_task_target"
            if metric_sim_ready
            else "derived_scene_bounded_sim_target"
            if bounded_sim_ready
            else "task_target_proposal_only"
        ),
    }


def compile_scene_task_targets(value: Mapping[str, Any]) -> dict[str, Any]:
    request = build_scene_task_target_request(value)
    views = request["rendered_views"]
    view_ids = {str(row["view_id"]) for row in views}
    candidates = [
        _proposal_result(row, request=request, view_ids=view_ids)
        for row in request["object_affordance_proposals"]
        if isinstance(row, Mapping)
    ]
    authorized = [row for row in candidates if row["status"] != "abstained"]
    authorized.sort(
        key=lambda row: (
            0 if row["status"] == "authorized_metric_sim_target" else 1,
            -float(row["visual_confidence"] or 0.0),
            row["proposal_id"],
        )
    )
    selected = authorized[0] if authorized else None
    source_video_available = bool(request["source_video_available"])
    derived_views = any(row.get("observation_source") == "reconstruction_render" for row in views)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "target_ready_for_bounded_sim" if selected else "abstained",
        "request_digest": request["request_digest"],
        "scene_id": request["scene_id"],
        "source_scene_digest": request["source_scene_digest"],
        "robot_id": request["robot_id"],
        "analyzer_provenance": dict(request["analyzer_provenance"]),
        "source_video_available": source_video_available,
        "source_video_required_for_bounded_sim_target": False,
        "derived_reconstruction_views_used": derived_views,
        "metric_scale_status": request["metric_scale_status"],
        "candidate_targets": candidates,
        "selected_target": selected,
        "blockers": [] if selected else ["no_qualified_3d_task_target"],
        "claim_boundary": {
            "raw_capture_authority": source_video_available,
            "derived_views_are_raw_capture": False,
            "metric_reach_or_clearance": bool(
                selected and selected["status"] == "authorized_metric_sim_target"
            ),
            "simulated_task_success": False,
            "physical_task_success": False,
            "deployment_readiness": False,
        },
    }
    result["target_analysis_digest"] = canonical_digest(
        result, digest_field="target_analysis_digest"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result-out", required=True)
    args = parser.parse_args(argv)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    if not isinstance(request, Mapping):
        raise SceneTaskTargetPipelineError(["target_request_not_json_object"])
    result = compile_scene_task_targets(request)
    write_json(Path(args.result_out), result)
    print(canonical_json(result))
    return 0


__all__ = [
    "REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "SceneTaskTargetPipelineError",
    "build_scene_task_target_request",
    "compile_scene_task_targets",
]


if __name__ == "__main__":
    raise SystemExit(main())
