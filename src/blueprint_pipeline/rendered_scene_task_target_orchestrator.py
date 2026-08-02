"""Orchestrate rendered-view proposals into fail-closed 3D task targets.

The semantic analyzer is deliberately replaceable: a vision model, detector,
or agent may propose visible objects and 2D regions, but it cannot authorize
its own target.  This module digest-binds the rendered observations, binds each
proposal to the analysis splat, and delegates authorization to the deterministic
scene-task target gate.

Missing raw source video is not a blocker when the rendered observations are
authorized external-reconstruction derivatives.  Failed or weak visual-to-3D
bindings produce an abstention rather than a fabricated target.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest
from .scene_task_target_pipeline import compile_scene_task_targets
from .splat_bbox_target_binding import (
    SplatBBoxTargetBindingError,
    bind_splat_bbox_target,
)


PROPOSAL_SET_SCHEMA = "rendered_scene_task_proposal_set.v1"
RESULT_SCHEMA = "rendered_scene_task_target_orchestration.v1"


class RenderedSceneTaskTargetOrchestratorError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _clone(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise RenderedSceneTaskTargetOrchestratorError(["rendered_target_value_not_json"]) from exc


def _digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _validate_proposal_set(value: Mapping[str, Any]) -> dict[str, Any]:
    proposal_set = _clone(dict(value))
    supplied = proposal_set.pop("proposal_set_digest", None)
    errors: list[str] = []
    if proposal_set.get("schema_version") != PROPOSAL_SET_SCHEMA:
        errors.append("rendered_target_proposal_set_schema_invalid")
    analyzer = proposal_set.get("analyzer_provenance")
    if not isinstance(analyzer, Mapping):
        errors.append("rendered_target_analyzer_provenance_missing")
    else:
        for key in ("analyzer_id", "implementation_version"):
            if not str(analyzer.get(key) or "").strip():
                errors.append(f"rendered_target_analyzer_{key}_missing")
        if not _digest(analyzer.get("analyzer_contract_digest")):
            errors.append("rendered_target_analyzer_contract_digest_invalid")
        if analyzer.get("proposal_generation_is_dynamic") is not True:
            errors.append("rendered_target_dynamic_analyzer_required")
        if analyzer.get("candidate_may_self_authorize") is not False:
            errors.append("rendered_target_analyzer_self_authorization_forbidden")
    if proposal_set.get("candidate_may_self_authorize") is not False:
        errors.append("rendered_target_proposal_set_self_authorization_forbidden")
    proposals = proposal_set.get("proposals")
    if not isinstance(proposals, list):
        errors.append("rendered_target_proposals_invalid")
        proposals = []
    identifiers: list[str] = []
    for index, proposal in enumerate(proposals):
        prefix = f"rendered_target_proposal_{index}"
        if not isinstance(proposal, Mapping):
            errors.append(f"{prefix}_invalid")
            continue
        proposal_id = str(proposal.get("proposal_id") or "").strip()
        if not proposal_id:
            errors.append(f"{prefix}_identity_missing")
        identifiers.append(proposal_id)
        for key in ("object_label", "task_family", "binding_view_id"):
            if not str(proposal.get(key) or "").strip():
                errors.append(f"{prefix}_{key}_missing")
        confidence = proposal.get("visual_confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            errors.append(f"{prefix}_confidence_invalid")
        bbox = proposal.get("bbox_xyxy_pixels")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in bbox)
        ):
            errors.append(f"{prefix}_bbox_invalid")
        supporting = proposal.get("supporting_view_ids")
        if not isinstance(supporting, list) or not supporting:
            errors.append(f"{prefix}_supporting_views_missing")
    if len(identifiers) != len(set(identifiers)):
        errors.append("rendered_target_proposal_ids_duplicate")
    expected = canonical_digest(proposal_set, digest_field="proposal_set_digest")
    if supplied is not None and supplied != expected:
        errors.append("rendered_target_proposal_set_digest_mismatch")
    if errors:
        raise RenderedSceneTaskTargetOrchestratorError(errors)
    proposal_set["proposal_set_digest"] = expected
    return proposal_set


def _validate_views(
    value: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows = _clone(list(value))
    errors: list[str] = []
    by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        prefix = f"rendered_target_view_{index}"
        if not isinstance(row, Mapping):
            errors.append(f"{prefix}_invalid")
            continue
        view = dict(row)
        view_id = str(view.get("view_id") or "").strip()
        image_path = Path(str(view.get("rgb_path") or ""))
        if not view_id or view_id in by_id:
            errors.append(f"{prefix}_identity_invalid")
        if not _digest(view.get("rgb_digest")):
            errors.append(f"{prefix}_rgb_digest_invalid")
        if image_path.is_symlink():
            errors.append(f"{prefix}_rgb_symlink_forbidden")
            continue
        try:
            resolved = image_path.resolve(strict=True)
        except (OSError, RuntimeError):
            errors.append(f"{prefix}_rgb_missing")
            continue
        if not resolved.is_file() or _sha256(resolved) != view.get("rgb_digest"):
            errors.append(f"{prefix}_rgb_binding_invalid")
        if view.get("observation_source") not in {"raw_capture", "reconstruction_render"}:
            errors.append(f"{prefix}_observation_source_invalid")
        if not _digest(view.get("camera_spec_digest")):
            errors.append(f"{prefix}_camera_spec_digest_invalid")
        if not isinstance(view.get("camera"), Mapping) or not isinstance(
            view.get("image_size"), Mapping
        ):
            errors.append(f"{prefix}_geometry_missing")
        view["rgb_path"] = str(resolved)
        by_id[view_id] = view
    if not rows:
        errors.append("rendered_target_views_missing")
    if errors:
        raise RenderedSceneTaskTargetOrchestratorError(errors)
    return [dict(row) for row in rows], by_id


def compile_rendered_scene_task_target(
    *,
    analysis_splat_path: str | Path,
    scene_id: str,
    source_scene_digest: str,
    rendered_views: Sequence[Mapping[str, Any]],
    proposal_set: Mapping[str, Any],
    source_video_available: bool,
    robot_id: str,
    metric_scale_status: str,
    collision_support: Mapping[str, Any],
    reach_support: Mapping[str, Any],
    minimum_visual_confidence: float = 0.8,
    minimum_opacity: float = 0.18,
    front_depth_fraction: float = 0.25,
    minimum_projected_splats: int = 32,
) -> dict[str, Any]:
    """Bind dynamic 2D proposals and deterministically authorize or abstain."""

    proposals = _validate_proposal_set(proposal_set)
    views, by_id = _validate_views(rendered_views)
    splat = Path(analysis_splat_path)
    if splat.is_symlink():
        raise RenderedSceneTaskTargetOrchestratorError(
            ["rendered_target_analysis_splat_symlink_forbidden"]
        )
    try:
        splat = splat.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RenderedSceneTaskTargetOrchestratorError(
            ["rendered_target_analysis_splat_missing"]
        ) from exc
    analysis_splat_digest = _sha256(splat)
    binding_results: list[dict[str, Any]] = []
    qualified_proposals: list[dict[str, Any]] = []
    for proposal in proposals["proposals"]:
        binding_view_id = str(proposal["binding_view_id"])
        view = by_id.get(binding_view_id)
        binding: dict[str, Any] | None = None
        binding_blockers: list[str] = []
        if view is None:
            binding_blockers.append("rendered_target_binding_view_missing")
        else:
            try:
                binding = bind_splat_bbox_target(
                    analysis_splat_path=splat,
                    request={
                        "schema_version": "splat_bbox_target_binding_request.v1",
                        "source_scene_digest": source_scene_digest,
                        "analysis_splat_digest": analysis_splat_digest,
                        "camera_spec_digest": view["camera_spec_digest"],
                        "rgb_digest": view["rgb_digest"],
                        "view_id": binding_view_id,
                        "image_size": view["image_size"],
                        "bbox_xyxy_pixels": proposal["bbox_xyxy_pixels"],
                        "camera": view["camera"],
                        "minimum_opacity": minimum_opacity,
                        "front_depth_fraction": front_depth_fraction,
                        "minimum_projected_splats": minimum_projected_splats,
                        "binding_may_self_authorize": False,
                    },
                )
            except SplatBBoxTargetBindingError as exc:
                binding_blockers.extend(exc.codes)
        binding_results.append(
            {
                "proposal_id": proposal["proposal_id"],
                "status": "candidate_bound" if binding is not None else "abstained",
                "binding": binding,
                "blockers": sorted(set(binding_blockers)),
            }
        )
        supporting_ids = sorted(set(str(item) for item in proposal["supporting_view_ids"]))
        evidence_digests = [by_id[item]["rgb_digest"] for item in supporting_ids if item in by_id]
        qualified_proposals.append(
            {
                "proposal_id": proposal["proposal_id"],
                "object_label": proposal["object_label"],
                "task_family": proposal["task_family"],
                "affordances": list(proposal.get("affordances") or []),
                "visual_confidence": proposal["visual_confidence"],
                "supporting_view_ids": supporting_ids,
                "visual_evidence_digests": evidence_digests,
                "target_binding": (
                    {
                        "method": binding["method"],
                        "position_scene": binding["position_scene"],
                        "spatial_uncertainty_scene_units": binding[
                            "spatial_uncertainty_scene_units"
                        ],
                        "binding_evidence_digest": binding["binding_evidence_digest"],
                    }
                    if binding is not None
                    else {}
                ),
            }
        )
    target_analysis = compile_scene_task_targets(
        {
            "schema_version": "scene_task_target_analysis_request.v1",
            "scene_id": scene_id,
            "source_scene_digest": source_scene_digest,
            "source_video_available": bool(source_video_available),
            "source_observation_profile": (
                "raw_capture_and_reconstruction_views"
                if source_video_available
                else "authorized_external_reconstruction_views"
            ),
            "rendered_views": [
                {
                    "view_id": view["view_id"],
                    "rgb_digest": view["rgb_digest"],
                    "observation_source": view["observation_source"],
                }
                for view in views
            ],
            "object_affordance_proposals": qualified_proposals,
            "minimum_visual_confidence": minimum_visual_confidence,
            "threshold_frozen_before_analysis": True,
            "candidate_may_self_authorize": False,
            "analyzer_provenance": proposals["analyzer_provenance"],
            "robot_id": robot_id,
            "metric_scale_status": metric_scale_status,
            "collision_support": dict(collision_support),
            "reach_support": dict(reach_support),
        }
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": target_analysis["status"],
        "scene_id": scene_id,
        "source_scene_digest": source_scene_digest,
        "analysis_splat_digest": analysis_splat_digest,
        "proposal_set_digest": proposals["proposal_set_digest"],
        "analyzer_provenance": proposals["analyzer_provenance"],
        "binding_results": binding_results,
        "target_analysis": target_analysis,
        "source_video_available": bool(source_video_available),
        "source_video_required_for_bounded_sim_target": False,
        "candidate_may_self_authorize": False,
        "proof_effect": "derived_rendered_view_task_target_candidate",
        "claim_ceiling": (
            "derived_scene_bounded_sim_target"
            if target_analysis["status"] == "target_ready_for_bounded_sim"
            else "task_target_proposal_only"
        ),
        "unsupported_claims": [
            "semantic_ground_truth",
            "metric_scale",
            "simulated_task_success",
            "physical_task_success",
            "deployment_readiness",
        ],
    }
    result["orchestration_digest"] = canonical_digest(result, digest_field="orchestration_digest")
    return result


__all__ = [
    "PROPOSAL_SET_SCHEMA",
    "RESULT_SCHEMA",
    "RenderedSceneTaskTargetOrchestratorError",
    "compile_rendered_scene_task_target",
]
