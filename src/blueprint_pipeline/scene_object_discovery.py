"""Fail-closed whole-splat object discovery and selection.

This module is the deep boundary between scene survey, replaceable visual
analyzers, semantic refinement, and the existing scene-construction contract.
It deliberately keeps RGB/model proposals candidate-only.  A proposal becomes
an automatically selectable ``source_object`` only when it carries independent,
digest-bound metric geometry from publisher truth or a production-qualified
semantic lifting backend.

The implementation is provider-neutral: Splat Analyzer, SAM 3.1, and a rendered
scene agent all enter through the same candidate interface.  The deterministic
compiler, rather than any model, owns admission and selection.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .scene_placement.perception_views import view_ring_for_bounds
from .sealed_camera_render import render_splat_at_exact_cameras
from .task_evaluation_splat_render_runtime import runtime_from_environment


CAMERA_PLAN_SCHEMA = "scene_object_discovery_camera_plan.v1"
DISCOVERY_SCHEMA = "scene_object_discovery.v1"
CALIBRATION_SCHEMA = "scene_object_discovery_calibration.v1"
SUPPORTED_ANALYZERS = {
    "publisher_semantics",
    "rendered_scene_agent",
    "sam31",
    "splat_analyzer",
}
RGB_ONLY_ANALYZERS = {"rendered_scene_agent", "sam31", "splat_analyzer"}
METRIC_GEOMETRY_AUTHORITIES = {
    "publisher_metric_label",
    "production_semantic_gaussian_obb",
}


class SceneObjectDiscoveryError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__(";".join(self.codes))


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SceneObjectDiscoveryError(["scene_discovery_value_not_canonical_json"]) from exc


def _digest_payload(value: Mapping[str, Any], *, field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return "sha256:" + hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _is_digest(value: Any) -> bool:
    text = str(value or "")
    return (
        len(text) == 71
        and text.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in text[7:])
    )


def _finite_xyz(value: Any) -> tuple[float, float, float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        return None
    try:
        row = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    return row if all(math.isfinite(item) for item in row) else None


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _look_at_opencv(
    eye: Sequence[float],
    target: Sequence[float],
    up: Sequence[float],
) -> list[list[float]]:
    """Build a camera-to-world OpenCV pose without duplicating renderer policy."""

    eye_row = _finite_xyz(eye)
    target_row = _finite_xyz(target)
    up_row = _finite_xyz(up)
    if eye_row is None or target_row is None or up_row is None:
        raise SceneObjectDiscoveryError(["scene_discovery_camera_vector_invalid"])
    forward = [target_row[index] - eye_row[index] for index in range(3)]
    forward_norm = math.sqrt(sum(value * value for value in forward))
    up_norm = math.sqrt(sum(value * value for value in up_row))
    if forward_norm <= 1e-9 or up_norm <= 1e-9:
        raise SceneObjectDiscoveryError(["scene_discovery_camera_degenerate"])
    forward = [value / forward_norm for value in forward]
    up_unit = [value / up_norm for value in up_row]
    # OpenCV camera axes are right, down, forward.  Match the production
    # renderer bridge: derive right from forward x world-down, then down from
    # forward x right.
    down_seed = [-value for value in up_unit]
    right = [
        forward[1] * down_seed[2] - forward[2] * down_seed[1],
        forward[2] * down_seed[0] - forward[0] * down_seed[2],
        forward[0] * down_seed[1] - forward[1] * down_seed[0],
    ]
    right_norm = math.sqrt(sum(value * value for value in right))
    if right_norm <= 1e-9:
        right = [1.0, 0.0, 0.0]
    else:
        right = [value / right_norm for value in right]
    down = [
        forward[1] * right[2] - forward[2] * right[1],
        forward[2] * right[0] - forward[0] * right[2],
        forward[0] * right[1] - forward[1] * right[0],
    ]
    down_norm = math.sqrt(sum(value * value for value in down))
    down = [value / down_norm for value in down]
    return [
        [right[0], down[0], forward[0], eye_row[0]],
        [right[1], down[1], forward[1], eye_row[1]],
        [right[2], down[2], forward[2], eye_row[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


def build_full_scene_camera_plan(
    *,
    scene_geometry: Mapping[str, Any],
    source_splat_digest: str,
    retained_gaussian_count: int,
    registration_digest: str,
    normalization_transform_digest: str | None = None,
    width: int = 1280,
    height: int = 960,
    n_azimuths: int = 8,
    elevations_deg: Sequence[float] = (12.0, 38.0),
) -> dict[str, Any]:
    """Build an exact, full-topology survey plan from analyzed scene bounds.

    ``view_ring_for_bounds`` is z-up.  Non-z-up scenes must first provide an
    explicit normalized geometry and its transform digest; silently orbiting a
    different axis would produce plausible but invalid survey evidence.
    """

    errors: list[str] = []
    bounds_min = _finite_xyz(scene_geometry.get("aabb_min"))
    bounds_max = _finite_xyz(scene_geometry.get("aabb_max"))
    if bounds_min is None or bounds_max is None or any(
        bounds_max[index] <= bounds_min[index] for index in range(3)
    ):
        errors.append("scene_discovery_bounds_invalid")
    if not _is_digest(source_splat_digest):
        errors.append("scene_discovery_source_digest_invalid")
    if not _is_digest(registration_digest):
        errors.append("scene_discovery_registration_digest_invalid")
    if (
        isinstance(retained_gaussian_count, bool)
        or not isinstance(retained_gaussian_count, int)
        or retained_gaussian_count <= 0
    ):
        errors.append("scene_discovery_retained_count_invalid")
    up_axis = scene_geometry.get("up_axis")
    up_sign = _finite_float(scene_geometry.get("up_sign"))
    if (up_axis, up_sign) != (2, 1.0):
        errors.append("scene_discovery_normalized_geometry_required")
    original_up_axis = scene_geometry.get("original_up_axis", up_axis)
    original_up_sign = _finite_float(scene_geometry.get("original_up_sign", up_sign))
    if (original_up_axis, original_up_sign) != (2, 1.0) and not _is_digest(
        normalization_transform_digest
    ):
        errors.append("scene_discovery_non_z_up_normalization_required")
    if width < 64 or height < 64 or n_azimuths < 4 or not elevations_deg:
        errors.append("scene_discovery_camera_plan_dimensions_invalid")
    if errors:
        raise SceneObjectDiscoveryError(errors)
    assert bounds_min is not None and bounds_max is not None
    ring = view_ring_for_bounds(
        bounds_min,
        bounds_max,
        margin=1.7,
        n_azimuths=n_azimuths,
        elevations_deg=elevations_deg,
        vfov_deg=60.0,
        width=width,
        height=height,
    )
    cameras: list[dict[str, Any]] = []
    for index, row in enumerate(ring):
        vfov = float(row["vfov"])
        fy = 0.5 * float(height) / math.tan(0.5 * vfov)
        camera_id = f"survey_{index:03d}"
        cameras.append(
            {
                "camera_id": camera_id,
                "T_world_camera_provider_frame": _look_at_opencv(
                    row["eye"], row["target"], row["up"]
                ),
                "intrinsics": {
                    "fx": fy,
                    "fy": fy,
                    "cx": 0.5 * float(width),
                    "cy": 0.5 * float(height),
                    "width": int(width),
                    "height": int(height),
                },
                "survey_role": "whole_scene_topology_before_target_closeup",
            }
        )
    result = {
        "schema_version": CAMERA_PLAN_SCHEMA,
        "source_splat_digest": source_splat_digest,
        "retained_gaussian_count": int(retained_gaussian_count),
        "registration_digest": registration_digest,
        "normalization_transform_digest": normalization_transform_digest,
        "scene_geometry": dict(scene_geometry),
        "coverage": {
            "strategy": "deterministic_stacked_full_scene_ring",
            "known_scene_bounds_covered": True,
            "missing_source_observations_recoverable_by_virtual_camera": False,
            "unseen_regions": list(scene_geometry.get("unseen_regions") or []),
        },
        "cameras": cameras,
    }
    result["camera_plan_digest"] = _digest_payload(result, field="camera_plan_digest")
    return result


def materialize_scene_object_discovery_renders(
    *,
    source_splat_path: str | Path,
    camera_plan: Mapping[str, Any],
    output_root: str | Path,
    repo_root: str | Path | None = None,
    graphics_backend: str = "egl",
    runtime_resolver: Callable[..., Mapping[str, Any]] = runtime_from_environment,
    renderer: Callable[..., Mapping[str, Any]] = render_splat_at_exact_cameras,
) -> dict[str, Any]:
    """Render the whole-scene plan through the immutable production renderer."""

    plan = dict(camera_plan)
    supplied_digest = plan.get("camera_plan_digest")
    if supplied_digest != _digest_payload(plan, field="camera_plan_digest"):
        raise SceneObjectDiscoveryError(["scene_discovery_camera_plan_digest_mismatch"])
    source = Path(source_splat_path).resolve()
    if source.is_symlink() or not source.is_file():
        raise SceneObjectDiscoveryError(["scene_discovery_source_splat_missing_or_symlink"])
    if _sha256_file(source) != plan.get("source_splat_digest"):
        raise SceneObjectDiscoveryError(["scene_discovery_source_splat_digest_mismatch"])
    cameras = plan.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise SceneObjectDiscoveryError(["scene_discovery_cameras_missing"])
    requested_root = Path(output_root)
    if requested_root.is_symlink():
        raise SceneObjectDiscoveryError(["scene_discovery_output_root_symlink_forbidden"])
    root = requested_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    calibration = {
        "schema_version": CALIBRATION_SCHEMA,
        "camera_plan_digest": supplied_digest,
        "cameras": [
            {
                "camera_id": row["camera_id"],
                "T_world_camera_provider_frame": row["T_world_camera_provider_frame"],
                "intrinsics": row["intrinsics"],
            }
            for row in cameras
        ],
    }
    calibration_path = root / "exact_scene_discovery_calibration.json"
    calibration_path.write_text(_canonical_json(calibration) + "\n", encoding="utf-8")
    resolved_repo = Path(repo_root).resolve() if repo_root else Path(__file__).resolve().parents[2]
    runtime = dict(runtime_resolver(repo_root=resolved_repo))
    render_manifest = dict(
        renderer(
            splat_path=source,
            cameras=calibration["cameras"],
            output_dir=root / "rendered",
            provider_splat_import_receipt_digest=plan["source_splat_digest"],
            alignment_digest=plan["registration_digest"],
            camera_set_label="scene-object-discovery-full-scene-v1",
            calibrated_camera_file=calibration_path,
            retained_gaussian_count=plan["retained_gaussian_count"],
            source_splat_digest=plan["source_splat_digest"],
            purpose="scene_object_discovery_full_scene_method_inputs",
            authorization_class="method_input",
            node=runtime["node"],
            renderer_runtime_root=runtime["renderer_root"],
            browser_executable=runtime["browser_executable"],
            renderer_runtime_identity=runtime["identity"],
            graphics_backend=graphics_backend,
        )
    )
    render_manifest_digest = render_manifest.get("sealed_camera_render_manifest_digest")
    if not _is_digest(render_manifest_digest):
        raise SceneObjectDiscoveryError(["scene_discovery_render_manifest_digest_invalid"])
    return {
        "camera_plan": plan,
        "calibration_path": str(calibration_path),
        "calibration_digest": _sha256_file(calibration_path),
        "render_manifest": render_manifest,
        "render_binding": {
            "source_splat_digest": plan["source_splat_digest"],
            "camera_plan_digest": supplied_digest,
            "render_manifest_digest": render_manifest_digest,
        },
        "renderer_capabilities": {
            "rgb": True,
            "depth": False,
            "per_gaussian_contributions": False,
            "semantic_lifting_from_this_renderer_supported": False,
        },
    }


def _candidate_metric_authority(candidate: Mapping[str, Any]) -> str | None:
    geometry = candidate.get("metric_geometry")
    if not isinstance(geometry, Mapping):
        return None
    authority = str(geometry.get("authority") or "")
    if authority not in METRIC_GEOMETRY_AUTHORITIES:
        return None
    if geometry.get("validated") is not True or not _is_digest(geometry.get("evidence_digest")):
        return None
    bounds_min = _finite_xyz(geometry.get("bounds_min"))
    bounds_max = _finite_xyz(geometry.get("bounds_max"))
    if bounds_min is None or bounds_max is None or any(
        bounds_max[index] <= bounds_min[index] for index in range(3)
    ):
        return None
    if authority == "production_semantic_gaussian_obb" and (
        geometry.get("production_large_scene_ready") is not True
        or geometry.get("independent_deterministic_validation_passed") is not True
    ):
        return None
    return authority


def _task_match_score(candidate: Mapping[str, Any], task_context: Mapping[str, Any]) -> float:
    label = str(candidate.get("label") or "").lower()
    task = " ".join(
        str(task_context.get(key) or "").lower()
        for key in ("task_statement", "target_hint", "workflow_context")
    )
    tokens = {token.strip(".,:;!?()[]{}") for token in label.split() if len(token) >= 3}
    overlap = sum(1 for token in tokens if token in task)
    supplied = candidate.get("task_relevance")
    supplied_score = _finite_float(supplied.get("score")) if isinstance(supplied, Mapping) else None
    supplied_score = supplied_score if supplied_score is not None else 0.0
    return round(max(supplied_score, min(1.0, 0.25 + 0.25 * overlap)), 6)


def compile_scene_object_discovery(
    *,
    source_binding: Mapping[str, Any],
    camera_plan: Mapping[str, Any],
    render_binding: Mapping[str, Any],
    analyzer_runs: Sequence[Mapping[str, Any]],
    task_context: Mapping[str, Any],
    minimum_confidence: float = 0.5,
    minimum_task_relevance: float = 0.5,
) -> dict[str, Any]:
    """Compile analyzer outputs into a sealed decision or explicit abstention."""

    errors: list[str] = []
    source_digest = source_binding.get("source_splat_digest")
    if not _is_digest(source_digest):
        errors.append("scene_discovery_source_binding_digest_invalid")
    camera_plan_digest = camera_plan.get("camera_plan_digest")
    if camera_plan_digest != _digest_payload(dict(camera_plan), field="camera_plan_digest"):
        errors.append("scene_discovery_camera_plan_invalid")
    render_manifest_digest = render_binding.get("render_manifest_digest")
    if not _is_digest(render_manifest_digest):
        errors.append("scene_discovery_render_manifest_digest_invalid")
    if render_binding.get("source_splat_digest") != source_digest:
        errors.append("scene_discovery_render_source_mismatch")
    if render_binding.get("camera_plan_digest") != camera_plan_digest:
        errors.append("scene_discovery_render_camera_plan_mismatch")
    minimum_confidence_value = _finite_float(minimum_confidence)
    minimum_task_relevance_value = _finite_float(minimum_task_relevance)
    if minimum_confidence_value is None or not 0.0 <= minimum_confidence_value <= 1.0:
        errors.append("scene_discovery_minimum_confidence_invalid")
    if (
        minimum_task_relevance_value is None
        or not 0.0 <= minimum_task_relevance_value <= 1.0
    ):
        errors.append("scene_discovery_minimum_task_relevance_invalid")

    candidates: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    analyzer_receipts: list[dict[str, Any]] = []
    for index, raw_run in enumerate(analyzer_runs):
        run = dict(raw_run)
        backend = str(run.get("backend") or "")
        if backend not in SUPPORTED_ANALYZERS:
            errors.append(f"scene_discovery_analyzer_{index}_backend_invalid")
            continue
        run_digest = run.get("run_digest")
        if not _is_digest(run_digest):
            errors.append(f"scene_discovery_analyzer_{index}_digest_invalid")
        if run.get("source_splat_digest") != source_digest:
            errors.append(f"scene_discovery_analyzer_{index}_source_mismatch")
        if backend in RGB_ONLY_ANALYZERS and run.get("render_manifest_digest") != render_manifest_digest:
            errors.append(f"scene_discovery_analyzer_{index}_render_mismatch")
        rows = run.get("candidates")
        if not isinstance(rows, list):
            errors.append(f"scene_discovery_analyzer_{index}_candidates_invalid")
            continue
        analyzer_receipts.append(
            {"backend": backend, "run_digest": run_digest, "candidate_count": len(rows)}
        )
        for candidate_index, raw_candidate in enumerate(rows):
            if not isinstance(raw_candidate, Mapping):
                errors.append(
                    f"scene_discovery_analyzer_{index}_candidate_{candidate_index}_invalid"
                )
                continue
            candidate = dict(raw_candidate)
            candidate_id = str(candidate.get("candidate_id") or "").strip()
            label = str(candidate.get("label") or "").strip()
            confidence = candidate.get("confidence")
            if not candidate_id or candidate_id in candidate_ids:
                errors.append("scene_discovery_candidate_identity_invalid_or_duplicate")
                continue
            if not label:
                errors.append(f"scene_discovery_candidate_{candidate_id}_label_missing")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0.0 <= float(confidence) <= 1.0
            ):
                errors.append(f"scene_discovery_candidate_{candidate_id}_confidence_invalid")
                continue
            supporting_views = candidate.get("supporting_view_ids")
            if backend in RGB_ONLY_ANALYZERS and (
                not isinstance(supporting_views, list) or not supporting_views
            ):
                errors.append(f"scene_discovery_candidate_{candidate_id}_views_missing")
            candidate_ids.add(candidate_id)
            metric_authority = _candidate_metric_authority(candidate)
            task_score = _task_match_score(candidate, task_context)
            admitted = (
                minimum_confidence_value is not None
                and minimum_task_relevance_value is not None
                and float(confidence) >= minimum_confidence_value
                and task_score >= minimum_task_relevance_value
                and metric_authority is not None
            )
            candidate.update(
                {
                    "candidate_id": candidate_id,
                    "label": label,
                    "backend": backend,
                    "analyzer_run_digest": run_digest,
                    "task_match_score": task_score,
                    "metric_geometry_authority": metric_authority,
                    "eligible_for_automatic_source_object": admitted,
                    "candidate_claim_boundary": (
                        "metric_source_object_candidate"
                        if metric_authority
                        else "model_derived_visual_candidate_not_metric_source_object"
                    ),
                }
            )
            candidates.append(candidate)
    if errors:
        raise SceneObjectDiscoveryError(errors)

    eligible = sorted(
        (candidate for candidate in candidates if candidate["eligible_for_automatic_source_object"]),
        key=lambda row: (
            -float(row["task_match_score"]),
            -float(row["confidence"]),
            str(row["candidate_id"]),
        ),
    )
    if len(eligible) == 1:
        status = "ready_auto_selected"
        selected = eligible[0]
        geometry = dict(selected["metric_geometry"])
        source_object = {
            "object_id": selected["candidate_id"],
            "publisher_instance_id": selected.get("publisher_instance_id"),
            "label": selected["label"],
            "bounds_min": geometry["bounds_min"],
            "bounds_max": geometry["bounds_max"],
            "metric_geometry_authority": selected["metric_geometry_authority"],
            "metric_geometry_evidence_digest": geometry["evidence_digest"],
        }
        next_action = "compile_standard_scene_configuration_from_selected_source_object"
    elif len(eligible) > 1:
        status = "selection_required"
        selected = None
        source_object = None
        next_action = "operator_or_policy_must_select_one_digest_bound_candidate"
    elif candidates:
        status = "metric_refinement_required"
        selected = None
        source_object = None
        next_action = "run_production_semantic_lifting_or_supply_publisher_metric_labels"
    else:
        status = "abstained_no_candidates"
        selected = None
        source_object = None
        next_action = "expand_prompts_or_acquire_missing_scene_observations"

    coverage = camera_plan.get("coverage") if isinstance(camera_plan.get("coverage"), Mapping) else {}
    output = {
        "schema_version": DISCOVERY_SCHEMA,
        "status": status,
        "source_binding": dict(source_binding),
        "camera_plan_digest": camera_plan_digest,
        "render_manifest_digest": render_manifest_digest,
        "analyzer_receipts": analyzer_receipts,
        "candidates": candidates,
        "eligible_candidate_ids": [row["candidate_id"] for row in eligible],
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "source_object": source_object,
        "next_action": next_action,
        "coverage": {
            "known_scene_bounds_covered": coverage.get("known_scene_bounds_covered") is True,
            "unseen_regions": list(coverage.get("unseen_regions") or []),
            "virtual_camera_recovers_missing_source_observations": False,
        },
        "claim_boundary": {
            "raw_capture_truth_preserved": True,
            "model_candidate_may_self_authorize": False,
            "splat_analyzer_boxes_are_metric_geometry": False,
            "rgb_render_proves_depth_or_gaussian_contributions": False,
            "selected_source_object_is_physical_truth": False,
            "selected_source_object_authorizes_robot_execution": False,
            "discovery_completes_adp_022": False,
        },
    }
    output["discovery_digest"] = _digest_payload(output, field="discovery_digest")
    return output


__all__ = [
    "CALIBRATION_SCHEMA",
    "CAMERA_PLAN_SCHEMA",
    "DISCOVERY_SCHEMA",
    "SceneObjectDiscoveryError",
    "build_full_scene_camera_plan",
    "compile_scene_object_discovery",
    "materialize_scene_object_discovery_renders",
]
