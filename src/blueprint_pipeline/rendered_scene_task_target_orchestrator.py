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

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .common import write_json
from .decision_evidence_contracts import canonical_digest
from .scene_task_target_pipeline import compile_scene_task_targets
from .splat_bbox_target_binding import (
    SplatBBoxTargetBindingError,
    bind_splat_bbox_target,
)


PROPOSAL_SET_SCHEMA = "rendered_scene_task_proposal_set.v1"
RESULT_SCHEMA = "rendered_scene_task_target_orchestration.v1"
ANALYZER_REQUEST_SCHEMA = "rendered_scene_task_analyzer_request.v1"
ANALYZER_RUN_SCHEMA = "rendered_scene_task_analyzer_run.v1"
PIPELINE_REQUEST_SCHEMA = "rendered_scene_task_target_pipeline_request.v1"
MAX_ANALYZER_OUTPUT_BYTES = 2 * 1024 * 1024

RenderedSceneAnalyzerBackend = Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]


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
        for key in ("analyzer_request_digest", "analyzer_run_digest"):
            if key in analyzer and not _digest(analyzer.get(key)):
                errors.append(f"rendered_target_analyzer_{key}_invalid")
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


def _resolved_analysis_splat(path: str | Path) -> tuple[Path, str]:
    splat = Path(path)
    if splat.is_symlink():
        raise RenderedSceneTaskTargetOrchestratorError(
            ["rendered_target_analysis_splat_symlink_forbidden"]
        )
    try:
        resolved = splat.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise RenderedSceneTaskTargetOrchestratorError(
            ["rendered_target_analysis_splat_missing"]
        ) from exc
    if not resolved.is_file():
        raise RenderedSceneTaskTargetOrchestratorError(["rendered_target_analysis_splat_missing"])
    return resolved, _sha256(resolved)


def build_rendered_scene_task_analyzer_request(
    *,
    analysis_splat_path: str | Path,
    scene_id: str,
    source_scene_digest: str,
    rendered_views: Sequence[Mapping[str, Any]],
    source_video_available: bool,
    robot_id: str,
    task_context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the immutable request and separate local-only analyzer inputs."""

    views, _ = _validate_views(rendered_views)
    splat, splat_digest = _resolved_analysis_splat(analysis_splat_path)
    request_errors: list[str] = []
    if not str(scene_id).strip():
        request_errors.append("rendered_target_analyzer_scene_id_missing")
    if not _digest(source_scene_digest):
        request_errors.append("rendered_target_analyzer_source_scene_digest_invalid")
    if not str(robot_id).strip():
        request_errors.append("rendered_target_analyzer_robot_id_missing")
    for index, view in enumerate(views):
        image_size = view["image_size"]
        width = image_size.get("width")
        height = image_size.get("height")
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            or isinstance(height, bool)
            or not isinstance(height, int)
            or height <= 0
        ):
            request_errors.append(f"rendered_target_analyzer_view_{index}_image_size_invalid")
    if request_errors:
        raise RenderedSceneTaskTargetOrchestratorError(request_errors)
    request = {
        "schema_version": ANALYZER_REQUEST_SCHEMA,
        "scene_id": str(scene_id),
        "source_scene_digest": str(source_scene_digest),
        "analysis_splat_digest": splat_digest,
        "rendered_views": [
            {
                "view_id": view["view_id"],
                "rgb_digest": view["rgb_digest"],
                "observation_source": view["observation_source"],
                "camera_spec_digest": view["camera_spec_digest"],
                "image_size": view["image_size"],
            }
            for view in views
        ],
        "source_video_available": bool(source_video_available),
        "robot_id": str(robot_id),
        "task_context": _clone(dict(task_context or {})),
        "candidate_may_self_authorize": False,
    }
    request["analyzer_request_digest"] = canonical_digest(
        request, digest_field="analyzer_request_digest"
    )
    runtime_inputs = {
        "analysis_splat_path": str(splat),
        "rendered_views": [
            {
                "view_id": view["view_id"],
                "rgb_path": view["rgb_path"],
                "camera": view["camera"],
                "image_size": view["image_size"],
            }
            for view in views
        ],
    }
    return request, runtime_inputs


class CommandRenderedSceneAnalyzer:
    """Invoke an owner-configured analyzer command through JSON stdin/stdout.

    The command is executed without a shell. It receives an immutable analyzer
    request plus local runtime paths on stdin and must return one JSON object on
    stdout. The result remains a proposal; it never authorizes a task target.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout_seconds: int = 300,
        maximum_output_bytes: int = MAX_ANALYZER_OUTPUT_BYTES,
    ) -> None:
        self.command = tuple(str(item) for item in command if str(item))
        self.timeout_seconds = int(timeout_seconds)
        self.maximum_output_bytes = int(maximum_output_bytes)
        if not self.command:
            raise RenderedSceneTaskTargetOrchestratorError(
                ["rendered_target_analyzer_command_missing"]
            )
        if self.timeout_seconds <= 0 or self.maximum_output_bytes <= 0:
            raise RenderedSceneTaskTargetOrchestratorError(
                ["rendered_target_analyzer_command_limits_invalid"]
            )

    def __call__(
        self,
        request: Mapping[str, Any],
        runtime_inputs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        payload = json.dumps(
            {
                "analyzer_request": dict(request),
                "runtime_inputs": dict(runtime_inputs),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        try:
            completed = subprocess.run(
                list(self.command),
                input=payload,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "abstained",
                "analyzer_request_digest": request["analyzer_request_digest"],
                "candidate_may_self_authorize": False,
                "proposals": [],
                "blockers": ["rendered_target_analyzer_command_timeout"],
            }
        except OSError:
            return {
                "status": "abstained",
                "analyzer_request_digest": request["analyzer_request_digest"],
                "candidate_may_self_authorize": False,
                "proposals": [],
                "blockers": ["rendered_target_analyzer_command_unavailable"],
            }
        if completed.returncode != 0:
            return {
                "status": "abstained",
                "analyzer_request_digest": request["analyzer_request_digest"],
                "candidate_may_self_authorize": False,
                "proposals": [],
                "blockers": ["rendered_target_analyzer_command_failed"],
            }
        encoded = completed.stdout.encode("utf-8")
        if len(encoded) > self.maximum_output_bytes:
            return {
                "status": "abstained",
                "analyzer_request_digest": request["analyzer_request_digest"],
                "candidate_may_self_authorize": False,
                "proposals": [],
                "blockers": ["rendered_target_analyzer_output_oversized"],
            }
        try:
            value = json.loads(completed.stdout)
        except json.JSONDecodeError:
            return {
                "status": "abstained",
                "analyzer_request_digest": request["analyzer_request_digest"],
                "candidate_may_self_authorize": False,
                "proposals": [],
                "blockers": ["rendered_target_analyzer_output_invalid_json"],
            }
        return (
            dict(value)
            if isinstance(value, Mapping)
            else {
                "status": "abstained",
                "analyzer_request_digest": request["analyzer_request_digest"],
                "candidate_may_self_authorize": False,
                "proposals": [],
                "blockers": ["rendered_target_analyzer_output_invalid"],
            }
        )


def _validate_analyzer_run(
    value: Mapping[str, Any],
    *,
    analyzer_request_digest: str,
) -> dict[str, Any]:
    run = _clone(dict(value))
    errors: list[str] = []
    if run.get("analyzer_request_digest") != analyzer_request_digest:
        errors.append("rendered_target_analyzer_request_digest_mismatch")
    if run.get("candidate_may_self_authorize") is not False:
        errors.append("rendered_target_analyzer_run_self_authorization_forbidden")
    status = run.get("status")
    if status not in {"completed", "abstained"}:
        errors.append("rendered_target_analyzer_run_status_invalid")
    proposals = run.get("proposals")
    if not isinstance(proposals, list):
        errors.append("rendered_target_analyzer_run_proposals_invalid")
        proposals = []
    blockers = run.get("blockers")
    if not isinstance(blockers, list) or any(
        not isinstance(item, str) or not item for item in blockers
    ):
        errors.append("rendered_target_analyzer_run_blockers_invalid")
        blockers = []
    if status == "completed" and blockers:
        errors.append("rendered_target_analyzer_completed_with_blockers")
    if status == "abstained" and (proposals or not blockers):
        errors.append("rendered_target_analyzer_abstention_invalid")
    if errors:
        raise RenderedSceneTaskTargetOrchestratorError(errors)
    normalized = {
        "schema_version": ANALYZER_RUN_SCHEMA,
        "status": status,
        "analyzer_request_digest": analyzer_request_digest,
        "candidate_may_self_authorize": False,
        "proposals": proposals,
        "blockers": sorted(set(blockers)),
    }
    normalized["analyzer_run_digest"] = canonical_digest(
        normalized, digest_field="analyzer_run_digest"
    )
    return normalized


def compile_rendered_scene_task_target_with_analyzer(
    *,
    analyzer_backend: RenderedSceneAnalyzerBackend,
    analyzer_id: str,
    analyzer_implementation_version: str,
    analyzer_contract_digest: str,
    analysis_splat_path: str | Path,
    scene_id: str,
    source_scene_digest: str,
    rendered_views: Sequence[Mapping[str, Any]],
    source_video_available: bool,
    robot_id: str,
    metric_scale_status: str,
    collision_support: Mapping[str, Any],
    reach_support: Mapping[str, Any],
    task_context: Mapping[str, Any] | None = None,
    minimum_visual_confidence: float = 0.8,
    minimum_opacity: float = 0.18,
    front_depth_fraction: float = 0.25,
    minimum_projected_splats: int = 32,
) -> dict[str, Any]:
    """Invoke a replaceable analyzer and compile its proposals fail closed."""

    if not str(analyzer_id).strip() or not str(analyzer_implementation_version).strip():
        raise RenderedSceneTaskTargetOrchestratorError(
            ["rendered_target_analyzer_identity_missing"]
        )
    if not _digest(analyzer_contract_digest):
        raise RenderedSceneTaskTargetOrchestratorError(
            ["rendered_target_analyzer_contract_digest_invalid"]
        )
    request, runtime_inputs = build_rendered_scene_task_analyzer_request(
        analysis_splat_path=analysis_splat_path,
        scene_id=scene_id,
        source_scene_digest=source_scene_digest,
        rendered_views=rendered_views,
        source_video_available=source_video_available,
        robot_id=robot_id,
        task_context=task_context,
    )
    try:
        raw_run = analyzer_backend(request, runtime_inputs)
    except Exception:  # noqa: BLE001 - backend failures must become bounded abstentions
        raw_run = {
            "status": "abstained",
            "analyzer_request_digest": request["analyzer_request_digest"],
            "candidate_may_self_authorize": False,
            "proposals": [],
            "blockers": ["rendered_target_analyzer_backend_failed"],
        }
    if not isinstance(raw_run, Mapping):
        raw_run = {
            "status": "abstained",
            "analyzer_request_digest": request["analyzer_request_digest"],
            "candidate_may_self_authorize": False,
            "proposals": [],
            "blockers": ["rendered_target_analyzer_output_invalid"],
        }
    analyzer_run = _validate_analyzer_run(
        raw_run, analyzer_request_digest=request["analyzer_request_digest"]
    )
    proposal_set = {
        "schema_version": PROPOSAL_SET_SCHEMA,
        "analyzer_provenance": {
            "analyzer_id": str(analyzer_id),
            "implementation_version": str(analyzer_implementation_version),
            "analyzer_contract_digest": analyzer_contract_digest,
            "analyzer_request_digest": request["analyzer_request_digest"],
            "analyzer_run_digest": analyzer_run["analyzer_run_digest"],
            "proposal_generation_is_dynamic": True,
            "candidate_may_self_authorize": False,
        },
        "candidate_may_self_authorize": False,
        "proposals": analyzer_run["proposals"],
    }
    result = compile_rendered_scene_task_target(
        analysis_splat_path=analysis_splat_path,
        scene_id=scene_id,
        source_scene_digest=source_scene_digest,
        rendered_views=rendered_views,
        proposal_set=proposal_set,
        source_video_available=source_video_available,
        robot_id=robot_id,
        metric_scale_status=metric_scale_status,
        collision_support=collision_support,
        reach_support=reach_support,
        minimum_visual_confidence=minimum_visual_confidence,
        minimum_opacity=minimum_opacity,
        front_depth_fraction=front_depth_fraction,
        minimum_projected_splats=minimum_projected_splats,
    )
    result["analyzer_request_digest"] = request["analyzer_request_digest"]
    result["analyzer_run"] = analyzer_run
    result["orchestration_digest"] = canonical_digest(result, digest_field="orchestration_digest")
    return result


def run_rendered_scene_task_target_pipeline(value: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the configured analyzer-to-qualified-target pipeline request."""

    request = _clone(dict(value))
    supplied_digest = request.pop("pipeline_request_digest", None)
    errors: list[str] = []
    if request.get("schema_version") != PIPELINE_REQUEST_SCHEMA:
        errors.append("rendered_target_pipeline_request_schema_invalid")
    analyzer = request.get("analyzer")
    if not isinstance(analyzer, Mapping):
        errors.append("rendered_target_pipeline_analyzer_missing")
        analyzer = {}
    command = analyzer.get("command")
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
    ):
        errors.append("rendered_target_pipeline_analyzer_command_invalid")
        command = []
    if analyzer.get("command_execution_authorized") is not True:
        errors.append("rendered_target_pipeline_analyzer_command_not_authorized")
    if analyzer.get("candidate_may_self_authorize") is not False:
        errors.append("rendered_target_pipeline_analyzer_self_authorization_forbidden")
    for key in ("collision_support", "reach_support"):
        if not isinstance(request.get(key), Mapping):
            errors.append(f"rendered_target_pipeline_{key}_invalid")
    if not isinstance(request.get("source_video_available"), bool):
        errors.append("rendered_target_pipeline_source_video_status_invalid")
    if not isinstance(request.get("rendered_views"), list):
        errors.append("rendered_target_pipeline_rendered_views_invalid")
    timeout_seconds = analyzer.get("timeout_seconds", 300)
    maximum_output_bytes = analyzer.get("maximum_output_bytes", MAX_ANALYZER_OUTPUT_BYTES)
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, int)
        or timeout_seconds <= 0
    ):
        errors.append("rendered_target_pipeline_analyzer_timeout_invalid")
    if (
        isinstance(maximum_output_bytes, bool)
        or not isinstance(maximum_output_bytes, int)
        or maximum_output_bytes <= 0
    ):
        errors.append("rendered_target_pipeline_analyzer_output_limit_invalid")
    numeric_defaults = {
        "minimum_visual_confidence": 0.8,
        "minimum_opacity": 0.18,
        "front_depth_fraction": 0.25,
    }
    numeric_values: dict[str, float] = {}
    for key, default in numeric_defaults.items():
        value = request.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"rendered_target_pipeline_{key}_invalid")
            numeric_values[key] = default
        else:
            numeric_values[key] = float(value)
    if not 0.0 <= numeric_values["minimum_visual_confidence"] <= 1.0:
        errors.append("rendered_target_pipeline_minimum_visual_confidence_invalid")
    if numeric_values["minimum_opacity"] < 0.0:
        errors.append("rendered_target_pipeline_minimum_opacity_invalid")
    if not 0.0 < numeric_values["front_depth_fraction"] <= 1.0:
        errors.append("rendered_target_pipeline_front_depth_fraction_invalid")
    minimum_projected_splats = request.get("minimum_projected_splats", 32)
    if (
        isinstance(minimum_projected_splats, bool)
        or not isinstance(minimum_projected_splats, int)
        or minimum_projected_splats <= 0
    ):
        errors.append("rendered_target_pipeline_minimum_projected_splats_invalid")
        minimum_projected_splats = 32
    expected_digest = canonical_digest(request, digest_field="pipeline_request_digest")
    if supplied_digest is not None and supplied_digest != expected_digest:
        errors.append("rendered_target_pipeline_request_digest_mismatch")
    if errors:
        raise RenderedSceneTaskTargetOrchestratorError(errors)
    assert isinstance(analyzer, Mapping)
    backend = CommandRenderedSceneAnalyzer(
        command,
        timeout_seconds=timeout_seconds,
        maximum_output_bytes=maximum_output_bytes,
    )
    result = compile_rendered_scene_task_target_with_analyzer(
        analyzer_backend=backend,
        analyzer_id=str(analyzer.get("analyzer_id") or ""),
        analyzer_implementation_version=str(analyzer.get("implementation_version") or ""),
        analyzer_contract_digest=str(analyzer.get("analyzer_contract_digest") or ""),
        analysis_splat_path=str(request.get("analysis_splat_path") or ""),
        scene_id=str(request.get("scene_id") or ""),
        source_scene_digest=str(request.get("source_scene_digest") or ""),
        rendered_views=request.get("rendered_views") or [],
        source_video_available=bool(request.get("source_video_available")),
        robot_id=str(request.get("robot_id") or ""),
        metric_scale_status=str(request.get("metric_scale_status") or ""),
        collision_support=request["collision_support"],
        reach_support=request["reach_support"],
        task_context=request.get("task_context")
        if isinstance(request.get("task_context"), Mapping)
        else {},
        minimum_visual_confidence=numeric_values["minimum_visual_confidence"],
        minimum_opacity=numeric_values["minimum_opacity"],
        front_depth_fraction=numeric_values["front_depth_fraction"],
        minimum_projected_splats=minimum_projected_splats,
    )
    result["pipeline_request_digest"] = expected_digest
    result["orchestration_digest"] = canonical_digest(result, digest_field="orchestration_digest")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze rendered scene views and compile one bounded 3D task target"
    )
    parser.add_argument("--request", required=True, help="Pipeline request JSON")
    parser.add_argument("--output", required=True, help="Target orchestration JSON")
    args = parser.parse_args(argv)
    request_path = Path(args.request).resolve(strict=True)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RenderedSceneTaskTargetOrchestratorError(["rendered_target_pipeline_request_invalid"])
    result = run_rendered_scene_task_target_pipeline(payload)
    write_json(Path(args.output), result)
    print(f"[rendered-scene-task-target] status={result['status']}")
    print(f"[rendered-scene-task-target] output={Path(args.output).resolve()}")
    return 0


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
    splat, analysis_splat_digest = _resolved_analysis_splat(analysis_splat_path)
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
    "ANALYZER_REQUEST_SCHEMA",
    "ANALYZER_RUN_SCHEMA",
    "CommandRenderedSceneAnalyzer",
    "PIPELINE_REQUEST_SCHEMA",
    "PROPOSAL_SET_SCHEMA",
    "RESULT_SCHEMA",
    "RenderedSceneTaskTargetOrchestratorError",
    "build_rendered_scene_task_analyzer_request",
    "compile_rendered_scene_task_target",
    "compile_rendered_scene_task_target_with_analyzer",
    "run_rendered_scene_task_target_pipeline",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
