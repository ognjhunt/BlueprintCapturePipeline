"""Repository-owned handlers for typed scene-configuration capabilities."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .task_evaluation_scene_configuration_disclosure import (
    renders_on_provider,
    stage_requests_upload,
)
from .source_collider_subtree_removal import remove_source_collider_subtree
from .task_evaluation_scene_configuration_adapters import (
    ADMITTED_STAGE_ADAPTER_IDENTITIES,
    SceneConfigurationAdapterIdentity,
    StageAdapter,
    TaskEvaluationSceneConfigurationAdapterError,
)
from .task_evaluation_scene_configuration_appearance_review import (
    AppearanceReviewContractError,
    PAUSED_UNGRADED_MODE,
    PAUSED_UNGRADED_WARNING,
    REQUIRED_MODE,
    appearance_review_mode,
    paused_review_receipt_valid,
)
from .task_evaluation_scene_configuration_orchestrator import (
    STAGE_RESULT_SCHEMA_VERSION,
)
from .task_evaluation_scene_configuration_render_handoff import (
    ARTIFACT_ROLE as PROVIDER_RENDER_REFERENCE_ROLE,
    TaskEvaluationSceneConfigurationRenderHandoffError,
    validate_provider_render_handoff,
)
from .task_evaluation_scene_configuration_stage_configuration import (
    native_import_checks_valid,
    static_qualification_checks_valid,
)
from .task_evaluation_scene_configuration_static_qualification import (
    SCHEMA_VERSION as STATIC_QUALIFICATION_SCHEMA_VERSION,
    TaskEvaluationSceneConfigurationStaticQualificationError,
    qualify_scene_configuration_rigid_asset_static,
)


_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ARTIFIXER_REVIEW_EXECUTION_SCHEMA_VERSION = (
    "task_evaluation_artifixer_ai_visual_review_execution.v1"
)
_PREDICATE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


def _require_named_predicates(
    base_code: str,
    predicates: Sequence[tuple[str, Callable[[], bool]]],
) -> None:
    """Raise the existing refusal prefix plus the first failed safe term."""

    for name, predicate in predicates:
        if _PREDICATE_NAME.fullmatch(name) is None:
            raise ValueError("scene_configuration_predicate_name_invalid")
        if not predicate():
            raise TaskEvaluationSceneConfigurationAdapterError(
                f"{base_code}:{name}"
            )


def _sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return "sha256:" + digest.hexdigest(), size


def _materialized_reference(
    envelope: Mapping[str, Any], *, contract_path: str
) -> tuple[Mapping[str, Any], Path]:
    matches = [
        row
        for row in envelope.get("materialized_references") or []
        if isinstance(row, Mapping) and row.get("contract_path") == contract_path
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "scene_configuration_materialized_reference_missing:"
            f"{contract_path}"
        )
    row = matches[0]
    path = Path(str(row.get("materialized_path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path) != (row.get("digest"), row.get("size_bytes"))
        or row.get("full_byte_service_account_readback_passed") is not True
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "scene_configuration_materialized_reference_invalid:"
            f"{contract_path}"
        )
    return row, path


def _dependency_artifact(
    dependency_results: tuple[Mapping[str, Any], ...], *, role: str
) -> tuple[Mapping[str, Any], Path]:
    matches = [
        artifact
        for result in dependency_results
        for artifact in result.get("output_artifacts") or []
        if isinstance(artifact, Mapping) and artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_dependency_artifact_missing:{role}"
        )
    artifact = matches[0]
    path = Path(str(artifact.get("path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path)
        != (artifact.get("digest"), artifact.get("size_bytes"))
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_dependency_artifact_invalid:{role}"
        )
    return artifact, path


def _provider_runtime_artifact(
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...], *, role: str
) -> tuple[Mapping[str, Any], Path]:
    matches = [
        artifact
        for artifact in provider_runtime_artifacts
        if isinstance(artifact, Mapping) and artifact.get("role") == role
    ]
    if len(matches) != 1:
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_provider_runtime_artifact_missing:{role}"
        )
    artifact = matches[0]
    path = Path(str(artifact.get("path") or "")).resolve()
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_and_size(path)
        != (artifact.get("digest"), artifact.get("size_bytes"))
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            f"scene_configuration_provider_runtime_artifact_invalid:{role}"
        )
    return artifact, path


def _copy_artifact(source: Path, destination: Path) -> dict[str, Any]:
    shutil.copyfile(source, destination)
    if _sha256_and_size(destination) != _sha256_and_size(source):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "scene_configuration_provider_runtime_artifact_copy_mismatch"
        )
    return {
        "path": str(destination),
        "digest": _sha256_and_size(destination)[0],
        "size_bytes": _sha256_and_size(destination)[1],
    }


def _stage_result(
    *,
    stage: Mapping[str, Any],
    configuration_path: Path,
    output_artifacts: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": STAGE_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "stage_id": stage["stage_id"],
        "capability": stage["capability"],
        "execution_class": stage["execution_class"],
        "configuration_digest": _sha256_and_size(configuration_path)[0],
        "retry_cap": 0,
        "raw_secret_values_recorded": False,
        "canonical_allocator": None,
        "provider_mutations_performed": 0,
        "paid_execution_requested": False,
        "executed_inside_parent_configuration_run": True,
        "output_artifacts": output_artifacts,
        "stage_result_digest": "",
    }
    result["stage_result_digest"] = canonical_digest(
        result, digest_field="stage_result_digest"
    )
    return result


def _extract_source_candidate_subtree(
    *, source_stage_path: Path, prim_path: str, output_path: Path
) -> dict[str, Any]:
    """Retain the exact SAGE target as candidate geometry before excision."""

    try:
        from pxr import Sdf, Usd, UsdGeom
    except ImportError as exc:  # pragma: no cover - production image owns USD
        raise TaskEvaluationSceneConfigurationAdapterError(
            "sage_source_candidate_usd_runtime_missing"
        ) from exc
    stage = Usd.Stage.Open(str(source_stage_path))
    prim = stage.GetPrimAtPath(prim_path) if stage is not None else None
    if stage is None or prim is None or not prim.IsValid():
        raise TaskEvaluationSceneConfigurationAdapterError(
            "sage_source_candidate_prim_missing"
        )
    flattened = stage.Flatten()
    layer = Sdf.Layer.CreateNew(str(output_path))
    if layer is None:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "sage_source_candidate_extraction_failed"
        )
    Sdf.CreatePrimInLayer(layer, Sdf.Path("/Root"))
    if not Sdf.CopySpec(
        flattened,
        Sdf.Path(prim_path),
        layer,
        Sdf.Path("/Root/SourceObjectCandidate"),
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "sage_source_candidate_extraction_failed"
        )
    layer.defaultPrim = "Root"
    layer.Save()
    candidate = Usd.Stage.Open(str(output_path))
    if candidate is None or not candidate.GetPrimAtPath(
        "/Root/SourceObjectCandidate"
    ).IsValid():
        raise TaskEvaluationSceneConfigurationAdapterError(
            "sage_source_candidate_extraction_failed"
        )
    UsdGeom.SetStageMetersPerUnit(
        candidate, UsdGeom.GetStageMetersPerUnit(stage)
    )
    UsdGeom.SetStageUpAxis(candidate, UsdGeom.GetStageUpAxis(stage))
    candidate.GetRootLayer().Save()
    digest, size = _sha256_and_size(output_path)
    return {
        "path": str(output_path),
        "digest": digest,
        "size_bytes": size,
        "source_prim_path": prim_path,
        "candidate_geometry_only": True,
        "observed_source_truth": False,
        "movable_physics_authority": False,
    }


def execute_artifixer3d_observed_object_removal(
    *,
    envelope: Mapping[str, Any],
    stage: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_path: Path,
    dependency_results: tuple[Mapping[str, Any], ...],
    output_root: Path,
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...] = (),
) -> Mapping[str, Any]:
    """Seal production-executed ArtiFixer outputs without exposing raw splats."""

    if dependency_results:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "artifixer3d_object_removal_dependency_invalid"
        )
    _appearance_record, appearance = _provider_runtime_artifact(
        provider_runtime_artifacts,
        role="configured_appearance_without_source_object",
    )
    _receipt_record, receipt_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="appearance_removal_receipt"
    )
    review_record, review_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="appearance_visual_review_receipt"
    )
    _thumbnail_record, thumbnail_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="configured_task_thumbnail"
    )
    render_reference_record, render_reference_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role=PROVIDER_RENDER_REFERENCE_ROLE
    )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        review = json.loads(review_path.read_text(encoding="utf-8"))
        render_reference, _render_reference_frames = (
            validate_provider_render_handoff(render_reference_path)
        )
    except (
        OSError,
        json.JSONDecodeError,
        TaskEvaluationSceneConfigurationRenderHandoffError,
    ) as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "artifixer3d_object_removal_receipt_invalid"
        ) from exc
    source_object = configuration.get("source_object")
    input_render = envelope.get("render_inputs_result") or {}
    request = envelope.get("request")
    try:
        review_mode = (
            appearance_review_mode(request, allow_historical_paused=True)
            if isinstance(request, Mapping)
            else REQUIRED_MODE
        )
    except AppearanceReviewContractError as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(str(exc)) from exc
    minimum_views = int(configuration.get("required_views", {}).get("minimum", 1))
    thumbnail_digest = _sha256_and_size(thumbnail_path)[0]
    source_publisher_id = (
        source_object.get("publisher_instance_id")
        if isinstance(source_object, Mapping)
        else None
    )
    reviewer = review.get("reviewer")
    paused_review_valid = (
        isinstance(source_object, Mapping)
        and paused_review_receipt_valid(
            review,
            publisher_instance_id=str(source_object.get("publisher_instance_id") or ""),
            minimum_frame_count=minimum_views,
            thumbnail_digest=thumbnail_digest,
        )
    )
    predicates: list[tuple[str, Callable[[], bool]]] = [
        (
            "configuration_schema",
            lambda: configuration.get("schema_version")
            == "observed_appearance_object_removal_configuration.v1",
        ),
        ("source_object", lambda: isinstance(source_object, Mapping)),
        (
            "production_render_required",
            lambda: configuration.get("production_render_required") is True,
        ),
        (
            "provider_render_site",
            lambda: stage_requests_upload(configuration)
            is renders_on_provider(input_render.get("disclosure_decision") or {}),
        ),
        (
            "generated_pixels_required",
            lambda: configuration.get("output_requirements", {}).get(
                "generated_pixels_labeled"
            )
            is True,
        ),
        (
            "receipt_schema",
            lambda: receipt.get("schema_version")
            == "task_evaluation_artifixer_object_removal_result.v1",
        ),
    ]
    if review_mode == REQUIRED_MODE:
        predicates.extend(
            [
                (
                    "receipt_status",
                    lambda: receipt.get("status")
                    == "qualified_generated_appearance_edit",
                ),
                (
                    "receipt_semantic_absence",
                    lambda: receipt.get(
                        "semantic_object_free_visual_review_passed"
                    )
                    is True,
                ),
                (
                    "receipt_multiview_consistency",
                    lambda: receipt.get("multiview_consistency_review_passed")
                    is True,
                ),
                (
                    "review_schema",
                    lambda: review.get("schema_version")
                    == "task_evaluation_artifixer_ai_visual_review.v1",
                ),
                ("review_status", lambda: review.get("status") == "accepted"),
                (
                    "review_publisher_instance",
                    lambda: review.get("publisher_instance_id")
                    == source_publisher_id,
                ),
                ("review_decision", lambda: review.get("decision") == "accepted"),
                (
                    "review_semantic_absence",
                    lambda: review.get("semantic_object_absence_review_passed")
                    is True,
                ),
                (
                    "review_multiview_consistency",
                    lambda: review.get("multiview_consistency_review_passed")
                    is True,
                ),
                (
                    "review_frame_count",
                    lambda: review.get("review_frame_count", 0) >= minimum_views,
                ),
                (
                    "review_frames_digest_bound",
                    lambda: review.get("all_review_frames_digest_bound") is True,
                ),
                (
                    "review_ai_completed",
                    lambda: review.get("ai_visual_review_completed") is True,
                ),
                (
                    "review_human_not_completed",
                    lambda: review.get("human_review_completed") is False,
                ),
                (
                    "review_generated_not_physical",
                    lambda: review.get(
                        "generated_output_is_capture_or_physical_evidence"
                    )
                    is False,
                ),
                ("reviewer_record", lambda: isinstance(reviewer, Mapping)),
                (
                    "reviewer_identity",
                    lambda: bool(str(reviewer.get("identity") or "")),
                ),
                (
                    "reviewer_runtime",
                    lambda: bool(str(reviewer.get("runtime") or "")),
                ),
                (
                    "reviewer_model",
                    lambda: bool(str(reviewer.get("model") or "")),
                ),
                (
                    "review_receipt_digest",
                    lambda: review.get("receipt_digest")
                    == canonical_digest(review, digest_field="receipt_digest"),
                ),
                (
                    "thumbnail_exact_review_frame",
                    lambda: review.get("task_thumbnail_is_exact_review_frame")
                    is True,
                ),
                (
                    "thumbnail_selection",
                    lambda: isinstance(
                        review.get("task_thumbnail_selection"), Mapping
                    ),
                ),
                (
                    "thumbnail_frame_digest",
                    lambda: review["task_thumbnail_selection"].get("frame_sha256")
                    == thumbnail_digest,
                ),
            ]
        )
    else:
        predicates.extend(
            [
                (
                    "receipt_status",
                    lambda: receipt.get("status")
                    == "completed_ungraded_generated_appearance_edit",
                ),
                (
                    "visual_review_mode",
                    lambda: receipt.get("visual_review_mode")
                    == PAUSED_UNGRADED_MODE,
                ),
                (
                    "receipt_semantic_ungraded",
                    lambda: receipt.get(
                        "semantic_object_free_visual_review_passed"
                    )
                    is False,
                ),
                (
                    "receipt_multiview_ungraded",
                    lambda: receipt.get("multiview_consistency_review_passed")
                    is False,
                ),
                (
                    "review_provider_call_not_performed",
                    lambda: receipt.get("review_provider_call_performed") is False,
                ),
                (
                    "ungraded_publication_acknowledged",
                    lambda: receipt.get("ungraded_publication_acknowledged")
                    is True,
                ),
                (
                    "warning_label",
                    lambda: receipt.get("warning_label")
                    == PAUSED_UNGRADED_WARNING,
                ),
                ("paused_review_receipt", lambda: paused_review_valid),
            ]
        )
    predicates.extend(
        [
            (
                "receipt_publisher_instance",
                lambda: receipt.get("publisher_instance_id")
                == source_publisher_id,
            ),
            (
                "raw_source_bytes_not_external",
                lambda: receipt.get(
                    "raw_interiorgs_bytes_sent_to_external_provider"
                )
                is False,
            ),
            (
                "visual_review_receipt_digest",
                lambda: receipt.get("visual_review_receipt_digest")
                == review.get("receipt_digest"),
            ),
            (
                "visual_review_receipt_sha256",
                lambda: receipt.get("visual_review_receipt_sha256")
                == review_record.get("digest"),
            ),
            (
                "generated_pixels_labeled",
                lambda: receipt.get("generated_pixels_labeled") is True,
            ),
            (
                "receipt_digest",
                lambda: receipt.get("result_digest")
                == canonical_digest(receipt, digest_field="result_digest"),
            ),
            (
                "handoff_control_plane_digest",
                lambda: render_reference.get("control_plane_render_result_digest")
                == (
                    input_render.get("control_plane_result_digest")
                    if renders_on_provider(
                        input_render.get("disclosure_decision") or {}
                    )
                    else input_render.get("result_digest")
                ),
            ),
            (
                "handoff_render_site",
                lambda: render_reference.get("render_completed_on_provider")
                is renders_on_provider(input_render.get("disclosure_decision") or {}),
            ),
        ]
    )
    _require_named_predicates(
        "artifixer3d_object_removal_result_invalid", predicates
    )
    copied_appearance = _copy_artifact(
        appearance, output_root / f"configured_appearance{appearance.suffix}"
    )
    copied_receipt = _copy_artifact(
        receipt_path, output_root / "appearance_removal_receipt.v1.json"
    )
    copied_review = _copy_artifact(
        review_path, output_root / "appearance_visual_review_receipt.v1.json"
    )
    copied_thumbnail = _copy_artifact(
        thumbnail_path, output_root / "configured_task_thumbnail.png"
    )
    return _stage_result(
        stage=stage,
        configuration_path=configuration_path,
        output_artifacts=[
            {
                "role": "configured_appearance_without_source_object",
                **copied_appearance,
            },
            {"role": "appearance_removal_receipt", **copied_receipt},
            {"role": "appearance_visual_review_receipt", **copied_review},
            {"role": "configured_task_thumbnail", **copied_thumbnail},
            dict(render_reference_record),
        ],
    )


def execute_artifixer3d_diagnostic_object_removal(
    *,
    envelope: Mapping[str, Any],
    stage: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_path: Path,
    dependency_results: tuple[Mapping[str, Any], ...],
    output_root: Path,
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...] = (),
) -> Mapping[str, Any]:
    """Carry one explicitly rejected appearance only through a diagnostic chain."""

    roles = {
        str(row.get("role") or "")
        for row in provider_runtime_artifacts
        if isinstance(row, Mapping)
    }
    if "appearance_rejection_receipt" not in roles:
        return execute_artifixer3d_observed_object_removal(
            envelope=envelope,
            stage=stage,
            configuration=configuration,
            configuration_path=configuration_path,
            dependency_results=dependency_results,
            output_root=output_root,
            provider_runtime_artifacts=provider_runtime_artifacts,
        )
    expected_roles = {
        "diagnostic_rejected_appearance_candidate",
        "appearance_rejection_receipt",
        "appearance_visual_review_execution",
        PROVIDER_RENDER_REFERENCE_ROLE,
    }
    if dependency_results or roles != expected_roles:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "artifixer3d_diagnostic_rejection_artifacts_invalid"
        )
    candidate_record, candidate = _provider_runtime_artifact(
        provider_runtime_artifacts,
        role="diagnostic_rejected_appearance_candidate",
    )
    _rejection_record, rejection_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="appearance_rejection_receipt"
    )
    execution_record, execution_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="appearance_visual_review_execution"
    )
    render_reference_record, render_reference_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role=PROVIDER_RENDER_REFERENCE_ROLE
    )
    try:
        rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
        execution = json.loads(execution_path.read_text(encoding="utf-8"))
        render_reference, _frames = validate_provider_render_handoff(
            render_reference_path
        )
    except (
        OSError,
        json.JSONDecodeError,
        TaskEvaluationSceneConfigurationRenderHandoffError,
    ) as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "artifixer3d_diagnostic_rejection_result_invalid"
        ) from exc
    frame_rows = execution.get("frames")
    receipt_rows = rejection.get("frame_decisions")
    minimum_views = int((configuration.get("required_views") or {}).get("minimum") or 0)
    accepted = [
        row
        for row in frame_rows or []
        if isinstance(row, Mapping) and row.get("decision") == "accepted"
    ]
    rejected = [
        row
        for row in frame_rows or []
        if isinstance(row, Mapping) and row.get("decision") == "rejected"
    ]
    receipt_identities = {
        (
            str(row.get("camera_id") or ""),
            str(row.get("frame_sha256") or ""),
            str(row.get("decision") or ""),
            str(row.get("rationale") or ""),
        )
        for row in receipt_rows or []
        if isinstance(row, Mapping)
    }
    execution_identities = {
        (
            str(row.get("camera_id") or ""),
            str(row.get("frame_sha256") or ""),
            str(row.get("decision") or ""),
            str(row.get("rationale") or ""),
        )
        for row in frame_rows or []
        if isinstance(row, Mapping)
    }
    input_render = envelope.get("render_inputs_result") or {}
    source_object = configuration.get("source_object")
    source_publisher_id = (
        source_object.get("publisher_instance_id")
        if isinstance(source_object, Mapping)
        else None
    )
    predicates: list[tuple[str, Callable[[], bool]]] = [
        (
            "configuration_schema",
            lambda: configuration.get("schema_version")
            == "observed_appearance_object_removal_configuration.v1",
        ),
        ("source_object", lambda: isinstance(source_object, Mapping)),
        (
            "production_render_required",
            lambda: configuration.get("production_render_required") is True,
        ),
        (
            "generated_pixels_required",
            lambda: configuration.get("output_requirements", {}).get(
                "generated_pixels_labeled"
            )
            is True,
        ),
        (
            "provider_render_site",
            lambda: stage_requests_upload(configuration)
            is renders_on_provider(input_render.get("disclosure_decision") or {}),
        ),
        (
            "rejection_schema",
            lambda: rejection.get("schema_version")
            == "task_evaluation_artifixer_object_removal_result.v1",
        ),
        (
            "rejection_status",
            lambda: rejection.get("status")
            == "diagnostic_generated_appearance_edit_visual_review_rejected",
        ),
        (
            "rejection_publisher_instance",
            lambda: rejection.get("publisher_instance_id") == source_publisher_id,
        ),
        (
            "candidate_digest",
            lambda: rejection.get(
                "diagnostic_rejected_appearance_candidate_sha256"
            )
            == candidate_record.get("digest"),
        ),
        (
            "review_execution_digest",
            lambda: rejection.get("visual_review_execution_digest")
            == execution.get("execution_digest"),
        ),
        (
            "review_execution_sha256",
            lambda: rejection.get("visual_review_execution_sha256")
            == execution_record.get("digest"),
        ),
        (
            "review_frame_count",
            lambda: rejection.get("review_frame_count") == minimum_views,
        ),
        (
            "accepted_frame_count",
            lambda: rejection.get("accepted_review_frame_count")
            == minimum_views - 1,
        ),
        (
            "rejected_frame_count",
            lambda: rejection.get("rejected_review_frame_count") == 1,
        ),
        ("accepted_frames", lambda: len(accepted) == minimum_views - 1),
        ("rejected_frames", lambda: len(rejected) == 1),
        (
            "execution_frame_count",
            lambda: len(execution_identities) == minimum_views,
        ),
        (
            "frame_decisions",
            lambda: receipt_identities == execution_identities,
        ),
        (
            "execution_schema",
            lambda: execution.get("schema_version")
            == _ARTIFIXER_REVIEW_EXECUTION_SCHEMA_VERSION,
        ),
        ("execution_status", lambda: execution.get("status") == "completed"),
        ("execution_decision", lambda: execution.get("decision") == "rejected"),
        (
            "execution_publisher_instance",
            lambda: execution.get("publisher_instance_id") == source_publisher_id,
        ),
        (
            "execution_review_frame_count",
            lambda: execution.get("review_frame_count") == minimum_views,
        ),
        (
            "execution_provider_called",
            lambda: execution.get("provider_called") is True,
        ),
        ("execution_provider", lambda: execution.get("provider") == "openai"),
        (
            "execution_response_store",
            lambda: execution.get("response_store") is False,
        ),
        (
            "execution_tracing_disabled",
            lambda: execution.get("tracing_disabled") is True,
        ),
        (
            "execution_secrets_redacted",
            lambda: execution.get("raw_secret_values_recorded") is False,
        ),
        (
            "execution_digest",
            lambda: execution.get("execution_digest")
            == canonical_digest(execution, digest_field="execution_digest"),
        ),
        (
            "accepted_frame_semantics",
            lambda: all(
                row.get("orientation_is_upright") is True
                and row.get("source_object_absent") is True
                and row.get("repair_is_locally_plausible") is True
                and row.get("preserves_non_target_content") is True
                for row in accepted
            ),
        ),
        (
            "raw_source_bytes_not_external",
            lambda: rejection.get(
                "raw_interiorgs_bytes_sent_to_external_provider"
            )
            is False,
        ),
        (
            "generated_pixels_labeled",
            lambda: rejection.get("generated_pixels_labeled") is True,
        ),
        ("diagnostic_only", lambda: rejection.get("diagnostic_only") is True),
        (
            "qualification_ineligible",
            lambda: rejection.get("qualification_eligible") is False,
        ),
        (
            "configured_revision_not_publishable",
            lambda: rejection.get("configured_revision_publication_permitted")
            is False,
        ),
        (
            "offering_not_publishable",
            lambda: rejection.get("offering_publication_permitted") is False,
        ),
        (
            "terminal_completion_not_permitted",
            lambda: rejection.get("terminal_e2e_completion_permitted") is False,
        ),
        (
            "semantic_absence_not_qualified",
            lambda: rejection.get("semantic_object_free_visual_review_passed")
            is False,
        ),
        (
            "multiview_not_qualified",
            lambda: rejection.get("multiview_consistency_review_passed") is False,
        ),
        (
            "rejection_digest",
            lambda: rejection.get("result_digest")
            == canonical_digest(rejection, digest_field="result_digest"),
        ),
        (
            "source_checkpoint_digest",
            lambda: _DIGEST.fullmatch(
                str(rejection.get("source_diagnostic_checkpoint_digest") or "")
            )
            is not None,
        ),
        (
            "post_training_binding_digest",
            lambda: _DIGEST.fullmatch(
                str(rejection.get("post_training_binding_digest") or "")
            )
            is not None,
        ),
        (
            "handoff_control_plane_digest",
            lambda: render_reference.get("control_plane_render_result_digest")
            == (
                input_render.get("control_plane_result_digest")
                if renders_on_provider(
                    input_render.get("disclosure_decision") or {}
                )
                else input_render.get("result_digest")
            ),
        ),
        (
            "handoff_render_site",
            lambda: render_reference.get("render_completed_on_provider")
            is renders_on_provider(input_render.get("disclosure_decision") or {}),
        ),
    ]
    _require_named_predicates(
        "artifixer3d_diagnostic_rejection_result_invalid", predicates
    )
    copied_candidate = _copy_artifact(
        candidate, output_root / f"diagnostic_rejected_appearance_candidate{candidate.suffix}"
    )
    copied_rejection = _copy_artifact(
        rejection_path, output_root / "appearance_rejection_receipt.v1.json"
    )
    copied_execution = _copy_artifact(
        execution_path, output_root / "appearance_visual_review_execution.v1.json"
    )
    result = _stage_result(
        stage=stage,
        configuration_path=configuration_path,
        output_artifacts=[
            {
                "role": "diagnostic_rejected_appearance_candidate",
                **copied_candidate,
            },
            {"role": "appearance_rejection_receipt", **copied_rejection},
            {"role": "appearance_visual_review_execution", **copied_execution},
            dict(render_reference_record),
        ],
    )
    result.update(
        {
            "appearance_visual_review_rejected": True,
            "diagnostic_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
        }
    )
    result["stage_result_digest"] = canonical_digest(
        result, digest_field="stage_result_digest"
    )
    return result


def execute_content_agents_rigid_replacement(
    *,
    envelope: Mapping[str, Any],
    stage: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_path: Path,
    dependency_results: tuple[Mapping[str, Any], ...],
    output_root: Path,
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...] = (),
) -> Mapping[str, Any]:
    """Seal a Content Agents rigid candidate for independent qualification."""

    if len(dependency_results) != 2:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "content_agents_replacement_dependency_invalid"
        )
    source_candidate_record, _source_candidate = _dependency_artifact(
        dependency_results, role="source_object_candidate_mesh"
    )
    asset_record, asset = _provider_runtime_artifact(
        provider_runtime_artifacts, role="replacement_asset"
    )
    _receipt_record, receipt_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="replacement_authoring_receipt"
    )
    _graph_record, graph_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="replacement_graph_spec"
    )
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "content_agents_replacement_result_invalid"
        ) from exc
    identity = configuration.get("replacement_identity")
    required_output = configuration.get("required_output")
    expected_physics_bounds = (
        {
            "mass_kg": required_output.get("mass_kg_bounds"),
            "static_friction": required_output.get("static_friction_bounds"),
            "dynamic_friction": required_output.get("dynamic_friction_bounds"),
            "restitution": required_output.get("restitution_bounds"),
        }
        if isinstance(required_output, Mapping)
        else {}
    )
    completion = receipt.get("candidate_physics_completion")
    if (
        configuration.get("schema_version")
        != "rigid_replacement_authoring_configuration.v1"
        or identity != envelope["recipe"]["subject_identity"]
        or not isinstance(required_output, Mapping)
        or required_output.get("format") != "OpenUSD"
        or required_output.get("rigid_body") is not True
        or required_output.get("single_movable_root") is not True
        or required_output.get("units") != "meters"
        or required_output.get("up_axis") != "Z"
        or configuration.get("physics_authority_granted_by_authoring") is not False
        or asset.suffix.lower() != ".usdz"
        or receipt.get("schema_version")
        != "task_evaluation_rigid_replacement_authoring_result.v1"
        or receipt.get("status") != "authored_candidate_pending_qualification"
        or receipt.get("replacement_identity") != identity
        or receipt.get("source_candidate_digest")
        != source_candidate_record.get("digest")
        or receipt.get("source_candidate_claim")
        != "sage_candidate_geometry_not_observed_truth_or_physics_authority"
        or receipt.get("output_usd", {}).get("sha256")
        != asset_record.get("digest")
        or receipt.get("output_usd", {}).get("size_bytes")
        != asset_record.get("size_bytes")
        or receipt.get("result_digest")
        != canonical_digest(receipt, digest_field="result_digest")
        or not isinstance(completion, Mapping)
        or completion.get("schema_version")
        != "task_evaluation_rigid_candidate_physics_completion.v1"
        or completion.get("status") != "bounded_candidate_completed"
        or completion.get("physics_bounds") != expected_physics_bounds
        or completion.get("candidate_prior_only") is not True
        or completion.get("physical_truth_claimed") is not False
        or completion.get("completion_digest")
        != canonical_digest(completion, digest_field="completion_digest")
        or graph.get("schema_version")
        != "task_evaluation_rigid_replacement_graph.v1"
        or graph.get("asset_id") != identity.get("id")
        or graph.get("asset_version") != identity.get("version")
        or graph.get("articulation_graph", {}).get("joints") != []
        or graph.get("single_rigid_candidate") is not True
        or graph.get("physics_bounds") != expected_physics_bounds
        or graph.get("physics_authority_granted") is not False
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "content_agents_replacement_result_invalid"
        )
    copied_asset = _copy_artifact(
        asset, output_root / f"replacement_asset{asset.suffix}"
    )
    copied_receipt = _copy_artifact(
        receipt_path, output_root / "replacement_authoring_receipt.v1.json"
    )
    copied_graph = _copy_artifact(
        graph_path, output_root / "replacement_graph_spec.v1.json"
    )
    return _stage_result(
        stage=stage,
        configuration_path=configuration_path,
        output_artifacts=[
            {"role": "replacement_asset", **copied_asset},
            {"role": "replacement_authoring_receipt", **copied_receipt},
            {"role": "replacement_graph_spec", **copied_graph},
        ],
    )


def execute_sage_exact_prim_excision(
    *,
    envelope: Mapping[str, Any],
    stage: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_path: Path,
    dependency_results: tuple[Mapping[str, Any], ...],
    output_root: Path,
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...] = (),
) -> Mapping[str, Any]:
    """Remove one exact SAGE prim and prove all unrelated prims unchanged."""

    if dependency_results[-1].get("status") != "completed":
        raise TaskEvaluationSceneConfigurationAdapterError(
            "sage_exact_prim_excision_dependency_invalid"
        )
    source_record, source = _materialized_reference(
        envelope, contract_path="scene.geometry.collision"
    )
    expected = configuration.get("expected_target")
    validation = configuration.get("validation")
    if (
        configuration.get("schema_version")
        != "collision_object_excision_configuration.v1"
        or configuration.get("operation") != "deactivate_exact_prim_only"
        or configuration.get("collision_source_digest")
        != source_record.get("digest")
        or not isinstance(expected, Mapping)
        or not isinstance(validation, Mapping)
        or validation.get("target_absent_after_excision") is not True
        or validation.get("all_non_target_prim_digests_unchanged") is not True
        or validation.get("stage_units_and_up_axis_unchanged") is not True
        or validation.get("before_and_after_prim_manifests_required") is not True
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "sage_exact_prim_excision_configuration_invalid"
        )
    target = str(configuration.get("exact_target_prim") or "")
    candidate = _extract_source_candidate_subtree(
        source_stage_path=source,
        prim_path=target,
        output_path=output_root / "source_object_candidate_mesh.usda",
    )
    removed = output_root / "collision_without_source_object.usda"
    receipt = remove_source_collider_subtree(
        source_usd_path=source,
        target_prim_path=target,
        output_usda_path=removed,
        expected_source_sha256=str(source_record["digest"]),
        removal_id=str(envelope["recipe"]["subject_identity"]["id"]),
    )
    if (
        receipt.get("status") != "exact_source_collider_subtree_removed"
        or receipt.get("removed_prim_path") != target
        or receipt.get("remaining_target_collision_prim_count") != 0
        or receipt.get("unrelated_prim_inventory_unchanged") is not True
        or receipt.get("source_bytes_unchanged") is not True
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "sage_exact_prim_excision_result_invalid"
        )
    receipt_path = output_root / "collision_excision_receipt.v1.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "schema_version": STAGE_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "stage_id": stage["stage_id"],
        "capability": stage["capability"],
        "execution_class": stage["execution_class"],
        "configuration_digest": _sha256_and_size(configuration_path)[0],
        "retry_cap": 0,
        "raw_secret_values_recorded": False,
        "canonical_allocator": None,
        "provider_mutations_performed": 0,
        "paid_execution_requested": False,
        "executed_inside_parent_configuration_run": True,
        "output_artifacts": [
            {
                "role": "configured_collision_without_source_object",
                "path": str(removed),
                "digest": _sha256_and_size(removed)[0],
                "size_bytes": _sha256_and_size(removed)[1],
            },
            {
                "role": "collision_excision_receipt",
                "path": str(receipt_path),
                "digest": _sha256_and_size(receipt_path)[0],
                "size_bytes": _sha256_and_size(receipt_path)[1],
            },
            {"role": "source_object_candidate_mesh", **candidate},
        ],
        "stage_result_digest": "",
    }
    result["stage_result_digest"] = canonical_digest(
        result, digest_field="stage_result_digest"
    )
    return result


# --- supplemental passive destination -------------------------------------
#
# A pick_and_place run may carry one passive destination (for example a tray)
# that has no source object to remove.  It enters the run as exact
# request-declared bytes plus the recipe-bound authoring receipt and SimReady
# result, and leaves with the same independent static and native-import
# qualifications the subject replacement receives.  Nothing here invents
# source-removal lineage for it.

SUPPLEMENTAL_DESTINATION_ASSET_CONTRACT_PATH = "task.destination.asset"
SUPPLEMENTAL_DESTINATION_STATIC_CONTRACT_PATH = (
    "task.destination.static_qualification"
)
SUPPLEMENTAL_DESTINATION_RIGHTS_CONTRACT_PATH = "task.destination.rights_admission"
SUPPLEMENTAL_DESTINATION_AUTHORING_CONTRACT_PATH = (
    "construction.recipe.supplemental_destination.authoring_receipt"
)
SUPPLEMENTAL_DESTINATION_SIMREADY_CONTRACT_PATH = (
    "construction.recipe.supplemental_destination.simready_result"
)
_SUPPLEMENTAL_DESTINATION_CONTRACT_PATHS = {
    "asset": SUPPLEMENTAL_DESTINATION_ASSET_CONTRACT_PATH,
    "static_qualification": SUPPLEMENTAL_DESTINATION_STATIC_CONTRACT_PATH,
    "rights_admission": SUPPLEMENTAL_DESTINATION_RIGHTS_CONTRACT_PATH,
    "authoring_receipt": SUPPLEMENTAL_DESTINATION_AUTHORING_CONTRACT_PATH,
    "simready_result": SUPPLEMENTAL_DESTINATION_SIMREADY_CONTRACT_PATH,
}
PASSIVE_DESTINATION_SIMREADY_SCHEMA_VERSION = (
    "task_evaluation_passive_destination_simready.v1"
)
DESTINATION_RIGHTS_SCHEMA_VERSION = (
    "task_evaluation_rigid_destination_rights_admission.v1"
)


def _load_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(code) from exc
    if not isinstance(value, Mapping):
        raise TaskEvaluationSceneConfigurationAdapterError(code)
    return dict(value)


def supplemental_destination_inputs(
    envelope: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Resolve and cross-bind every supplemental destination input, or ``None``.

    Returns the recipe binding, the exact materialized files, and their parsed
    receipts after every digest join has been checked.  Raises a typed adapter
    error when the recipe, request references, or receipts disagree.
    """

    recipe = envelope.get("recipe")
    destination = (
        recipe.get("supplemental_destination") if isinstance(recipe, Mapping) else None
    )
    if destination is None:
        return None
    if not isinstance(destination, Mapping) or not isinstance(
        destination.get("identity"), Mapping
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_supplemental_destination_recipe_invalid"
        )
    identity = dict(destination["identity"])
    rows: dict[str, Mapping[str, Any]] = {}
    paths: dict[str, Path] = {}
    for name, contract_path in _SUPPLEMENTAL_DESTINATION_CONTRACT_PATHS.items():
        row, path = _materialized_reference(envelope, contract_path=contract_path)
        expected = destination.get(name)
        if not isinstance(expected, Mapping) or any(
            row.get(key) != expected.get(key)
            for key in ("uri", "digest", "size_bytes")
        ):
            raise TaskEvaluationSceneConfigurationAdapterError(
                f"simready_supplemental_destination_recipe_binding_invalid:{name}"
            )
        rows[name] = row
        paths[name] = path
    asset_digest, asset_size = _sha256_and_size(paths["asset"])
    static = _load_json(
        paths["static_qualification"],
        code="simready_supplemental_destination_input_invalid:static_qualification",
    )
    rights = _load_json(
        paths["rights_admission"],
        code="simready_supplemental_destination_input_invalid:rights_admission",
    )
    authoring = _load_json(
        paths["authoring_receipt"],
        code="simready_supplemental_destination_input_invalid:authoring_receipt",
    )
    simready = _load_json(
        paths["simready_result"],
        code="simready_supplemental_destination_input_invalid:simready_result",
    )
    completion = authoring.get("candidate_physics_completion")
    interior = simready.get("interior_bounds_body_frame_m")
    support_paths = simready.get("intended_support_prim_paths")
    if (
        static.get("schema_version") != STATIC_QUALIFICATION_SCHEMA_VERSION
        or static.get("status") != "authored_structure_statically_qualified"
        or static.get("authored_structure_statically_qualified") is not True
        or static.get("structural_findings") != []
        or static.get("replacement_identity") != identity
        or (static.get("replacement_usd") or {}).get("sha256") != asset_digest
        or (static.get("replacement_usd") or {}).get("size_bytes") != asset_size
        or (static.get("claim_boundary") or {}).get("native_simulator_import_qualified")
        is not False
        or static.get("result_digest")
        != canonical_digest(static, digest_field="result_digest")
        or rights.get("schema_version") != DESTINATION_RIGHTS_SCHEMA_VERSION
        or rights.get("status") != "admitted"
        or rights.get("destination_identity") != identity
        or rights.get("private_provider_processing_allowed") is not True
        or rights.get("rights_admission_digest")
        != canonical_digest(rights, digest_field="rights_admission_digest")
        or authoring.get("schema_version")
        != "task_evaluation_rigid_replacement_authoring_result.v1"
        or authoring.get("status") != "authored_candidate_pending_qualification"
        or authoring.get("replacement_identity") != identity
        or authoring.get("physics_authority_granted") is not False
        or (authoring.get("output_usd") or {}).get("sha256") != asset_digest
        or (authoring.get("output_usd") or {}).get("size_bytes") != asset_size
        or authoring.get("result_digest")
        != canonical_digest(authoring, digest_field="result_digest")
        or not isinstance(completion, Mapping)
        or not isinstance(completion.get("physics_bounds"), Mapping)
        or simready.get("schema_version") != PASSIVE_DESTINATION_SIMREADY_SCHEMA_VERSION
        or simready.get("destination_identity") != identity
        or (simready.get("asset") or {}).get("sha256") != asset_digest
        or (simready.get("static_qualification") or {}).get("sha256")
        != rows["static_qualification"]["digest"]
        or (simready.get("authoring_receipt") or {}).get("sha256")
        != rows["authoring_receipt"]["digest"]
        or (simready.get("rights_admission") or {}).get("sha256")
        != rows["rights_admission"]["digest"]
        or simready.get("static_result_digest") != static.get("result_digest")
        or not isinstance(support_paths, list)
        or not support_paths
        or any(
            not str(path).startswith("/")
            or path not in (static.get("observed_structure") or {}).get(
                "collision_prim_paths", []
            )
            for path in support_paths
        )
        or not isinstance(interior, Mapping)
        or simready.get("result_digest")
        != canonical_digest(simready, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_supplemental_destination_binding_invalid"
        )
    return {
        "identity": identity,
        "relation": destination.get("relation"),
        "binding": dict(destination),
        "rows": rows,
        "paths": paths,
        "asset_digest": asset_digest,
        "asset_size_bytes": asset_size,
        "static_qualification": static,
        "rights_admission": rights,
        "authoring_receipt": authoring,
        "simready_result": simready,
    }


def _requalification_comparable(receipt: Mapping[str, Any]) -> dict[str, Any]:
    comparable = json.loads(json.dumps(dict(receipt)))
    comparable.pop("result_digest", None)
    usd = comparable.get("replacement_usd")
    if isinstance(usd, dict):
        usd.pop("path", None)
    return comparable


def _supplemental_destination_static_artifacts(
    *, envelope: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    inputs = supplemental_destination_inputs(envelope)
    if inputs is None:
        return []
    identity = inputs["identity"]
    authoring = inputs["authoring_receipt"]
    graph = {
        "schema_version": "task_evaluation_rigid_replacement_graph.v1",
        "asset_id": identity["id"],
        "asset_version": identity["version"],
        "articulation_graph": {"joints": []},
        "single_rigid_candidate": True,
        "physics_bounds": dict(
            authoring["candidate_physics_completion"]["physics_bounds"]
        ),
        "physics_authority_granted": False,
    }
    requalification_path = (
        output_root / "destination_static_requalification_receipt.v1.json"
    )
    try:
        requalified = qualify_scene_configuration_rigid_asset_static(
            asset_path=inputs["paths"]["asset"],
            graph_spec=graph,
            authoring_receipt=authoring,
            replacement_identity=identity,
            output_path=requalification_path,
        )
    except TaskEvaluationSceneConfigurationStaticQualificationError as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_static_destination_requalification_failed:"
            + ";".join(exc.codes)
        ) from exc
    if _requalification_comparable(requalified) != _requalification_comparable(
        inputs["static_qualification"]
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_static_destination_requalification_mismatch"
        )
    asset = inputs["paths"]["asset"]
    retained_asset = _copy_artifact(
        asset, output_root / f"statically_qualified_destination_asset{asset.suffix}"
    )
    retained_static = _copy_artifact(
        inputs["paths"]["static_qualification"],
        output_root / "destination_static_qualification_receipt.v1.json",
    )
    return [
        {"role": "statically_qualified_destination_asset", **retained_asset},
        {"role": "destination_static_qualification_receipt", **retained_static},
        {
            "role": "destination_static_requalification_receipt",
            "path": str(requalification_path),
            "digest": _sha256_and_size(requalification_path)[0],
            "size_bytes": _sha256_and_size(requalification_path)[1],
        },
    ]


def execute_simready_static_rigid_qualification(
    *,
    envelope: Mapping[str, Any],
    stage: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_path: Path,
    dependency_results: tuple[Mapping[str, Any], ...],
    output_root: Path,
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...] = (),
) -> Mapping[str, Any]:
    """Statically qualify exactly the replacement authored by stage 3."""

    _asset_record, asset = _dependency_artifact(
        dependency_results, role="replacement_asset"
    )
    _receipt_record, authoring_receipt = _dependency_artifact(
        dependency_results, role="replacement_authoring_receipt"
    )
    _spec_record, graph_spec_path = _dependency_artifact(
        dependency_results, role="replacement_graph_spec"
    )
    try:
        graph_spec = json.loads(graph_spec_path.read_text(encoding="utf-8"))
        authoring = json.loads(authoring_receipt.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_static_rigid_dependency_json_invalid"
        ) from exc
    checks = configuration.get("required_checks")
    identity = configuration.get("replacement_identity")
    graph = graph_spec.get("articulation_graph") if isinstance(graph_spec, Mapping) else None
    if (
        configuration.get("schema_version")
        != "replacement_static_qualification_configuration.v1"
        or not isinstance(identity, Mapping)
        or identity != envelope["recipe"]["subject_identity"]
        or not static_qualification_checks_valid(checks)
        or configuration.get("center_of_mass_must_lie_inside_collision_bounds")
        is not True
        or graph_spec.get("asset_id") != identity.get("id")
        or not isinstance(graph, Mapping)
        or graph.get("joints") != []
        or authoring.get("output_usd", {}).get("sha256")
        != _sha256_and_size(asset)[0]
        or authoring.get("output_usd", {}).get("size_bytes")
        != _sha256_and_size(asset)[1]
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_static_rigid_configuration_or_binding_invalid"
        )
    qualification_path = output_root / "static_qualification_receipt.v1.json"
    qualification = qualify_scene_configuration_rigid_asset_static(
        asset_path=asset,
        graph_spec=graph_spec,
        authoring_receipt=authoring,
        replacement_identity=identity,
        output_path=qualification_path,
    )
    if (
        qualification.get("schema_version")
        != STATIC_QUALIFICATION_SCHEMA_VERSION
        or qualification.get("result_digest")
        != canonical_digest(qualification, digest_field="result_digest")
        or qualification.get("status")
        != "authored_structure_statically_qualified"
        or qualification.get("authored_structure_statically_qualified") is not True
        or qualification.get("structural_findings") != []
        or qualification.get("replacement_usd", {}).get("sha256")
        != _sha256_and_size(asset)[0]
        or qualification.get("claim_boundary", {}).get(
            "native_simulator_import_qualified"
        )
        is not False
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_static_rigid_qualification_failed"
        )
    retained_asset = output_root / f"replacement_asset{asset.suffix}"
    shutil.copyfile(asset, retained_asset)
    if _sha256_and_size(retained_asset) != _sha256_and_size(asset):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_static_rigid_asset_copy_mismatch"
        )
    destination_artifacts = _supplemental_destination_static_artifacts(
        envelope=envelope, output_root=output_root
    )
    result: dict[str, Any] = {
        "schema_version": STAGE_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "stage_id": stage["stage_id"],
        "capability": stage["capability"],
        "execution_class": stage["execution_class"],
        "configuration_digest": _sha256_and_size(configuration_path)[0],
        "retry_cap": 0,
        "raw_secret_values_recorded": False,
        "canonical_allocator": None,
        "provider_mutations_performed": 0,
        "paid_execution_requested": False,
        "executed_inside_parent_configuration_run": True,
        "output_artifacts": [
            {
                "role": "statically_qualified_replacement_asset",
                "path": str(retained_asset),
                "digest": _sha256_and_size(retained_asset)[0],
                "size_bytes": _sha256_and_size(retained_asset)[1],
            },
            {
                "role": "static_qualification_receipt",
                "path": str(qualification_path),
                "digest": _sha256_and_size(qualification_path)[0],
                "size_bytes": _sha256_and_size(qualification_path)[1],
            },
            *destination_artifacts,
        ],
        "stage_result_digest": "",
    }
    result["stage_result_digest"] = canonical_digest(
        result, digest_field="stage_result_digest"
    )
    return result


def execute_simready_native_import_qualification(
    *,
    envelope: Mapping[str, Any],
    stage: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_path: Path,
    dependency_results: tuple[Mapping[str, Any], ...],
    output_root: Path,
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...] = (),
) -> Mapping[str, Any]:
    """Admit the exact statically-qualified asset after native Isaac readback."""

    _asset_record, asset = _dependency_artifact(
        dependency_results, role="statically_qualified_replacement_asset"
    )
    _static_record, static_receipt = _dependency_artifact(
        dependency_results, role="static_qualification_receipt"
    )
    _runtime_record, runtime_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="native_import_runtime_result"
    )
    try:
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_native_import_result_invalid"
        ) from exc
    identity = configuration.get("replacement_identity")
    checks = configuration.get("required_checks")
    if (
        configuration.get("schema_version")
        != "replacement_native_import_qualification_configuration.v1"
        or identity != envelope["recipe"]["subject_identity"]
        or not native_import_checks_valid(checks)
        or runtime.get("schema_version")
        != "task_evaluation_replacement_native_import_result.v1"
        or runtime.get("status") != "qualified"
        or runtime.get("replacement_identity") != identity
        or runtime.get("asset_digest") != _sha256_and_size(asset)[0]
        or runtime.get("static_qualification_digest")
        != _sha256_and_size(static_receipt)[0]
        or runtime.get("native_isaac_executed") is not True
        or runtime.get("native_simulator_import_qualified") is not True
        or runtime.get("support_contact_observed") is not True
        or runtime.get("deterministic_reset_state_digest_repeat_count") != 3
        or runtime.get("blockers") != []
        or runtime.get("result_digest")
        != canonical_digest(runtime, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_native_import_result_invalid"
        )
    retained_asset = _copy_artifact(
        asset, output_root / f"native_qualified_replacement_asset{asset.suffix}"
    )
    retained_receipt = _copy_artifact(
        runtime_path, output_root / "native_import_qualification_receipt.v1.json"
    )
    output_artifacts = [
        {"role": "native_qualified_replacement_asset", **retained_asset},
        {"role": "native_import_qualification_receipt", **retained_receipt},
    ]
    output_artifacts.extend(
        _supplemental_destination_native_artifacts(
            envelope=envelope,
            dependency_results=dependency_results,
            provider_runtime_artifacts=provider_runtime_artifacts,
            output_root=output_root,
        )
    )
    return _stage_result(
        stage=stage,
        configuration_path=configuration_path,
        output_artifacts=output_artifacts,
    )


def _supplemental_destination_native_artifacts(
    *,
    envelope: Mapping[str, Any],
    dependency_results: tuple[Mapping[str, Any], ...],
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...],
    output_root: Path,
) -> list[dict[str, Any]]:
    """Admit the destination's native readback with the subject's exact rules."""

    recipe = envelope.get("recipe")
    destination = (
        recipe.get("supplemental_destination") if isinstance(recipe, Mapping) else None
    )
    if destination is None:
        return []
    identity = destination.get("identity") if isinstance(destination, Mapping) else None
    if not isinstance(identity, Mapping):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_supplemental_destination_recipe_invalid"
        )
    _asset_record, asset = _dependency_artifact(
        dependency_results, role="statically_qualified_destination_asset"
    )
    _static_record, static_receipt = _dependency_artifact(
        dependency_results, role="destination_static_qualification_receipt"
    )
    _runtime_record, runtime_path = _provider_runtime_artifact(
        provider_runtime_artifacts, role="destination_native_import_runtime_result"
    )
    runtime = _load_json(
        runtime_path, code="simready_native_import_destination_result_invalid"
    )
    if (
        runtime.get("schema_version")
        != "task_evaluation_replacement_native_import_result.v1"
        or runtime.get("status") != "qualified"
        or runtime.get("replacement_identity") != identity
        or runtime.get("asset_digest") != _sha256_and_size(asset)[0]
        or runtime.get("static_qualification_digest")
        != _sha256_and_size(static_receipt)[0]
        or runtime.get("native_isaac_executed") is not True
        or runtime.get("native_simulator_import_qualified") is not True
        or runtime.get("support_contact_observed") is not True
        or runtime.get("deterministic_reset_state_digest_repeat_count") != 3
        or runtime.get("blockers") != []
        or runtime.get("result_digest")
        != canonical_digest(runtime, digest_field="result_digest")
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "simready_native_import_destination_result_invalid"
        )
    retained_asset = _copy_artifact(
        asset, output_root / f"native_qualified_destination_asset{asset.suffix}"
    )
    retained_receipt = _copy_artifact(
        runtime_path,
        output_root / "destination_native_import_qualification_receipt.v1.json",
    )
    return [
        {"role": "native_qualified_destination_asset", **retained_asset},
        {"role": "destination_native_import_qualification_receipt", **retained_receipt},
    ]


def execute_native_task_scene_assembly(
    *,
    envelope: Mapping[str, Any],
    stage: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_path: Path,
    dependency_results: tuple[Mapping[str, Any], ...],
    output_root: Path,
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...] = (),
) -> Mapping[str, Any]:
    """Assemble robot-neutral provider outputs for control-plane publication."""

    appearance_record, appearance = _dependency_artifact(
        dependency_results, role="configured_appearance_without_source_object"
    )
    collision_record, collision = _dependency_artifact(
        dependency_results, role="configured_collision_without_source_object"
    )
    replacement_record, replacement = _dependency_artifact(
        dependency_results, role="native_qualified_replacement_asset"
    )
    native_receipt_record, _native_receipt = _dependency_artifact(
        dependency_results, role="native_import_qualification_receipt"
    )
    scene_identity = configuration.get("scene_identity")
    replacement_config = configuration.get("replacement")
    robot_mount = configuration.get("robot_mount_interface")
    if (
        configuration.get("schema_version")
        != "task_evaluation_scene_assembly_configuration.v1"
        or scene_identity != envelope["recipe"]["scene_identity"]
        or not isinstance(replacement_config, Mapping)
        or replacement_config.get("qualified_asset_from_stage") != "stage-5"
        or replacement_config.get(
            "source_and_replacement_visual_instances_must_not_coexist"
        )
        is not True
        or replacement_config.get(
            "source_and_replacement_collision_instances_must_not_coexist"
        )
        is not True
        or not isinstance(robot_mount, Mapping)
        or robot_mount.get("publish_robot_neutral_scene_mount_frame") is not True
        or robot_mount.get(
            "robot_specific_base_transform_and_reachability_deferred_to_each_evaluation"
        )
        is not True
        or configuration.get("evaluation_episode_executed_in_this_run") is not False
        or configuration.get("scene_construction_repeated_per_evaluation") is not False
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "native_task_scene_assembly_configuration_invalid"
        )
    assembled = output_root / "configured_scene_bundle_candidate"
    assembled.mkdir(mode=0o750)
    copied: dict[str, Path] = {}
    for role, source in (
        ("appearance", appearance),
        ("collision", collision),
        ("replacement", replacement),
    ):
        destination = assembled / f"{role}{source.suffix}"
        shutil.copyfile(source, destination)
        if _sha256_and_size(destination) != _sha256_and_size(source):
            raise TaskEvaluationSceneConfigurationAdapterError(
                f"native_task_scene_assembly_copy_mismatch:{role}"
            )
        copied[role] = destination
    manifest: dict[str, Any] = {
        "schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
        "status": "assembled_pending_control_plane_publication",
        "configuration_run_id": envelope["run_id"],
        "team_namespace": envelope["team_namespace"],
        "scene_identity": dict(scene_identity),
        "task_identity": dict(envelope["recipe"]["task_identity"]),
        "subject_identity": dict(envelope["recipe"]["subject_identity"]),
        "source_commit": envelope["expected_production_commit"],
        "assets": [
            {
                "role": role,
                "relative_path": path.relative_to(assembled).as_posix(),
                "digest": _sha256_and_size(path)[0],
                "size_bytes": _sha256_and_size(path)[1],
            }
            for role, path in sorted(copied.items())
        ],
        "upstream_stage_artifacts": {
            "appearance": {
                "digest": appearance_record["digest"],
                "size_bytes": appearance_record["size_bytes"],
            },
            "collision": {
                "digest": collision_record["digest"],
                "size_bytes": collision_record["size_bytes"],
            },
            "replacement": {
                "digest": replacement_record["digest"],
                "size_bytes": replacement_record["size_bytes"],
            },
            "native_import_qualification": {
                "digest": native_receipt_record["digest"],
                "size_bytes": native_receipt_record["size_bytes"],
            },
        },
        "robot_neutral": True,
        "robot_specific_base_registration_included": False,
        "robot_specific_kinematics_included": False,
        "evaluation_episode_executed": False,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_path = assembled / "configured_scene_bundle_candidate.v1.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    receipt: dict[str, Any] = {
        "schema_version": "task_evaluation_scene_assembly_receipt.v1",
        "status": "assembled_pending_control_plane_publication",
        "manifest_digest": manifest["manifest_digest"],
        "asset_count": len(copied),
        "robot_neutral": True,
        "evaluation_episode_executed": False,
        "control_plane_publication_required": True,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = output_root / "scene_assembly_receipt.v1.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "schema_version": STAGE_RESULT_SCHEMA_VERSION,
        "status": "completed",
        "stage_id": stage["stage_id"],
        "capability": stage["capability"],
        "execution_class": stage["execution_class"],
        "configuration_digest": _sha256_and_size(configuration_path)[0],
        "retry_cap": 0,
        "raw_secret_values_recorded": False,
        "canonical_allocator": None,
        "provider_mutations_performed": 0,
        "paid_execution_requested": False,
        "executed_inside_parent_configuration_run": True,
        "output_artifacts": [
            {
                "role": "configured_scene_bundle_candidate_manifest",
                "path": str(manifest_path),
                "digest": _sha256_and_size(manifest_path)[0],
                "size_bytes": _sha256_and_size(manifest_path)[1],
            },
            {
                "role": "scene_assembly_receipt",
                "path": str(receipt_path),
                "digest": _sha256_and_size(receipt_path)[0],
                "size_bytes": _sha256_and_size(receipt_path)[1],
            },
        ],
        "stage_result_digest": "",
    }
    result["stage_result_digest"] = canonical_digest(
        result, digest_field="stage_result_digest"
    )
    return result


def execute_native_task_scene_diagnostic_assembly(
    *,
    envelope: Mapping[str, Any],
    stage: Mapping[str, Any],
    configuration: Mapping[str, Any],
    configuration_path: Path,
    dependency_results: tuple[Mapping[str, Any], ...],
    output_root: Path,
    provider_runtime_artifacts: tuple[Mapping[str, Any], ...] = (),
) -> Mapping[str, Any]:
    """Assemble rejected appearance bytes with an irrevocable diagnostic ceiling."""

    diagnostic_sources = [
        result
        for result in dependency_results
        if any(
            isinstance(row, Mapping)
            and row.get("role") == "diagnostic_rejected_appearance_candidate"
            for row in result.get("output_artifacts") or []
        )
    ]
    if not diagnostic_sources:
        return execute_native_task_scene_assembly(
            envelope=envelope,
            stage=stage,
            configuration=configuration,
            configuration_path=configuration_path,
            dependency_results=dependency_results,
            output_root=output_root,
            provider_runtime_artifacts=provider_runtime_artifacts,
        )
    if (
        len(diagnostic_sources) != 1
        or diagnostic_sources[0].get("diagnostic_only") is not True
        or diagnostic_sources[0].get("qualification_eligible") is not False
        or diagnostic_sources[0].get("configured_revision_publication_permitted")
        is not False
        or diagnostic_sources[0].get("offering_publication_permitted") is not False
        or diagnostic_sources[0].get("terminal_e2e_completion_permitted") is not False
        or diagnostic_sources[0].get("appearance_visual_review_rejected") is not True
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "native_task_scene_diagnostic_assembly_source_invalid"
        )
    appearance_record, appearance = _dependency_artifact(
        dependency_results, role="diagnostic_rejected_appearance_candidate"
    )
    rejection_record, rejection_path = _dependency_artifact(
        dependency_results, role="appearance_rejection_receipt"
    )
    execution_record, _execution_path = _dependency_artifact(
        dependency_results, role="appearance_visual_review_execution"
    )
    collision_record, collision = _dependency_artifact(
        dependency_results, role="configured_collision_without_source_object"
    )
    replacement_record, replacement = _dependency_artifact(
        dependency_results, role="native_qualified_replacement_asset"
    )
    native_receipt_record, _native_receipt = _dependency_artifact(
        dependency_results, role="native_import_qualification_receipt"
    )
    try:
        rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "native_task_scene_diagnostic_assembly_source_invalid"
        ) from exc
    scene_identity = configuration.get("scene_identity")
    replacement_config = configuration.get("replacement")
    robot_mount = configuration.get("robot_mount_interface")
    if (
        configuration.get("schema_version")
        != "task_evaluation_scene_assembly_configuration.v1"
        or scene_identity != envelope["recipe"]["scene_identity"]
        or not isinstance(replacement_config, Mapping)
        or replacement_config.get("qualified_asset_from_stage") != "stage-5"
        or replacement_config.get(
            "source_and_replacement_visual_instances_must_not_coexist"
        )
        is not True
        or replacement_config.get(
            "source_and_replacement_collision_instances_must_not_coexist"
        )
        is not True
        or not isinstance(robot_mount, Mapping)
        or robot_mount.get("publish_robot_neutral_scene_mount_frame") is not True
        or robot_mount.get(
            "robot_specific_base_transform_and_reachability_deferred_to_each_evaluation"
        )
        is not True
        or configuration.get("evaluation_episode_executed_in_this_run") is not False
        or configuration.get("scene_construction_repeated_per_evaluation") is not False
        or rejection.get("result_digest")
        != canonical_digest(rejection, digest_field="result_digest")
        or rejection.get("diagnostic_only") is not True
        or rejection.get("qualification_eligible") is not False
        or rejection.get("configured_revision_publication_permitted") is not False
        or rejection.get("offering_publication_permitted") is not False
        or rejection.get("terminal_e2e_completion_permitted") is not False
        or rejection.get("diagnostic_rejected_appearance_candidate_sha256")
        != appearance_record.get("digest")
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "native_task_scene_diagnostic_assembly_configuration_invalid"
        )
    assembled = output_root / "diagnostic_configured_scene_bundle_candidate"
    assembled.mkdir(mode=0o750)
    copied: dict[str, Path] = {}
    for role, source in (
        ("diagnostic_rejected_appearance", appearance),
        ("collision", collision),
        ("replacement", replacement),
    ):
        destination = assembled / f"{role}{source.suffix}"
        shutil.copyfile(source, destination)
        if _sha256_and_size(destination) != _sha256_and_size(source):
            raise TaskEvaluationSceneConfigurationAdapterError(
                f"native_task_scene_diagnostic_assembly_copy_mismatch:{role}"
            )
        copied[role] = destination
    manifest: dict[str, Any] = {
        "schema_version": "task_evaluation_configured_scene_bundle_candidate.v1",
        "status": "assembled_diagnostic_with_rejected_appearance_not_publishable",
        "configuration_run_id": envelope["run_id"],
        "team_namespace": envelope["team_namespace"],
        "scene_identity": dict(scene_identity),
        "task_identity": dict(envelope["recipe"]["task_identity"]),
        "subject_identity": dict(envelope["recipe"]["subject_identity"]),
        "source_commit": envelope["expected_production_commit"],
        "assets": [
            {
                "role": role,
                "relative_path": path.relative_to(assembled).as_posix(),
                "digest": _sha256_and_size(path)[0],
                "size_bytes": _sha256_and_size(path)[1],
            }
            for role, path in sorted(copied.items())
        ],
        "upstream_stage_artifacts": {
            "appearance_rejection": {
                "digest": rejection_record["digest"],
                "size_bytes": rejection_record["size_bytes"],
            },
            "appearance_visual_review_execution": {
                "digest": execution_record["digest"],
                "size_bytes": execution_record["size_bytes"],
            },
            "collision": {
                "digest": collision_record["digest"],
                "size_bytes": collision_record["size_bytes"],
            },
            "replacement": {
                "digest": replacement_record["digest"],
                "size_bytes": replacement_record["size_bytes"],
            },
            "native_import_qualification": {
                "digest": native_receipt_record["digest"],
                "size_bytes": native_receipt_record["size_bytes"],
            },
        },
        "robot_neutral": True,
        "robot_specific_base_registration_included": False,
        "robot_specific_kinematics_included": False,
        "evaluation_episode_executed": False,
        "appearance_visual_review_rejected": True,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "manifest_digest": "",
    }
    manifest["manifest_digest"] = canonical_digest(
        manifest, digest_field="manifest_digest"
    )
    manifest_path = assembled / "diagnostic_configured_scene_bundle_candidate.v1.json"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")
    receipt: dict[str, Any] = {
        "schema_version": "task_evaluation_scene_assembly_receipt.v1",
        "status": "assembled_diagnostic_with_rejected_appearance_not_publishable",
        "manifest_digest": manifest["manifest_digest"],
        "asset_count": len(copied),
        "robot_neutral": True,
        "evaluation_episode_executed": False,
        "appearance_visual_review_rejected": True,
        "control_plane_publication_required": False,
        "diagnostic_only": True,
        "qualification_eligible": False,
        "configured_revision_publication_permitted": False,
        "offering_publication_permitted": False,
        "terminal_e2e_completion_permitted": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = output_root / "diagnostic_scene_assembly_receipt.v1.json"
    receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    result = _stage_result(
        stage=stage,
        configuration_path=configuration_path,
        output_artifacts=[
            {
                "role": "diagnostic_configured_scene_bundle_candidate_manifest",
                "path": str(manifest_path),
                "digest": _sha256_and_size(manifest_path)[0],
                "size_bytes": _sha256_and_size(manifest_path)[1],
            },
            {
                "role": "diagnostic_scene_assembly_receipt",
                "path": str(receipt_path),
                "digest": _sha256_and_size(receipt_path)[0],
                "size_bytes": _sha256_and_size(receipt_path)[1],
            },
        ],
    )
    result.update(
        {
            "appearance_visual_review_rejected": True,
            "diagnostic_only": True,
            "qualification_eligible": False,
            "configured_revision_publication_permitted": False,
            "offering_publication_permitted": False,
            "terminal_e2e_completion_permitted": False,
        }
    )
    result["stage_result_digest"] = canonical_digest(
        result, digest_field="stage_result_digest"
    )
    return result


def builtin_scene_configuration_adapter_handlers(
) -> dict[SceneConfigurationAdapterIdentity, StageAdapter]:
    """Return installed handlers; admission stays separate from installation."""

    sage_identity = next(
        identity
        for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES
        if identity.adapter_id == "sage_exact_prim_excision"
    )
    static_identity = next(
        identity
        for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES
        if identity.adapter_id == "simready_static_rigid_qualification"
    )
    assembly_identity = next(
        identity
        for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES
        if identity.adapter_id == "native_task_scene_assembly"
    )
    artifixer_identity = next(
        identity
        for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES
        if identity.adapter_id == "artifixer3d_observed_object_removal"
    )
    content_agents_identity = next(
        identity
        for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES
        if identity.adapter_id == "content_agents_rigid_replacement"
    )
    native_import_identity = next(
        identity
        for identity in ADMITTED_STAGE_ADAPTER_IDENTITIES
        if identity.adapter_id == "simready_native_import_qualification"
    )
    return {
        artifixer_identity: execute_artifixer3d_observed_object_removal,
        sage_identity: execute_sage_exact_prim_excision,
        content_agents_identity: execute_content_agents_rigid_replacement,
        static_identity: execute_simready_static_rigid_qualification,
        native_import_identity: execute_simready_native_import_qualification,
        assembly_identity: execute_native_task_scene_assembly,
    }


def builtin_scene_configuration_diagnostic_adapter_handlers(
) -> dict[SceneConfigurationAdapterIdentity, StageAdapter]:
    """Install rejected-appearance handlers only for diagnostic execution."""

    handlers = builtin_scene_configuration_adapter_handlers()
    for identity in tuple(handlers):
        if identity.adapter_id == "artifixer3d_observed_object_removal":
            handlers[identity] = execute_artifixer3d_diagnostic_object_removal
        elif identity.adapter_id == "native_task_scene_assembly":
            handlers[identity] = execute_native_task_scene_diagnostic_assembly
    return handlers


__all__ = [
    "builtin_scene_configuration_adapter_handlers",
    "builtin_scene_configuration_diagnostic_adapter_handlers",
    "execute_sage_exact_prim_excision",
    "execute_simready_static_rigid_qualification",
    "execute_native_task_scene_assembly",
    "execute_artifixer3d_observed_object_removal",
    "execute_artifixer3d_diagnostic_object_removal",
    "execute_native_task_scene_diagnostic_assembly",
    "execute_content_agents_rigid_replacement",
    "execute_simready_native_import_qualification",
]
