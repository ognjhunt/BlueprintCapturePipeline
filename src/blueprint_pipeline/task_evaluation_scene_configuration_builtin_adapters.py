"""Repository-owned handlers for typed scene-configuration capabilities."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, canonical_json
from .source_collider_subtree_removal import remove_source_collider_subtree
from .simready_graph_asset_static_qualification import (
    qualify_simready_graph_asset_static,
)
from .task_evaluation_scene_configuration_adapters import (
    ADMITTED_STAGE_ADAPTER_IDENTITIES,
    SceneConfigurationAdapterIdentity,
    StageAdapter,
    TaskEvaluationSceneConfigurationAdapterError,
)
from .task_evaluation_scene_configuration_orchestrator import (
    STAGE_RESULT_SCHEMA_VERSION,
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


def _positive_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0.0
    )


def _native_import_checks_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "stage_import",
        "rigid_body_enabled",
        "collider_enabled",
        "gravity_settle_seconds",
        "maximum_settle_translation_m",
        "maximum_settle_rotation_rad",
        "support_contact_required",
        "explosion_or_tunneling_forbidden",
        "deterministic_reset_required",
        "state_digest_repeat_count",
    }:
        return False
    return (
        value["stage_import"] is True
        and value["rigid_body_enabled"] is True
        and value["collider_enabled"] is True
        and _positive_finite(value["gravity_settle_seconds"])
        and _positive_finite(value["maximum_settle_translation_m"])
        and _positive_finite(value["maximum_settle_rotation_rad"])
        and value["support_contact_required"] is True
        and value["explosion_or_tunneling_forbidden"] is True
        and value["deterministic_reset_required"] is True
        and value["state_digest_repeat_count"] == 3
    )


def _static_qualification_checks_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or set(value) != {
        "usd_parses",
        "meters_per_unit",
        "up_axis",
        "single_movable_rigid_root",
        "collision_geometry_present",
        "collision_geometry_nonempty_and_finite",
        "mass_and_inertia_positive_finite",
        "materials_within_preregistered_bounds",
        "no_external_unpinned_dependencies",
        "no_articulation",
        "no_scripts_or_credentials",
    }:
        return False
    return (
        value["usd_parses"] is True
        and value["meters_per_unit"] == 1.0
        and value["up_axis"] == "Z"
        and all(
            value[name] is True
            for name in (
                "single_movable_rigid_root",
                "collision_geometry_present",
                "collision_geometry_nonempty_and_finite",
                "mass_and_inertia_positive_finite",
                "materials_within_preregistered_bounds",
                "no_external_unpinned_dependencies",
                "no_articulation",
                "no_scripts_or_credentials",
            )
        )
    )


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
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        review = json.loads(review_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskEvaluationSceneConfigurationAdapterError(
            "artifixer3d_object_removal_receipt_invalid"
        ) from exc
    source_object = configuration.get("source_object")
    if (
        configuration.get("schema_version")
        != "observed_appearance_object_removal_configuration.v1"
        or not isinstance(source_object, Mapping)
        or configuration.get("production_render_required") is not True
        or configuration.get("provider_disclosure", {}).get(
            "raw_interiorgs_bytes"
        )
        is not False
        or configuration.get("output_requirements", {}).get(
            "generated_pixels_labeled"
        )
        is not True
        or receipt.get("schema_version")
        != "task_evaluation_artifixer_object_removal_result.v1"
        or receipt.get("status") != "qualified_generated_appearance_edit"
        or receipt.get("publisher_instance_id")
        != source_object.get("publisher_instance_id")
        or receipt.get("raw_interiorgs_bytes_sent_to_external_provider")
        is not False
        or receipt.get("visual_review_receipt_digest")
        != review.get("receipt_digest")
        or receipt.get("visual_review_receipt_sha256")
        != review_record.get("digest")
        or receipt.get("semantic_object_free_visual_review_passed") is not True
        or receipt.get("multiview_consistency_review_passed") is not True
        or receipt.get("generated_pixels_labeled") is not True
        or receipt.get("result_digest")
        != canonical_digest(receipt, digest_field="result_digest")
        or review.get("schema_version")
        != "task_evaluation_artifixer_ai_visual_review.v1"
        or review.get("status") != "accepted"
        or review.get("publisher_instance_id")
        != source_object.get("publisher_instance_id")
        or review.get("decision") != "accepted"
        or review.get("semantic_object_absence_review_passed") is not True
        or review.get("multiview_consistency_review_passed") is not True
        or review.get("review_frame_count", 0)
        < configuration.get("required_views", {}).get("minimum", 1)
        or review.get("all_review_frames_digest_bound") is not True
        or review.get("ai_visual_review_completed") is not True
        or review.get("human_review_completed") is not False
        or review.get("generated_output_is_capture_or_physical_evidence")
        is not False
        or not isinstance(review.get("reviewer"), Mapping)
        or not str(review["reviewer"].get("identity") or "")
        or not str(review["reviewer"].get("runtime") or "")
        or not str(review["reviewer"].get("model") or "")
        or review.get("receipt_digest")
        != canonical_digest(review, digest_field="receipt_digest")
    ):
        raise TaskEvaluationSceneConfigurationAdapterError(
            "artifixer3d_object_removal_result_invalid"
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
        ],
    )


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
    if (
        configuration.get("schema_version")
        != "rigid_replacement_authoring_configuration.v1"
        or identity != envelope["recipe"]["subject_identity"]
        or configuration.get("required_output", {}).get("rigid_body") is not True
        or configuration.get("required_output", {}).get("single_movable_root")
        is not True
        or configuration.get("physics_authority_granted_by_authoring") is not False
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
        or graph.get("asset_id") != identity.get("id")
        or graph.get("articulation_graph", {}).get("joints") != []
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
        or not _static_qualification_checks_valid(checks)
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
    qualification = qualify_simready_graph_asset_static(
        spec=graph_spec,
        authoring_receipt_path=authoring_receipt,
        output_path=qualification_path,
    )
    if (
        qualification.get("status")
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
        or not _native_import_checks_valid(checks)
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
    return _stage_result(
        stage=stage,
        configuration_path=configuration_path,
        output_artifacts=[
            {"role": "native_qualified_replacement_asset", **retained_asset},
            {"role": "native_import_qualification_receipt", **retained_receipt},
        ],
    )


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


__all__ = [
    "builtin_scene_configuration_adapter_handlers",
    "execute_sage_exact_prim_excision",
    "execute_simready_static_rigid_qualification",
    "execute_native_task_scene_assembly",
    "execute_artifixer3d_observed_object_removal",
    "execute_content_agents_rigid_replacement",
    "execute_simready_native_import_qualification",
]
