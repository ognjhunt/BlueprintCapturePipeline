"""Compile executable Gaussian-training requests from accepted evidence only."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json
from .reconstruction_worker_contracts import (
    ReconstructionWorkerContractError,
    build_training_request,
    build_worker_build_receipt,
    build_worker_smoke_receipt,
    build_worker_stack_manifest,
)


_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class ReconstructionTrainingRequestCompilationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _digest(value: Any) -> bool:
    return _DIGEST.fullmatch(str(value or "")) is not None


def _mapping(value: Any) -> dict[str, Any]:
    return json.loads(canonical_json(dict(value))) if isinstance(value, Mapping) else {}


def _finite(value: Any, *, minimum: float = 0.0) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def compile_reconstruction_training_request(
    *,
    stable_run_identity: str,
    capture_evidence: Mapping[str, Any],
    dataset_export: Mapping[str, Any],
    worker_stack_manifest: Mapping[str, Any],
    worker_build_receipt: Mapping[str, Any],
    worker_smoke_receipt: Mapping[str, Any],
    pose_binding: Mapping[str, Any],
    evaluation_contract: Mapping[str, Any],
    execution_configuration: Mapping[str, Any],
    execution_authority: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Bind qualified worker and candidate-only evidence into one request.

    This compiler never selects capture truth, poses, calibration, splits,
    thresholds, rights, budgets, or provider authority. All such values must be
    present in accepted deterministic artifacts or explicit authority input.
    """

    capture = _mapping(capture_evidence)
    dataset = _mapping(dataset_export)
    pose = _mapping(pose_binding)
    evaluation = _mapping(evaluation_contract)
    configuration = _mapping(execution_configuration)
    authority = _mapping(execution_authority)
    errors: list[str] = []

    try:
        stack = build_worker_stack_manifest(worker_stack_manifest)
    except ReconstructionWorkerContractError:
        stack = {}
        errors.append("training_worker_stack_manifest_invalid")
    try:
        build = build_worker_build_receipt(worker_build_receipt)
    except ReconstructionWorkerContractError:
        build = {}
        errors.append("training_worker_build_receipt_invalid")
    try:
        smoke = build_worker_smoke_receipt(worker_smoke_receipt)
    except ReconstructionWorkerContractError:
        smoke = {}
        errors.append("training_worker_smoke_receipt_invalid")

    dataset_digest = dataset.get("colmap_training_dataset_export_result_digest")
    if (
        dataset.get("schema_version") != "colmap_training_dataset_export_result.v1"
        or not _digest(dataset_digest)
        or dataset_digest
        != canonical_digest(dataset, digest_field="colmap_training_dataset_export_result_digest")
    ):
        errors.append("training_dataset_export_receipt_invalid")
    if (
        dataset.get("status") != "exported_candidate_only_colmap_text_dataset"
        or dataset.get("hidden_heldout_pixels_included") is not False
        or dataset.get("trainer_self_grading_permitted") is not False
        or dataset.get("raw_input_poses_modified") is not False
    ):
        errors.append("training_dataset_candidate_isolation_invalid")
    observation_ids = dataset.get("observation_ids")
    if (
        not isinstance(observation_ids, list)
        or not observation_ids
        or any(not isinstance(value, str) or not value for value in observation_ids)
        or len(set(observation_ids)) != len(observation_ids)
    ):
        errors.append("training_dataset_observation_ledger_invalid")

    capture_digest = capture.get("source_capture_digest")
    if not _digest(capture_digest) or dataset.get("source_capture_digest") != capture_digest:
        errors.append("training_source_capture_binding_invalid")
    original_files = capture.get("original_file_references")
    if (
        not isinstance(original_files, list)
        or not original_files
        or any(
            not isinstance(row, Mapping)
            or not str(row.get("artifact_id") or row.get("relative_path") or "")
            or not _digest(row.get("digest"))
            for row in original_files or []
        )
    ):
        errors.append("training_original_file_lineage_invalid")

    image = build.get("resolved_image_digest")
    if (
        build.get("status") != "built"
        or build.get("blockers") != []
        or build.get("worker_stack_manifest_digest") != stack.get("worker_stack_manifest_digest")
        or build.get("source_commit_sha") != stack.get("source_commit_sha")
    ):
        errors.append("training_worker_build_not_accepted")
    if (
        smoke.get("status") != "passed"
        or smoke.get("build_receipt_digest") != build.get("build_receipt_digest")
        or smoke.get("resolved_image_digest") != image
        or smoke.get("source_commit_sha") != build.get("source_commit_sha")
        or smoke.get("display_attached") is not False
    ):
        errors.append("training_worker_smoke_not_accepted")

    camera_digest = dataset.get("camera_observation_digest")
    initialization_digest = dataset.get("initialization_surface_digest")
    pose_digest = pose.get("pose_result_digest")
    if not _digest(camera_digest) or not _digest(initialization_digest):
        errors.append("training_dataset_camera_or_initialization_binding_missing")
    if pose.get("source_capture_digest") != capture_digest or not _digest(pose_digest):
        errors.append("training_pose_binding_invalid")
    if pose.get("train_heldout_split_digest") != dataset.get("frozen_split_digest"):
        errors.append("training_pose_split_binding_invalid")
    if pose.get("raw_capture_poses_modified") is not False:
        errors.append("training_pose_truth_mutation_forbidden")
    if pose.get("binding_kind") == "unrefined_camera_observations" and pose_digest != camera_digest:
        errors.append("training_unrefined_pose_digest_mismatch")
    if pose.get("binding_kind") not in {
        "unrefined_camera_observations",
        "qualified_pose_refinement_result",
    }:
        errors.append("training_pose_binding_kind_invalid")

    evaluation_digest = evaluation.get("evaluation_contract_digest")
    if (
        not _digest(evaluation_digest)
        or evaluation.get("source_capture_digest") != capture_digest
        or evaluation.get("train_heldout_split_digest") != dataset.get("frozen_split_digest")
        or evaluation.get("candidate_hidden_pixel_access_permitted") is not False
        or evaluation.get("candidate_self_grading_permitted") is not False
        or evaluation.get("split_mutation_permitted") is not False
    ):
        errors.append("training_evaluation_contract_invalid")

    provider = configuration.get("provider_runtime_identity")
    provider = provider if isinstance(provider, Mapping) else {}
    smoke_provider = smoke.get("provider_runtime_identity")
    if (
        not str(provider.get("provider") or "").strip()
        or not str(provider.get("runtime") or "").strip()
        or not isinstance(smoke_provider, Mapping)
        or dict(smoke_provider) != dict(provider)
    ):
        errors.append("training_provider_runtime_binding_invalid")
    is_remote = provider.get("provider") not in {None, "local"}
    max_spend = authority.get("max_spend_usd")
    spend_cap = configuration.get("spend_cap_usd")
    hard_ttl = authority.get("hard_ttl_seconds")
    timeout = configuration.get("timeout_seconds")
    retry_cap = authority.get("retry_cap")
    if (
        not str(authority.get("authority_id") or "").strip()
        or authority.get("paid_compute_authorized") is not True
        or not _finite(max_spend, minimum=0.01)
        or not _finite(spend_cap, minimum=0.01)
        or float(spend_cap or 0) > float(max_spend or 0)
        or isinstance(hard_ttl, bool)
        or not isinstance(hard_ttl, int)
        or hard_ttl <= 0
        or not _finite(timeout, minimum=1)
        or float(timeout or 0) > hard_ttl
        or isinstance(retry_cap, bool)
        or not isinstance(retry_cap, int)
        or retry_cap < 0
    ):
        errors.append("training_execution_authority_invalid")
    if is_remote and (
        authority.get("provider_processing_authorized") is not True
        or authority.get("provider_upload_authorized") is not True
    ):
        errors.append("training_remote_processing_authority_missing")
    capture_authority = capture.get("authority_used")
    if not isinstance(capture_authority, Mapping):
        errors.append("training_capture_authority_invalid")
    if is_remote and (
        not isinstance(capture_authority, Mapping)
        or capture_authority.get("provider_upload_authorized") is not True
    ):
        errors.append("training_capture_provider_upload_authority_missing")

    if errors:
        raise ReconstructionTrainingRequestCompilationError(errors)

    assert isinstance(original_files, list)
    assert isinstance(capture_authority, Mapping)
    normalized_original_files = [
        {
            "artifact_id": str(row.get("artifact_id") or row.get("relative_path")),
            "digest": row["digest"],
        }
        for row in original_files
    ]
    deterministic_configuration_digest = canonical_digest(
        {
            "execution_configuration": configuration,
            "execution_authority_bounds": {
                "max_spend_usd": max_spend,
                "hard_ttl_seconds": hard_ttl,
                "retry_cap": retry_cap,
            },
            "dataset_export_result_digest": dataset_digest,
            "worker_build_receipt_digest": build["build_receipt_digest"],
            "worker_smoke_receipt_digest": smoke["smoke_test_receipt_digest"],
            "pose_result_digest": pose_digest,
            "evaluation_contract_digest": evaluation_digest,
        }
    )
    request = {
        "stable_run_identity": stable_run_identity,
        "source_capture_identity": capture.get("source_capture_identity"),
        "source_capture_digest": capture_digest,
        "original_file_references": normalized_original_files,
        "producing_method": "blueprint.reconstruction_training_request_compiler",
        "implementation_version": "1.0.0",
        "container_image_digest": image,
        "source_commit_sha": build["source_commit_sha"],
        "deterministic_configuration_digest": deterministic_configuration_digest,
        "input_digests": [
            {
                "artifact_id": "colmap_training_dataset",
                "digest": dataset["colmap_training_dataset_digest"],
            },
            {"artifact_id": "dataset_export_result", "digest": dataset_digest},
            {"artifact_id": "initialization_geometry", "digest": initialization_digest},
            {"artifact_id": "pose_binding", "digest": pose_digest},
            {"artifact_id": "evaluation_contract", "digest": evaluation_digest},
            {"artifact_id": "worker_build_receipt", "digest": build["build_receipt_digest"]},
            {"artifact_id": "worker_smoke_receipt", "digest": smoke["smoke_test_receipt_digest"]},
        ],
        "output_digests": [],
        "train_heldout_split_digest": dataset["frozen_split_digest"],
        "camera_calibration_binding": {"calibration_digest": camera_digest},
        "coordinate_frame_declaration": dataset.get("coordinate_frame_declaration"),
        "units": dataset.get("units"),
        "metric_scale_status": dataset.get("metric_scale_status"),
        "provider_runtime_identity": dict(provider),
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": {
            **authority,
            "capture_authority": dict(capture_authority),
        },
        "warnings": list(dataset.get("warnings") or []),
        "blockers": [],
        "proof_effect": "none",
        "claim_ceiling": "request_only",
        "parent_artifact_or_event": {
            "dataset_export_result_digest": dataset_digest,
            "worker_build_receipt_digest": build["build_receipt_digest"],
            "worker_smoke_receipt_digest": smoke["smoke_test_receipt_digest"],
        },
        "timestamp": timestamp,
        "method_profile_id": configuration.get("method_profile_id"),
        "reconstruction_dataset_digest": dataset["colmap_training_dataset_digest"],
        "calibration_digest": camera_digest,
        "initialization_geometry_digest": initialization_digest,
        "pose_result_digest": pose_digest,
        "worker_stack_manifest_digest": stack["worker_stack_manifest_digest"],
        "worker_build_receipt_digest": build["build_receipt_digest"],
        "worker_smoke_receipt_digest": smoke["smoke_test_receipt_digest"],
        "dataset_export_result_digest": dataset_digest,
        "evaluation_contract_digest": evaluation_digest,
        "camera_model": configuration.get("camera_model"),
        "densification_configuration": configuration.get("densification_configuration"),
        "random_seed": configuration.get("random_seed"),
        "iteration_budget": configuration.get("iteration_budget"),
        "resource_request": configuration.get("resource_request"),
        "timeout_seconds": timeout,
        "hard_ttl_seconds": hard_ttl,
        "retry_cap": retry_cap,
        "spend_cap_usd": spend_cap,
        "output_contract": configuration.get("output_contract"),
        "candidate_dataset_contains_hidden_heldout_pixels": False,
        "candidate_can_change_split": False,
        "candidate_may_read_hidden_heldout": False,
        "trainer_may_grade_heldout": False,
    }
    try:
        return build_training_request(request)
    except ReconstructionWorkerContractError as exc:
        raise ReconstructionTrainingRequestCompilationError(
            ["compiled_training_request_contract_invalid", *exc.codes]
        ) from exc


__all__ = [
    "ReconstructionTrainingRequestCompilationError",
    "compile_reconstruction_training_request",
]
