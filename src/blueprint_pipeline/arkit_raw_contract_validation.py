"""Replayable receipt for strict Blueprint ARKit Raw Contract 3.2 validation."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

from .decision_evidence_contracts import canonical_digest, canonical_json


SCHEMA_VERSION = "arkit_raw_contract_validation.v1"


class ArkitRawContractValidationError(ValueError):
    def __init__(self, codes: Sequence[str]) -> None:
        self.codes = tuple(sorted(set(str(code) for code in codes if str(code))))
        super().__init__("; ".join(self.codes))


def _digest(value: Any) -> bool:
    return re.fullmatch(r"sha256:[0-9a-f]{64}", str(value or "")) is not None


def build_arkit_raw_contract_validation(
    *,
    intake_id: str,
    source_capture_digest: str,
    source_artifact_digests: Mapping[str, str],
    implementation_digest: str,
    source_commit_sha: str,
    runtime_identity: str,
    runtime_digest: str,
    frozen_split_digest: str,
    metric_scaffold_digest: str,
    reconstruction_dataset_export_digest: str,
    coordinate_frame_declaration: Mapping[str, Any],
    retained_frame_count: int,
    dropped_attempt_count: int,
    depth_confidence_pair_count: int,
    authority_used: Mapping[str, Any],
    timestamp: str,
) -> dict[str, Any]:
    """Build a claim-limited receipt from already validated raw-contract facts."""

    sources = [
        {"relative_path": str(path), "digest": str(digest)}
        for path, digest in sorted(source_artifact_digests.items())
    ]
    if (
        not str(intake_id).strip()
        or not _digest(source_capture_digest)
        or not sources
        or any(not row["relative_path"] or not _digest(row["digest"]) for row in sources)
        or not _digest(implementation_digest)
        or re.fullmatch(r"[0-9a-f]{40}", source_commit_sha) is None
        or not str(runtime_identity).strip()
        or not _digest(runtime_digest)
        or any(
            not _digest(value)
            for value in (
                frozen_split_digest,
                metric_scaffold_digest,
                reconstruction_dataset_export_digest,
            )
        )
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (
                retained_frame_count,
                dropped_attempt_count,
                depth_confidence_pair_count,
            )
        )
    ):
        raise ArkitRawContractValidationError(["arkit_raw_contract_request_invalid"])
    configuration_digest = canonical_digest(
        {
            "source_capture_digest": source_capture_digest,
            "source_artifacts": sources,
            "implementation_digest": implementation_digest,
            "source_commit_sha": source_commit_sha,
            "runtime_digest": runtime_digest,
            "frozen_split_digest": frozen_split_digest,
            "metric_scaffold_digest": metric_scaffold_digest,
        }
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "stable_run_identity": f"arkit-raw-contract-{configuration_digest[7:31]}",
        "source_capture_identity": intake_id,
        "source_capture_digest": source_capture_digest,
        "capture_profile": "iphone_arkit_lidar",
        "raw_contract_version": "3.2.0",
        "original_file_references_and_digests": sources,
        "producing_method": "blueprint.arkit_raw_contract_validator.v1",
        "implementation_version": implementation_digest,
        "container_image_digest": None,
        "source_commit_sha": source_commit_sha,
        "deterministic_configuration_digest": configuration_digest,
        "input_digests": sorted(row["digest"] for row in sources),
        "output_digests": sorted(
            {metric_scaffold_digest, reconstruction_dataset_export_digest}
        ),
        "train_heldout_split_digest": frozen_split_digest,
        "camera_calibration_binding": metric_scaffold_digest,
        "coordinate_frame_declaration": dict(coordinate_frame_declaration),
        "units": "meters",
        "metric_scale_status": "sensor_declared_not_independently_validated",
        "provider_runtime_identity": {
            "provider": "local",
            "runtime_identity": runtime_identity,
            "runtime_digest": runtime_digest,
        },
        "cost_usd": 0.0,
        "duration_seconds": 0.0,
        "authority_used": dict(authority_used),
        "warnings": [
            "metric_scale_requires_independent_validation",
            "depth_confidence_filtering_and_rgb_alignment_remain_separate_gates",
        ],
        "blockers": [],
        "proof_effect": "raw_contract_and_calibrated_trajectory_only",
        "claim_ceiling": "calibrated_camera_trajectory",
        "parent_artifact_or_event": {
            "metric_scaffold_digest": metric_scaffold_digest,
            "reconstruction_dataset_export_digest": reconstruction_dataset_export_digest,
        },
        "timestamp": str(timestamp),
        "retained_frame_count": retained_frame_count,
        "dropped_attempt_count": dropped_attempt_count,
        "depth_confidence_pair_count": depth_confidence_pair_count,
        "decoded_pts_verified": True,
        "retention_mapping_verified": True,
        "poses_intrinsics_bound": True,
        "depth_confidence_presence_bound": True,
        "raw_capture_remains_authoritative": True,
        "agent_changed_profile_or_evidence": False,
        "metric_scale_proven": False,
        "metric_geometry_proven": False,
        "collision_geometry_proven": False,
        "isaac_compatibility_proven": False,
        "physical_success_proven": False,
        "deployment_readiness_proven": False,
    }
    result["arkit_raw_contract_validation_digest"] = canonical_digest(
        result, digest_field="arkit_raw_contract_validation_digest"
    )
    return validate_arkit_raw_contract_validation(result)


def validate_arkit_raw_contract_validation(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(canonical_json(dict(value)))
    digest = result.get("arkit_raw_contract_validation_digest")
    sources = result.get("original_file_references_and_digests")
    forbidden_true = (
        "agent_changed_profile_or_evidence",
        "metric_scale_proven",
        "metric_geometry_proven",
        "collision_geometry_proven",
        "isaac_compatibility_proven",
        "physical_success_proven",
        "deployment_readiness_proven",
    )
    if (
        result.get("schema_version") != SCHEMA_VERSION
        or result.get("capture_profile") != "iphone_arkit_lidar"
        or result.get("raw_contract_version") != "3.2.0"
        or not _digest(result.get("source_capture_digest"))
        or not isinstance(sources, list)
        or not sources
        or any(
            not isinstance(row, Mapping)
            or not str(row.get("relative_path") or "")
            or not _digest(row.get("digest"))
            for row in sources
        )
        or result.get("decoded_pts_verified") is not True
        or result.get("retention_mapping_verified") is not True
        or result.get("poses_intrinsics_bound") is not True
        or result.get("depth_confidence_presence_bound") is not True
        or result.get("raw_capture_remains_authoritative") is not True
        or any(result.get(key) is not False for key in forbidden_true)
        or result.get("proof_effect") != "raw_contract_and_calibrated_trajectory_only"
        or result.get("claim_ceiling") != "calibrated_camera_trajectory"
        or result.get("metric_scale_status")
        != "sensor_declared_not_independently_validated"
        or digest
        != canonical_digest(result, digest_field="arkit_raw_contract_validation_digest")
    ):
        raise ArkitRawContractValidationError(["arkit_raw_contract_result_invalid"])
    return result


__all__ = [
    "ArkitRawContractValidationError",
    "SCHEMA_VERSION",
    "build_arkit_raw_contract_validation",
    "validate_arkit_raw_contract_validation",
]
