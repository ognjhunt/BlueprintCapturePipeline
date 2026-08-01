from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.arkit_raw_contract_validation import (
    ArkitRawContractValidationError,
    build_arkit_raw_contract_validation,
    validate_arkit_raw_contract_validation,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _build() -> dict:
    return build_arkit_raw_contract_validation(
        intake_id="intake-arkit-1",
        source_capture_digest="sha256:" + "a" * 64,
        source_artifact_digests={
            "walkthrough.mov": "sha256:" + "b" * 64,
            "sync_map.jsonl": "sha256:" + "c" * 64,
            "video_frame_retention.jsonl": "sha256:" + "d" * 64,
        },
        implementation_digest="sha256:" + "e" * 64,
        source_commit_sha="f" * 40,
        runtime_identity="ffmpeg-fixture-7.1.1",
        runtime_digest="sha256:" + "1" * 64,
        frozen_split_digest="sha256:" + "2" * 64,
        metric_scaffold_digest="sha256:" + "3" * 64,
        reconstruction_dataset_export_digest="sha256:" + "4" * 64,
        coordinate_frame_declaration={
            "units": "meters",
            "handedness": "right_handed",
            "gravity_aligned": True,
        },
        retained_frame_count=2,
        dropped_attempt_count=1,
        depth_confidence_pair_count=1,
        authority_used={"local_processing_authorized": True},
        timestamp="2026-08-01T12:00:00Z",
    )


def test_arkit_raw_contract_receipt_is_replayable_and_schema_valid() -> None:
    first = _build()
    second = _build()

    assert first == second
    assert first["claim_ceiling"] == "calibrated_camera_trajectory"
    assert first["metric_scale_proven"] is False
    assert first["metric_geometry_proven"] is False
    assert first["collision_geometry_proven"] is False
    assert first["isaac_compatibility_proven"] is False
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/arkit_raw_contract_validation.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(schema).validate(first)


def test_rehashed_raw_contract_proof_upgrade_is_rejected() -> None:
    tampered = _build()
    tampered["metric_scale_proven"] = True
    tampered["arkit_raw_contract_validation_digest"] = canonical_digest(
        tampered, digest_field="arkit_raw_contract_validation_digest"
    )

    with pytest.raises(ArkitRawContractValidationError, match="result_invalid"):
        validate_arkit_raw_contract_validation(tampered)
