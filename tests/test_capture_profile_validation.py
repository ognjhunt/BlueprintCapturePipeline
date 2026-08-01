from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from blueprint_pipeline.capture_profile_validation import (
    CaptureProfileValidationError,
    build_capture_profile_validation,
    validate_capture_profile_validation,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_360_normalization import build_native_360_probe_receipt


CAPTURE_DIGEST = "sha256:" + "a" * 64
IMPLEMENTATION_DIGEST = "sha256:" + "b" * 64
SOURCE_COMMIT = "c" * 40
RUNTIME_DIGEST = "sha256:" + "d" * 64


def _probe(*, lane: str, source_digit: str = "e") -> dict:
    return build_native_360_probe_receipt(
        source_file_digest="sha256:" + source_digit * 64,
        runtime_identity="ffprobe-fixture-7.1.1",
        runtime_digest=RUNTIME_DIGEST,
        streams=[
            {
                "stream_index": 0,
                "media_type": "video",
                "codec_name": "h264",
                "width": 3840,
                "height": 1920,
                "time_base": "1/50000",
                "pts_seconds": [0.0, 0.02],
                "metadata": {},
            }
        ],
        format_metadata={
            "compatible_processing_lane": lane,
            "processing_lane_claim_ceiling": "container_stream_topology_only",
            "capture_profile_fully_validated": False,
        },
    )


def _native_normalization() -> dict:
    value = {
        "schema_version": "native_360_capture_normalization.v1",
        "source_capture_digest": CAPTURE_DIGEST,
        "status": "normalized",
        "blockers": [],
        "claim_ceiling": "calibrated_camera_rig",
        "proof_effect": "calibrated_native_360_rig_only",
    }
    value["native_360_normalization_digest"] = canonical_digest(
        value, digest_field="native_360_normalization_digest"
    )
    return value


def _build(*, declared: str, lane: str, normalization: dict | None = None) -> dict:
    return build_capture_profile_validation(
        source_capture_digest=CAPTURE_DIGEST,
        declared_capture_authority_profile=declared,
        probe_receipts=[_probe(lane=lane)],
        native_normalization_result=normalization,
        source_commit_sha=SOURCE_COMMIT,
        implementation_digest=IMPLEMENTATION_DIGEST,
        timestamp="2026-08-01T12:00:00-05:00",
        parent_artifact_or_event={"capture_intake_digest": "sha256:" + "f" * 64},
    )


def test_stitched_profile_validation_is_replayable_and_schema_valid() -> None:
    first = _build(
        declared="camera_360_equirectangular",
        lane="camera_360_equirectangular",
    )
    second = _build(
        declared="camera_360_equirectangular",
        lane="camera_360_equirectangular",
    )

    assert first == second
    assert first["validation_status"] == "validated"
    assert first["compatible_capture_authority_profile"] == (
        "camera_360_equirectangular"
    )
    assert first["blockers"] == []
    assert first["agent_selected_capture_profile"] is False
    assert first["agent_may_change_capture_profile"] is False
    assert first["proof_effect"] == "capture_profile_validation_only"
    assert first["capture_profile_routing_binding_digest"].startswith("sha256:")
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/capture_profile_validation.v1.schema.json"
        ).read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    ).validate(first)


def test_declared_profile_conflict_blocks_without_agent_rerouting() -> None:
    result = _build(
        declared="camera_360_native",
        lane="camera_360_equirectangular",
    )

    assert result["validation_status"] == "blocked"
    assert result["declared_capture_authority_profile"] == "camera_360_native"
    assert result["compatible_capture_authority_profile"] == (
        "camera_360_equirectangular"
    )
    assert result["blockers"] == [
        "declared_profile_conflicts_with_observed_topology"
    ]
    assert result["proof_effect"] == "none"
    assert result["legal_next_actions"] == [
        "preserve_evidence_and_stop",
        "request_corrected_capture_intake",
    ]


def test_native_profile_is_selected_before_separate_rig_calibration() -> None:
    pending = _build(
        declared="camera_360_native",
        lane="camera_360_native_candidate_requires_calibration",
    )
    accepted = _build(
        declared="camera_360_native",
        lane="camera_360_native_candidate_requires_calibration",
        normalization=_native_normalization(),
    )

    assert pending["validation_status"] == "validated"
    assert pending["blockers"] == []
    assert pending["warnings"] == ["native_360_rig_calibration_pending"]
    assert pending["native_normalization_digest"] is None
    assert pending["claim_ceiling"] == "capture_profile_compatibility"
    assert accepted["validation_status"] == "validated"
    assert accepted["warnings"] == []
    assert accepted["native_normalization_digest"] == _native_normalization()[
        "native_360_normalization_digest"
    ]


def test_mixed_topology_abstains_and_tampered_inputs_fail_closed() -> None:
    mixed = build_capture_profile_validation(
        source_capture_digest=CAPTURE_DIGEST,
        declared_capture_authority_profile="camera_360_equirectangular",
        probe_receipts=[
            _probe(lane="camera_360_equirectangular", source_digit="e"),
            _probe(
                lane="camera_360_native_candidate_requires_calibration",
                source_digit="f",
            ),
        ],
        source_commit_sha=SOURCE_COMMIT,
        implementation_digest=IMPLEMENTATION_DIGEST,
        timestamp="2026-08-01T12:00:00Z",
    )
    assert mixed["validation_status"] == "blocked"
    assert mixed["compatible_capture_authority_profile"] is None
    assert mixed["blockers"] == ["unsupported_or_ambiguous_360_stream_topology"]

    tampered_probe = _probe(lane="camera_360_equirectangular")
    tampered_probe["format_metadata"]["compatible_processing_lane"] = (
        "camera_360_native_candidate_requires_calibration"
    )
    with pytest.raises(CaptureProfileValidationError, match="probe_receipt_invalid"):
        build_capture_profile_validation(
            source_capture_digest=CAPTURE_DIGEST,
            declared_capture_authority_profile="camera_360_native",
            probe_receipts=[tampered_probe],
            source_commit_sha=SOURCE_COMMIT,
            implementation_digest=IMPLEMENTATION_DIGEST,
            timestamp="2026-08-01T12:00:00Z",
        )

    tampered_result = dict(mixed)
    tampered_result["blockers"] = []
    with pytest.raises(CaptureProfileValidationError, match="result_invalid"):
        validate_capture_profile_validation(tampered_result)


def test_stitched_profile_rejects_native_normalization_input() -> None:
    with pytest.raises(
        CaptureProfileValidationError,
        match="native_normalization_incompatible_with_stitched_topology",
    ):
        _build(
            declared="camera_360_equirectangular",
            lane="camera_360_equirectangular",
            normalization=_native_normalization(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_commit_sha", "f" * 40),
        ("compatible_capture_authority_profile", "camera_360_native"),
        ("native_normalization_digest", "sha256:" + "9" * 64),
        ("legal_next_actions", ["compile_profile_specific_reconstruction_plan", "retry"]),
    ],
)
def test_profile_validation_rejects_rehashed_semantic_tampering(
    field: str, value: object
) -> None:
    result = _build(
        declared="camera_360_equirectangular",
        lane="camera_360_equirectangular",
    )
    result[field] = value
    result["capture_profile_validation_digest"] = canonical_digest(
        result, digest_field="capture_profile_validation_digest"
    )

    with pytest.raises(CaptureProfileValidationError, match="result_invalid"):
        validate_capture_profile_validation(result)
