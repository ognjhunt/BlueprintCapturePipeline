from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.capture_v32_candidate_admission import (
    CaptureV32CandidateAdmissionError,
    MISSING_SELECTION_PROFILE,
    build_capture_v32_reconstruction_admission,
    validate_capture_v32_candidate_manifest,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


FIXTURE = Path(__file__).parent / "fixtures" / "capture_v32_downstream_candidate_manifest.json"


def _manifest() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _profile(
    manifest: dict,
    *,
    selector: str = "explicit_encoded_frame_ordinals",
    **parameters: object,
) -> dict:
    profile = {
        "schema_version": "task_site_frame_selection_profile.v1",
        "site_id": "site-warehouse-a",
        "task_id": "task-pallet-transfer",
        "candidate_manifest_digest": manifest["manifest_digest"],
        "selector": selector,
        "parameters": parameters or {"encoded_frame_ordinals": [0]},
        "rights_authorization": {
            "status": "authorized",
            "latest_revocation_check_status": "clear",
            "rights_evidence_digest": "sha256:" + "b" * 64,
        },
    }
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    return profile


def _manifest_with_encoded_frame_gap() -> dict:
    manifest = _manifest()
    second = json.loads(json.dumps(manifest["candidates"][0]))
    second.update(
        {
            "candidate_id": "rgb_000001",
            "output_image_relative_path": "candidate_rgb/000001.png",
            "decoded_frame_ordinal": 1,
            "encoded_frame_index": 7,
            "decoded_pts_sec": 1.0,
            "decoded_source_pts_sec": 101.25,
            "t_capture_sec": 1.0,
            "frame_id": "000008",
            "pose_frame_id": "000008",
            "arkit_frame_row_ordinal": 7,
            "arkit_pose_row_ordinal": 7,
        }
    )
    manifest["candidates"].append(second)
    manifest["candidate_count"] = 2
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    return manifest


def test_versioned_capture_fixture_validates_with_exact_source_binding() -> None:
    manifest = validate_capture_v32_candidate_manifest(
        _manifest(),
        expected_source_video_digest="sha256:" + "a" * 64,
        expected_source_video_uri="raw/walkthrough.mov",
        expected_coordinate_frame_session_id="cfs_fixture_001",
    )
    assert manifest["candidate_count"] == 1
    assert manifest["candidates"][0]["frame_id"] == "000001"


def test_missing_task_site_profile_abstains_with_smallest_missing_input() -> None:
    admission = build_capture_v32_reconstruction_admission(
        candidate_manifest=_manifest(), task_site_selection_profile=None
    )
    assert admission["status"] == "abstained"
    assert admission["blockers"] == [MISSING_SELECTION_PROFILE]
    assert admission["selected_candidates"] == []
    assert admission["provider_selected"] is None
    assert admission["provider_upload_authorized"] is False


def test_explicit_task_site_profile_admits_exact_candidate_without_provider_selection() -> None:
    manifest = _manifest()
    admission = build_capture_v32_reconstruction_admission(
        candidate_manifest=manifest,
        task_site_selection_profile=_profile(manifest),
    )
    assert admission["status"] == "admitted"
    assert [row["candidate_id"] for row in admission["selected_candidates"]] == ["rgb_000000"]
    assert admission["next_stage"] == "materialize_exact_decoded_candidate_images"
    assert admission["claim_ceiling"] == "retained_observation_selection_only"


def test_explicit_selector_uses_encoded_frame_identity_not_list_offset() -> None:
    manifest = _manifest_with_encoded_frame_gap()
    admission = build_capture_v32_reconstruction_admission(
        candidate_manifest=manifest,
        task_site_selection_profile=_profile(
            manifest, encoded_frame_ordinals=[7]
        ),
    )
    assert [row["candidate_id"] for row in admission["selected_candidates"]] == [
        "rgb_000001"
    ]


def test_quality_selector_abstains_when_profile_qualifies_no_observation() -> None:
    manifest = _manifest()
    profile = _profile(
        manifest,
        selector="profile_bound_quality_filter",
        allowed_tracking_states=["limited"],
        require_pose_assisted_eligible=True,
        exclude_relocalization_events=True,
        maximum_frames=4,
    )
    admission = build_capture_v32_reconstruction_admission(
        candidate_manifest=manifest,
        task_site_selection_profile=profile,
    )
    assert admission["status"] == "abstained"
    assert admission["blockers"] == [
        "task_site_frame_selection_no_candidates_qualified"
    ]
    assert admission["selected_candidates"] == []


def test_provider_authorization_and_digest_drift_fail_closed() -> None:
    manifest = _manifest()
    manifest["provider_neutrality"]["third_party_provider_upload_authorized"] = True
    with pytest.raises(CaptureV32CandidateAdmissionError) as exc:
        validate_capture_v32_candidate_manifest(manifest)
    assert "capture_v32_candidate_provider_neutrality_invalid" in exc.value.codes
    assert "capture_v32_candidate_digest_mismatch" in exc.value.codes


def test_capture_contract_authority_drift_fails_closed() -> None:
    manifest = _manifest()
    manifest["selection_contract"]["capture_default_selection"] = [0]
    manifest["claim_boundary"]["candidate_manifest_qualifies_reconstruction"] = True
    manifest["candidates"][0]["camera_intrinsics"]["authority"] = "nominal_camera"
    manifest["manifest_digest"] = canonical_digest(manifest, digest_field="manifest_digest")
    with pytest.raises(CaptureV32CandidateAdmissionError) as exc:
        validate_capture_v32_candidate_manifest(manifest)
    assert "capture_v32_candidate_selection_contract_invalid" in exc.value.codes
    assert "capture_v32_candidate_claim_boundary_invalid" in exc.value.codes
    assert "capture_v32_candidate_intrinsics_invalid:0" in exc.value.codes


def test_profile_must_be_digest_bound_to_exact_capture_manifest() -> None:
    manifest = _manifest()
    profile = _profile(manifest)
    profile["candidate_manifest_digest"] = "sha256:" + "f" * 64
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    with pytest.raises(CaptureV32CandidateAdmissionError) as exc:
        build_capture_v32_reconstruction_admission(
            candidate_manifest=manifest,
            task_site_selection_profile=profile,
        )
    assert "task_site_frame_selection_profile_capture_binding_mismatch" in exc.value.codes


def test_profile_requires_latest_authoritative_revocation_check() -> None:
    manifest = _manifest()
    profile = _profile(manifest)
    profile.pop("rights_authorization")
    profile["profile_digest"] = canonical_digest(profile, digest_field="profile_digest")
    with pytest.raises(CaptureV32CandidateAdmissionError) as exc:
        build_capture_v32_reconstruction_admission(
            candidate_manifest=manifest,
            task_site_selection_profile=profile,
        )
    assert "latest_authoritative_revocation_check_required" in exc.value.codes
