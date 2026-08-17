"""Contract tests for the capture -> canonical 3DGS launch dispatcher.

The dispatcher is the only automatic bridge from an admitted production capture
to the canonical 3DGS campaign.  It must mint a per-capture frame-selection
profile from a pre-registered site/task policy, refuse to invent selection
authority, and enqueue exactly once per capture digest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.capture_reconstruction_launch_dispatcher import (
    LAUNCH_REQUEST_SCHEMA_VERSION,
    MISSING_POLICY,
    POLICY_SCHEMA_VERSION,
    SELECTION_PROFILE_SCHEMA_VERSION,
    CaptureReconstructionLaunchError,
    build_launch_request,
    mint_frame_selection_profile,
    process_launch_queue,
    resolve_site_task_policy,
    stage_launch_request,
    validate_launch_request,
    validate_site_task_reconstruction_policy,
)
from blueprint_pipeline.capture_v32_candidate_admission import (
    _validate_profile as validate_selection_profile_against_admission_gate,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _sha(seed: str) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest()


def _policy(**overrides: object) -> dict:
    policy = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "policy_id": "policy-warehouse-a",
        "site_id": "site-warehouse-a",
        "task_id": "task-shelf-restock",
        "selector": "profile_bound_quality_filter",
        "parameters": {
            "allowed_tracking_states": ["normal"],
            "require_pose_assisted_eligible": True,
            "exclude_relocalization_events": True,
        },
        "rights_authority": {
            "rights_profile": "operator_authorized_commercial",
            "rights_evidence_digest": _sha("rights"),
        },
        "arms": ["postshot-primary"],
        "max_spend_usd": 10.0,
        "hard_ttl_seconds": 5400,
        "authority_id": "authority-founder-20260817",
    }
    policy.update(overrides)  # type: ignore[arg-type]
    policy["policy_digest"] = canonical_digest(policy, digest_field="policy_digest")
    return policy


def _request(**overrides: object) -> dict:
    policy = _policy()
    request = build_launch_request(
        policy=policy,
        capture_id="capture-live-001",
        scene_id="scene-live-001",
        intake_id="intake-live-001",
        capture_digest=_sha("capture"),
        candidate_manifest_digest=_sha("manifest"),
        capture_root_uri="gs://blueprint-captures/scenes/scene-live-001/captures/capture-live-001",
        source_commit_sha="a" * 40,
        rights_evidence_digest=_sha("rights"),
        revocation_check_status="clear",
        requested_at="2026-08-17T00:00:00Z",
    )
    request.update(overrides)  # type: ignore[arg-type]
    if "request_digest" in request:
        del request["request_digest"]
    request["request_digest"] = canonical_digest(request, digest_field="request_digest")
    return request


# --------------------------------------------------------------------------
# Policy validation: selection authority is never invented
# --------------------------------------------------------------------------


def test_valid_policy_has_no_blockers() -> None:
    assert validate_site_task_reconstruction_policy(_policy()) == []


def test_policy_requires_known_selector() -> None:
    blockers = validate_site_task_reconstruction_policy(
        _policy(selector="whatever_the_dispatcher_feels_like")
    )
    assert "site_task_reconstruction_policy_selector_invalid" in blockers


def test_policy_digest_must_bind_its_own_content() -> None:
    policy = _policy()
    policy["max_spend_usd"] = 999.0
    blockers = validate_site_task_reconstruction_policy(policy)
    assert "site_task_reconstruction_policy_digest_mismatch" in blockers


def test_absent_policy_abstains_and_never_defaults(tmp_path: Path) -> None:
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        resolve_site_task_policy(
            policy_root=tmp_path,
            site_id="site-warehouse-a",
            task_id="task-shelf-restock",
        )
    assert MISSING_POLICY in str(excinfo.value)


def test_policy_resolves_by_exact_site_and_task(tmp_path: Path) -> None:
    policy = _policy()
    (tmp_path / "policy-warehouse-a.json").write_text(json.dumps(policy), encoding="utf-8")
    resolved = resolve_site_task_policy(
        policy_root=tmp_path,
        site_id="site-warehouse-a",
        task_id="task-shelf-restock",
    )
    assert resolved["policy_digest"] == policy["policy_digest"]


def test_policy_for_a_different_task_does_not_satisfy_this_capture(tmp_path: Path) -> None:
    (tmp_path / "other.json").write_text(json.dumps(_policy()), encoding="utf-8")
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        resolve_site_task_policy(
            policy_root=tmp_path,
            site_id="site-warehouse-a",
            task_id="task-something-else",
        )
    assert MISSING_POLICY in str(excinfo.value)


# --------------------------------------------------------------------------
# Per-capture minting: the profile binds THIS capture's manifest
# --------------------------------------------------------------------------


def test_minted_profile_binds_candidate_manifest_digest() -> None:
    manifest_digest = _sha("manifest")
    profile = mint_frame_selection_profile(
        policy=_policy(),
        candidate_manifest_digest=manifest_digest,
        rights_evidence_digest=_sha("rights"),
        revocation_check_status="clear",
    )
    assert profile["schema_version"] == SELECTION_PROFILE_SCHEMA_VERSION
    assert profile["candidate_manifest_digest"] == manifest_digest
    assert profile["profile_digest"] == canonical_digest(
        {k: v for k, v in profile.items() if k != "profile_digest"},
        digest_field="profile_digest",
    )


def test_minted_profile_is_deterministic() -> None:
    kwargs = {
        "policy": _policy(),
        "candidate_manifest_digest": _sha("manifest"),
        "rights_evidence_digest": _sha("rights"),
        "revocation_check_status": "clear",
    }
    assert mint_frame_selection_profile(**kwargs) == mint_frame_selection_profile(**kwargs)


def test_minted_profile_differs_per_capture() -> None:
    common = {
        "policy": _policy(),
        "rights_evidence_digest": _sha("rights"),
        "revocation_check_status": "clear",
    }
    first = mint_frame_selection_profile(candidate_manifest_digest=_sha("a"), **common)
    second = mint_frame_selection_profile(candidate_manifest_digest=_sha("b"), **common)
    assert first["profile_digest"] != second["profile_digest"]


def test_minted_profile_is_accepted_by_the_real_admission_gate() -> None:
    """The minted profile must satisfy the existing consumer, not a parallel one.

    This binds the producer (dispatcher) to the exact gate the canonical 3DGS
    preparation runs, so the two cannot drift apart silently.
    """
    manifest_digest = _sha("manifest")
    profile = mint_frame_selection_profile(
        policy=_policy(),
        candidate_manifest_digest=manifest_digest,
        rights_evidence_digest=_sha("rights"),
        revocation_check_status="clear",
    )
    normalized = validate_selection_profile_against_admission_gate(
        profile, candidate_manifest_digest=manifest_digest
    )
    assert normalized["selector"] == "profile_bound_quality_filter"


def test_minted_profile_is_rejected_for_a_different_capture() -> None:
    """A profile minted for capture A must not admit capture B."""
    profile = mint_frame_selection_profile(
        policy=_policy(),
        candidate_manifest_digest=_sha("manifest-a"),
        rights_evidence_digest=_sha("rights"),
        revocation_check_status="clear",
    )
    with pytest.raises(Exception) as excinfo:
        validate_selection_profile_against_admission_gate(
            profile, candidate_manifest_digest=_sha("manifest-b")
        )
    assert "capture_binding_mismatch" in str(excinfo.value)


def test_revoked_rights_abstain_before_any_work() -> None:
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        mint_frame_selection_profile(
            policy=_policy(),
            candidate_manifest_digest=_sha("manifest"),
            rights_evidence_digest=_sha("rights"),
            revocation_check_status="revoked",
        )
    assert "capture_reconstruction_rights_revoked" in str(excinfo.value)


def test_rights_evidence_must_match_the_policy_authority() -> None:
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        mint_frame_selection_profile(
            policy=_policy(),
            candidate_manifest_digest=_sha("manifest"),
            rights_evidence_digest=_sha("some-other-rights-document"),
            revocation_check_status="clear",
        )
    assert "capture_reconstruction_rights_evidence_mismatch" in str(excinfo.value)


# --------------------------------------------------------------------------
# Launch request validation
# --------------------------------------------------------------------------


def test_built_launch_request_is_valid() -> None:
    request = _request()
    assert request["schema_version"] == LAUNCH_REQUEST_SCHEMA_VERSION
    assert validate_launch_request(request) == []


def test_launch_request_carries_no_secret_material() -> None:
    request = _request()
    serialized = json.dumps(request).lower()
    for fragment in ("password", "secret", "token", "private_key", "credential"):
        assert fragment not in serialized


def test_launch_request_refuses_paid_authority_beyond_policy_ceiling() -> None:
    blockers = validate_launch_request(_request(max_spend_usd=250.0))
    assert "capture_reconstruction_launch_spend_exceeds_policy" in blockers


def test_launch_request_binds_capture_digest() -> None:
    request = _request()
    assert request["capture_digest"] == _sha("capture")
    tampered = dict(request)
    tampered["capture_digest"] = _sha("different-capture")
    assert "capture_reconstruction_launch_request_digest_mismatch" in validate_launch_request(
        tampered
    )


# --------------------------------------------------------------------------
# Exactly-once enqueue
# --------------------------------------------------------------------------


def test_enqueue_is_idempotent_for_the_same_capture(tmp_path: Path) -> None:
    request = _request()
    first = stage_launch_request(value=request, queue_root=tmp_path)
    second = stage_launch_request(value=request, queue_root=tmp_path)
    assert first["status"] == "queued"
    assert first["already_exists"] is False
    assert second["already_exists"] is True
    assert second["queue_path"] == first["queue_path"]
    pending = list((tmp_path / "pending").glob("*.json"))
    assert len(pending) == 1


def test_duplicate_submit_with_mutated_content_is_refused(tmp_path: Path) -> None:
    request = _request()
    stage_launch_request(value=request, queue_root=tmp_path)
    mutated = dict(request)
    mutated["max_spend_usd"] = 9.0
    mutated["request_digest"] = canonical_digest(
        {k: v for k, v in mutated.items() if k != "request_digest"},
        digest_field="request_digest",
    )
    # Same capture, different paid authority: must not silently create a second
    # billable request under the same idempotency key.
    with pytest.raises(CaptureReconstructionLaunchError) as excinfo:
        stage_launch_request(value=mutated, queue_root=tmp_path)
    assert "immutable" in str(excinfo.value) or "conflict" in str(excinfo.value)


def test_enqueue_never_performs_provider_mutation(tmp_path: Path) -> None:
    receipt = stage_launch_request(value=_request(), queue_root=tmp_path)
    assert receipt["provider_mutation_performed"] is False


def test_queue_run_without_execute_authority_performs_no_paid_work(tmp_path: Path) -> None:
    queue_root = tmp_path / "queue"
    stage_launch_request(value=_request(), queue_root=queue_root)
    result = process_launch_queue(
        queue_root=queue_root,
        state_root=tmp_path / "state",
        execute=False,
    )
    assert result["provider_mutation_performed"] is False
    assert result["dispatched"] == 0
    assert result["previewed"] == 1
