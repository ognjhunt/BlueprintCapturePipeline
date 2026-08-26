from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_episode_lifecycle import (
    REQUIRED_READINESS_CHECKS,
    TERMINAL_PLANNED_DURATION,
    TERMINAL_POLICY_SAFETY,
    PolicyEpisodeLifecycleError,
    build_lifecycle,
    seal_prestart_readiness,
    validate_policy_episode_lifecycle,
)


def _readiness():
    return seal_prestart_readiness(
        {
            "candidate_id": "groot_n17_droid",
            "candidate_policy_queried": False,
            "policy_state_advanced": False,
            "canonical_reset_restored": True,
            "checks": {check: True for check in REQUIRED_READINESS_CHECKS},
            "storage_reservation": {
                "required_free_bytes": 1,
                "observed_free_bytes": 2,
                "projection_is_conservative": True,
            },
            "policy_control_plane": {
                "identity_verified": True,
                "candidate_policy_queried": False,
                "candidate_inference_performed": False,
                "policy_state_advanced": False,
            },
            "visual_evidence": {
                "status": "complete",
                "required_camera_ids": ["external", "wrist", "overview"],
                "review_only_camera_ids": ["overview"],
                "terminal_observation_present": True,
                "videos": {camera: {} for camera in ("external", "wrist", "overview")},
            },
            "media_artifacts": [{"role": "test"}],
        }
    )


def _receipt(*, terminal_class: str = TERMINAL_PLANNED_DURATION):
    readiness = _readiness()
    planned = terminal_class == TERMINAL_PLANNED_DURATION
    lifecycle = build_lifecycle(
        readiness=readiness,
        terminal_class=terminal_class,
        planned_policy_queries=4,
        planned_action_steps=32,
        planned_settle_steps=6,
        actual_policy_queries=4 if planned else 2,
        actual_action_steps=32 if planned else 8,
        actual_settle_steps=6 if planned else 0,
        terminal_reason="test",
        retained_terminal_result=True,
    )
    receipt = {
        "schema_version": "adp009d_policy_episode.v4",
        "prestart_readiness": readiness,
        "lifecycle": lifecycle,
        "visual_evidence": {
            "status": "complete",
            "required_camera_ids": ["external", "wrist", "overview"],
            "review_only_camera_ids": ["overview"],
            "terminal_observation_present": True,
            "videos": {camera: {} for camera in ("external", "wrist", "overview")},
        },
        "candidate_exact_policy_input_frames": [{"frame": 0}],
        "policy_observations_retained": 1,
        "candidate_exact_policy_input_manifest_digest": canonical_digest(
            {"frames": [{"frame": 0}]}
        ),
        "observation_trace_digest": canonical_digest(
            {"observations": [{"frame": 0}]}
        ),
        "media_artifacts": [{"role": "test"}],
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    return receipt


@pytest.mark.parametrize(
    "terminal_class", [TERMINAL_PLANNED_DURATION, TERMINAL_POLICY_SAFETY]
)
def test_lifecycle_accepts_only_digest_bound_retained_terminal_classes(
    terminal_class: str,
) -> None:
    receipt = _receipt(terminal_class=terminal_class)

    lifecycle = validate_policy_episode_lifecycle(receipt)

    assert lifecycle["terminal_class"] == terminal_class
    assert lifecycle["post_start_infrastructure_failure"] is False


def test_lifecycle_rejects_resealed_early_planned_duration_claim() -> None:
    receipt = copy.deepcopy(_receipt())
    receipt["lifecycle"]["actual_action_steps"] = 31
    receipt["lifecycle"]["lifecycle_digest"] = canonical_digest(
        receipt["lifecycle"], digest_field="lifecycle_digest"
    )
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )

    with pytest.raises(
        PolicyEpisodeLifecycleError,
        match="policy_episode_planned_duration_not_completed",
    ):
        validate_policy_episode_lifecycle(receipt)


def test_readiness_rejects_candidate_query_before_episode_start() -> None:
    payload = {
        "candidate_policy_queried": True,
        "policy_state_advanced": False,
        "canonical_reset_restored": True,
        "checks": {check: True for check in REQUIRED_READINESS_CHECKS},
        "storage_reservation": {
            "required_free_bytes": 1,
            "observed_free_bytes": 2,
            "projection_is_conservative": True,
        },
        "policy_control_plane": {"identity_verified": True},
        "visual_evidence": {
            "status": "complete",
            "required_camera_ids": ["external", "wrist", "overview"],
            "review_only_camera_ids": ["overview"],
            "terminal_observation_present": True,
            "videos": {camera: {} for camera in ("external", "wrist", "overview")},
        },
        "media_artifacts": [{"role": "test"}],
    }

    with pytest.raises(
        PolicyEpisodeLifecycleError,
        match="policy_episode_readiness_queried_candidate",
    ):
        seal_prestart_readiness(payload)
