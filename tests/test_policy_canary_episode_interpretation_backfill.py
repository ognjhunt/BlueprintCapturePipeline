from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import (
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.policy_canary_episode_interpretation_backfill import (
    PolicyCanaryEpisodeInterpretationBackfillError,
    build_policy_canary_episode_interpretation_sidecar,
)


def _projection(*, interpreted: bool) -> dict:
    episodes = []
    for index in range(20):
        candidate = "pi05_droid" if index < 10 else "groot_n17_droid"
        cell = index % 10
        row = {
            "episode_id": f"episode-{index}",
            "candidate_id": candidate,
            "cell_id": f"scene839873.quick10.{cell:02d}.canonical_anchor",
            "seed": 1_000 + cell,
        }
        if interpreted:
            row["interpretation"] = {
                "status": "completed",
                "abstention_reason": None,
                "episode_outcome": "appears_complete",
                "summary": "The episode appears complete.",
                "events": [],
                "possible_missed_events": [],
                "contract_considerations": [],
                "confidence": 0.8,
                "deterministic_agreement": "agrees",
                "receipt": {
                    "artifact_id": f"interpretation-{index}",
                    "digest": "sha256:" + f"{index + 1:064x}",
                    "size_bytes": 512,
                },
                "learned_interpretation_only": True,
                "authoritative_task_success_unchanged": True,
                "ranking_or_promotion_effect": "none",
            }
        episodes.append(row)
    value = {
        "schema_version": "task_evaluation_policy_canary_result_projection.v1",
        "run_id": "scene839873-quick10",
        "request_digest": "sha256:" + "1" * 64,
        "configuration_digest": "sha256:" + "2" * 64,
        "result_delivery_digest": "sha256:" + ("4" if interpreted else "3") * 64,
        "run_kind": "internal_policy_canary",
        "claim_ceiling": "diagnostic_policy_execution",
        "matrix_digest": "sha256:" + "5" * 64,
        "candidate_ids": ["pi05_droid", "groot_n17_droid"],
        "episodes": episodes,
        "projection_digest": "",
    }
    if interpreted:
        value["episode_interpretation"] = {
            "schema_version": "policy_canary_episode_interpretation_closeout.v1",
            "status": "completed",
            "episode_count": 20,
            "receipt_count": 20,
            "completed_count": 20,
            "abstained_count": 0,
            "disagreement_count": 0,
            "reused_receipt_count": 0,
            "provider_call_count": 20,
            "provider_invocation_attempt_count": 20,
            "input_bundle_unavailable_count": 0,
            "interpreter": {"model": "gpt-5.6-luna"},
            "interpreter_profile_digest": "sha256:" + "6" * 64,
            "official_cost_completion_error_type": None,
            "closeout_error_type": None,
            "authoritative_deterministic_result_unchanged": True,
            "score_overwrite_performed": False,
            "ranking_or_promotion_effect": "none",
            "summary_digest": "sha256:" + "7" * 64,
        }
    value["projection_digest"] = cross_runtime_canonical_digest(
        value, digest_field="projection_digest"
    )
    return value


def _site_record() -> dict:
    projection = _projection(interpreted=False)
    return {
        "record_id": "capture-run-c257ae6e11a18e883637739477e5ded8",
        "publication": {
            "schema_version": "task_evaluation_run_publication.v4",
            "run_id": projection["run_id"],
            "result_delivery": {
                "delivery_digest": projection["result_delivery_digest"]
            },
            "policy_canary_result": projection,
        },
        "score_correction": {
            "sidecar_digest": "sha256:" + "8" * 64,
        },
    }


def test_builds_exact_historical_interpretation_sidecar() -> None:
    value = build_policy_canary_episode_interpretation_sidecar(
        source_site_record=_site_record(),
        backfill_projection=_projection(interpreted=True),
        record_id="capture-run-c257ae6e11a18e883637739477e5ded8",
        verified_at_iso="2026-09-03T23:00:00Z",
    )

    assert value["source_binding"]["source_score_correction_sidecar_digest"] == (
        "sha256:" + "8" * 64
    )
    assert len(value["episodes"]) == 20
    assert value["audit"]["deterministic_scores_unchanged"] is True
    assert value["sidecar_digest"] == cross_runtime_canonical_digest(
        value, digest_field="sidecar_digest"
    )


def test_refuses_episode_or_scientific_identity_drift() -> None:
    projection = _projection(interpreted=True)
    projection["episodes"][0]["episode_id"] = "different-episode"
    projection["projection_digest"] = cross_runtime_canonical_digest(
        projection, digest_field="projection_digest"
    )
    with pytest.raises(
        PolicyCanaryEpisodeInterpretationBackfillError,
        match="episode_inventory_invalid",
    ):
        build_policy_canary_episode_interpretation_sidecar(
            source_site_record=_site_record(),
            backfill_projection=projection,
            record_id="capture-run-c257ae6e11a18e883637739477e5ded8",
            verified_at_iso="2026-09-03T23:00:00Z",
        )

    projection = _projection(interpreted=True)
    projection["configuration_digest"] = "sha256:" + "9" * 64
    projection["projection_digest"] = cross_runtime_canonical_digest(
        projection, digest_field="projection_digest"
    )
    with pytest.raises(
        PolicyCanaryEpisodeInterpretationBackfillError,
        match="scientific_identity_changed",
    ):
        build_policy_canary_episode_interpretation_sidecar(
            source_site_record=_site_record(),
            backfill_projection=projection,
            record_id="capture-run-c257ae6e11a18e883637739477e5ded8",
            verified_at_iso="2026-09-03T23:00:00Z",
        )


def test_refuses_learned_interpretation_with_ranking_authority() -> None:
    projection = copy.deepcopy(_projection(interpreted=True))
    projection["episodes"][0]["interpretation"]["ranking_or_promotion_effect"] = (
        "winner_selected"
    )
    projection["projection_digest"] = cross_runtime_canonical_digest(
        projection, digest_field="projection_digest"
    )
    with pytest.raises(
        PolicyCanaryEpisodeInterpretationBackfillError,
        match="receipt_invalid",
    ):
        build_policy_canary_episode_interpretation_sidecar(
            source_site_record=_site_record(),
            backfill_projection=projection,
            record_id="capture-run-c257ae6e11a18e883637739477e5ded8",
            verified_at_iso="2026-09-03T23:00:00Z",
        )
