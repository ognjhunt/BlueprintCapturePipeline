from __future__ import annotations

import json
from pathlib import Path

from blueprint_pipeline.policy_ranking_thesis import canonical_sha256


EXPERIMENT_DOCS = (
    Path(__file__).resolve().parents[1]
    / "docs/experiments/policy_ranking_roboarena_powered_droid_confirmation_20260729"
)


def _artifact(name: str) -> dict:
    payload = json.loads((EXPERIMENT_DOCS / name).read_text(encoding="utf-8"))
    recorded = payload.pop("manifest_sha256")
    assert recorded == canonical_sha256(payload)
    return payload


def test_powered_split_replaces_exposed_session_without_opening_labels() -> None:
    split = _artifact("disjoint_session_candidate_split_amendment_v3.json")

    assert split["selection"]["selected_session_count"] == 17
    assert "005387dc-76ab-405e-b363-b2182a075b5c" not in split["selection"]["session_ids"]
    assert "08bf285a-2a05-4deb-bfba-37080457e9e6" in split["selection"]["session_ids"]
    assert split["selection"]["outcome_labels_accessed"] is False
    assert split["claim_ceiling"]["independent_new_snapshot_confirmation_proven"] is False


def test_powered_protocol_freezes_complete_causal_matrix_and_claim_ceiling() -> None:
    protocol = _artifact("protocol_v1.json")

    execution = protocol["execution"]
    assert execution["structured_canary_requests"] == 1
    assert execution["scientific_request_count"] == 17 * 3 * 6 * 2
    assert execution["evaluator_calls"] == 0
    assert protocol["frozen_session_reliability_rule"]["minimum_eligible_timing_windows"] == 3
    assert protocol["frozen_causal_gates"]["action_effect_to_cross_seed_noise_ratio_min"] == 1.0
    assert protocol["paid_execution_admitted"] is False
    assert protocol["provider_called"] is False
    assert protocol["claim_ceiling"]["policy_ranking_fidelity"] is False
    assert protocol["claim_ceiling"]["live_policy_wam_policy_closed_loop"] is False


def test_powered_protocol_amendment_adds_only_preexisting_dynamic_canary_gate() -> None:
    amendment = _artifact("protocol_amendment_v2.json")

    assert amendment["amendment_basis"]["provider_called"] is False
    assert amendment["amendment_basis"]["new_generated_output_seen"] is False
    gates = amendment["changed_fields"]["structured_canary_gates"]
    assert gates["temporal_absolute_difference_mean_minimum_gray_0_255"] == 1.0
    assert gates["first_to_last_absolute_difference_mean_minimum_gray_0_255"] == 3.0
    assert gates["all_required"] is True
    assert (
        amendment["changed_fields"]["execution"]["canary_failure_submits_zero_untouched_requests"]
        is True
    )


def test_powered_environment_and_compute_authority_preserve_campaign_caps() -> None:
    environment = _artifact("environment_and_source_manifest_v1.json")
    authorization = json.loads(
        (EXPERIMENT_DOCS / "compute_authorization_allocation_1.json").read_text(encoding="utf-8")
    )

    assert environment["execution"]["maximum_concurrent_gpus"] == 1
    assert environment["execution"]["target_spend_usd"] == 6.2
    assert environment["execution"]["hard_cap_usd"] == 10.0
    assert authorization["maximum_provider_allocations"] == 1
    assert authorization["authorized_compute_cap_usd"] == 10.0
    assert authorization["gpu_category_ceiling_usd"] == 50.0
    assert authorization["campaign_total_ceiling_usd"] == 100.0
    assert authorization["physical_robot_endpoint_access_allowed"] is False
    assert authorization["evaluator_or_vlm_spend_authorized_by_this_record"] is False

    amendment = _artifact("cost_admission_amendment_v3.json")
    assert amendment["trigger"]["provider_mutations_performed"] == 0
    assert amendment["change"]["target_spend_usd_after"] == 6.2
    assert amendment["change"]["allocation_hard_cap_usd_unchanged"] == 10.0
    assert amendment["change"]["scientific_fields_changed"] is False


def test_runtime_fix_amendment_preserves_science_and_binds_replacement_bundle() -> None:
    amendment = _artifact("runtime_fix_amendment_v4.json")
    authorization = json.loads(
        (EXPERIMENT_DOCS / "compute_authorization_allocation_2.json").read_text(encoding="utf-8")
    )
    environment = _artifact("environment_and_source_manifest_v2.json")

    assert amendment["allocation_1_result"]["structured_canary"]["status"] == "passed"
    assert (
        amendment["allocation_1_result"]["untouched_matrix"]["scientific_result_available"] is False
    )
    assert amendment["prospective_change"]["request_serialization_changed"] is False
    assert amendment["prospective_change"]["outcome_label_accessed"] is False
    assert amendment["replacement_execution"]["replacement_provider_called"] is False
    assert amendment["claim_boundary"]["cosmos_wam_qualification_proven"] is False
    assert authorization["allocation_index"] == 2
    assert authorization["runtime_fix_amendment_sha256"] == (
        "d64a338a232119881e24e00ef6193df6031f398eb2dc2be23c86c8c300017b82"
    )
    assert authorization["authorized_compute_cap_usd"] == 10.0
    assert environment["replacement_bundle"]["bundle_sha256"] == (
        "bae438e48fa4ac2544840c91e713cdfc1274334820f1d7649d0c772876f1831a"
    )
