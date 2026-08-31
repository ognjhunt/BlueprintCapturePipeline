from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_evaluation_control_search_funnel import (
    CLAIM_CEILING,
    ControlSearchFunnelError,
    OUTCOME_SCHEMA_VERSION,
    build_control_search_funnel_plan,
    build_control_search_sweep_result,
    validate_control_search_funnel_plan,
    validate_control_search_sweep_result,
)


def _candidate(index: int) -> dict:
    value = {
        "schema_version": "task_evaluation_native_construction_candidate.v1",
        "candidate_id": f"curobo-r0-candidate-{index:04d}",
        "deterministic_rank": index,
        "candidate_digest": "",
    }
    value["candidate_digest"] = canonical_digest(
        value, digest_field="candidate_digest"
    )
    return value


def _inventory(count: int = 300) -> dict:
    value = {
        "schema_version": (
            "task_evaluation_native_construction_candidate_inventory.v1"
        ),
        "run_id": "scene-839873-control-search",
        "round_index": 0,
        "source_native_feedback_digest": "sha256:" + "1" * 64,
        "model_authored_candidates": False,
        "candidates": [_candidate(index) for index in range(count)],
        "inventory_digest": "",
    }
    value["inventory_digest"] = canonical_digest(
        value, digest_field="inventory_digest"
    )
    return value


def _plan(**overrides):
    values = {
        "run_id": "scene-839873-control-search",
        "source_commit": "a" * 40,
        "packet_request_digest": "sha256:" + "2" * 64,
        "candidate_inventory": _inventory(),
        "runtime_source_packet_digest": "sha256:" + "3" * 64,
        "scene_collision_digest": "sha256:" + "4" * 64,
        "robot_configuration_digest": "sha256:" + "5" * 64,
        "task_scoring_digest": "sha256:" + "6" * 64,
        "requested_vector_env_count": 256,
        "maximum_vector_env_count": 1_024,
        "seeds_per_candidate": 4,
        "shortlist_size": 16,
    }
    values.update(overrides)
    return build_control_search_funnel_plan(**values)


def test_plan_freezes_search_without_granting_qualification() -> None:
    plan = _plan()

    assert validate_control_search_funnel_plan(plan) == plan
    assert plan["claim_ceiling"] == CLAIM_CEILING
    assert plan["vector_sweep"] == {
        "backend": "isaac_lab_vectorized_lightweight_physics",
        "requested_vector_env_count": 256,
        "maximum_vector_env_count": 1_024,
        "resolved_vector_env_count": 256,
        "seeds_per_candidate": 4,
        "assignment_count": 1_200,
        "wave_count": 5,
        "appearance_mode": "omitted",
        "camera_mode": "disabled",
        "collision_authority": "exact_scene_collision_digest",
        "robot_object_task_scoring_exact": True,
    }
    assert plan["shortlist"]["resolved_maximum_size"] == 16
    assert plan["shortlist"]["learned_grader_used"] is False
    assert plan["full_fidelity_replay"]["particlefield_required"] is True
    assert plan["full_fidelity_replay"][
        "search_result_alone_may_not_qualify_controls"
    ] is True


def test_plan_caps_vector_envs_and_shortlist_to_real_inventory() -> None:
    plan = _plan(
        candidate_inventory=_inventory(10),
        requested_vector_env_count=512,
        seeds_per_candidate=1,
        shortlist_size=8,
    )

    assert plan["vector_sweep"]["resolved_vector_env_count"] == 10
    assert plan["vector_sweep"]["wave_count"] == 1
    assert plan["shortlist"]["resolved_maximum_size"] == 8


def test_plan_refuses_tampered_or_model_authored_inventory() -> None:
    inventory = _inventory()
    inventory["model_authored_candidates"] = True
    inventory["inventory_digest"] = canonical_digest(
        inventory, digest_field="inventory_digest"
    )
    with pytest.raises(
        ControlSearchFunnelError,
        match="control_search_candidate_inventory_invalid",
    ):
        _plan(candidate_inventory=inventory)

    plan = _plan()
    tampered = copy.deepcopy(plan)
    tampered["vector_sweep"]["camera_mode"] = "enabled"
    with pytest.raises(ControlSearchFunnelError, match="control_search_plan_invalid"):
        validate_control_search_funnel_plan(tampered)


def _outcome(
    candidate: dict,
    *,
    seed_index: int,
    wave_index: int,
    environment_index: int,
    collision: float,
    coverage: float,
    path_error: float,
    destination_error: float,
    stability_error: float,
    displacement: float,
    reset_passed: bool = True,
) -> dict:
    value = {
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "candidate_id": candidate["candidate_id"],
        "candidate_digest": candidate["candidate_digest"],
        "seed_index": seed_index,
        "resolved_seed": 839873104 + seed_index,
        "wave_index": wave_index,
        "environment_index": environment_index,
        "reset_readback_passed": reset_passed,
        "forbidden_collision_peak_force_n": collision,
        "required_task_contact_coverage_fraction": coverage,
        "push_path_tracking_error_m": path_error,
        "destination_error_m": destination_error,
        "support_stability_error_m": stability_error,
        "task_displacement_m": displacement,
        "physics_steps": 220,
        "measurement_authority": (
            "isaac_lab_simulator_state_and_contact_sensors"
        ),
        "learned_grader_used": False,
        "outcome_digest": "",
    }
    value["outcome_digest"] = canonical_digest(
        value, digest_field="outcome_digest"
    )
    return value


def test_sweep_ranks_worst_case_physics_and_emits_bounded_shortlist() -> None:
    plan = _plan(
        candidate_inventory=_inventory(10),
        requested_vector_env_count=20,
        maximum_vector_env_count=1_024,
        seeds_per_candidate=2,
        shortlist_size=8,
    )
    outcomes = []
    for index, candidate in enumerate(plan["candidate_index"]):
        for seed_index in range(2):
            outcomes.append(
                _outcome(
                    candidate,
                    seed_index=seed_index,
                    wave_index=0,
                    environment_index=index * 2 + seed_index,
                    collision=0.1 + index,
                    coverage=0.95 - index * 0.01,
                    path_error=0.01 + index * 0.001,
                    destination_error=0.02 + index * 0.001,
                    stability_error=0.001 + index * 0.0001,
                    displacement=0.12 - index * 0.001,
                    reset_passed=index != 9,
                )
            )

    result = build_control_search_sweep_result(
        plan=plan,
        outcomes=outcomes,
        actual_vector_env_count=20,
        peak_gpu_memory_bytes=18_000_000_000,
    )

    assert validate_control_search_sweep_result(result, plan=plan) == result
    assert result["qualification_effect"] == "none_until_full_fidelity_replay"
    assert len(result["shortlist"]) == 8
    assert result["shortlist"][0]["candidate_id"] == (
        "curobo-r0-candidate-0000"
    )
    assert result["ranked_candidates"][-1]["candidate_id"] == (
        "curobo-r0-candidate-0009"
    )
    assert result["learned_grader_used"] is False


def test_sweep_requires_every_candidate_seed_and_unique_env_assignment() -> None:
    plan = _plan(
        candidate_inventory=_inventory(10),
        requested_vector_env_count=10,
        seeds_per_candidate=1,
        shortlist_size=8,
    )
    outcomes = [
        _outcome(
            candidate,
            seed_index=0,
            wave_index=0,
            environment_index=index,
            collision=0.0,
            coverage=1.0,
            path_error=0.0,
            destination_error=0.0,
            stability_error=0.0,
            displacement=0.12,
        )
        for index, candidate in enumerate(plan["candidate_index"])
    ]
    with pytest.raises(
        ControlSearchFunnelError, match="control_search_sweep_result_incomplete"
    ):
        build_control_search_sweep_result(
            plan=plan,
            outcomes=outcomes[:-1],
            actual_vector_env_count=10,
            peak_gpu_memory_bytes=1,
        )

    duplicate = copy.deepcopy(outcomes)
    duplicate[1]["environment_index"] = duplicate[0]["environment_index"]
    duplicate[1]["outcome_digest"] = canonical_digest(
        duplicate[1], digest_field="outcome_digest"
    )
    with pytest.raises(
        ControlSearchFunnelError, match="control_search_sweep_result_invalid"
    ):
        build_control_search_sweep_result(
            plan=plan,
            outcomes=duplicate,
            actual_vector_env_count=10,
            peak_gpu_memory_bytes=1,
        )
