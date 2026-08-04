from __future__ import annotations

import copy

import pytest

from blueprint_pipeline.adp_prospective_design import (
    INVALID_TRIAL_RULE,
    MULTIPLICITY_RULE,
    POWER_METHOD,
    STOP_RULE,
    UNCERTAINTY_METHOD,
    ADPProspectiveDesignError,
    compile_decision,
    compile_power_requirement,
    compile_trial_schedule,
    validate_episode_evidence_contract,
    validate_schedule_for_execution,
    validate_secondary_metrics,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest


def _design() -> dict:
    return {
        "method": POWER_METHOD,
        "planning_variance_rate": 0.5,
        "minimum_decision_relevant_difference": 0.20,
        "alpha": 0.05,
        "power": 0.80,
        "uncertainty_method": UNCERTAINTY_METHOD,
        "invalid_trial_handling": INVALID_TRIAL_RULE,
        "stop_rule": STOP_RULE,
        "multiplicity": MULTIPLICITY_RULE,
    }


def _schedule() -> dict:
    return compile_trial_schedule(
        candidate_pair={
            # Deliberately reverse alphabetical order: the baseline is declared.
            "baseline_candidate_id": "z-baseline",
            "alternative_candidate_id": "a-alternative",
        },
        conditions=[
            {"condition_id": "condition-a", "reset_digest": "sha256:" + "1" * 64},
            {"condition_id": "condition-b", "reset_digest": "sha256:" + "2" * 64},
            {"condition_id": "condition-c", "reset_digest": "sha256:" + "3" * 64},
        ],
        statistical_design=_design(),
        randomization_seed=73011,
    )


def _binary_results(schedule: dict, *, baseline_successes: int, alternative_successes: int) -> list:
    seen = {"z-baseline": 0, "a-alternative": 0}
    limits = {
        "z-baseline": baseline_successes,
        "a-alternative": alternative_successes,
    }
    results = []
    for row in schedule["rows"]:
        candidate_id = row["candidate_id"]
        success = seen[candidate_id] < limits[candidate_id]
        seen[candidate_id] += 1
        results.append(
            {
                "trial_id": row["trial_id"],
                "status": "completed",
                "success": success,
            }
        )
    return results


def _complete_episode() -> dict:
    return {
        "episode_id": "episode-1",
        "status": "completed",
        "policy_query_count": 1,
        "visual_evidence": {
            "status": "complete",
            "human_review_available": True,
            "terminal_observation_frame_present": True,
            "frame_manifest_digest": "sha256:" + "4" * 64,
            "video": {
                "relative_path": "media/episode-1/review.mp4",
                "sha256": "sha256:" + "5" * 64,
            },
        },
        "evaluator": {
            "owner": "environment_not_policy",
            "grader_type": "deterministic_simulator_state",
            "success_source": "environment_step_info.success",
            "policy_self_report_used": False,
        },
        "success_evidence": {
            "grader_type": "deterministic_simulator_state",
            "policy_self_report_used": False,
        },
        "artifacts": [
            {
                "role": "observation_frame_manifest",
                "relative_path": "media/episode-1/frame_manifest.json",
                "sha256": "sha256:" + "6" * 64,
            },
            {
                "role": "policy_input_frame",
                "relative_path": "media/episode-1/policy-input-0.png",
                "sha256": "sha256:" + "7" * 64,
                "raw_rgb_sha256": "sha256:" + "8" * 64,
            },
            {
                "role": "terminal_observation_frame",
                "relative_path": "media/episode-1/terminal.png",
                "sha256": "sha256:" + "9" * 64,
                "raw_rgb_sha256": "sha256:" + "a" * 64,
            },
            {
                "role": "episode_video",
                "relative_path": "media/episode-1/review.mp4",
                "sha256": "sha256:" + "5" * 64,
            },
        ],
    }


def test_power_requirement_and_schedule_are_the_same_declared_design() -> None:
    requirement = compile_power_requirement(_design())
    schedule = _schedule()
    admission = validate_schedule_for_execution(schedule)

    assert requirement["minimum_trials_per_candidate"] == 99
    assert schedule["statistical_design"]["method"] == requirement["method"]
    assert schedule["power_requirement"] == requirement
    assert schedule["candidate_pair"] == {
        "baseline_candidate_id": "z-baseline",
        "alternative_candidate_id": "a-alternative",
    }
    assert schedule["repetitions_per_candidate_condition"] == 33
    assert schedule["trials_per_candidate"] == 99
    assert schedule["total_trial_budget"] == 198
    assert admission["status"] == "admitted_for_execution"


def test_execution_admission_rejects_schedule_below_frozen_requirement() -> None:
    schedule = _schedule()
    schedule["rows"] = [
        row
        for index, row in enumerate(schedule["rows"])
        if not (
            row["candidate_id"] == "a-alternative"
            and index
            == next(
                i
                for i, candidate_row in enumerate(schedule["rows"])
                if candidate_row["candidate_id"] == "a-alternative"
            )
        )
    ]
    schedule["total_trial_budget"] = len(schedule["rows"])
    schedule["schedule_digest"] = canonical_digest(schedule, digest_field="schedule_digest")

    with pytest.raises(ADPProspectiveDesignError) as caught:
        validate_schedule_for_execution(schedule)

    assert "schedule_below_frozen_power_requirement:a-alternative" in caught.value.blockers
    assert "schedule_condition_seed_repetition_matrix_mismatch" in caught.value.blockers


def test_execution_admission_rejects_broken_matched_candidate_interleaving() -> None:
    schedule = _schedule()
    schedule["rows"][1], schedule["rows"][2] = schedule["rows"][2], schedule["rows"][1]
    for index, row in enumerate(schedule["rows"]):
        row["execution_order"] = index
    schedule["schedule_digest"] = canonical_digest(schedule, digest_field="schedule_digest")

    with pytest.raises(ADPProspectiveDesignError) as caught:
        validate_schedule_for_execution(schedule)

    assert "schedule_randomization_order_mismatch" in caught.value.blockers
    assert any(blocker.startswith("schedule_interleaving_") for blocker in caught.value.blockers)


@pytest.mark.parametrize(
    ("baseline_successes", "alternative_successes", "expected"),
    [
        (0, 99, "select"),
        (99, 0, "eliminate"),
        (99, 99, "equivalent_inconclusive"),
        (50, 60, "abstain"),
    ],
)
def test_all_four_decision_branches_are_deterministic(
    baseline_successes: int,
    alternative_successes: int,
    expected: str,
) -> None:
    schedule = _schedule()
    results = _binary_results(
        schedule,
        baseline_successes=baseline_successes,
        alternative_successes=alternative_successes,
    )

    first = compile_decision(schedule=schedule, trial_results=results)
    second = compile_decision(schedule=schedule, trial_results=copy.deepcopy(results))

    assert first == second
    assert first["decision"] == expected
    assert first["baseline_candidate_id"] == "z-baseline"
    assert first["alternative_candidate_id"] == "a-alternative"


def test_failed_timed_out_invalid_and_missing_trials_stay_in_denominator() -> None:
    schedule = _schedule()
    results = _binary_results(schedule, baseline_successes=99, alternative_successes=99)
    baseline_ids = [
        row["trial_id"] for row in schedule["rows"] if row["candidate_id"] == "z-baseline"
    ]
    by_id = {row["trial_id"]: row for row in results}
    by_id[baseline_ids[0]]["status"] = "failed"
    by_id[baseline_ids[1]]["status"] = "timed_out"
    by_id[baseline_ids[2]]["status"] = "invalid"
    missing_id = baseline_ids[3]
    results = [row for row in results if row["trial_id"] != missing_id]

    decision = compile_decision(schedule=schedule, trial_results=results)
    summary = decision["candidate_summaries"]["z-baseline"]

    assert summary["frozen_denominator"] == 99
    assert summary["successes"] == 95
    assert summary["status_counts"]["failed"] == 1
    assert summary["status_counts"]["timed_out"] == 1
    assert summary["status_counts"]["invalid"] == 1
    assert summary["status_counts"]["missing"] == 1
    assert decision["all_scheduled_trials_retained"] is True


def test_secondary_metric_requires_partner_task_owner_preregistration() -> None:
    with pytest.raises(ADPProspectiveDesignError) as caught:
        validate_secondary_metrics(
            [
                {
                    "metric_id": "time_to_success_seconds",
                    "preregistered_by_partner_task_owner": False,
                    "owner_evidence_digest": None,
                }
            ]
        )

    assert "secondary_metric_not_owner_preregistered:time_to_success_seconds" in (
        caught.value.blockers
    )
    assert "secondary_metric_owner_evidence_missing:time_to_success_seconds" in (
        caught.value.blockers
    )


def test_future_completed_episode_requires_all_visual_and_grader_evidence() -> None:
    episode = _complete_episode()

    admission = validate_episode_evidence_contract(episode)

    assert admission["status"] == "admitted"
    assert admission["completed_media_contract"] is True
    assert admission["independent_grader_type"] == "deterministic_simulator_state"


@pytest.mark.parametrize("status", ["failed", "timed_out", "invalid", "interrupted"])
def test_future_non_success_after_observation_requires_and_admits_complete_media(
    status: str,
) -> None:
    episode = _complete_episode()
    episode["status"] = status

    admission = validate_episode_evidence_contract(episode)

    assert admission["status"] == "admitted"
    assert admission["completed_media_contract"] is True


@pytest.mark.parametrize(
    ("role", "expected_blocker"),
    [
        ("policy_input_frame", "episode_policy_input_frame_count_mismatch"),
        (
            "observation_frame_manifest",
            "episode_artifact_role_count_invalid:observation_frame_manifest",
        ),
        (
            "terminal_observation_frame",
            "episode_artifact_role_count_invalid:terminal_observation_frame",
        ),
        ("episode_video", "episode_artifact_role_count_invalid:episode_video"),
    ],
)
def test_future_completed_episode_rejects_each_missing_visual_artifact(
    role: str,
    expected_blocker: str,
) -> None:
    episode = _complete_episode()
    episode["artifacts"] = [row for row in episode["artifacts"] if row["role"] != role]

    with pytest.raises(ADPProspectiveDesignError) as caught:
        validate_episode_evidence_contract(episode)

    assert expected_blocker in caught.value.blockers


def test_future_episode_rejects_missing_terminal_flag_and_policy_grader() -> None:
    episode = _complete_episode()
    episode["visual_evidence"]["terminal_observation_frame_present"] = False
    episode["evaluator"]["grader_type"] = "policy_self_report"

    with pytest.raises(ADPProspectiveDesignError) as caught:
        validate_episode_evidence_contract(episode)

    assert "episode_terminal_observation_required" in caught.value.blockers
    assert "episode_independent_grader_provenance_invalid" in caught.value.blockers


def test_failure_before_first_observation_requires_typed_media_gap() -> None:
    episode = _complete_episode()
    episode.update(
        {
            "status": "timed_out",
            "policy_query_count": 0,
            "visual_evidence": {
                "status": "unavailable_before_first_observation",
                "media_gap": {
                    "type": "before_first_observation",
                    "reason": "runtime_startup_timeout",
                },
            },
            "artifacts": [],
        }
    )

    admission = validate_episode_evidence_contract(episode)

    assert admission["typed_pre_observation_media_gap"] is True
