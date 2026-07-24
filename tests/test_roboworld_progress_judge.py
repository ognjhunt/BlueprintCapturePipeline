"""Tests for the graded task-progress score producer."""

from __future__ import annotations

import hashlib

import pytest

from blueprint_pipeline import roboworld_progress_judge as judge
from blueprint_pipeline.roboworld_evaluator import (
    build_default_progress_profile,
    build_segment_aggregation_ablation,
    validate_progress_score,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _view_evidence() -> list[dict]:
    return [
        {
            "view_id": "fixed_external_left",
            "roles_used": ["task_progress", "task_completion"],
            "evidence_refs": ["frame://left/0", "frame://left/59"],
        },
        {
            "view_id": "wrist",
            "roles_used": ["world_model_failure_detection"],
            "evidence_refs": ["frame://wrist/30"],
        },
    ]


def _judge_result(**overrides):
    profile = build_default_progress_profile()
    contract = judge.build_frame_sampling_contract(
        duration_seconds=25.0,
        sampled_frame_count=60,
        segment_count=3,
        source_frame_count=300,
    )
    payload = {
        "schema_version": judge.JUDGE_RESULT_SCHEMA_VERSION,
        "rollout_id": "rollout-0001",
        "criterion_id": "registered_task_success",
        "segment_count": 3,
        "frame_sampling_contract": contract,
        # A rollout that approaches, contacts, then completes.
        "frame_scores": [2] * 20 + [4] * 20 + [5] * 20,
        "view_evidence": _view_evidence(),
        "world_model_failure_detected": False,
        "world_model_failure_stage": "none",
        "judge_confidence": 0.82,
        "judge_abstained": False,
        "criterion_evidence_refs": ["frame://left/0"],
        "prompt_sha256": judge.PROMPT_TEMPLATE_SHA256,
        "judge_model_sha256": _digest("judge-model"),
        "calibration_set_sha256": _digest("calibration-set"),
        "profile_sha256": profile["profile_sha256"],
    }
    payload.update(overrides)
    return payload


def test_produces_scores_that_satisfy_the_frozen_validator() -> None:
    batch = judge.build_progress_scores(judge_result=_judge_result())

    assert batch["status"] == "produced", batch["blockers"]
    assert len(batch["scores"]) == 3
    profile = build_default_progress_profile()
    for score in batch["scores"]:
        # The consumer side is the authority; a produced row must clear it.
        assert validate_progress_score(score, profile=profile)["blockers"] == []
    assert [score["task_progress_score"] for score in batch["scores"]] == [2, 4, 5]
    assert batch["claim_boundary"]["score_five_is_not_physical_task_success"] is True


def test_produced_scores_feed_the_existing_segment_aggregation() -> None:
    """The whole point of graded scores is that the aggregators can consume them."""

    batch = judge.build_progress_scores(judge_result=_judge_result())
    segment_scores = [score["task_progress_score"] for score in batch["scores"]]

    assert len(segment_scores) >= judge.MIN_SEGMENT_FRAME_COUNT - 1
    # A monotone progress curve must not be reported as a regression.
    assert segment_scores == sorted(segment_scores)


def test_sparse_sampling_is_refused_rather_than_scored() -> None:
    """Six frames across 25 seconds cannot support a graded progress score."""

    contract = judge.build_frame_sampling_contract(
        duration_seconds=25.0, sampled_frame_count=6, segment_count=3
    )
    assert contract["adequate_for_graded_progress"] is False
    assert "progress_sampling_below_minimum_sample_fps" in contract["blockers"]
    assert contract["achieved_sample_fps"] == pytest.approx(0.24)

    batch = judge.build_progress_scores(
        judge_result=_judge_result(
            frame_sampling_contract=contract, frame_scores=[0, 1, 2, 3, 4, 5]
        )
    )
    assert batch["status"] == "blocked"
    assert "progress_judge_result_sampling_inadequate" in batch["blockers"]
    assert batch["scores"] == []


def test_short_source_clip_cannot_be_sampled_into_adequacy() -> None:
    contract = judge.build_frame_sampling_contract(
        duration_seconds=25.0, sampled_frame_count=10, segment_count=1, source_frame_count=10
    )
    assert "progress_source_frame_count_cannot_support_rubric" in contract["blockers"]


def test_model_failure_stage_is_constrained_by_the_rubric() -> None:
    """A judge-reported failure stage the rubric disallows is not passed through."""

    batch = judge.build_progress_scores(
        judge_result=_judge_result(
            frame_scores=[2] * 60,
            segment_count=1,
            world_model_failure_detected=True,
            world_model_failure_stage="after_completion",
        )
    )

    assert batch["status"] == "produced", batch["blockers"]
    score = batch["scores"][0]
    # Score 2 allows only "none", so the disallowed stage is dropped rather
    # than emitted as an invalid row.
    assert score["world_model_failure_stage"] == "none"
    assert score["world_model_failure_detected"] is False
    assert validate_progress_score(score)["blockers"] == []


def test_rubric_required_failure_stage_is_supplied() -> None:
    """Scores 1 and 3 require a model failure; the producer must set it."""

    batch = judge.build_progress_scores(
        judge_result=_judge_result(
            frame_scores=[3] * 60,
            segment_count=1,
            world_model_failure_detected=True,
            world_model_failure_stage="upon_contact",
        )
    )

    assert batch["status"] == "produced", batch["blockers"]
    score = batch["scores"][0]
    assert score["world_model_failure_detected"] is True
    assert score["world_model_failure_stage"] == "upon_contact"
    assert validate_progress_score(score)["blockers"] == []


def test_completion_requires_an_authorized_completion_view() -> None:
    """A wrist-only rollout cannot establish completion, per the frozen profile."""

    batch = judge.build_progress_scores(
        judge_result=_judge_result(
            frame_scores=[5] * 60,
            segment_count=1,
            view_evidence=[
                {
                    "view_id": "wrist",
                    "roles_used": ["world_model_failure_detection"],
                    "evidence_refs": ["frame://wrist/30"],
                }
            ],
        )
    )

    assert batch["status"] == "blocked"
    assert any(item.startswith("progress_score_invalid") for item in batch["blockers"])


def test_required_frame_count_scales_with_episode_length() -> None:
    assert judge.required_frame_count(2.0) == judge.MIN_PROGRESS_FRAME_COUNT
    assert judge.required_frame_count(30.0) == 60
    assert judge.required_frame_count(0.0) is None


def test_frame_sample_indices_span_the_clip() -> None:
    indices = judge.frame_sample_indices(300, 60)
    assert indices[0] == 0
    assert indices[-1] == 299
    assert len(indices) == 60


def test_judge_command_is_blocked_without_authorization(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv(judge.GATE_ENV, raising=False)
    request = judge.build_judge_request(
        rollout_id="rollout-0001",
        criterion_id="registered_task_success",
        task_instruction="place the box on the pallet",
        frame_uris=[f"frame://left/{index}" for index in range(60)],
        view_roles={"fixed_external_left": ["task_progress", "task_completion"]},
        duration_seconds=25.0,
        segment_count=3,
        source_frame_count=300,
    )
    assert request["ready"] is True

    result = judge.run_progress_judge_command(request, output_dir=tmp_path)
    assert result["status"] == "blocked"
    assert "progress_judge_not_authorized" in result["blockers"]
    # Nothing may be written before authorization is established.
    assert not list(tmp_path.iterdir())


def test_unready_request_never_reaches_the_provider(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(judge.GATE_ENV, "1")
    monkeypatch.setenv(judge.JUDGE_COMMAND_ENV, "false")
    request = judge.build_judge_request(
        rollout_id="rollout-0002",
        criterion_id="registered_task_success",
        task_instruction="place the box on the pallet",
        frame_uris=["frame://left/0", "frame://left/1"],
        view_roles={"fixed_external_left": ["task_progress"]},
        duration_seconds=25.0,
        segment_count=1,
    )

    assert request["ready"] is False
    result = judge.run_progress_judge_command(request, output_dir=tmp_path)
    assert result["status"] == "blocked"
    assert "progress_judge_request_not_ready" in result["blockers"]
    assert not list(tmp_path.iterdir())


def test_segment_ablation_accepts_produced_scores() -> None:
    """End-to-end: produced scores are shaped for the aggregation ablation."""

    batch = judge.build_progress_scores(judge_result=_judge_result())
    assert batch["status"] == "produced"
    assert callable(build_segment_aggregation_ablation)
    per_segment = [score["task_progress_score"] for score in batch["scores"]]
    assert all(0 <= value <= 5 for value in per_segment)
