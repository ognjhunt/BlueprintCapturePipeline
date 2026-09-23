"""Graded episode reports: RoboLab-style detail from sealed canary evidence.

Hermetic: every score here comes from the real deterministic scorer on a
fixture trace, and every sidecar binds to a fixture publication.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.adp_task_scoring import score_task_episode_from_spec
from blueprint_pipeline.decision_evidence_contracts import (
    canonical_digest,
    cross_runtime_canonical_digest,
)
from blueprint_pipeline.policy_episode_graded_report import (
    GradedReportError,
    build_policy_canary_graded_report_sidecar,
    grade_episode,
    sparc,
    summarize_candidate,
)
from tests.test_task_evaluation_policy_canary_rescore import _fixture as _retained_result


def _task_spec(**overrides: object) -> dict[str, object]:
    spec: dict[str, object] = {
        "schema_version": "adp_task_spec.v2",
        "task_kind": "rigid_pick_place",
        "subject_asset_id": "cup",
        "start_pose_world": [1.0, 2.0, 0.8, 0.0, 0.0, 0.0, 1.0],
        "destination_position_bounds_world_m": {
            "minimum": [1.14, 1.99, 0.79],
            "maximum": [1.16, 2.01, 0.81],
        },
        "destination_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
        "destination_orientation_tolerance_rad": 0.1,
        "support_height_interval_m": [0.79, 0.81],
        "minimum_translation_m": 0.14,
        "minimum_lift_m": 0.02,
        "movement_epsilon_m": 0.001,
        "reset_translation_tolerance_m": 0.001,
        "reset_orientation_tolerance_rad": 0.01,
        "settle_window_samples": 3,
        "settle_position_tolerance_m": 0.002,
        "settle_orientation_tolerance_rad": 0.01,
        "release_required": True,
        "release_gripper_width_min_m": 0.07,
        "task_contact_minimum_force_n": 0.5,
    }
    spec.update(overrides)
    return spec


def _sample(step: int, position: list[float], *, hand: list[float] | None = None) -> dict[str, object]:
    return {
        "step_index": step,
        "task_object_pose_world": [*position, 0.0, 0.0, 0.0, 1.0],
        "controlled_body_pose_world": [*(hand or position), 0.0, 0.0, 0.0, 1.0],
        "gripper_width_m": 0.08 if step >= 3 else 0.04,
        "task_contact_active": step < 3,
        "support_contact_active": step >= 3,
        "containment_violation": False,
        "robot_collision_failure": False,
        "scene_collision_failure": False,
        "forbidden_robot_task_collision_failure": False,
        "locked_joint_containment_violation": False,
    }


def _placed_samples() -> list[dict[str, object]]:
    return [
        _sample(0, [1.0, 2.0, 0.8]),
        _sample(1, [1.0, 2.0, 0.83]),
        _sample(2, [1.15, 2.0, 0.83]),
        _sample(3, [1.15, 2.0, 0.8]),
        _sample(4, [1.15, 2.0, 0.8]),
        _sample(5, [1.15, 2.0, 0.8]),
    ]


def _untouched_samples() -> list[dict[str, object]]:
    rows = [_sample(step, [1.0, 2.0, 0.8]) for step in range(6)]
    for row in rows:
        row["task_contact_active"] = False
    return rows


def _joints(count: int) -> list[dict[str, object]]:
    return [
        {"step_index": step, "joint_positions_rad": [0.1 * step, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        for step in range(count)
    ]


# ------------------------------------------------------------------ SPARC


def test_sparc_orders_smooth_jerky_and_repeated_motion_like_the_literature() -> None:
    t = [i / 100 for i in range(101)]
    smooth = [30 * x**2 * (1 - x) ** 2 for x in t]
    jerky = [v * (1 + 0.3 * math.sin(2 * math.pi * 8 * x)) for v, x in zip(smooth, t)]
    repeated = smooth + smooth
    smooth_value = sparc(smooth, 100.0)
    assert smooth_value is not None and -1.5 < smooth_value < -1.3
    assert sparc(jerky, 100.0) < smooth_value
    assert sparc(repeated, 100.0) < smooth_value


def test_sparc_refuses_to_invent_a_value() -> None:
    assert sparc([], 15.0) is None
    assert sparc([0.0] * 10, 15.0) is None
    assert sparc([1.0, float("nan"), 1.0, 1.0], 15.0) is None
    assert sparc([1.0, 2.0, 1.0, 2.0], 0.0) is None


# -------------------------------------------------------- episode grading


def test_a_placed_and_settled_episode_earns_every_stage() -> None:
    samples = _placed_samples()
    score = score_task_episode_from_spec(task_spec=_task_spec(), samples=samples)
    assert score["task_succeeded"] is True
    report = grade_episode(score=score, samples=samples, joint_states=_joints(6))

    assert report["status"] == "graded"
    assert report["graded_score"] == 1.0
    assert [stage["id"] for stage in report["subtasks"]] == [
        "contact", "moved", "lifted", "translated", "placed", "released_and_settled",
    ]
    assert all(stage["achieved"] for stage in report["subtasks"])
    assert report["timing"]["first_task_contact_s"] == 0.0
    assert report["timing"]["first_object_motion_s"] == round(1 / 15, 4)
    assert report["timing"]["settled_at_s"] == round(3 / 15, 4)
    assert report["timing"]["episode_duration_s"] == round(6 / 15, 4)
    assert report["failure_events"]["drops"] == 0
    assert report["failure_events"]["wrong_object_interactions"] == {
        "status": "not_measurable", "reason": "single_task_object_scene",
    }
    assert report["smoothness"]["end_effector_path_length_m"] is not None
    assert report["inputs"]["score_report_digest"] == score["report_digest"]
    assert report["authority"]["ranking_or_promotion_effect"] == "none"
    assert report["report_digest"] == cross_runtime_canonical_digest(report, digest_field="report_digest")


def test_an_untouched_object_earns_no_stage_and_the_score_still_says_so() -> None:
    samples = _untouched_samples()
    score = score_task_episode_from_spec(task_spec=_task_spec(), samples=samples)
    report = grade_episode(score=score, samples=samples)
    assert score["task_succeeded"] is False
    assert report["graded_score"] == 0.0
    assert not any(stage["achieved"] for stage in report["subtasks"])
    # "Released and settled" is trivially true of an untouched object; it is
    # reported as met and credited as nothing.
    assert report["subtasks"][-1]["condition_met"] is True
    assert report["timing"]["settled_at_s"] is None
    assert report["task_succeeded"] is False


def test_partial_progress_is_credited_between_nothing_and_success() -> None:
    samples = _placed_samples()[:3] + [_sample(step, [1.15, 2.0, 0.83]) for step in (3, 4, 5)]
    for row in samples[3:]:
        row["gripper_width_m"] = 0.04
        row["task_contact_active"] = True
        row["support_contact_active"] = False
    score = score_task_episode_from_spec(task_spec=_task_spec(), samples=samples)
    report = grade_episode(score=score, samples=samples)
    assert score["task_succeeded"] is False
    assert 0.0 < report["graded_score"] < 1.0
    achieved = {stage["id"] for stage in report["subtasks"] if stage["achieved"]}
    assert {"contact", "moved", "lifted", "translated"} <= achieved
    assert "released_and_settled" not in achieved


def test_a_push_task_is_graded_on_push_stages() -> None:
    score = {
        "status": "scored", "task_succeeded": False, "outcome": "moved_below_success_contract",
        "manipulation_strategy": "planar_push",
        "criteria_satisfied": {"minimum_translation": True, "destination_containment": False,
                               "settling": True, "terminal_task_contact": True, "safety": True},
        "measurements": {"maximum_translation_m": 0.1},
        "event_ledger": {"drop_events": [], "safety_events": []},
    }
    report = grade_episode(score=score, samples=_placed_samples())
    assert [stage["id"] for stage in report["subtasks"]] == [
        "contact", "moved", "translated", "reached_destination", "settled_and_cleared",
    ]
    # Settled and let go, but never reached the destination: the condition is
    # reported, the credit is not, because stages count only in order.
    assert report["graded_score"] == 0.6
    last = report["subtasks"][-1]
    assert last["condition_met"] is True and last["achieved"] is False


def test_failure_events_come_from_the_score_ledger_and_are_named_plainly() -> None:
    score = score_task_episode_from_spec(task_spec=_task_spec(), samples=_placed_samples())
    score = copy.deepcopy(score)
    score["event_ledger"]["drop_events"] = [{"drop_step_index": 2, "fall_m": 0.03}]
    score["event_ledger"]["safety_events"] = [
        {"event_type": "forbidden_robot_object_contact_force_exceeded", "step_index": 1},
        {"event_type": "robot_scene_contact_force_exceeded", "step_index": 4},
    ]
    events = grade_episode(score=score, samples=_placed_samples())["failure_events"]
    assert events["drops"] == 1 and events["drop_steps"] == [2]
    assert events["robot_body_hit_object"] == 1
    assert events["robot_hit_scene"] == 1
    assert events["object_hit_scene"] == 0


def test_contact_with_another_object_is_counted_when_the_scene_reports_it() -> None:
    samples = _placed_samples()
    for index, row in enumerate(samples):
        row["non_task_object_contact_active"] = index == 2
    score = score_task_episode_from_spec(task_spec=_task_spec(), samples=_placed_samples())
    events = grade_episode(score=score, samples=samples)["failure_events"]
    assert events["wrong_object_interactions"] == {"status": "measured", "count": 1}


def test_an_undetermined_score_is_not_graded() -> None:
    report = grade_episode(score={"status": "undetermined", "outcome": "native_safety_readback_missing"},
                           samples=_placed_samples())
    assert report["status"] == "not_gradable"
    assert report["graded_score"] is None
    assert report["not_gradable_reason"] == "score_undetermined"


def test_the_summary_describes_a_run_and_never_ranks_it() -> None:
    placed = grade_episode(score=score_task_episode_from_spec(task_spec=_task_spec(), samples=_placed_samples()),
                           samples=_placed_samples())
    untouched = grade_episode(
        score=score_task_episode_from_spec(task_spec=_task_spec(), samples=_untouched_samples()),
        samples=_untouched_samples(),
    )
    summary = summarize_candidate("pi05_droid", [placed, untouched])
    assert summary["episode_count"] == 2 and summary["success_count"] == 1
    assert summary["mean_graded_score"] == 0.5
    assert summary["subtask_completion_rate"]["placed"] == 0.5
    assert summary["ranking_permitted"] is False


# ------------------------------------------------------------------ sidecar


def _site_record(tmp_path: Path, *, correction: bool = False) -> tuple[dict[str, object], Path, Path]:
    """A published run over the rescorer's retained-result fixture.

    Its recorded scores say every episode failed; the correction's rescored
    scores say every episode succeeded, which is what the fixture's traces
    actually show.
    """

    tmp_path.mkdir(parents=True, exist_ok=True)
    source, evidence, result = _retained_result(tmp_path)
    episodes = []
    updates = []
    for row in result["episodes"]:
        episodes.append({
            "episode_id": f"ep-{row['candidate_id']}-{row['cell_id']}",
            "candidate_id": row["candidate_id"],
            "cell_id": row["cell_id"],
            "seed": row["seed"],
        })
        raw = row["episode"]
        updates.append({
            "candidate_id": row["candidate_id"], "cell_id": row["cell_id"], "seed": row["seed"],
            "new_score": score_task_episode_from_spec(
                task_spec=raw["task_spec"], samples=raw["state_trace"]["task_state_samples"],
            ),
        })
    projection: dict[str, object] = {
        "schema_version": "task_evaluation_policy_canary_result_projection.v1",
        "run_id": result["run_id"],
        "result_delivery_digest": "sha256:" + "3" * 64,
        "episodes": episodes,
        "report": {"result_digest": result["result_digest"]},
        "projection_digest": "",
    }
    projection["projection_digest"] = cross_runtime_canonical_digest(projection, digest_field="projection_digest")
    record: dict[str, object] = {
        "publication": {
            "schema_version": "task_evaluation_run_publication.v4",
            "run_id": result["run_id"],
            "policy_canary_result": projection,
            "result_delivery": {"delivery_digest": "sha256:" + "3" * 64},
        },
    }
    if correction:
        record["score_correction"] = {
            "sidecar_digest": "sha256:" + "9" * 64,
            "correction": {"score_updates": updates},
        }
    return record, source, evidence


def _build(record: dict[str, object], source: Path, evidence: Path, **overrides: object) -> dict[str, object]:
    arguments: dict[str, object] = {
        "source_site_record": record, "source_result_path": source, "evidence_root": evidence,
        "record_id": "record-1", "generated_at_iso": "2026-09-23T21:00:00Z",
    }
    arguments.update(overrides)
    return build_policy_canary_graded_report_sidecar(**arguments)  # type: ignore[arg-type]


def test_sidecar_binding_summaries_and_digest(tmp_path: Path) -> None:
    record, source, evidence = _site_record(tmp_path)
    sidecar = _build(record, source, evidence)
    binding = sidecar["source_binding"]
    assert binding["record_id"] == "record-1"
    assert binding["source_run_id"] == record["publication"]["run_id"]
    assert binding["source_projection_digest"] == record["publication"]["policy_canary_result"]["projection_digest"]
    assert binding["source_score_correction_sidecar_digest"] is None
    assert [c["candidate_id"] for c in sidecar["candidates"]] == ["groot_n17_droid", "pi05_droid"]
    assert all(c["ranking_permitted"] is False for c in sidecar["candidates"])
    assert all(c["episode_count"] == 10 for c in sidecar["candidates"])
    assert len(sidecar["episodes"]) == 20
    # Without a correction the recorded scores are the published ones.
    assert all(e["graded"]["task_succeeded"] is False for e in sidecar["episodes"])
    assert all(e["graded"]["inputs"]["state_trace_digest"] for e in sidecar["episodes"])
    assert sidecar["audit"]["deterministic_scores_unchanged"] is True
    assert sidecar["sidecar_digest"] == cross_runtime_canonical_digest(sidecar, digest_field="sidecar_digest")


def test_a_published_score_correction_is_what_gets_graded(tmp_path: Path) -> None:
    record, source, evidence = _site_record(tmp_path, correction=True)
    sidecar = _build(record, source, evidence)
    assert sidecar["source_binding"]["source_score_correction_sidecar_digest"] == "sha256:" + "9" * 64
    assert all(e["graded"]["task_succeeded"] is True for e in sidecar["episodes"])
    assert all(e["graded"]["graded_score"] == 1.0 for e in sidecar["episodes"])
    assert all(c["success_count"] == 10 for c in sidecar["candidates"])


def test_tampered_or_unbound_evidence_is_refused(tmp_path: Path) -> None:
    record, source, evidence = _site_record(tmp_path)
    artifact = next(evidence.glob("*.state_trace.json"))
    artifact.write_text(artifact.read_text().replace("0.83", "0.84"))
    with pytest.raises(GradedReportError, match="artifact_inventory_invalid"):
        _build(record, source, evidence)

    record, source, evidence = _site_record(tmp_path / "second")
    result = json.loads(source.read_text())
    result["episodes"][0]["episode"]["state_trace"]["task_state_samples"][0]["step_index"] = 99
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    source.write_text(json.dumps(result))
    with pytest.raises(GradedReportError, match="source_result_unbound"):
        _build(record, source, evidence)

    record, source, evidence = _site_record(tmp_path / "third")
    record["publication"]["policy_canary_result"]["episodes"][0]["seed"] = 7
    projection = record["publication"]["policy_canary_result"]
    projection["projection_digest"] = cross_runtime_canonical_digest(projection, digest_field="projection_digest")
    with pytest.raises(GradedReportError, match="episode_evidence_missing"):
        _build(record, source, evidence)


def test_a_publication_that_does_not_verify_is_refused(tmp_path: Path) -> None:
    record, source, evidence = _site_record(tmp_path)
    record["publication"]["policy_canary_result"]["run_id"] = "another-run"
    with pytest.raises(GradedReportError, match="projection_invalid"):
        _build(record, source, evidence)
    record, source, evidence = _site_record(tmp_path / "b")
    with pytest.raises(GradedReportError, match="generated_at_invalid"):
        _build(record, source, evidence, generated_at_iso="2026-09-23T21:00:00")
