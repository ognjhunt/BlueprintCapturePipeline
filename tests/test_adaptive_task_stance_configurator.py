"""Hermetic tests for the bounded adaptive task-stance configurator loop."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from blueprint_pipeline.adaptive_task_stance_configurator import (
    AGENT_WAIVER_BLOCKER,
    ALL_STANCE_GATE_IDS,
    BUDGET_EXHAUSTED_BLOCKER,
    EVALUATIONS_ARTIFACT_NAME,
    MAX_CANDIDATE_RADIUS_M,
    MEASUREMENT_ERROR_BLOCKER,
    RENDER_EVIDENCE_GATE_ID,
    RESULT_ARTIFACT_NAME,
    SCHEMA_VERSION,
    STANCE_GATE_IDS,
    default_search_region,
    evaluate_stance_gates,
    generate_stance_candidate,
    normalize_stance_search_request,
    refine_search_region,
    run_adaptive_stance_search,
    validate_agent_proposal,
)


REFERENCE_POSE = [-1.229635, 1.471274, 0.84]
REFERENCE_YAW = 3.141593


def make_request(**overrides):
    request = {
        "kitchen_scene_digest": "sha256:kitchen-scene-digest",
        "task_id": "microwave_door",
        "completion_contract": {
            "registered_criteria": [{"criterion": "microwave_door_open_angle_ge_60deg"}]
        },
        "target_prim_path": "/root/Microwave017",
        "affordance_prim_path": "/root/Microwave017/Microwave017_Door",
        "robot_profile": {
            "collision_radius_m": 0.35,
            "min_reach_m": 0.35,
            "max_reach_m": 0.95,
        },
        "reference_pose_xyz": list(REFERENCE_POSE),
        "reference_yaw_rad": REFERENCE_YAW,
        "camera_limits": {
            "min_affordance_visible_fraction": 0.5,
            "min_robot_in_frame_fraction": 0.2,
            "min_target_in_frame_fraction": 0.2,
        },
        "search_budget": {"max_candidates": 4},
        "seed": 7,
    }
    request.update(overrides)
    return request


def make_render_pngs(tmp_path):
    robot = tmp_path / "robot_pov.png"
    third = tmp_path / "third_person.png"
    robot.write_bytes(b"\x89PNG\r\n\x1a\n fake-robot-pov")
    third.write_bytes(b"\x89PNG\r\n\x1a\n fake-third-person")
    return robot, third


def passing_metrics(robot_png, third_png):
    return {
        "floor_support_error_m": 0.004,
        "uprightness_deg": 1.2,
        "min_clearance_m": 0.55,
        "yaw_error_rad": 0.04,
        "reach_distance_m": 0.62,
        "affordance_visible_fraction": 0.82,
        "robot_in_frame_fraction": 0.44,
        "target_in_frame_fraction": 0.61,
        "robot_pov_png": str(robot_png),
        "third_person_png": str(third_png),
        "render_source": "isaac_rtx_rgb",
    }


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Request normalization
# ---------------------------------------------------------------------------


def test_normalize_good_request():
    normalized = normalize_stance_search_request(make_request())
    assert normalized["task_id"] == "microwave_door"
    assert normalized["kitchen_scene_digest"] == "sha256:kitchen-scene-digest"
    assert normalized["reference_pose_xyz"] == pytest.approx(REFERENCE_POSE)
    assert normalized["reference_yaw_rad"] == pytest.approx(REFERENCE_YAW)
    assert normalized["robot_profile"]["collision_radius_m"] == pytest.approx(0.35)
    assert normalized["robot_profile"]["min_reach_m"] == pytest.approx(0.35)
    assert normalized["robot_profile"]["max_reach_m"] == pytest.approx(0.95)
    assert normalized["search_budget"] == {"max_candidates": 4}
    assert normalized["seed"] == 7
    assert normalized["yaw_tolerance_rad"] > 0


def test_normalize_optional_max_seconds_and_seed_default():
    request = make_request(search_budget={"max_candidates": 2, "max_seconds": 12.5})
    del request["seed"]
    normalized = normalize_stance_search_request(request)
    assert normalized["search_budget"] == {"max_candidates": 2, "max_seconds": 12.5}
    assert normalized["seed"] == 0


@pytest.mark.parametrize(
    "missing",
    [
        "kitchen_scene_digest",
        "task_id",
        "completion_contract",
        "target_prim_path",
        "affordance_prim_path",
        "robot_profile",
        "reference_pose_xyz",
        "reference_yaw_rad",
        "camera_limits",
        "search_budget",
    ],
)
def test_normalize_missing_required_field(missing):
    request = make_request()
    del request[missing]
    with pytest.raises(ValueError):
        normalize_stance_search_request(request)


def test_normalize_rejects_affordance_not_descendant():
    request = make_request(affordance_prim_path="/root/Refrigerator001/Door")
    with pytest.raises(ValueError, match="affordance_not_descendant"):
        normalize_stance_search_request(request)
    request = make_request(affordance_prim_path="/root/Microwave017")
    with pytest.raises(ValueError, match="affordance_not_descendant"):
        normalize_stance_search_request(request)


def test_normalize_rejects_bad_robot_profile_and_budget():
    bad_profile = make_request(robot_profile={"collision_radius_m": 0.35})
    with pytest.raises(ValueError, match="reach_envelope"):
        normalize_stance_search_request(bad_profile)
    inverted = make_request(
        robot_profile={"collision_radius_m": 0.35, "min_reach_m": 1.0, "max_reach_m": 0.5}
    )
    with pytest.raises(ValueError, match="reach_envelope"):
        normalize_stance_search_request(inverted)
    no_radius = make_request(robot_profile={"min_reach_m": 0.35, "max_reach_m": 0.95})
    with pytest.raises(ValueError, match="collision_radius"):
        normalize_stance_search_request(no_radius)
    zero_budget = make_request(search_budget={"max_candidates": 0})
    with pytest.raises(ValueError, match="max_candidates"):
        normalize_stance_search_request(zero_budget)
    bad_pose = make_request(reference_pose_xyz=[1.0, 2.0])
    with pytest.raises(ValueError, match="reference_pose_xyz"):
        normalize_stance_search_request(bad_pose)


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------


GATE_FAILURE_CASES = [
    ("floor_support", "floor_support_error_m", 0.2, "floor_support_error_exceeds_tolerance"),
    ("uprightness", "uprightness_deg", 25.0, "uprightness_tilt_exceeds_tolerance"),
    ("collision_clearance", "min_clearance_m", 0.05, "collision_clearance_below_robot_radius"),
    (
        "target_facing_yaw",
        "yaw_error_rad",
        1.2,
        "target_facing_yaw_error_exceeds_tolerance",
    ),
    ("reach_envelope", "reach_distance_m", 2.4, "reach_distance_outside_envelope"),
    (
        "affordance_visibility",
        "affordance_visible_fraction",
        0.1,
        "affordance_visibility_below_minimum",
    ),
    (
        "robot_target_framing",
        "robot_in_frame_fraction",
        0.0,
        "robot_target_framing_insufficient",
    ),
]


@pytest.mark.parametrize("gate_id,metric,bad_value,reason", GATE_FAILURE_CASES)
def test_each_gate_fails_with_specific_reason(tmp_path, gate_id, metric, bad_value, reason):
    request = normalize_stance_search_request(make_request())
    robot, third = make_render_pngs(tmp_path)
    metrics = passing_metrics(robot, third)
    metrics[metric] = bad_value
    evaluation = evaluate_stance_gates(metrics, request=request)
    assert evaluation["all_gates_passed"] is False
    assert evaluation["rejection_reasons"] == [reason]
    by_gate = {row["gate_id"]: row for row in evaluation["gates"]}
    assert by_gate[gate_id]["status"] == "FAIL"
    assert by_gate[gate_id]["rejection_reason"] == reason
    for other_id, row in by_gate.items():
        if other_id != gate_id:
            assert row["status"] == "PASS"


def test_gate_rows_cover_fixed_gate_tuple(tmp_path):
    request = normalize_stance_search_request(make_request())
    robot, third = make_render_pngs(tmp_path)
    evaluation = evaluate_stance_gates(passing_metrics(robot, third), request=request)
    assert tuple(row["gate_id"] for row in evaluation["gates"]) == ALL_STANCE_GATE_IDS
    assert ALL_STANCE_GATE_IDS == STANCE_GATE_IDS + (RENDER_EVIDENCE_GATE_ID,)
    assert evaluation["all_gates_passed"] is True


def test_missing_metric_fails_closed(tmp_path):
    request = normalize_stance_search_request(make_request())
    robot, third = make_render_pngs(tmp_path)
    metrics = passing_metrics(robot, third)
    del metrics["reach_distance_m"]
    evaluation = evaluate_stance_gates(metrics, request=request)
    assert "reach_distance_outside_envelope" in evaluation["rejection_reasons"]
    assert evaluation["all_gates_passed"] is False


# ---------------------------------------------------------------------------
# Render evidence gate
# ---------------------------------------------------------------------------


def render_gate_row(evaluation):
    return next(
        row for row in evaluation["gates"] if row["gate_id"] == RENDER_EVIDENCE_GATE_ID
    )


def test_render_evidence_missing_png_fails(tmp_path):
    request = normalize_stance_search_request(make_request())
    robot, third = make_render_pngs(tmp_path)
    metrics = passing_metrics(robot, third)
    metrics["robot_pov_png"] = str(tmp_path / "does_not_exist.png")
    evaluation = evaluate_stance_gates(metrics, request=request)
    assert render_gate_row(evaluation)["status"] == "FAIL"
    assert "render_evidence_not_fresh_isaac_rgb" in evaluation["rejection_reasons"]
    assert evaluation["all_gates_passed"] is False

    metrics = passing_metrics(robot, third)
    del metrics["third_person_png"]
    evaluation = evaluate_stance_gates(metrics, request=request)
    assert render_gate_row(evaluation)["status"] == "FAIL"


@pytest.mark.parametrize("source", ["schematic", "dry_preview", "", "isaac_rtx_rgb_stale"])
def test_render_evidence_non_isaac_source_fails(tmp_path, source):
    request = normalize_stance_search_request(make_request())
    robot, third = make_render_pngs(tmp_path)
    metrics = passing_metrics(robot, third)
    metrics["render_source"] = source
    evaluation = evaluate_stance_gates(metrics, request=request)
    assert render_gate_row(evaluation)["status"] == "FAIL"
    assert evaluation["all_gates_passed"] is False


def test_render_evidence_real_files_and_isaac_source_passes(tmp_path):
    request = normalize_stance_search_request(make_request())
    robot, third = make_render_pngs(tmp_path)
    evaluation = evaluate_stance_gates(passing_metrics(robot, third), request=request)
    assert render_gate_row(evaluation)["status"] == "PASS"
    assert evaluation["all_gates_passed"] is True


# ---------------------------------------------------------------------------
# Full accepted run
# ---------------------------------------------------------------------------


def test_accepted_run_writes_artifacts_and_claim_boundary(tmp_path):
    robot, third = make_render_pngs(tmp_path)
    out_dir = tmp_path / "out"
    calls = []

    def measure(candidate):
        calls.append(candidate)
        return passing_metrics(robot, third)

    result = run_adaptive_stance_search(
        request=make_request(), measure_candidate=measure, out_dir=out_dir
    )
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["status"] == "accepted"
    assert result["blockers"] == []
    assert result["candidates_evaluated"] == 1
    assert result["accepted_candidate"]["candidate_source"] == "deterministic_generator"
    assert result["evidence"]["robot_pov_png"] == str(robot)
    assert result["evidence"]["third_person_png"] == str(third)
    assert result["evidence"]["render_source"] == "isaac_rtx_rgb"
    assert result["provider_revalidation_required"] is True
    assert "provider" in result["provider_acceptance_note"].lower()
    assert result["claim_boundary"] == {
        "local_gate_acceptance_only": True,
        "provider_acceptance_proven": False,
        "proves_task_success": False,
        "agent_cannot_waive_gates": True,
    }
    final = read_json(out_dir / RESULT_ARTIFACT_NAME)
    assert final["status"] == "accepted"
    evaluations = read_json(out_dir / EVALUATIONS_ARTIFACT_NAME)
    assert evaluations["schema_version"] == SCHEMA_VERSION
    assert len(evaluations["candidate_evaluations"]) == 1
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Budget exhaustion is blocked, never accepted
# ---------------------------------------------------------------------------


def test_budget_exhaustion_blocked_even_when_last_candidate_is_close(tmp_path):
    robot, third = make_render_pngs(tmp_path)
    out_dir = tmp_path / "out"

    def measure(candidate):
        # Every physical gate passes; only the render evidence is a dry preview.
        metrics = passing_metrics(robot, third)
        metrics["render_source"] = "dry_preview"
        return metrics

    result = run_adaptive_stance_search(
        request=make_request(search_budget={"max_candidates": 3}),
        measure_candidate=measure,
        out_dir=out_dir,
    )
    assert result["status"] == "blocked"
    assert result["status"] != "accepted"
    assert result["status"] != "completed"
    assert BUDGET_EXHAUSTED_BLOCKER in result["blockers"]
    assert result["candidates_evaluated"] == 3
    assert result["accepted_candidate"] is None
    final = read_json(out_dir / RESULT_ARTIFACT_NAME)
    assert final["status"] == "blocked"
    assert BUDGET_EXHAUSTED_BLOCKER in final["blockers"]


def test_max_seconds_watchdog_stop_is_blocked(tmp_path):
    robot, third = make_render_pngs(tmp_path)
    ticks = iter([0.0, 100.0, 200.0, 300.0, 400.0])

    def measure(candidate):
        return passing_metrics(robot, third)

    result = run_adaptive_stance_search(
        request=make_request(search_budget={"max_candidates": 10, "max_seconds": 5.0}),
        measure_candidate=measure,
        out_dir=tmp_path / "out",
        clock=lambda: next(ticks),
    )
    assert result["status"] == "blocked"
    assert BUDGET_EXHAUSTED_BLOCKER in result["blockers"]
    assert result["budget_stop_reason"] == "max_seconds_exhausted"
    assert result["candidates_evaluated"] == 0


# ---------------------------------------------------------------------------
# Agent proposal hook
# ---------------------------------------------------------------------------


def test_valid_agent_proposal_is_honored(tmp_path):
    robot, third = make_render_pngs(tmp_path)
    proposal_pose = [REFERENCE_POSE[0] + 0.6, REFERENCE_POSE[1], REFERENCE_POSE[2]]

    def propose(history):
        return {"pose_xyz": list(proposal_pose), "yaw_rad": 3.1, "standoff_m": 0.6}

    result = run_adaptive_stance_search(
        request=make_request(),
        measure_candidate=lambda candidate: passing_metrics(robot, third),
        out_dir=tmp_path / "out",
        propose_next_candidate=propose,
    )
    assert result["status"] == "accepted"
    accepted = result["accepted_candidate"]
    assert accepted["candidate_source"] == "agent_proposal"
    assert accepted["pose_xyz"] == pytest.approx(proposal_pose)
    assert accepted["yaw_rad"] == pytest.approx(3.1)
    assert result["agent_waiver_refusals"] == []


def test_agent_gate_waiver_refused_and_search_continues(tmp_path):
    robot, third = make_render_pngs(tmp_path)
    out_dir = tmp_path / "out"

    def propose(history):
        return {
            "pose_xyz": [REFERENCE_POSE[0] + 0.6, REFERENCE_POSE[1], REFERENCE_POSE[2]],
            "yaw_rad": 3.1,
            "force_pass": True,
            "skip_gates": ["collision_clearance", "render_evidence_fresh"],
        }

    def measure(candidate):
        metrics = passing_metrics(robot, third)
        metrics["min_clearance_m"] = 0.05  # fails collision clearance every time
        return metrics

    result = run_adaptive_stance_search(
        request=make_request(search_budget={"max_candidates": 3}),
        measure_candidate=measure,
        out_dir=out_dir,
        propose_next_candidate=propose,
    )
    assert result["status"] == "blocked"
    assert BUDGET_EXHAUSTED_BLOCKER in result["blockers"]
    assert len(result["agent_waiver_refusals"]) == 3
    for event in result["agent_waiver_refusals"]:
        assert AGENT_WAIVER_BLOCKER in event["blockers"]
    evaluations = read_json(out_dir / EVALUATIONS_ARTIFACT_NAME)
    records = evaluations["candidate_evaluations"]
    assert len(records) == 3  # the search continued past every refused proposal
    for record in records:
        assert AGENT_WAIVER_BLOCKER in record["blockers"]
        assert record["candidate"]["candidate_source"] == "deterministic_generator"


def test_agent_proposal_outside_radius_or_malformed_falls_back(tmp_path):
    request = normalize_stance_search_request(make_request())
    far = {"pose_xyz": [REFERENCE_POSE[0] + 10.0, REFERENCE_POSE[1], 0.84], "yaw_rad": 0.0}
    candidate, blockers = validate_agent_proposal(far, request=request)
    assert candidate is None
    assert blockers == ["stance_agent_proposal_outside_sane_radius"]
    candidate, blockers = validate_agent_proposal({"yaw_rad": 0.0}, request=request)
    assert candidate is None
    assert blockers == ["stance_agent_proposal_invalid"]
    candidate, blockers = validate_agent_proposal("not-a-mapping", request=request)
    assert candidate is None
    assert blockers == ["stance_agent_proposal_invalid"]


@pytest.mark.parametrize(
    "waiver_key", ["gates", "thresholds", "force_pass", "waive", "skip_gates", "render_source"]
)
def test_agent_waiver_keys_each_refused(waiver_key):
    request = normalize_stance_search_request(make_request())
    proposal = {
        "pose_xyz": [REFERENCE_POSE[0] + 0.6, REFERENCE_POSE[1], REFERENCE_POSE[2]],
        "yaw_rad": 3.1,
        waiver_key: True,
    }
    candidate, blockers = validate_agent_proposal(proposal, request=request)
    assert candidate is None
    assert blockers == [AGENT_WAIVER_BLOCKER]


# ---------------------------------------------------------------------------
# refine_search_region
# ---------------------------------------------------------------------------


def test_refine_search_region_clearance_failure_grows_standoff():
    region = {"standoff_m": 0.65, "yaw_correction_rad": 0.0}
    gate_rows = [{"gate_id": "collision_clearance", "status": "FAIL", "threshold": {}}]
    refined = refine_search_region(region, gate_rows, {"min_clearance_m": 0.1})
    assert refined["standoff_m"] > region["standoff_m"]
    assert region == {"standoff_m": 0.65, "yaw_correction_rad": 0.0}  # pure, not mutated


def test_refine_search_region_reach_too_far_shrinks_standoff():
    region = {"standoff_m": 1.0, "yaw_correction_rad": 0.0}
    gate_rows = [
        {
            "gate_id": "reach_envelope",
            "status": "FAIL",
            "threshold": {"min_reach_m": 0.35, "max_reach_m": 0.95},
        }
    ]
    refined = refine_search_region(region, gate_rows, {"reach_distance_m": 1.4})
    assert refined["standoff_m"] < region["standoff_m"]


def test_refine_search_region_reach_too_close_grows_standoff():
    region = {"standoff_m": 0.4, "yaw_correction_rad": 0.0}
    gate_rows = [
        {
            "gate_id": "reach_envelope",
            "status": "FAIL",
            "threshold": {"min_reach_m": 0.35, "max_reach_m": 0.95},
        }
    ]
    refined = refine_search_region(region, gate_rows, {"reach_distance_m": 0.1})
    assert refined["standoff_m"] > region["standoff_m"]


def test_refine_search_region_yaw_failure_applies_correction():
    region = {"standoff_m": 0.65, "yaw_correction_rad": 0.0}
    gate_rows = [{"gate_id": "target_facing_yaw", "status": "FAIL", "threshold": {}}]
    refined = refine_search_region(region, gate_rows, {"yaw_error_rad": 0.5})
    assert refined["yaw_correction_rad"] == pytest.approx(-0.5)
    assert refined["standoff_m"] == pytest.approx(region["standoff_m"])


def test_refine_search_region_no_failures_keeps_region():
    region = {"standoff_m": 0.65, "yaw_correction_rad": 0.1}
    refined = refine_search_region(region, [], {})
    assert refined["standoff_m"] == pytest.approx(0.65)
    assert refined["yaw_correction_rad"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Per-candidate evaluations artifact
# ---------------------------------------------------------------------------


def test_evaluations_file_records_every_candidate_with_exact_metrics(tmp_path):
    robot, third = make_render_pngs(tmp_path)
    out_dir = tmp_path / "out"
    clearances = [0.05, 0.11, 0.22]

    def measure(candidate):
        metrics = passing_metrics(robot, third)
        metrics["min_clearance_m"] = clearances[len_records()]
        return metrics

    seen = []

    def len_records():
        return len(seen)

    def measure_tracked(candidate):
        metrics = measure(candidate)
        seen.append(metrics)
        return metrics

    result = run_adaptive_stance_search(
        request=make_request(search_budget={"max_candidates": 3}),
        measure_candidate=measure_tracked,
        out_dir=out_dir,
    )
    assert result["status"] == "blocked"
    evaluations = read_json(out_dir / EVALUATIONS_ARTIFACT_NAME)
    records = evaluations["candidate_evaluations"]
    assert [record["candidate_index"] for record in records] == [0, 1, 2]
    for record, clearance in zip(records, clearances):
        assert record["metrics"]["min_clearance_m"] == pytest.approx(clearance)
        assert record["rejection_reasons"] == ["collision_clearance_below_robot_radius"]
        assert record["robot_pov_png"] == str(robot)
        assert record["third_person_png"] == str(third)
        gate_ids = [row["gate_id"] for row in record["gates"]]
        assert tuple(gate_ids) == ALL_STANCE_GATE_IDS


def test_measure_candidate_exception_recorded_and_loop_continues(tmp_path):
    robot, third = make_render_pngs(tmp_path)
    out_dir = tmp_path / "out"
    calls = {"count": 0}

    def measure(candidate):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("isaac worker crashed during settle")
        return passing_metrics(robot, third)

    result = run_adaptive_stance_search(
        request=make_request(search_budget={"max_candidates": 3}),
        measure_candidate=measure,
        out_dir=out_dir,
    )
    assert result["status"] == "accepted"
    assert result["candidates_evaluated"] == 2
    evaluations = read_json(out_dir / EVALUATIONS_ARTIFACT_NAME)
    records = evaluations["candidate_evaluations"]
    assert len(records) == 2
    assert MEASUREMENT_ERROR_BLOCKER in records[0]["blockers"]
    assert any(
        MEASUREMENT_ERROR_BLOCKER in reason for reason in records[0]["rejection_reasons"]
    )
    assert records[1]["rejection_reasons"] == []


# ---------------------------------------------------------------------------
# Deterministic candidate generation
# ---------------------------------------------------------------------------


def test_generate_stance_candidate_is_deterministic_and_scene_grounded():
    request = normalize_stance_search_request(make_request(seed=42))
    region = default_search_region(request)
    for index in range(6):
        first = generate_stance_candidate(
            request=request, candidate_index=index, region=region
        )
        second = generate_stance_candidate(
            request=request, candidate_index=index, region=region
        )
        assert first == second
        distance = math.hypot(
            first["pose_xyz"][0] - REFERENCE_POSE[0],
            first["pose_xyz"][1] - REFERENCE_POSE[1],
        )
        assert distance <= MAX_CANDIDATE_RADIUS_M
        assert first["pose_xyz"][2] == pytest.approx(REFERENCE_POSE[2])
    other_seed = normalize_stance_search_request(make_request(seed=43))
    different = generate_stance_candidate(
        request=other_seed, candidate_index=0, region=region
    )
    baseline = generate_stance_candidate(request=request, candidate_index=0, region=region)
    assert different != baseline


def test_same_seed_reproduces_same_candidate_sequence_end_to_end(tmp_path):
    robot, third = make_render_pngs(tmp_path)

    def failing_measure(candidate):
        metrics = passing_metrics(robot, third)
        metrics["min_clearance_m"] = 0.05
        return metrics

    candidate_lists = []
    for run in ("a", "b"):
        out_dir = tmp_path / f"out_{run}"
        run_adaptive_stance_search(
            request=make_request(seed=99, search_budget={"max_candidates": 4}),
            measure_candidate=failing_measure,
            out_dir=out_dir,
        )
        evaluations = read_json(out_dir / EVALUATIONS_ARTIFACT_NAME)
        candidate_lists.append(
            [record["candidate"] for record in evaluations["candidate_evaluations"]]
        )
    assert candidate_lists[0] == candidate_lists[1]
    assert len(candidate_lists[0]) == 4
