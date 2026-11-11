"""How far each learned policy got: ``progress`` on a beta result, publication v4, and its downgrades.

Hermetic and synthetic: every score here comes from the real deterministic scorer
(``score_task_episode_from_spec``) on a fixture trace, every summary from the real
``candidate_result_summary`` and every publication from the real coordinator. None of
it was pulled from a host or derived from a real capture.

``BLUEPRINT_WRITE_SCREENING_PUBLICATION_FIXTURES=1`` rewrites the v4 fixture. The
WebApp commits the same bytes and pins the same sha256, so a change on either side
fails a test there.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
import pytest

from blueprint_pipeline import beta_policy_screening_coordinator as coordinator
from blueprint_pipeline import beta_policy_screening_progress as progress
from blueprint_pipeline import beta_policy_screening_worker as worker
from blueprint_pipeline.adp_task_scoring import score_task_episode_from_spec
from blueprint_pipeline.beta_policy_screening_plan import validate_screening_plan
from blueprint_pipeline.beta_policy_screening_publication_flags import (
    DOWNGRADE_FIELD, PUBLICATION_SCHEMA_V2, PUBLICATION_SCHEMA_V3, PUBLICATION_SCHEMA_V4, as_v2, as_v3,
)
from blueprint_pipeline.beta_policy_screening_result import candidate_result_summary, screening_publication
from blueprint_pipeline.decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest
from blueprint_pipeline.native_task_arena_policy_canary_session import validate_session_result
from blueprint_pipeline.policy_episode_graded_report import (
    APPROACHED_DISTANCE_M, PROGRESS_GRADER_VERSION, PROGRESS_STAGE_IDS, progress_episode, summarize_progress,
)
from blueprint_pipeline.policy_execution_evidence import SCRIPTED, rollout_counts, scripted_fields
from tests.test_beta_policy_screening_coordinator import native_result, plan as queue_plan, seal
from tests.test_beta_policy_screening_known_good_control import PerRow, closed, with_control
from tests.test_beta_policy_screening_partial_cells import _webapp_contract
from tests.test_beta_policy_screening_publication_flags import Website, state_of
from tests.test_policy_episode_graded_report import _task_spec

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests/fixtures/beta_policy_screening_progress_publication.v4.json"
#: The fixture's bytes. The WebApp pins the same value for its copy.
FIXTURE_SHA256 = "5ad8e51f64357ff520ad2b961c0db169e21ec9f5d82b17583cc04cdd33c75da6"
SCHEMA = {version: json.loads((ROOT / f"docs/schemas/beta_policy_screening_publication.{version}.schema.json").read_text())
          for version in ("v2", "v3", "v4")}

START, DESTINATION = (1.0, 2.0, 0.8), (1.15, 2.0, 0.8)
#: A rigid pick-and-place onto a support: the scorer measures the grasp frame's closest approach to the object's box.
SPEC = _task_spec(
    destination_relation="on", destination_pose_world=[*DESTINATION, 0.0, 0.0, 0.0, 1.0],
    destination_position_bounds_destination_frame_m={"minimum": [-0.01, -0.01, -0.01], "maximum": [0.01, 0.01, 0.01]},
    subject_collision_bounds_scoring_frame_m={"minimum": [-0.03, -0.03, -0.05], "maximum": [0.03, 0.03, 0.05]},
    destination_interior_bounds_body_frame_m={"minimum": [-0.1, -0.1, -0.06], "maximum": [0.1, 0.1, 0.06]},
    destination_reset_translation_tolerance_m=0.001, destination_reset_rotation_tolerance_rad=0.01)
#: How far each trace gets: the deepest stage of ``PROGRESS_STAGE_IDS`` it reaches (0: not even within reach).
DEPTH = {stage: index + 1 for index, stage in enumerate(PROGRESS_STAGE_IDS)} | {"none": 0}


def trace(depth: int, *, approach: bool = True, height: float | None = None) -> list[dict]:
    """A 14-step trace that gets exactly ``depth`` stages along the ladder; ``approach=False`` records no grasp frame.

    ``height``: where a trace that never gets within reach holds the grasp frame above the object's centre.
    """
    x0, y0, z0 = START
    path = [START] * 14
    if depth == DEPTH["moved"]:
        path = [START] * 4 + [(x0 + 0.01, y0, z0)] * 10
    elif depth == DEPTH["lifted"]:
        path = [START] * 3 + [(x0, y0, z0 + 0.03)] * 4 + [(x0 + 0.02, y0, z0)] * 7
    elif depth >= DEPTH["translated"]:
        end = DESTINATION if depth >= DEPTH["placed"] else (1.15, 2.2, z0)
        path = ([START] * 3 + [(x0, y0, z0 + 0.03), (1.07, (y0 + end[1]) / 2, z0 + 0.03), (*end[:2], z0 + 0.03)]
                + [end] * 8)
    holding = range(2, 8 if depth == DEPTH["released_and_settled"] else 14) if depth >= DEPTH["contact"] else range(0)
    offset = (height or 0.3) if depth == DEPTH["none"] else 0.06 if depth == DEPTH["approached"] else 0.04
    samples = []
    for step, position in enumerate(path):
        held = step in holding
        sample = {
            "step_index": step, "task_object_pose_world": [*position, 0.0, 0.0, 0.0, 1.0],
            "task_scoring_pose_world": [*position, 0.0, 0.0, 0.0, 1.0],
            "destination_pose_world": [*DESTINATION, 0.0, 0.0, 0.0, 1.0],
            "controlled_body_pose_world": [position[0], position[1], position[2] + offset, 0.0, 0.0, 0.0, 1.0],
            "gripper_width_m": 0.04 if held else 0.08, "task_contact_active": held,
            "support_contact_active": abs(position[2] - z0) < 1e-9, "containment_violation": False,
            "robot_collision_failure": False, "scene_collision_failure": False,
            "forbidden_robot_task_collision_failure": False, "locked_joint_containment_violation": False}
        if approach:
            sample["grasp_frame_position_world_m"] = [position[0], position[1], position[2] + offset]
        samples.append(sample)
    return samples


def episode(depth: int, *, approach: bool = True, height: float | None = None) -> tuple[dict, list[dict]]:
    samples = trace(depth, approach=approach, height=height)
    return score_task_episode_from_spec(task_spec=SPEC, samples=samples), samples


def session_result(value, candidate, depths, *, approach=True, scripted=False, heights=None):
    """The candidate's validated native session result, each episode carrying its real score and sealed trace."""
    if len(candidate["scenario_cells"]) == 10:
        result = native_result(value, candidate)
    else:  # The control's three cells: the same builder's result, rebound to that row.
        template = next(row for row in value["candidates"] if len(row["scenario_cells"]) == 10)
        result = deepcopy(native_result(value, template))
        binding = result["beta_screening_admission"]
        binding.update(candidate_id=candidate["candidate_id"], admission_digest=candidate["admission_digest"],
                       configuration_digest=candidate["configuration_digest"])
        result.update(candidate_ids=[candidate["candidate_id"]], configuration_digest=candidate["configuration_digest"],
                      episodes_per_policy=len(depths), learned_policy_rollout_count=len(depths))
        rows = result["episodes"][:len(depths)]
        for row, cell in zip(rows, candidate["scenario_cells"], strict=True):
            row.update(cell, candidate_id=candidate["candidate_id"],
                       runtime_identity_digest=candidate["execution_spec"]["runtime_digest"])
        result["episodes"] = rows
    if scripted:
        result.update(rollout_counts({candidate["candidate_id"]: SCRIPTED}, len(depths)),
                      scripted_controller_invoked=True, candidate_policy_queried=False)
    for index, (row, depth) in enumerate(zip(result["episodes"], depths, strict=True)):
        score, samples = episode(depth, approach=approach, height=heights[index] if heights else None)
        state = {"task_state_samples": samples}
        row["episode"] = {"score": score, "state_trace": state}
        row["deterministic_score_digest"] = canonical_digest({"value": score})
        row["state_trace_digest"] = canonical_digest({"value": state})
        if scripted:
            row.update(scripted_fields(invoked=True), scripted_outcome_interpretable=True)
    return validate_session_result(seal(result, "result_digest"))


#: Four learned policies with different ladders, one with no approach measurement, and the scripted control.
CAMPAIGN = {
    "policy-a": {"display_name": "Policy A", "depths": [7, 7, 6, 5, 5, 4, 3, 2, 1, 0]},
    "policy-b": {"display_name": "Policy B", "depths": [5, 4, 4, 3, 2, 2, 1, 1, 0, 0]},
    # Never within reach: the ladder is all zeros and the distance is what tells it apart.
    "policy-c": {"display_name": "Policy C", "depths": [0] * 10,
                 "heights": [0.12, 0.13, 0.14, 0.15, 0.16, 0.18, 0.2, 0.22, 0.25, 0.3]},
    "policy-d": {"display_name": "Policy D", "depths": [4, 3, 3, 2, 2, 2, 0, 0, 0, 0], "approach": False},
}


def campaign_plan():
    value = deepcopy(queue_plan(cost="3.00", count=len(CAMPAIGN)))
    for row, (candidate_id, options) in zip(value["candidates"], CAMPAIGN.items(), strict=True):
        row.update(candidate_id=candidate_id, display_name=options["display_name"])
        row["execution_spec"]["candidate_id"] = candidate_id
        row["admission_digest"] = canonical_digest(row["execution_spec"])
    value["campaign_id"] = "campaign-progress-1"
    return with_control(validate_screening_plan(seal(value, "plan_digest")))


def run_to_completion(root):
    value = campaign_plan()
    observations = {row["candidate_id"]: closed(session_result(
        value, row, CAMPAIGN[row["candidate_id"]]["depths"], approach=CAMPAIGN[row["candidate_id"]].get("approach", True),
        heights=CAMPAIGN[row["candidate_id"]].get("heights")))
        for row in value["candidates"][:-1]}
    control = value["candidates"][-1]
    observations[control["candidate_id"]] = closed(session_result(value, control, [7, 7, 6], scripted=True))
    coordinator.enqueue_screening_campaign(value, queue_root=root)
    backend = PerRow(observations)
    for _ in range(3 * len(value["candidates"])):
        coordinator.process_screening_queue(require_publication_ack=False, queue_root=root, executor=backend)
    [publication] = coordinator.pending_publications(queue_root=root)
    return value, publication


def stages(row):
    return [(item["stage"], item["episodes"]) for item in row["progress"]["stages"]]


# ------------------------------------------------------------------ the rung and the ladder


def test_each_trace_gets_exactly_as_far_as_built_and_the_grade_matches_the_score():
    for depth in range(len(PROGRESS_STAGE_IDS) + 1):
        score, samples = episode(depth)
        assert score["status"] == "scored"
        assert score["task_succeeded"] is (depth == len(PROGRESS_STAGE_IDS))
        value = summarize_progress([progress_episode(score=score, samples=samples)])
        assert [row["episodes"] for row in value["stages"]] == [1] * depth + [0] * (7 - depth)
        assert value["mean_score"] == round(depth / 7, 4)


def test_approached_is_the_scores_own_closest_approach_against_the_fixed_threshold():
    assert (PROGRESS_GRADER_VERSION, APPROACHED_DISTANCE_M) == ("policy_episode_progress.v1", 0.02)
    near, samples = episode(DEPTH["approached"])
    distance = near["behaviour"]["gripper"]["closest_approach_to_subject_bounds"]["distance_m"]
    assert distance == pytest.approx(0.01)
    assert progress_episode(score=near, samples=samples)["approach_distance_m"] == distance
    far, samples = episode(DEPTH["none"])
    assert far["behaviour"]["gripper"]["closest_approach_to_subject_bounds"]["distance_m"] > APPROACHED_DISTANCE_M
    assert summarize_progress([progress_episode(score=far, samples=samples)])["stages"][0] == {
        "stage": "approached", "episodes": 0}


def test_a_stage_counts_only_in_order_and_the_counts_never_increase():
    # The untouched object "released and settled" trivially; that earns nothing, nor does contact without approach.
    score, samples = episode(DEPTH["none"])
    assert score["criteria_satisfied"]["settling"] is True
    value = summarize_progress([progress_episode(score=score, samples=samples)])
    assert [row["episodes"] for row in value["stages"]] == [0] * 7 and value["mean_score"] == 0
    item = progress_episode(score=episode(DEPTH["placed"])[0], samples=trace(DEPTH["placed"]))
    item["approach_distance_m"] = 0.05  # contact and carry met, approach not: nothing counts
    assert summarize_progress([item])["mean_score"] == 0
    graded = [progress_episode(score=s, samples=t) for s, t in (episode(d) for d in CAMPAIGN["policy-a"]["depths"])]
    counts = [row["episodes"] for row in summarize_progress(graded)["stages"]]
    assert counts == sorted(counts, reverse=True) == [9, 8, 7, 6, 5, 3, 2]
    assert summarize_progress(graded)["mean_score"] == round(sum(CAMPAIGN["policy-a"]["depths"]) / 70, 4)
    assert summarize_progress([]) is None


def test_approached_is_left_out_when_any_graded_episode_lacks_the_measurement():
    with_it = [progress_episode(score=s, samples=t) for s, t in (episode(4), episode(2))]
    without = progress_episode(score=episode(4, approach=False)[0], samples=trace(4, approach=False))
    assert without["approach_distance_m"] is None
    value = summarize_progress([*with_it, without])
    assert [row["stage"] for row in value["stages"]] == list(PROGRESS_STAGE_IDS[1:])
    # On the six-stage ladder, depth 4 (lifted) is three stages, depth 2 (contact) is one.
    assert [row["episodes"] for row in value["stages"]] == [3, 2, 2, 0, 0, 0]
    assert value["mean_score"] == round((3 + 1 + 3) / 18, 4)
    assert "closest_approach_m" not in value
    assert [row["stage"] for row in summarize_progress(with_it)["stages"]] == list(PROGRESS_STAGE_IDS)
    # With the rung comes the distance itself: median and minimum over the graded episodes, 4 decimals.
    far = [progress_episode(score=s, samples=t) for s, t in (episode(0), episode(0), episode(1))]
    assert [item["approach_distance_m"] for item in far] == pytest.approx([0.25, 0.25, 0.01])
    assert summarize_progress(far)["closest_approach_m"] == {"median": 0.25, "min": 0.01}
    assert summarize_progress(with_it)["closest_approach_m"] == {"median": 0.0, "min": 0.0}


def test_only_the_pick_and_place_ladder_is_graded():
    score, samples = episode(5)
    assert progress_episode(score={**score, "manipulation_strategy": "planar_push"}, samples=samples) is None
    assert progress_episode(score={**score, "status": "undetermined"}, samples=samples) is None
    assert progress_episode(score=score, samples=[]) is None
    assert progress_episode(score={key: item for key, item in score.items() if key != "criteria_satisfied"},
                            samples=samples) is None


# ------------------------------------------------------------------ the summary


def test_a_grading_error_or_an_unbound_episode_is_skipped_and_never_fails_the_summary(monkeypatch):
    value = queue_plan()
    candidate = value["candidates"][0]
    result = session_result(value, candidate, [5] * 10)
    rows = deepcopy(result["episodes"])
    rows[0]["episode"]["state_trace"]["task_state_samples"][3]["task_contact_active"] = False  # trace not as sealed
    rows[1]["episode"]["score"] = {**rows[1]["episode"]["score"], "task_succeeded": True}  # score not as sealed
    rows[2]["episode"].pop("state_trace")
    graded, skipped = progress.grade_rows(rows)
    assert graded["episodes_graded"] == 7
    assert skipped == {"score_digest_mismatch": 1, "state_trace_digest_mismatch": 1, "state_trace_missing": 1}

    calls = []
    real = progress.progress_episode

    def flaky(**kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ZeroDivisionError("grader bug")
        return real(**kwargs)
    monkeypatch.setattr(progress, "progress_episode", flaky)
    summary = candidate_result_summary(result, plan=value, candidate=candidate)
    assert summary["episodes_scored"] == 10 and summary["progress"]["episodes_graded"] == 9
    monkeypatch.setattr(progress, "summarize_progress", lambda _: 1 / 0)
    assert "progress" not in candidate_result_summary(result, plan=value, candidate=candidate)


def test_a_summary_with_nothing_to_grade_keeps_its_bytes():
    value = queue_plan()
    candidate = value["candidates"][0]
    # Today's results: a score without a trace, so nothing is graded and nothing is added.
    plain = native_result(value)
    summary = candidate_result_summary(plain, plan=value, candidate=candidate)
    assert "progress" not in summary
    assert set(summary) == {"episodes_scored", "episodes_succeeded", "episodes_failed", "episodes_void", "success_rate",
                            "session_result_digest", "episode_outcomes"}
    # Every episode off the ladder: still no field, not null.
    result = native_result(value)
    for row in result["episodes"]:
        score, samples = episode(5)
        score = {**score, "manipulation_strategy": "planar_push"}
        row["episode"] = {"score": score, "state_trace": {"task_state_samples": samples}}
        row["deterministic_score_digest"] = canonical_digest({"value": score})
        row["state_trace_digest"] = canonical_digest({"value": row["episode"]["state_trace"]})
    assert "progress" not in candidate_result_summary(validate_session_result(seal(result, "result_digest")),
                                                      plan=value, candidate=candidate)


def test_graded_never_exceeds_scored_and_scripted_or_control_rows_carry_no_progress(tmp_path):
    value, publication = run_to_completion(tmp_path)
    ledger = state_of(tmp_path, value["campaign_id"])["candidates"]
    control_id = value["candidates"][-1]["candidate_id"]
    assert ledger[control_id]["summary"]["execution_evidence_kind"] == SCRIPTED
    assert "progress" not in ledger[control_id]["summary"]
    rows = {row["candidate_id"]: row for row in publication["candidates"]}
    assert "progress" not in rows[control_id]
    for candidate_id in CAMPAIGN:
        assert rows[candidate_id]["progress"] == ledger[candidate_id]["summary"]["progress"]
        assert rows[candidate_id]["progress"]["episodes_graded"] <= rows[candidate_id]["episodes_scored"]
    # A scripted run of a learned row's own cells is never graded either.
    summary = candidate_result_summary(session_result(value, value["candidates"][0], [7] * 10, scripted=True),
                                       plan=value, candidate=value["candidates"][0])
    assert summary["execution_evidence_kind"] == SCRIPTED and "progress" not in summary
    assert progress.publication_progress({"candidate_id": "x"}, {"status": "completed"},
                                         {**ledger["policy-a"]["summary"], "execution_evidence_kind": SCRIPTED}) == {}
    assert progress.publication_progress(value["candidates"][-1], {"status": "completed"},
                                         ledger["policy-a"]["summary"]) == {}


@pytest.mark.parametrize("change", ["version", "order", "increase", "over_graded", "over_scored", "mean", "extra",
                                    "distance_without_rung", "rung_without_distance", "min_over_median",
                                    "distance_precision"])
def test_a_malformed_progress_is_never_published(change):
    good = {"grader_version": PROGRESS_GRADER_VERSION, "episodes_graded": 4, "mean_score": 0.25,
            "stages": [{"stage": stage, "episodes": count} for stage, count in zip(PROGRESS_STAGE_IDS[1:], [3, 2, 1, 0, 0, 0])]}
    near = {**deepcopy(good), "stages": [{"stage": "approached", "episodes": 4}, *deepcopy(good["stages"])],
            "closest_approach_m": {"median": 0.0123, "min": 0.0}}
    assert progress.progress_valid(good, episodes_scored=4) and progress.progress_valid(near, episodes_scored=4)
    bad = deepcopy(near if change in {"rung_without_distance", "min_over_median", "distance_precision"} else good)
    if change == "distance_without_rung":
        bad["closest_approach_m"] = {"median": 0.1, "min": 0.1}
    elif change == "rung_without_distance":
        bad.pop("closest_approach_m")
    elif change == "min_over_median":
        bad["closest_approach_m"]["min"] = 0.5
    elif change == "distance_precision":
        bad["closest_approach_m"]["median"] = 0.012345

    elif change == "version":
        bad["grader_version"] = "policy_episode_progress.v0"
    elif change == "order":
        bad["stages"][0], bad["stages"][1] = bad["stages"][1], bad["stages"][0]
    elif change == "increase":
        bad["stages"][2]["episodes"] = 3
    elif change == "over_graded":
        bad["stages"][0]["episodes"] = 5
    elif change == "over_scored":
        bad["episodes_graded"] = 5
    elif change == "mean":
        bad["mean_score"] = 1.5
    else:
        bad["stages"][0]["label"] = "Touched the object"
    assert not progress.progress_valid(bad, episodes_scored=4)
    assert progress.publication_progress({"candidate_id": "x"}, {"status": "completed"},
                                         {"episodes_scored": 4, "progress": bad}) == {}


# ------------------------------------------------------------------ publication v4 and its downgrades


def test_a_campaign_with_progress_publishes_v4_and_each_earlier_form_is_the_same_result(tmp_path):
    _, publication = run_to_completion(tmp_path)
    assert publication["schema_version"] == PUBLICATION_SCHEMA_V4 and publication["status"] == "completed"
    Draft202012Validator(SCHEMA["v4"], format_checker=FormatChecker()).validate(publication)
    rows = {row["candidate_id"]: row for row in publication["candidates"]}
    assert stages(rows["policy-a"]) == [("approached", 9), ("contact", 8), ("moved", 7), ("lifted", 6),
                                        ("translated", 5), ("placed", 3), ("released_and_settled", 2)]
    assert stages(rows["policy-d"])[0] == ("contact", 6) and len(stages(rows["policy-d"])) == 6
    assert "closest_approach_m" not in rows["policy-d"]["progress"]
    assert [rows[key]["progress"]["closest_approach_m"] for key in ("policy-a", "policy-b", "policy-c")] == [
        {"median": 0.0, "min": 0.0}, {"median": 0.0, "min": 0.0}, {"median": 0.12, "min": 0.07}]
    assert [row["episodes"] for row in rows["policy-c"]["progress"]["stages"]] == [0] * 7
    assert [rows[key]["progress"]["mean_score"] for key in CAMPAIGN] == [0.5714, 0.3143, 0.0, 0.1667]
    assert [rows[key]["episodes_succeeded"] for key in CAMPAIGN] == [2, 0, 0, 0]

    v3 = as_v3(publication)
    assert v3["schema_version"] == PUBLICATION_SCHEMA_V3 and not any("progress" in row for row in v3["candidates"])
    Draft202012Validator(SCHEMA["v3"], format_checker=FormatChecker()).validate(v3)
    assert v3["publication_digest"] == cross_runtime_canonical_digest(v3, digest_field="publication_digest")
    assert as_v2(publication) == as_v2(v3)
    Draft202012Validator(SCHEMA["v2"]).validate(as_v2(publication))
    _webapp_contract(as_v2(publication))
    with pytest.raises(Exception):
        Draft202012Validator(SCHEMA["v3"]).validate({**publication, "schema_version": PUBLICATION_SCHEMA_V3})
    facts = ("candidate_id", "status", "episodes_scored", "episodes_succeeded", "episodes_failed", "success_rate")
    assert [{key: row[key] for key in facts} for row in v3["candidates"]] == [
        {key: row[key] for key in facts} for row in publication["candidates"]]

    if os.environ.get("BLUEPRINT_WRITE_SCREENING_PUBLICATION_FIXTURES") == "1":
        FIXTURE.write_text(json.dumps(publication, indent=2, sort_keys=True) + "\n")
    retained = json.loads(FIXTURE.read_text())
    progress_of = lambda value: [(row["candidate_id"], row.get("progress")) for row in value["candidates"]]  # noqa: E731
    assert progress_of(retained) == progress_of(publication)
    assert [{key: row[key] for key in facts} for row in retained["candidates"]] == [
        {key: row[key] for key in facts} for row in publication["candidates"]]


def test_the_v4_fixture_shared_with_the_webapp_is_valid_and_unchanged():
    raw = FIXTURE.read_bytes()
    value = json.loads(raw)
    Draft202012Validator(SCHEMA["v4"], format_checker=FormatChecker()).validate(value)
    assert value["publication_digest"] == cross_runtime_canonical_digest(value, digest_field="publication_digest")
    rows = [row for row in value["candidates"] if "progress" in row]
    assert len(rows) == 4 and all(progress.progress_valid(row["progress"], episodes_scored=row["episodes_scored"])
                                  for row in rows)
    assert {len(row["progress"]["stages"]) for row in rows} == {6, 7}
    assert all(("closest_approach_m" in row["progress"]) == (len(row["progress"]["stages"]) == 7) for row in rows)
    assert any(row.get("execution_evidence_kind") == SCRIPTED and "progress" not in row for row in value["candidates"])
    # Change the fixture in both repositories together, and this value in both tests.
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256


def test_a_publication_without_progress_is_v3_byte_for_byte(tmp_path):
    from tests.test_beta_policy_screening_publication_flags import completed, flagged_plan

    value = flagged_plan()
    campaign_id = completed(tmp_path, value)
    [publication] = coordinator.pending_publications(queue_root=tmp_path)
    assert publication["schema_version"] == PUBLICATION_SCHEMA_V3
    assert as_v3(publication) == publication
    ledger = json.loads((tmp_path / coordinator.LEDGER_FILENAME).read_text())
    state = ledger["campaigns"][campaign_id]
    assert screening_publication(plan=value, state=state, budget=coordinator._published_budget(ledger, state)) == publication


@pytest.mark.parametrize("typed", [True, False])
def test_a_website_that_refuses_v4_is_sent_v3_in_the_same_pass_and_v2_if_it_refuses_that(tmp_path, typed):
    _, offered = run_to_completion(tmp_path)
    website = Website(accepts=("v1", "v2", "v3"), typed=typed)
    assert worker.deliver_publications(queue_root=tmp_path, transport=website) == {
        "acknowledged": 1, "pending": 0, "downgraded_to_v3": 1}
    first, second = website.received
    assert first == offered and second == as_v3(offered)
    state = state_of(tmp_path, offered["campaign_id"])
    assert state[DOWNGRADE_FIELD] | {"recorded_at": None} == {
        "revision": offered["revision"], "from_schema_version": PUBLICATION_SCHEMA_V4,
        "to_schema_version": PUBLICATION_SCHEMA_V3, "reason": "schema_not_accepted" if typed else "http_400",
        "refused_publication_digest": offered["publication_digest"], "recorded_at": None}
    assert state["published_digest"] == second["publication_digest"]
    # The revision stays v3: recomputed from the ledger, it is the bytes the website holds.
    assert coordinator.pending_publications(queue_root=tmp_path) == []
    assert worker.deliver_publications(queue_root=tmp_path, transport=website) == {"acknowledged": 0, "pending": 0}

    # A website that accepts only v2: v4, v3 and v2 in one pass, each step recorded before it is sent.
    root = tmp_path / "older"
    run_to_completion(root)
    older = Website(accepts=("v1", "v2"), typed=typed)
    assert worker.deliver_publications(queue_root=root, transport=older) == {
        "acknowledged": 1, "pending": 0, "downgraded_to_v2": 1, "downgraded_to_v3": 1}
    assert [row["schema_version"] for row in older.received] == [
        PUBLICATION_SCHEMA_V4, PUBLICATION_SCHEMA_V3, PUBLICATION_SCHEMA_V2]
    assert older.received[1] == as_v3(older.received[0]) and older.received[2] == as_v2(older.received[0])
    record = state_of(root, offered["campaign_id"])[DOWNGRADE_FIELD]
    assert (record["from_schema_version"], record["to_schema_version"]) == (PUBLICATION_SCHEMA_V3, PUBLICATION_SCHEMA_V2)
    assert coordinator.pending_publications(queue_root=root) == []


def test_a_revision_the_website_holds_as_v3_is_never_rewritten_as_v4(tmp_path):
    value, offered = run_to_completion(tmp_path)
    held = as_v3(offered)
    with coordinator._lock(tmp_path):
        ledger = coordinator._ledger(tmp_path)
        ledger["campaigns"][value["campaign_id"]].update(published_revision=held["revision"],
                                                         published_digest=held["publication_digest"])
        coordinator._save(tmp_path, ledger)
    assert coordinator.pending_publications(queue_root=tmp_path) == []
    state = state_of(tmp_path, value["campaign_id"])
    ledger = json.loads((tmp_path / coordinator.LEDGER_FILENAME).read_text())
    assert screening_publication(plan=value, state=state, budget=coordinator._published_budget(ledger, state)) == held
    # The next revision is offered as v4 again.
    with coordinator._lock(tmp_path):
        ledger = coordinator._ledger(tmp_path)
        coordinator._changed(ledger["campaigns"][value["campaign_id"]])
        coordinator._save(tmp_path, ledger)
    [following] = coordinator.pending_publications(queue_root=tmp_path)
    assert following["schema_version"] == PUBLICATION_SCHEMA_V4 and following["revision"] == held["revision"] + 1


def test_a_refusal_that_is_not_about_the_schema_never_downgrades_v4(tmp_path):
    value, offered = run_to_completion(tmp_path)

    def refuse(publication):
        raise worker.PublicationRefused(503, "screening_store_unavailable")
    summary = worker.deliver_publications(queue_root=tmp_path, transport=refuse)
    assert (summary["acknowledged"], summary["pending"]) == (0, 1) and "downgraded_to_v3" not in summary
    assert DOWNGRADE_FIELD not in state_of(tmp_path, value["campaign_id"])
    assert coordinator.pending_publications(queue_root=tmp_path) == [offered]


# ------------------------------------------------------------------ the offline driver


def test_the_offline_driver_grades_a_partial_retained_result_and_writes_nothing_else(tmp_path, capsys):
    value = campaign_plan()
    candidate = value["candidates"][0]
    full = session_result(value, candidate, CAMPAIGN["policy-a"]["depths"])
    partial = deepcopy(full)
    partial["status"] = "blocked"
    for row in partial["episodes"][7:]:
        row.update(status="blocked")
        row["episode"].pop("score")
    seal(partial, "result_digest")
    paths = []
    for name, result in (("full.json", full), ("partial.json", partial)):
        (tmp_path / name).write_text(json.dumps(result))
        paths += ["--session-result", str(tmp_path / name)]
    before = sorted(tmp_path.iterdir())
    assert progress.main(paths) == 0
    table = json.loads(capsys.readouterr().out)
    assert sorted(tmp_path.iterdir()) == before
    assert (table["grader_version"], table["approached_distance_m"]) == (PROGRESS_GRADER_VERSION, 0.02)
    [whole], [cut] = (row["candidates"] for row in table["results"])
    expected = summarize_progress([progress_episode(score=s, samples=t)
                                   for s, t in (episode(d) for d in CAMPAIGN["policy-a"]["depths"])])
    assert whole["progress"] == expected and whole["episodes_skipped"] == {}
    assert table["results"][0]["session_result_valid"] is True
    assert cut["episodes_graded"] == 7 and cut["episodes_skipped"] == {"episode_not_scored": 3}
    assert cut["progress"]["stages"][0] == {"stage": "approached", "episodes": 7}
    assert table["results"][1]["result_digest_valid"] is True

    out = tmp_path / "table.json"
    assert progress.main([*paths, "--output", str(out)]) == 0
    assert json.loads(out.read_text()) == table
    with pytest.raises(SystemExit):
        progress.main([*paths, "--output", str(out)])


def test_an_older_score_without_behaviour_has_no_rung_unless_the_driver_is_asked_to_recompute(tmp_path, capsys):
    """The summary reads only the sealed score; the offline driver recomputes only on its explicit flag."""
    value = campaign_plan()
    candidate = value["candidates"][0]
    result = native_result(value, candidate)
    for row, depth in zip(result["episodes"], [2, 1, 1, 1, 1, 0, 0, 0, 0, 0], strict=True):
        score, samples = episode(depth)
        score = {key: item for key, item in score.items() if key != "behaviour"}  # sealed before it carried one
        state = {"task_state_samples": samples}
        row["episode"] = {"score": score, "state_trace": state, "task_spec": SPEC}
        row["deterministic_score_digest"] = canonical_digest({"value": score})
        row["state_trace_digest"] = canonical_digest({"value": state})
    result = validate_session_result(seal(result, "result_digest"))
    summary = candidate_result_summary(result, plan=value, candidate=candidate)["progress"]
    assert summary["stages"][0]["stage"] == "contact" and "closest_approach_m" not in summary

    path = tmp_path / "older.json"
    path.write_text(json.dumps(result))
    progress.main(["--session-result", str(path)])
    [plain] = json.loads(capsys.readouterr().out)["results"][0]["candidates"]
    assert plain["progress"] == summary and plain["closest_approach_source"] is None
    progress.main(["--session-result", str(path), "--recompute-behaviour"])
    table = json.loads(capsys.readouterr().out)
    [row] = table["results"][0]["candidates"]
    assert table["behaviour_recompute_requested"] is True
    assert row["closest_approach_source"] == "recomputed_from_recorded_trace"
    assert row["episodes_skipped"] == {} and row["episodes_behaviour_recomputed"] == 10
    assert plain["episodes_behaviour_recomputed"] == 0
    assert row["progress"]["stages"][0] == {"stage": "approached", "episodes": 5}
    assert row["progress"]["closest_approach_m"] == {"median": 0.13, "min": 0.0}
