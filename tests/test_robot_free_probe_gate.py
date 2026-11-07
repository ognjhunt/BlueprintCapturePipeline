"""A website scene admits a team evaluation only after its robot-free probe passed.

Beta journey step: L5 team policy evaluation. Receipts here come from the real
robot-free readiness scorer over a synthetic probe trace, so the gate is pinned
against the exact receipt shape operators produce, not a hand-written copy.
Navigation receipts come from the real navigation scorer probe
(``navigation_robot_free_probe``) over a sealed G1 navigation plan.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import native_g1_navigation_goal as navigation_goal
from blueprint_pipeline import navigation_robot_free_probe as navigation_probe
from blueprint_pipeline import robot_free_probe_gate as gate
from blueprint_pipeline import task_evaluation_scene_robot_free_readiness as readiness
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from tests.test_native_task_navigation_scene import _navigation_task
from tests.test_rigid_robot_free_probe import passed_rigid_receipt
from tests.test_task_evaluation_scene_robot_free_readiness import _BINDINGS, _probe, _result, _spec

REVISION = "sha256:" + "2" * 64


def scored_probe_receipt(revision: str = REVISION, *, passes: bool = True) -> dict[str, Any]:
    """A real sealed readiness receipt for ``revision``; shared with admission tests."""

    return readiness.score_robot_free_readiness(
        task_spec=_spec(), task_joint_bindings=_BINDINGS,
        native_import_result=_result(_probe(rests_at=1.57 if passes else 0.0)),
        configured_scene_revision_digest=revision)


def _reason(receipt: Any, **kwargs: Any) -> str:
    with pytest.raises(gate.RobotFreeProbeGateError) as raised:
        gate.require_robot_free_probe(receipt, **{"scene_revision_digest": REVISION, **kwargs})
    assert str(raised.value) == "team_evaluation_robot_free_probe_" + raised.value.reason
    return raised.value.reason


def test_a_passed_probe_at_the_current_revision_admits_and_binds_its_digest() -> None:
    receipt = scored_probe_receipt()
    admitted = gate.require_robot_free_probe(receipt, scene_revision_digest=REVISION,
                                             task_spec_digest=canonical_digest(_spec()),
                                             task_kind="articulated_open_close")
    assert admitted == {"receipt_digest": receipt["receipt_digest"], "scene_revision_digest": REVISION,
                        "task_spec_digest": canonical_digest(_spec())}
    assert gate.RECEIPT_SCHEMA == readiness.SCHEMA
    # The operator CLI writes the receipt under the name the gate reads.
    import inspect
    assert f'"{gate.RECEIPT_FILENAME}"' in inspect.getsource(readiness.main)


def test_every_way_a_probe_can_fail_to_admit_has_its_own_reason() -> None:
    passed = scored_probe_receipt()
    resealed = lambda **change: {**(row := {**passed, **change}),  # noqa: E731
                                 "receipt_digest": canonical_digest(row, digest_field="receipt_digest")}
    assert _reason(None) == "receipt_missing"
    assert _reason({**passed, "schema_version": "other.v1"}) == "receipt_schema_invalid"
    assert _reason({**passed, "status": "failed"}) == "receipt_digest_invalid"
    assert _reason(scored_probe_receipt(passes=False)) == "not_passed"
    assert _reason(resealed(learned_policy_evaluation_admitted=False)) == "not_passed"
    assert _reason(resealed(proof_boundary={**passed["proof_boundary"], "robot_embodiment_used": True})) == "robot_used"
    assert _reason(scored_probe_receipt("sha256:" + "9" * 64)) == "scene_revision_mismatch"
    assert _reason(passed, task_spec_digest="sha256:" + "8" * 64) == "task_spec_mismatch"
    assert _reason(passed, scene_revision_digest=None) == "scene_revision_unbound"
    unavailable = readiness.score_robot_free_readiness(task_spec=_spec(), task_joint_bindings=_BINDINGS,
                                                       native_import_result=_result(None),
                                                       configured_scene_revision_digest=REVISION)
    assert _reason(unavailable) == "not_passed"


def test_task_kinds_without_a_robot_free_probe_are_refused_by_name() -> None:
    assert gate.PROBE_TASK_KINDS == {"articulated_open_close", "rigid_pick_place", "goal_region_navigation"}
    assert _reason(scored_probe_receipt(), task_kind="deformable_fold") == (
        "unavailable_for_task_kind:deformable_fold")
    # An articulated receipt never admits a rigid task (its own probe: tests/test_rigid_robot_free_probe.py).
    assert _reason(scored_probe_receipt(), task_kind="rigid_pick_place") == "task_kind_mismatch"


def test_the_retained_receipt_is_read_beside_the_source_launch_without_following_links(tmp_path: Path) -> None:
    assert gate.retained_probe_receipt(tmp_path) is None
    (tmp_path / gate.RECEIPT_FILENAME).write_text(json.dumps(scored_probe_receipt()))
    assert gate.retained_probe_receipt(tmp_path)["status"] == "passed"
    linked = tmp_path / "linked"
    linked.mkdir()
    (linked / gate.RECEIPT_FILENAME).symlink_to(tmp_path / gate.RECEIPT_FILENAME)
    with pytest.raises(gate.RobotFreeProbeGateError, match="receipt_path_unsafe"):
        gate.retained_probe_receipt(linked)
    (tmp_path / gate.RECEIPT_FILENAME).write_text("{not json")
    with pytest.raises(gate.RobotFreeProbeGateError, match="receipt_unreadable"):
        gate.retained_probe_receipt(tmp_path)


# --- the rigid and articulated paths, pinned --------------------------------------------------


def test_rigid_and_articulated_validation_results_are_pinned(tmp_path: Path) -> None:
    """Computed with the gate as it was before the navigation receipt kind was added."""

    receipt, spec, configured = passed_rigid_receipt(tmp_path)
    revision = configured["revision_digest"]
    kwargs = {"scene_revision_digest": revision, "task_spec_digest": canonical_digest(spec),
              "task_kind": "rigid_pick_place", "task_success_contract_digest": receipt["task_success_contract_digest"]}
    assert gate.require_robot_free_probe(receipt, **kwargs) == {
        "receipt_digest": "sha256:f861049c3b1c37b58ca1072ef0c3768cd913454502901297ff10cc33de5aebd9",
        "scene_revision_digest": "sha256:890687750fcc6cbb43c0b6e0526117418fa149f367f53fc210fa5473fbd68903",
        "task_spec_digest": "sha256:eb0f0d8f73bd0334bd05ae6dbb910cce49b4d003dead63b536ed9616cc32f7b8"}

    def reseal(**change: Any) -> dict[str, Any]:
        row = {**receipt, **change}
        return {**row, "receipt_digest": canonical_digest(row, digest_field="receipt_digest")}

    def reason(value: Any, **change: Any) -> str:
        with pytest.raises(gate.RobotFreeProbeGateError) as raised:
            gate.require_robot_free_probe(value, **{**kwargs, **change})
        return raised.value.reason

    missing = {key: flag for key, flag in receipt["checks"].items() if key != "contract_drift_not_success"}
    assert reason({**receipt, "status": "failed"}) == "receipt_digest_invalid"
    assert reason(reseal(checks=missing)) == "not_passed"
    assert reason(reseal(checks={**receipt["checks"], "contract_drift_not_success": False})) == "not_passed"
    assert reason(reseal(proof_boundary={**receipt["proof_boundary"], "robot_embodiment_used": True})) == "robot_used"
    assert reason(receipt, scene_revision_digest="sha256:" + "9" * 64) == "scene_revision_mismatch"
    assert reason(receipt, task_spec_digest="sha256:" + "8" * 64) == "task_spec_mismatch"
    assert reason(receipt, task_success_contract_digest="sha256:" + "7" * 64) == "task_success_contract_mismatch"
    assert reason(receipt, task_kind="articulated_open_close") == "task_kind_mismatch"
    # The navigation-only claim ceiling check never reads a rigid receipt.
    assert gate.require_robot_free_probe(reseal(claim_ceiling="physical"), **kwargs)["receipt_digest"] == (
        "sha256:17ab06bb5a9f4ef323e4af152f1903167d05260fdb9e2a1ca27064d6b1b27965")
    assert gate.task_success_contract_binding("rigid_pick_place", {"contract_digest": "sha256:" + "a" * 64}) == (
        "sha256:" + "a" * 64)
    assert gate.task_success_contract_binding("rigid_pick_place", {}) == ""
    assert gate.task_success_contract_binding("articulated_open_close", {"contract_digest": "x"}) is None
    assert gate.require_robot_free_probe(scored_probe_receipt(), scene_revision_digest=REVISION,
                                         task_kind="articulated_open_close") == {
        "receipt_digest": "sha256:7a1d3d633bf34407e523dae91ae2df9359fd6bb33b298dcfbf0cc6f6b992b54f",
        "scene_revision_digest": REVISION,
        "task_spec_digest": "sha256:0d9ef9dacf25849039dc9bf88102a71ea2453b76c02a1d480883efd338f33868"}


# --- goal-region navigation -------------------------------------------------------------------


def navigation_plan(task: dict[str, Any] | None = None, **changes: Any) -> dict[str, Any]:
    """A sealed G1 navigation plan's fields the probe reads (the full packet: website_navigation_scene_inputs)."""

    task = copy.deepcopy(task or _navigation_task())
    contract = navigation_goal.navigation_task_success_contract(task)
    plan = {"schema_version": "native_task_arena_scene_plan.v1", "scene_id": task["site_id"],
            "task_id": task["task_id"], "task_kind": "goal_region_navigation",
            "task_spec": {**task, "task_success_contract": contract,
                          "task_success_contract_digest": contract["contract_digest"]},
            "robot": {"robot_id": "unitree_g1"}, **changes, "plan_digest": ""}
    plan["plan_digest"] = canonical_digest(plan, digest_field="plan_digest")
    return plan


def _moved_goal_task(dx: float = 0.5) -> dict[str, Any]:
    task = _navigation_task()
    task["g1_navigation_goal"]["center_world_m"][0] += dx
    task["g1_navigation_goal"]["visible_target_marker"]["surface_position_world_m"][0] += dx
    return task


def _planning_kwargs(plan: dict[str, Any]) -> dict[str, Any]:
    """What beta planning passes (build_planning_inputs) for a scene whose scene revision is its plan."""

    task = plan["task_spec"]
    return {"scene_revision_digest": plan["plan_digest"], "task_kind": task["task_kind"],
            "task_spec_digest": canonical_digest(task),
            "task_success_contract_digest": gate.task_success_contract_binding(task["task_kind"],
                                                                               task["task_success_contract"])}


def test_navigation_constants_are_pinned_to_the_probe() -> None:
    assert gate.NAVIGATION_RECEIPT_SCHEMA == navigation_probe.SCHEMA
    assert gate.RECEIPT_TASK_KINDS[navigation_probe.SCHEMA] == navigation_probe.TASK_KIND == "goal_region_navigation"
    assert gate.REQUIRED_CHECKS["goal_region_navigation"] == frozenset(navigation_probe.REQUIRED_CHECKS) == {
        "zero_action_not_success", "direct_arrival_held_success", "arrival_without_hold_not_success",
        "root_height_drift_not_success"}
    assert gate.NAVIGATION_CLAIM_CEILING == navigation_probe.CLAIM_CEILING == "development_only"


def test_the_synthetic_traces_are_built_from_the_plans_start_and_goal() -> None:
    plan = navigation_plan()
    task = plan["task_spec"]
    goal = task["g1_navigation_goal"]
    start, centre = task["start_pose_world"], goal["center_world_m"]
    radius, settle, drift = goal["acceptance_radius_m"], goal["settle_window_samples"], goal["max_root_height_drift_m"]
    traces = navigation_probe.synthetic_traces(plan)

    def inside(trace: list[dict]) -> list[bool]:
        return [math.dist(row["root_position_world_m"][:2], centre[:2]) <= radius for row in trace]

    def longest_inside_run(trace: list[dict]) -> int:
        best = run = 0
        for flag in inside(trace):
            run = run + 1 if flag else 0
            best = max(best, run)
        return best

    for trace in traces.values():
        assert [row["step_index"] for row in trace] == list(range(len(trace)))
        assert trace[0]["root_position_world_m"] == start[:3] and not inside(trace)[0]
    assert all(row["root_position_world_m"] == start[:3] for row in traces["zero_action"])
    held = traces["direct_arrival_held"]
    assert [row["root_position_world_m"] for row in held[-settle:]] == [[*centre[:2], start[2]]] * settle
    assert longest_inside_run(held) == settle and inside(held)[-settle:] == [True] * settle
    # Arrives, stays one sample short of the settle window, and leaves.
    assert longest_inside_run(traces["arrival_without_hold"]) == settle - 1
    assert not inside(traces["arrival_without_hold"])[-1]
    lowered = [abs(row["root_position_world_m"][2] - start[2]) for row in traces["root_height_drift"]]
    assert max(lowered) == pytest.approx(drift + navigation_probe.DRIFT_MARGIN_M)
    assert lowered[-settle:] == [0.0] * settle


def test_each_check_is_the_real_scorer_on_its_trace_and_the_receipt_binds_the_plan() -> None:
    plan = navigation_plan()
    task = plan["task_spec"]
    receipt = navigation_probe.score_navigation_robot_free_probe(plan=plan)
    assert receipt["status"] == "passed" and receipt["learned_policy_evaluation_admitted"] is True
    assert receipt["checks"] == dict.fromkeys(navigation_probe.REQUIRED_CHECKS, True)
    assert receipt["required_checks"] == list(navigation_probe.REQUIRED_CHECKS)
    for name, trace in navigation_probe.synthetic_traces(plan).items():
        report = navigation_goal.score_g1_navigation_episode(task_spec=task, samples=trace)
        case = receipt["cases"][name]
        assert case["trace_digest"] == canonical_digest({"samples": trace})
        assert (case["score_digest"], case["outcome"], case["sample_count"]) == (
            report["score_digest"], report["outcome"], len(trace))
        assert case["must_succeed"] is (name == "direct_arrival_held")
    assert receipt["cases"]["arrival_without_hold"]["root_height_stable_throughout"] is True
    assert receipt["cases"]["root_height_drift"]["terminal_goal_hold"] is True
    # The binding: the plan (the scene revision of a navigation scene), the scored spec, its contract, the scorer.
    assert receipt["configured_scene_revision_digest"] == receipt["plan_digest"] == plan["plan_digest"]
    assert receipt["task_spec_digest"] == canonical_digest(task)
    assert receipt["navigation_task_spec_digest"] == task["task_spec_digest"]
    assert receipt["task_success_contract_digest"] == task["task_success_contract"]["contract_digest"]
    import hashlib
    assert receipt["scorer"] == {
        "function": "blueprint_pipeline.native_g1_navigation_goal.score_g1_navigation_episode",
        "score_schema_version": navigation_goal.SCORE_SCHEMA_VERSION,
        "module_sha256": "sha256:" + hashlib.sha256(Path(navigation_goal.__file__).read_bytes()).hexdigest()}
    # Only the scorer on synthetic traces: no robot, simulator, scene geometry or outcome claimed.
    assert receipt["claim_ceiling"] == "development_only"
    boundary = receipt["proof_boundary"]
    assert boundary["robot_embodiment_used"] is False and boundary["simulator_used"] is False
    assert boundary["synthetic_root_traces_only"] is True and boundary["scene_geometry_read"] is False
    assert boundary["simulator_outcome_claimed"] is False and boundary["physical_outcome_claimed"] is False
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")
    assert navigation_probe.score_navigation_robot_free_probe(plan=plan) == receipt  # deterministic
    assert navigation_probe.validate_navigation_robot_free_probe_receipt(receipt, plan=plan) == receipt

    admitted = gate.require_robot_free_probe(receipt, **_planning_kwargs(plan))
    assert admitted == {"receipt_digest": receipt["receipt_digest"], "scene_revision_digest": plan["plan_digest"],
                        "task_spec_digest": canonical_digest(task)}


def test_the_gate_refuses_a_tampered_another_plans_or_incomplete_navigation_receipt() -> None:
    plan = navigation_plan()
    receipt = navigation_probe.score_navigation_robot_free_probe(plan=plan)
    kwargs = _planning_kwargs(plan)

    def reseal(**change: Any) -> dict[str, Any]:
        row = {**copy.deepcopy(receipt), **change}
        return {**row, "receipt_digest": canonical_digest(row, digest_field="receipt_digest")}

    def reason(value: Any, **change: Any) -> str:
        with pytest.raises(gate.RobotFreeProbeGateError) as raised:
            gate.require_robot_free_probe(value, **{**kwargs, **change})
        return raised.value.reason

    # Tampered: any change without the seal.
    assert reason({**receipt, "plan_digest": "sha256:" + "9" * 64}) == "receipt_digest_invalid"
    assert reason({**receipt, "checks": {**receipt["checks"], "root_height_drift_not_success": False}}) == (
        "receipt_digest_invalid")
    # Another plan: the same task in another scene, or another goal.
    other_scene = navigation_plan(scene_id="site-other")
    assert reason(receipt, **_planning_kwargs(other_scene)) == "scene_revision_mismatch"
    other_goal = navigation_plan(_moved_goal_task())
    assert reason(navigation_probe.score_navigation_robot_free_probe(plan=other_goal)) == "scene_revision_mismatch"
    assert reason(receipt, task_spec_digest=canonical_digest(other_goal["task_spec"])) == "task_spec_mismatch"
    assert reason(receipt, task_success_contract_digest=gate.task_success_contract_binding(
        "goal_region_navigation", other_goal["task_spec"]["task_success_contract"])) == (
        "task_success_contract_mismatch")
    # A contract without a digest is refused, not skipped.
    assert gate.task_success_contract_binding("goal_region_navigation", {}) == ""
    assert reason(receipt, task_success_contract_digest="") == "task_success_contract_mismatch"
    # Missing or failing a required check, or not passed.
    for name in navigation_probe.REQUIRED_CHECKS:
        assert reason(reseal(checks={key: flag for key, flag in receipt["checks"].items() if key != name})) == (
            "not_passed")
        assert reason(reseal(checks={**receipt["checks"], name: False})) == "not_passed"
    assert reason(reseal(status="failed")) == "not_passed"
    assert reason(reseal(learned_policy_evaluation_admitted=False)) == "not_passed"
    # Claims beyond the scorer on synthetic traces.
    assert reason(reseal(proof_boundary={**receipt["proof_boundary"], "robot_embodiment_used": True})) == "robot_used"
    assert reason(reseal(claim_ceiling="diagnostic_policy_execution")) == "claim_ceiling_invalid"
    no_ceiling = {key: value for key, value in receipt.items() if key != "claim_ceiling"}
    no_ceiling["receipt_digest"] = canonical_digest(no_ceiling, digest_field="receipt_digest")
    assert reason(no_ceiling) == "claim_ceiling_invalid"
    for claim in ("simulator_outcome_claimed", "physical_outcome_claimed"):
        assert reason(reseal(proof_boundary={**receipt["proof_boundary"], claim: True})) == "claim_ceiling_invalid"
    # One kind never admits another.
    assert reason(receipt, task_kind="rigid_pick_place") == "task_kind_mismatch"
    assert reason(scored_probe_receipt(), **kwargs) == "task_kind_mismatch"


def test_a_scorer_that_cannot_tell_the_cases_apart_fails_the_probe(monkeypatch) -> None:
    plan = navigation_plan()
    real = navigation_goal.score_g1_navigation_episode

    def ever_reached(*, task_spec: Any, samples: Any) -> dict[str, Any]:
        # Calls an episode a success once the root has ever been inside the goal: no hold, no height.
        report = dict(real(task_spec=task_spec, samples=samples))
        if report["minimum_distance_m"] <= report["goal"]["acceptance_radius_m"]:
            report["outcome"] = "success"
        return report

    monkeypatch.setattr(navigation_goal, "score_g1_navigation_episode", ever_reached)
    receipt = navigation_probe.score_navigation_robot_free_probe(plan=plan)
    assert receipt["status"] == "failed" and receipt["learned_policy_evaluation_admitted"] is False
    assert receipt["checks"] == {"zero_action_not_success": True, "direct_arrival_held_success": True,
                                 "arrival_without_hold_not_success": False, "root_height_drift_not_success": False}
    assert navigation_probe.validate_navigation_robot_free_probe_receipt(receipt, plan=plan) == receipt
    with pytest.raises(gate.RobotFreeProbeGateError, match="not_passed"):
        gate.require_robot_free_probe(receipt, **_planning_kwargs(plan))

    def fails_everything(*, task_spec: Any, samples: Any) -> dict[str, Any]:
        return {**real(task_spec=task_spec, samples=samples), "outcome": "failure"}

    monkeypatch.setattr(navigation_goal, "score_g1_navigation_episode", fails_everything)
    receipt = navigation_probe.score_navigation_robot_free_probe(plan=plan)
    assert receipt["status"] == "failed" and receipt["checks"]["direct_arrival_held_success"] is False


def test_a_start_already_inside_the_goal_is_unscorable_and_fails() -> None:
    task = _navigation_task()
    task["start_pose_world"][:2] = task["g1_navigation_goal"]["center_world_m"][:2]
    receipt = navigation_probe.score_navigation_robot_free_probe(plan=navigation_plan(task))
    assert receipt["status"] == "failed" and not any(receipt["checks"].values())
    assert {case["error"] for case in receipt["cases"].values()} == {"g1_navigation_start_already_at_goal"}


def test_the_probe_refuses_a_plan_it_cannot_bind_and_a_receipt_of_another_plan() -> None:
    plan = navigation_plan()
    receipt = navigation_probe.score_navigation_robot_free_probe(plan=plan)

    def plan_refused(value: Any) -> None:
        with pytest.raises(navigation_probe.NavigationRobotFreeProbeError,
                           match="^navigation_robot_free_probe_plan_invalid$"):
            navigation_probe.score_navigation_robot_free_probe(plan=value)

    plan_refused({**plan, "scene_id": "site-other"})  # unsealed
    plan_refused(navigation_plan(task_kind="rigid_pick_place"))
    no_contract = copy.deepcopy(plan)
    del no_contract["task_spec"]["task_success_contract"]
    no_contract["plan_digest"] = canonical_digest(no_contract, digest_field="plan_digest")
    plan_refused(no_contract)
    wrong_contract = copy.deepcopy(plan)
    wrong_contract["task_spec"]["task_success_contract"]["criterion"] = "root_xy_inside_visible_goal"
    wrong_contract["plan_digest"] = canonical_digest(wrong_contract, digest_field="plan_digest")
    plan_refused(wrong_contract)

    def receipt_refused(value: Any, against: dict[str, Any] = plan) -> None:
        with pytest.raises(navigation_probe.NavigationRobotFreeProbeError,
                           match="^navigation_robot_free_probe_receipt_invalid$"):
            navigation_probe.validate_navigation_robot_free_probe_receipt(value, plan=against)

    def reseal(**change: Any) -> dict[str, Any]:
        row = {**copy.deepcopy(receipt), **change}
        return {**row, "receipt_digest": canonical_digest(row, digest_field="receipt_digest")}

    receipt_refused(receipt, navigation_plan(scene_id="site-other"))
    receipt_refused(receipt, navigation_plan(_moved_goal_task()))
    receipt_refused({**receipt, "status": "failed"})
    receipt_refused(reseal(checks={**receipt["checks"], "zero_action_not_success": False}))  # disagrees with status
    receipt_refused(reseal(checks={key: True for key in navigation_probe.REQUIRED_CHECKS[1:]}))
    receipt_refused(reseal(claim_ceiling="diagnostic_policy_execution"))
    receipt_refused(reseal(proof_boundary={**receipt["proof_boundary"], "scene_geometry_read": True}))
    receipt_refused(None)
