"""The rigid robot-free probe tells success from failure before any team policy runs.

Beta journey step: L5 team policy evaluation. Specs come from the real rigid
relocation adapter over a configured revision, receipts from the real scorer,
and the physics trace from the real recorder driving a deterministic flat
support stand-in for the simulator (the Isaac scene is pinned in the slow lane
by tests/test_rigid_robot_free_probe_isaac.py).
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import robot_free_probe_gate as gate
from blueprint_pipeline import task_evaluation_scene_rigid_robot_free_readiness as rigid
from blueprint_pipeline import task_evaluation_scene_robot_free_readiness as articulated
from blueprint_pipeline.adp_task_scoring import (
    _compatibility_rigid_success_criteria,
    seal_rigid_task_success_contract,
)
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.rigid_robot_free_probe_plan import PHYSICS_HZ, physics_plan, probe_context
from blueprint_pipeline.rigid_robot_free_probe_recorder import record_rigid_robot_free_probe
from blueprint_pipeline.task_evaluation_rigid_relocation_native_adapter import (
    NATIVE_PHYSICS_FREQUENCY_HZ,
    adapt_rigid_relocation_task_template,
)
from tests.test_task_evaluation_rigid_relocation_native_adapter import DEFINITION, EXECUTION, _case, _rewrite

SUPPORT_TOP = 0.7545
REST_HEIGHT = 0.063819  # the fixture's qualified centre of mass above the support


class FlatSupport:
    """Gravity onto the registered support's top, or the floor beyond its edge; no robot."""

    def __init__(self, start: Sequence[float], *, support_max_xy: Sequence[float] = (4.5, -1.4),
                 slide_to: Sequence[float] | None = None):
        self.pose, self.vz, self.steps = list(start), 0.0, 0
        self.support_min, self.support_max, self.slide_to = (2.5, -9.5), tuple(support_max_xy), slide_to

    def _ground(self) -> float:
        on = all(a <= v <= b for a, v, b in zip(self.support_min, self.pose[:2], self.support_max))
        return (SUPPORT_TOP if on else 0.0) + REST_HEIGHT

    def set_scoring_pose(self, pose: Sequence[float]) -> None:
        self.pose, self.vz = [float(v) for v in pose], 0.0

    def step(self) -> None:
        self.steps += 1
        if self.slide_to is not None and self.steps == 60:  # a scene that moves the object by itself
            self.pose[:2] = list(self.slide_to)
        self.vz -= 9.81 / PHYSICS_HZ
        self.pose[2] = max(self._ground(), self.pose[2] + self.vz / PHYSICS_HZ)
        if self.pose[2] == self._ground():
            self.vz = 0.0

    def read_sample(self) -> dict[str, Any]:
        return {"task_scoring_pose_world": list(self.pose), "support_contact_active": self.pose[2] <= self._ground(),
                "scene_collision_force_n": 0.0}


def rigid_case(tmp_path: Path, *, strategy: str = "planar_push") -> tuple[dict, dict, dict, dict]:
    """A configured rigid revision, its references, documents and the adapted task specification."""

    launch, configured, references, docs = _case(tmp_path)
    if strategy == "pick_and_place":
        definition = copy.deepcopy(docs[DEFINITION])
        definition["strategy"] = "pick_and_place"
        definition["interaction_affordance"] = {
            "contact_point_scoring_frame_m": [-0.06, 0.0, 0.0], "approach_unit_scoring_frame": [-1.0, 0.0, 0.0],
            "jaw_unit_scoring_frame": [0.0, 1.0, 0.0], "lift_unit_world": [0.0, 0.0, 1.0],
            "pregrasp_clearance_m": 0.12, "minimum_lift_m": 0.08}
        execution = copy.deepcopy(docs[EXECUTION])
        execution["strategy"] = "pick_and_place"
        for path, document in ((DEFINITION, definition), (EXECUTION, execution)):
            _rewrite(tmp_path=tmp_path, configured=configured, references=references, contract_path=path,
                     document=document)
            docs[path] = document
    adapted = adapt_rigid_relocation_task_template(configured_revision=configured, materialized_references=references)
    return configured, references, docs, adapted["native_task_definition"]["task_spec"]


def binding(spec: dict, revision_digest: str, asset: str | None = None) -> dict[str, Any]:
    return {"configured_scene_revision_digest": revision_digest, "task_spec_digest": canonical_digest(spec),
            "replacement_asset_digest": asset, "scene_context": "flat_support_stand_in",
            "scene_collisions_modeled": False}


def recorded_trace(spec: dict, revision_digest: str, **scene: Any) -> dict[str, Any]:
    return record_rigid_robot_free_probe(scene=FlatSupport(spec["start_pose_world"], **scene),
                                         plan=physics_plan(probe_context(spec)),
                                         binding=binding(spec, revision_digest))


def passed_rigid_receipt(tmp_path: Path) -> tuple[dict, dict, dict]:
    configured, _references, _docs, spec = rigid_case(tmp_path)
    receipt = rigid.score_rigid_robot_free_readiness(
        task_spec=spec, physics_probe=recorded_trace(spec, configured["revision_digest"]),
        configured_scene_revision_digest=configured["revision_digest"])
    return receipt, spec, configured


def _sealed_contract(spec: dict, **changes: Any) -> dict:
    criteria = _compatibility_rigid_success_criteria(spec)
    for path, value in changes.items():
        section, key = path.split("__")
        criteria[section][key] = value
    return seal_rigid_task_success_contract(
        task_spec=spec, site_id="site-1", task_id="task-1", author_source="task_owner", author_id="owner-1",
        confirmation_status="confirmed", confirmed_by_team_id="team-1", criteria=criteria)


def _with_contract(spec: dict, **changes: Any) -> dict:
    value = dict(spec, site_id="site-1", task_id="task-1")
    value["task_success_contract"] = _sealed_contract(value, **changes)
    return value


def test_constants_are_pinned_to_their_owners() -> None:
    assert PHYSICS_HZ == NATIVE_PHYSICS_FREQUENCY_HZ
    assert gate.RECEIPT_SCHEMA == articulated.SCHEMA
    assert gate.RIGID_RECEIPT_SCHEMA == rigid.SCHEMA
    assert gate.REQUIRED_CHECKS["rigid_pick_place"] == frozenset(rigid.REQUIRED_CHECKS)
    assert gate.RECEIPT_TASK_KINDS[rigid.SCHEMA] == rigid.TASK_KIND


@pytest.mark.parametrize("strategy", ["planar_push", "pick_and_place"])
def test_placed_by_state_setting_passes_and_every_wrong_state_fails(tmp_path: Path, strategy: str) -> None:
    configured, _references, _docs, spec = rigid_case(tmp_path, strategy=strategy)
    receipt = rigid.score_rigid_robot_free_readiness(
        task_spec=spec, physics_probe=recorded_trace(spec, configured["revision_digest"]),
        configured_scene_revision_digest=configured["revision_digest"])
    assert receipt["status"] == "passed" and receipt["learned_policy_evaluation_admitted"] is True
    assert all(receipt["checks"].values()) and set(rigid.REQUIRED_CHECKS) <= set(receipt["checks"])
    cases = receipt["contract_cases"]
    assert cases["goal"]["task_succeeded"] is True
    for name in ("untouched", "short_hold", "drift", "outside_x_plus", "outside_x_minus", "outside_y_plus",
                 "outside_y_minus"):
        assert cases[name]["status"] == "scored" and cases[name]["task_succeeded"] is False, name
    assert "destination_containment" in cases["outside_x_plus"]["failed_criteria"]
    assert cases["drift"]["failed_criteria"] == ["settling"]
    if strategy == "pick_and_place":
        assert cases["tipped"]["failed_criteria"] == ["orientation"]
    else:  # a push contract ignores orientation, so a tipped object is not a wrong state for it
        assert receipt["not_applicable_checks"]["contract_tipped_not_success"] == "orientation_not_required"
    physics = receipt["physics"]
    assert physics["zero_action"]["outcome"] == "never_moved" and physics["zero_action"]["task_succeeded"] is False
    assert physics["direct_placement"]["task_succeeded"] is True
    # 3 s untouched, then 2 s of carry and 5 s released, each read once per 20 Hz control step.
    assert (physics["zero_action"]["sample_count"], physics["direct_placement"]["sample_count"]) == (60, 141)
    assert receipt["proof_boundary"]["robot_embodiment_used"] is False
    assert "gripper_state" in receipt["proof_boundary"]["robot_side_criteria_not_exercised"]
    assert receipt["task_spec_digest"] == canonical_digest(spec)
    assert receipt["task_success_contract_digest"] == probe_context(spec)["criteria_contract_digest"]
    assert receipt["receipt_digest"] == canonical_digest(receipt, digest_field="receipt_digest")


def test_a_goal_the_scene_cannot_hold_fails_on_physics_alone(tmp_path: Path) -> None:
    configured, _references, _docs, spec = rigid_case(tmp_path)
    off_support = rigid.score_rigid_robot_free_readiness(
        task_spec=spec, physics_probe=recorded_trace(spec, configured["revision_digest"], support_max_xy=(3.05, -1.4)),
        configured_scene_revision_digest=configured["revision_digest"])
    assert off_support["status"] == "failed" and off_support["learned_policy_evaluation_admitted"] is False
    assert off_support["checks"]["direct_placement_success"] is False
    assert all(value for key, value in off_support["checks"].items() if key.startswith("contract_"))
    assert "support_height" in off_support["physics"]["direct_placement"]["failed_criteria"]


def test_a_scene_that_moves_the_object_into_the_goal_by_itself_fails(tmp_path: Path) -> None:
    configured, _references, _docs, spec = rigid_case(tmp_path)
    target = spec["target_position_world_m"]
    receipt = rigid.score_rigid_robot_free_readiness(
        task_spec=spec, physics_probe=recorded_trace(spec, configured["revision_digest"], slide_to=target[:2]),
        configured_scene_revision_digest=configured["revision_digest"])
    assert receipt["checks"]["zero_action_not_success"] is False and receipt["status"] == "failed"


def test_contracts_that_cannot_tell_success_from_failure_fail(tmp_path: Path) -> None:
    configured, _references, _docs, spec = rigid_case(tmp_path)
    trace = recorded_trace(spec, configured["revision_digest"])

    def checks(value: dict) -> dict[str, bool]:
        return rigid.score_rigid_robot_free_readiness(task_spec=value, physics_probe=trace)["checks"]

    # Any object resting anywhere counts: the start already succeeds and outside is not wrong.
    loose = _with_contract(spec, destination_containment__mode="ignored", motion__minimum_translation_m=None)
    assert checks(loose)["contract_untouched_not_success"] is False
    assert checks(loose)["contract_outside_not_success"] is False
    # A one-sample hold window cannot show the released object stayed put.
    instant = dict(spec, settle_window_samples=1)
    assert checks(instant)["contract_drift_not_success"] is False
    # Neither support nor release is required, so a held object counts before the hold window ends.
    held = _with_contract(spec, support__contact_mode="ignored", terminal_task_contact__mode="ignored")
    assert checks(held)["contract_short_hold_not_success"] is False
    # The authored target lies outside its own success region: the goal itself fails.
    unreachable = dict(spec, target_position_world_m=[spec["target_position_world_m"][0] + 0.2,
                                                     *spec["target_position_world_m"][1:]])
    assert checks(unreachable)["contract_goal_success"] is False


def test_a_drop_is_a_wrong_state_only_when_the_contract_forbids_drops(tmp_path: Path) -> None:
    _configured, _references, _docs, spec = rigid_case(tmp_path, strategy="pick_and_place")
    strict = rigid.score_rigid_robot_free_readiness(
        task_spec=_with_contract(spec, temporal_invariants__no_drop={"mode": "required", "minimum_fall_m": 0.02}),
        physics_probe=None)
    assert strict["contract_cases"]["dropped"]["failed_criteria"] == ["no_drop"]
    assert strict["checks"]["contract_dropped_not_success"] is True
    lenient = rigid.score_rigid_robot_free_readiness(task_spec=spec, physics_probe=None)
    assert lenient["not_applicable_checks"]["contract_dropped_not_success"] == "no_drop_not_required"


def test_success_that_is_a_robot_state_or_has_no_hold_window_is_refused_by_name(tmp_path: Path) -> None:
    _configured, _references, _docs, spec = rigid_case(tmp_path)
    for changes, reason in (({"gripper_state__mode": "closed_at_most", "gripper_state__threshold_m": 0.02},
                             "terminal_grasp_success"),
                            ({"terminal_task_contact__mode": "maintained"}, "terminal_grasp_success"),
                            ({"settling__mode": "ignored"}, "hold_window_not_required")):
        receipt = rigid.score_rigid_robot_free_readiness(task_spec=_with_contract(spec, **changes), physics_probe=None)
        assert (receipt["status"], receipt["probe_error"]) == ("probe_unavailable", reason)
        assert receipt["learned_policy_evaluation_admitted"] is False
    unpaced = {key: value for key, value in spec.items() if key != "control_frequency_hz"}
    assert rigid.score_rigid_robot_free_readiness(task_spec=unpaced, physics_probe=None)["probe_error"] == (
        "control_frequency_missing")


def test_physics_evidence_must_be_recorded_for_this_revision_task_and_plan(tmp_path: Path) -> None:
    configured, _references, _docs, spec = rigid_case(tmp_path)
    revision = configured["revision_digest"]
    trace = recorded_trace(spec, revision)

    def error(probe: Any, **kwargs: Any) -> str:
        receipt = rigid.score_rigid_robot_free_readiness(task_spec=spec, physics_probe=probe,
                                                         configured_scene_revision_digest=revision, **kwargs)
        assert receipt["status"] == "probe_unavailable" and receipt["learned_policy_evaluation_admitted"] is False
        # The contract checks are still sealed, so an operator sees which half is missing.
        assert receipt["checks"]["contract_goal_success"] is True
        return receipt["probe_error"]

    def resealed(value: dict) -> dict:
        return {**value, "trace_digest": canonical_digest(value, digest_field="trace_digest")}

    assert error(None) == "physics_trace_missing"
    assert error({**trace, "schema_version": "other.v1"}) == "physics_trace_schema_invalid"
    assert error({**trace, "samples": trace["samples"][:-1]}) == "physics_trace_digest_invalid"
    assert error(recorded_trace(spec, "sha256:" + "9" * 64)) == (
        "physics_trace_binding_mismatch:configured_scene_revision_digest")
    assert error(trace, replacement_asset_digest="sha256:" + "8" * 64) == (
        "physics_trace_binding_mismatch:replacement_asset_digest")
    assert error(resealed({**trace, "plan": {**trace["plan"], "release_seconds": 1.0}})) == (
        "physics_trace_plan_mismatch")
    assert error(resealed({**trace, "proof_boundary": {"robot_embodiment_used": True}})) == (
        "physics_trace_incomplete")

    class Broken(FlatSupport):
        def read_sample(self) -> dict[str, Any]:
            return {**super().read_sample(), "support_contact_active": None}

    failed = record_rigid_robot_free_probe(scene=Broken(spec["start_pose_world"]),
                                           plan=physics_plan(probe_context(spec)), binding=binding(spec, revision))
    assert failed["status"] == "failed" and "readback_invalid" in failed["error"]
    assert error(failed).startswith("physics_trace_not_recorded:RuntimeError:rigid_robot_free_probe_readback")


def _reason(receipt: Any, **kwargs: Any) -> str:
    with pytest.raises(gate.RobotFreeProbeGateError) as raised:
        gate.require_robot_free_probe(receipt, **kwargs)
    return raised.value.reason


def test_the_gate_admits_a_passed_rigid_receipt_and_binds_it(tmp_path: Path) -> None:
    receipt, spec, configured = passed_rigid_receipt(tmp_path)
    revision = configured["revision_digest"]
    admitted = gate.require_robot_free_probe(
        receipt, scene_revision_digest=revision, task_spec_digest=canonical_digest(spec), task_kind="rigid_pick_place",
        task_success_contract_digest=receipt["task_success_contract_digest"])
    assert admitted == {"receipt_digest": receipt["receipt_digest"], "scene_revision_digest": revision,
                        "task_spec_digest": canonical_digest(spec)}
    # The team run controller's source admission reads it beside the source launch.
    from blueprint_pipeline.task_evaluation_team_run_controller import _require_passed_probe

    (tmp_path / gate.RECEIPT_FILENAME).write_text(json.dumps(receipt))
    _require_passed_probe(tmp_path, {"configured_scene_revision_digest": revision})


def test_the_gate_refuses_a_stale_failed_mismatched_or_incomplete_rigid_receipt(tmp_path: Path) -> None:
    receipt, spec, configured = passed_rigid_receipt(tmp_path)
    revision = configured["revision_digest"]
    kwargs = {"scene_revision_digest": revision, "task_kind": "rigid_pick_place"}

    def resealed(**change: Any) -> dict:
        value = {**receipt, **change}
        return {**value, "receipt_digest": canonical_digest(value, digest_field="receipt_digest")}

    assert _reason(receipt, **{**kwargs, "scene_revision_digest": "sha256:" + "9" * 64}) == "scene_revision_mismatch"
    assert _reason(receipt, **kwargs, task_spec_digest="sha256:" + "8" * 64) == "task_spec_mismatch"
    assert _reason(receipt, **kwargs, task_success_contract_digest="sha256:" + "7" * 64) == (
        "task_success_contract_mismatch")
    failed = rigid.score_rigid_robot_free_readiness(
        task_spec=spec, physics_probe=recorded_trace(spec, revision, support_max_xy=(3.05, -1.4)),
        configured_scene_revision_digest=revision)
    assert _reason(failed, **kwargs) == "not_passed"
    unavailable = rigid.score_rigid_robot_free_readiness(task_spec=spec, physics_probe=None,
                                                         configured_scene_revision_digest=revision)
    assert _reason(unavailable, **kwargs) == "not_passed"
    without_physics = dict(receipt["checks"])
    without_physics.pop("direct_placement_success")
    assert _reason(resealed(checks=without_physics), **kwargs) == "not_passed"
    assert _reason(resealed(checks={**receipt["checks"], "contract_tipped_not_success": False}), **kwargs) == (
        "not_passed")
    assert _reason({**receipt, "status": "failed"}, **kwargs) == "receipt_digest_invalid"
    # A receipt of one task kind never admits the other.
    assert _reason(receipt, **{**kwargs, "task_kind": "articulated_open_close"}) == "task_kind_mismatch"
