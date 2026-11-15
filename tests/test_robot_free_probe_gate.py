"""A website scene admits a team evaluation only after its robot-free probe passed.

Beta journey step: L5 team policy evaluation. Receipts here come from the real
robot-free readiness scorer over a synthetic probe trace, so the gate is pinned
against the exact receipt shape operators produce, not a hand-written copy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import robot_free_probe_gate as gate
from blueprint_pipeline import task_evaluation_scene_robot_free_readiness as readiness
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
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
