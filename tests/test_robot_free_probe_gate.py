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
    assert _reason(scored_probe_receipt(), task_kind="rigid_pick_place") == (
        "unavailable_for_task_kind:rigid_pick_place")


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
