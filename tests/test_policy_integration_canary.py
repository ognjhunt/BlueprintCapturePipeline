"""Integration canary: cheap wiring checks before a paid site session.

Hermetic. Reference sessions reuse the rescorer's retained-result fixture,
extended with the commanded actions and hand poses a real session records.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.policy_integration_canary import (
    IntegrationCanaryError,
    build_integration_canary_receipt,
    candidate_verdict,
    integration_canary_blockers,
    integration_checks,
    session_checkpoint_mismatches,
)
from tests.test_task_evaluation_policy_canary_rescore import _fixture, _sha

REVISION = "0123456789abcdef0123456789abcdef01234567"
NOW = datetime(2026, 9, 23, 21, 0, tzinfo=timezone.utc)


def _action(step: int, gripper: float, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "step_index": step,
        "clipped_droid_action": [0.1] * 7 + [gripper],
        "isaac_action": [0.1] * 7 + [gripper],
        "joint_limit_clamped": False,
        "environment_step_applied": True,
    }
    row.update(overrides)
    return row


def _healthy(steps: int = 12) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """A reach toward the object, then a close that narrows the fingers."""

    actions = [_action(step, 1.0 if step >= 6 else 0.0) for step in range(1, steps + 1)]
    samples = [
        {
            "step_index": step,
            "task_object_pose_world": [1.0, 2.0, 0.8, 0, 0, 0, 1],
            "controlled_body_pose_world": [1.0 - 0.3 * max(0, 6 - step) / 6, 2.0, 0.9, 0, 0, 0, 1],
            "gripper_width_m": 0.08 if step < 6 else 0.02,
        }
        for step in range(0, steps + 1)
    ]
    joints = [{"step_index": step, "joint_positions_rad": [0.02 * step] + [0.0] * 6} for step in range(steps + 1)]
    return actions, samples, joints


def _by_id(checks: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {str(check["id"]): check for check in checks}


# ------------------------------------------------------------------ checks


def test_a_healthy_reference_episode_passes_every_check() -> None:
    actions, samples, joints = _healthy()
    checks = _by_id(integration_checks(actions=actions, samples=samples, joint_states=joints))
    assert all(check["passed"] for check in checks.values()), checks
    assert checks["gripper_polarity"]["status"] == "passed"


def test_wiring_faults_each_fail_their_own_check() -> None:
    actions, samples, joints = _healthy()
    nan = [dict(row) for row in actions]
    nan[3]["isaac_action"] = [float("nan")] * 8
    assert not _by_id(integration_checks(actions=nan, samples=samples, joint_states=joints))["actions_finite"]["passed"]

    short = [dict(row, clipped_droid_action=[0.1] * 7) for row in actions]
    assert not _by_id(integration_checks(actions=short, samples=samples, joint_states=joints))["action_shape"]["passed"]

    clamped = [dict(row, joint_limit_clamped=index % 2 == 0) for index, row in enumerate(actions)]
    assert not _by_id(integration_checks(actions=clamped, samples=samples, joint_states=joints))["joint_limits"]["passed"]

    dropped = [dict(row, environment_step_applied=index < 3) for index, row in enumerate(actions)]
    assert not _by_id(integration_checks(actions=dropped, samples=samples, joint_states=joints))["actions_reached_robot"]["passed"]

    inverted = [dict(sample, gripper_width_m=0.02 if sample["step_index"] < 6 else 0.08) for sample in samples]
    polarity = _by_id(integration_checks(actions=actions, samples=inverted, joint_states=joints))["gripper_polarity"]
    assert polarity["status"] == "failed" and "opened" in str(polarity["detail"])

    still = [dict(row, joint_positions_rad=[0.0] * 7) for row in joints]
    assert not _by_id(integration_checks(actions=actions, samples=samples, joint_states=still))["arm_moved"]["passed"]

    away = [dict(sample, controlled_body_pose_world=[0.4, 2.0, 0.9, 0, 0, 0, 1]) for sample in samples]
    assert not _by_id(integration_checks(actions=actions, samples=away, joint_states=joints))["approached_object"]["passed"]

    assert not _by_id(integration_checks(actions=[], samples=samples, joint_states=joints))["actions_returned"]["passed"]


def test_a_policy_that_never_closes_is_not_called_inverted() -> None:
    actions, samples, joints = _healthy()
    open_only = [dict(row, clipped_droid_action=[0.1] * 7 + [0.0]) for row in actions]
    polarity = _by_id(integration_checks(actions=open_only, samples=samples, joint_states=joints))["gripper_polarity"]
    assert polarity == {"id": "gripper_polarity", "status": "not_exercised", "passed": True,
                        "detail": "the policy never commanded a close"}


def test_wiring_must_hold_every_time_and_reach_at_least_once() -> None:
    actions, samples, joints = _healthy()
    good = {"checks": integration_checks(actions=actions, samples=samples, joint_states=joints)}
    missed = {"checks": integration_checks(
        actions=actions,
        samples=[dict(sample, controlled_body_pose_world=[0.4, 2.0, 0.9, 0, 0, 0, 1]) for sample in samples],
        joint_states=joints,
    )}
    assert candidate_verdict([good, missed]) == (True, [])
    assert candidate_verdict([missed]) == (False, ["approached_object"])
    broken = {"checks": [dict(c, passed=False) if c["id"] == "actions_finite" else c for c in good["checks"]]}
    assert candidate_verdict([good, broken]) == (False, ["actions_finite"])
    assert candidate_verdict([]) == (False, ["no_reference_episodes"])


# ----------------------------------------------------------------- receipt


def _reference_session(tmp_path: Path, *, break_candidate: str | None = None) -> tuple[Path, Path]:
    source, evidence, result = _fixture(tmp_path)
    inventory = [dict(record) for record in result["artifact_inventory"]]
    for row in result["episodes"]:
        actions, samples, joints = _healthy()
        if row["candidate_id"] == break_candidate:
            actions = [dict(action, isaac_action=[float("inf")] * 8) for action in actions]
            actions = json.loads(json.dumps(actions).replace("Infinity", "1e999"))
        state = {"task_state_samples": samples, "joint_states": joints}
        episode_id = row["episode"]["episode_id"]
        state_path = evidence / f"{episode_id}.state_trace.json"
        state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        action_path = evidence / f"{episode_id}.action_sequence.json"
        action_path.write_text(json.dumps(actions, sort_keys=True) + "\n", encoding="utf-8")
        records = {
            "state_trace": {"relative_path": state_path.name, "size_bytes": state_path.stat().st_size,
                            "sha256": _sha(state_path), "role": "state_trace"},
            "action_sequence": {"relative_path": action_path.name, "size_bytes": action_path.stat().st_size,
                                "sha256": _sha(action_path), "role": "action_sequence"},
        }
        inventory = [record for record in inventory if record["relative_path"] != state_path.name]
        inventory.extend(records.values())
        row["evidence_artifacts"].update(records)
        row["episode"]["state_trace"] = state
        row["state_trace_digest"] = canonical_digest({"value": state})
    result["artifact_inventory"] = inventory
    result["artifact_inventory_digest"] = canonical_digest({"value": inventory})
    result["result_digest"] = canonical_digest(result, digest_field="result_digest")
    source.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return source, evidence


def _receipt(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    source, evidence = _reference_session(tmp_path, **kwargs)  # type: ignore[arg-type]
    return build_integration_canary_receipt(
        source_result_path=source, evidence_root=evidence,
        reference_task_ids=["RubiksCubeInBowlTask"], robolab_revision=REVISION,
        generated_at_iso="2026-09-23T20:00:00Z",
    )


def test_a_clean_reference_session_seals_a_passing_receipt(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    assert receipt["passed"] is True
    assert [row["candidate_id"] for row in receipt["candidates"]] == ["groot_n17_droid", "pi05_droid"]
    assert all(row["episode_count"] == 10 and row["checkpoint_digest"].startswith("sha256:")
               for row in receipt["candidates"])
    assert receipt["authority"] == {"proves": "integration_wiring_only", "policy_quality_claimed": False,
                                    "ranking_or_promotion_effect": "none"}
    assert integration_canary_blockers(receipt, candidate_ids=["pi05_droid", "groot_n17_droid"], now=NOW) == []


def test_a_broken_candidate_blocks_the_paid_session(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, break_candidate="groot_n17_droid")
    assert receipt["passed"] is False
    assert integration_canary_blockers(receipt, candidate_ids=["pi05_droid", "groot_n17_droid"], now=NOW) == [
        "policy_integration_canary_candidate_failed:groot_n17_droid:actions_finite,action_shape"
    ]


def test_the_gate_fails_closed(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    pair = ["pi05_droid", "groot_n17_droid"]
    assert integration_canary_blockers(None, candidate_ids=pair) == ["policy_integration_canary_receipt_missing"]
    edited = dict(receipt, passed=True, thresholds={"maximum_joint_limit_clamp_fraction": 1.0})
    assert integration_canary_blockers(edited, candidate_ids=pair, now=NOW) == ["policy_integration_canary_receipt_invalid"]
    assert integration_canary_blockers(
        receipt, candidate_ids=pair, now=datetime(2026, 10, 30, tzinfo=timezone.utc)
    ) == ["policy_integration_canary_receipt_stale"]
    assert integration_canary_blockers(receipt, candidate_ids=[*pair, "octo_base"], now=NOW) == [
        "policy_integration_canary_candidate_missing:octo_base"
    ]


def test_evidence_that_does_not_verify_is_refused(tmp_path: Path) -> None:
    source, evidence = _reference_session(tmp_path)
    kwargs = {"source_result_path": source, "evidence_root": evidence, "reference_task_ids": ["RubiksCubeInBowlTask"],
              "generated_at_iso": "2026-09-23T20:00:00Z"}
    with pytest.raises(IntegrationCanaryError, match="robolab_revision_unpinned"):
        build_integration_canary_receipt(robolab_revision="main", **kwargs)  # type: ignore[arg-type]
    next(evidence.glob("*.action_sequence.json")).write_text("[]\n")
    with pytest.raises(IntegrationCanaryError, match="artifact_inventory_invalid"):
        build_integration_canary_receipt(robolab_revision=REVISION, **kwargs)  # type: ignore[arg-type]


def test_a_paid_run_on_other_checkpoints_is_caught_afterwards(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    (tmp_path / "paid").mkdir()
    _source, _evidence, session = _fixture(tmp_path / "paid")
    assert session_checkpoint_mismatches(receipt, session) == []
    session["episodes"][0]["checkpoint_digest"] = "sha256:" + "f" * 64
    assert session_checkpoint_mismatches(receipt, session) == [session["episodes"][0]["candidate_id"]]
