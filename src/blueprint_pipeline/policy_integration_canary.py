"""Integration canary: a cheap check that each candidate is wired up correctly.

Before a site run spends a full paired session, each frozen candidate runs a
few short episodes on a public reference task (NVIDIA RoboLab assets: the
Rubik's cube, bowl and table the founder protocol already uses). A healthy
integration there does not prove the policy is good. It proves the plumbing:
actions come back finite and in shape, reach the robot, stay inside the joint
limits, close the gripper when the policy says close, and move the arm toward
the object. Each of those failures otherwise looks like a bad policy on the
site scene, after the money is spent.

The receipt is derived only from a retained session result and the artifacts
its inventory names, verified the way the rescorer verifies them. The paid
allocator can require a passing, fresh receipt for the same candidates, and
after a run the receipt can be checked against the checkpoints that actually
ran.

Backlog: ADP-050 (normalized observation/action contract; complete execution
receipts). It changes no scorer source and ranks nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest, cross_runtime_canonical_digest

RECEIPT_SCHEMA_VERSION = "policy_integration_canary_receipt.v1"

# DROID actions: seven arm values then the gripper, where above 0.5 is closed.
DROID_ACTION_LENGTH = 8
DROID_GRIPPER_INDEX = 7
DROID_GRIPPER_CLOSED_ABOVE = 0.5

THRESHOLDS: dict[str, float] = {
    "minimum_applied_step_fraction": 0.95,
    "maximum_joint_limit_clamp_fraction": 0.25,
    "minimum_arm_motion_rad": 0.05,
    "minimum_approach_fraction": 0.3,
    "minimum_approach_m": 0.05,
    "gripper_response_window_steps": 10,
    "gripper_response_m": 0.005,
}
DEFAULT_MAX_AGE_DAYS = 14

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,191}")


class IntegrationCanaryError(ValueError):
    """Stable failure for evidence that cannot back an integration receipt."""


# ------------------------------------------------------------------ checks


def _finite_list(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    out: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            return None
        out.append(float(item))
    return out


def _xyz(value: Any) -> list[float] | None:
    vector = _finite_list(value[:3] if isinstance(value, list) else None)
    return vector if vector and len(vector) == 3 else None


def _check(check_id: str, status: str, detail: str) -> dict[str, Any]:
    return {"id": check_id, "status": status, "passed": status in {"passed", "not_exercised"}, "detail": detail}


def _gripper_polarity(
    actions: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    widths = {
        int(sample["step_index"]): float(sample["gripper_width_m"])
        for sample in samples
        if isinstance(sample.get("step_index"), int)
        and isinstance(sample.get("gripper_width_m"), (int, float))
        and not isinstance(sample.get("gripper_width_m"), bool)
        and math.isfinite(sample["gripper_width_m"])
    }
    window = int(THRESHOLDS["gripper_response_window_steps"])
    response = THRESHOLDS["gripper_response_m"]
    previous_closed: bool | None = None
    for record in actions:
        droid = _finite_list(record.get("clipped_droid_action"))
        step = record.get("step_index")
        if not droid or len(droid) <= DROID_GRIPPER_INDEX or not isinstance(step, int):
            continue
        closed = droid[DROID_GRIPPER_INDEX] > DROID_GRIPPER_CLOSED_ABOVE
        if closed and previous_closed is False and step - 1 in widths:
            later = [widths[s] for s in range(step, step + window + 1) if s in widths]
            if later:
                change = min(later) - widths[step - 1]
                if change <= -response:
                    return _check("gripper_polarity", "passed", f"closing narrowed the gripper by {-change:.3f} m")
                if max(later) - widths[step - 1] >= response:
                    return _check("gripper_polarity", "failed", "a close command opened the gripper")
                return _check("gripper_polarity", "not_exercised", "a close command did not move the fingers")
        previous_closed = closed
    return _check("gripper_polarity", "not_exercised", "the policy never commanded a close")


def integration_checks(
    *,
    actions: Sequence[Mapping[str, Any]],
    samples: Sequence[Mapping[str, Any]],
    joint_states: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """The integration checks for one reference episode, from sealed evidence."""

    rows = [row for row in actions if isinstance(row, Mapping)]
    checks: list[dict[str, Any]] = []
    checks.append(_check(
        "actions_returned",
        "passed" if rows else "failed",
        f"{len(rows)} commanded actions" if rows else "the policy returned no actions",
    ))
    droid = [_finite_list(row.get("clipped_droid_action")) for row in rows]
    isaac = [_finite_list(row.get("isaac_action")) for row in rows]
    finite = bool(rows) and all(droid) and all(isaac)
    checks.append(_check(
        "actions_finite",
        "passed" if finite else "failed",
        "every action is finite" if finite else "an action is missing or not finite",
    ))
    shapes = {len(vector) for vector in droid if vector}
    isaac_shapes = {len(vector) for vector in isaac if vector}
    shaped = finite and shapes == {DROID_ACTION_LENGTH} and len(isaac_shapes) == 1
    checks.append(_check(
        "action_shape",
        "passed" if shaped else "failed",
        f"{DROID_ACTION_LENGTH}-value DROID actions" if shaped
        else f"action lengths {sorted(shapes)} / {sorted(isaac_shapes)}",
    ))
    applied = sum(1 for row in rows if row.get("environment_step_applied") is True)
    applied_fraction = applied / len(rows) if rows else 0.0
    checks.append(_check(
        "actions_reached_robot",
        "passed" if rows and applied_fraction >= THRESHOLDS["minimum_applied_step_fraction"] else "failed",
        f"{applied} of {len(rows)} actions were applied",
    ))
    clamped = sum(1 for row in rows if row.get("joint_limit_clamped") is True)
    clamp_fraction = clamped / len(rows) if rows else 1.0
    checks.append(_check(
        "joint_limits",
        "passed" if rows and clamp_fraction <= THRESHOLDS["maximum_joint_limit_clamp_fraction"] else "failed",
        f"{clamped} of {len(rows)} actions hit a joint limit",
    ))
    checks.append(_gripper_polarity(rows, samples))

    joints = [
        vector for vector in (_finite_list(row.get("joint_positions_rad")) for row in joint_states
                              if isinstance(row, Mapping))
        if vector
    ]
    motion = max(
        (max(abs(a - b) for a, b in zip(vector, joints[0])) for vector in joints[1:]
         if len(vector) == len(joints[0])),
        default=0.0,
    ) if joints else 0.0
    checks.append(_check(
        "arm_moved",
        "passed" if motion >= THRESHOLDS["minimum_arm_motion_rad"] else "failed",
        f"largest joint travel {motion:.3f} rad",
    ))

    distances = []
    for sample in samples:
        hand = _xyz(sample.get("controlled_body_pose_world"))
        target = _xyz(sample.get("task_scoring_pose_world")) or _xyz(sample.get("task_object_pose_world"))
        if hand and target:
            distances.append(math.dist(hand, target))
    if len(distances) < 2:
        checks.append(_check("approached_object", "failed", "hand or object position was not recorded"))
    else:
        closed_in = distances[0] - min(distances)
        approached = (
            closed_in >= THRESHOLDS["minimum_approach_m"]
            or closed_in >= THRESHOLDS["minimum_approach_fraction"] * distances[0]
        )
        checks.append(_check(
            "approached_object",
            "passed" if approached else "failed",
            f"hand came {closed_in:.3f} m closer (from {distances[0]:.3f} m)",
        ))
    return checks


_EVERY_EPISODE = ("actions_returned", "actions_finite", "action_shape", "actions_reached_robot",
                  "joint_limits", "gripper_polarity")
_ANY_EPISODE = ("arm_moved", "approached_object")


def candidate_verdict(episodes: Sequence[Mapping[str, Any]]) -> tuple[bool, list[str]]:
    """Wiring checks must hold in every episode; motion and approach in at least one.

    A healthy policy may miss a reach on one reset. It does not return NaNs,
    inverted grippers or out-of-range targets on any.
    """

    failures: list[str] = []
    if not episodes:
        return False, ["no_reference_episodes"]
    for check_id in _EVERY_EPISODE:
        if any(
            not next((c["passed"] for c in episode["checks"] if c["id"] == check_id), False)
            for episode in episodes
        ):
            failures.append(check_id)
    for check_id in _ANY_EPISODE:
        if not any(
            next((c["passed"] for c in episode["checks"] if c["id"] == check_id), False)
            for episode in episodes
        ):
            failures.append(check_id)
    return not failures, failures


# ----------------------------------------------------------------- receipt


def build_integration_canary_receipt(
    *,
    source_result_path: str | Path,
    evidence_root: str | Path,
    reference_task_ids: Sequence[str],
    robolab_revision: str,
    generated_at_iso: str,
) -> dict[str, Any]:
    """Seal an integration receipt from one retained reference-task session."""

    from .native_task_arena_policy_canary_session import validate_session_result
    from .task_evaluation_policy_canary_rescore import (
        PolicyCanaryRescoreError,
        _verified_episode_artifact,
        _verify_artifact_inventory,
    )

    if not _COMMIT.fullmatch(str(robolab_revision)):
        raise IntegrationCanaryError("integration_canary_robolab_revision_unpinned")
    tasks = [str(task) for task in reference_task_ids]
    if not tasks or any(not _IDENTIFIER.fullmatch(task) for task in tasks):
        raise IntegrationCanaryError("integration_canary_reference_tasks_invalid")
    try:
        generated_at = datetime.fromisoformat(generated_at_iso.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IntegrationCanaryError("integration_canary_generated_at_invalid") from exc
    if generated_at.tzinfo is None:
        raise IntegrationCanaryError("integration_canary_generated_at_invalid")

    path = Path(source_result_path).expanduser()
    if path.is_symlink() or not path.is_file():
        raise IntegrationCanaryError("integration_canary_source_result_invalid")
    try:
        source = json.loads(path.read_text(encoding="utf-8"))
        validate_session_result(source, allow_legacy_missing_task_success_contract=True)
    except (ValueError, UnicodeDecodeError) as exc:
        raise IntegrationCanaryError("integration_canary_source_result_invalid") from exc
    evidence = Path(evidence_root).expanduser()
    if evidence.is_symlink() or not evidence.is_dir():
        raise IntegrationCanaryError("integration_canary_evidence_root_invalid")
    evidence = evidence.resolve()
    try:
        inventory = _verify_artifact_inventory(source, evidence_root=evidence)
    except PolicyCanaryRescoreError as exc:
        raise IntegrationCanaryError("integration_canary_artifact_inventory_invalid") from exc

    by_candidate: dict[str, list[dict[str, Any]]] = {}
    identities: dict[str, set[tuple[str, str]]] = {}
    for raw in source.get("episodes") or []:
        row = dict(raw) if isinstance(raw, Mapping) else {}
        candidate_id = str(row.get("candidate_id") or "")
        if not _IDENTIFIER.fullmatch(candidate_id):
            raise IntegrationCanaryError("integration_canary_episode_identity_invalid")
        identities.setdefault(candidate_id, set()).add(
            (str(row.get("checkpoint_digest") or ""), str(row.get("runtime_identity_digest") or ""))
        )
        episode = row.get("episode") if isinstance(row.get("episode"), Mapping) else {}
        if row.get("status") != "completed":
            reason = str(row.get("typed_harness_failure") or "not_completed")
            checks = [_check("actions_returned", "failed", f"episode did not complete: {reason}")]
        else:
            state = episode.get("state_trace") if isinstance(episode.get("state_trace"), Mapping) else None
            if state is None or row.get("state_trace_digest") != canonical_digest({"value": state}):
                raise IntegrationCanaryError("integration_canary_state_trace_invalid")
            try:
                _, state_path = _verified_episode_artifact(
                    row=row, role="state_trace", inventory=inventory, evidence_root=evidence
                )
                retained_state = json.loads(state_path.read_text(encoding="utf-8"))
            except (PolicyCanaryRescoreError, ValueError) as exc:
                raise IntegrationCanaryError("integration_canary_state_trace_invalid") from exc
            if retained_state != state:
                raise IntegrationCanaryError("integration_canary_state_trace_invalid")
            try:
                _, action_path = _verified_episode_artifact(
                    row=row, role="action_sequence", inventory=inventory, evidence_root=evidence
                )
                actions = json.loads(action_path.read_text(encoding="utf-8"))
            except (PolicyCanaryRescoreError, ValueError) as exc:
                raise IntegrationCanaryError("integration_canary_action_sequence_invalid") from exc
            if not isinstance(actions, list):
                raise IntegrationCanaryError("integration_canary_action_sequence_invalid")
            checks = integration_checks(
                actions=actions,
                samples=state.get("task_state_samples") or [],
                joint_states=state.get("joint_states") or [],
            )
        by_candidate.setdefault(candidate_id, []).append({
            "cell_id": str(row.get("cell_id") or ""),
            "seed": row.get("seed"),
            "checks": checks,
        })

    candidates = []
    for candidate_id, episodes in sorted(by_candidate.items()):
        identity = identities[candidate_id]
        if len(identity) != 1:
            raise IntegrationCanaryError("integration_canary_candidate_identity_ambiguous")
        checkpoint, runtime = next(iter(identity))
        if not _DIGEST.fullmatch(checkpoint) or not _DIGEST.fullmatch(runtime):
            raise IntegrationCanaryError("integration_canary_candidate_identity_invalid")
        passed, failures = candidate_verdict(episodes)
        candidates.append({
            "candidate_id": candidate_id,
            "checkpoint_digest": checkpoint,
            "runtime_identity_digest": runtime,
            "episode_count": len(episodes),
            "passed": passed,
            "failed_checks": failures,
            "episodes": episodes,
        })
    if not candidates:
        raise IntegrationCanaryError("integration_canary_no_episodes")

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "backlog_item": "ADP-050",
        "reference": {
            "suite": "nvidia_robolab",
            "revision": robolab_revision,
            "task_ids": tasks,
        },
        "source": {
            "run_id": str(source.get("run_id") or ""),
            "result_digest": str(source.get("result_digest") or ""),
        },
        "thresholds": dict(THRESHOLDS),
        "candidates": candidates,
        "passed": all(candidate["passed"] for candidate in candidates),
        "authority": {
            "proves": "integration_wiring_only",
            "policy_quality_claimed": False,
            "ranking_or_promotion_effect": "none",
        },
        "generated_at_iso": generated_at_iso,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = cross_runtime_canonical_digest(receipt, digest_field="receipt_digest")
    return receipt


# -------------------------------------------------------------------- gates


def integration_canary_blockers(
    receipt: Mapping[str, Any] | None,
    *,
    candidate_ids: Sequence[str],
    now: datetime | None = None,
    max_age_days: int = DEFAULT_MAX_AGE_DAYS,
) -> list[str]:
    """Why a paid session may not rely on this receipt; empty when it may."""

    if receipt is None:
        return ["policy_integration_canary_receipt_missing"]
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("receipt_digest")
        != cross_runtime_canonical_digest(dict(receipt), digest_field="receipt_digest")
    ):
        return ["policy_integration_canary_receipt_invalid"]
    blockers: list[str] = []
    reference = receipt.get("reference") if isinstance(receipt.get("reference"), Mapping) else {}
    if not _COMMIT.fullmatch(str(reference.get("revision") or "")):
        blockers.append("policy_integration_canary_robolab_revision_unpinned")
    try:
        generated = datetime.fromisoformat(str(receipt.get("generated_at_iso")).replace("Z", "+00:00"))
        current = now or datetime.now(timezone.utc)
        if generated.tzinfo is None or current - generated > timedelta(days=max_age_days) or generated > current:
            blockers.append("policy_integration_canary_receipt_stale")
    except ValueError:
        blockers.append("policy_integration_canary_receipt_invalid")
    rows = {
        str(row.get("candidate_id")): row
        for row in receipt.get("candidates") or []
        if isinstance(row, Mapping)
    }
    for candidate_id in candidate_ids:
        row = rows.get(str(candidate_id))
        if row is None:
            blockers.append(f"policy_integration_canary_candidate_missing:{candidate_id}")
        elif row.get("passed") is not True:
            failed = ",".join(str(value) for value in row.get("failed_checks") or []) or "unknown"
            blockers.append(f"policy_integration_canary_candidate_failed:{candidate_id}:{failed}")
    return sorted(set(blockers))


def session_checkpoint_mismatches(receipt: Mapping[str, Any], session_result: Mapping[str, Any]) -> list[str]:
    """Candidates whose paid-session checkpoint differs from the one the canary cleared."""

    cleared = {
        str(row.get("candidate_id")): (row.get("checkpoint_digest"), row.get("runtime_identity_digest"))
        for row in receipt.get("candidates") or []
        if isinstance(row, Mapping)
    }
    mismatched = set()
    for row in session_result.get("episodes") or []:
        if not isinstance(row, Mapping):
            continue
        candidate_id = str(row.get("candidate_id"))
        observed = (row.get("checkpoint_digest"), row.get("runtime_identity_digest"))
        if candidate_id not in cleared or cleared[candidate_id] != observed:
            mismatched.add(candidate_id)
    return sorted(mismatched)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source-result", required=True, help="Retained reference-task session result.")
    parser.add_argument("--evidence-root", required=True)
    parser.add_argument("--robolab-revision", required=True, help="The pinned RoboLab commit.")
    parser.add_argument("--reference-task", action="append", required=True, dest="reference_tasks")
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    receipt = build_integration_canary_receipt(
        source_result_path=args.source_result,
        evidence_root=args.evidence_root,
        reference_task_ids=args.reference_tasks,
        robolab_revision=args.robolab_revision,
        generated_at_iso=args.generated_at,
    )
    out = Path(args.out)
    if out.exists():
        raise IntegrationCanaryError("integration_canary_output_exists")
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "receipt_digest": receipt["receipt_digest"]}))
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "IntegrationCanaryError",
    "RECEIPT_SCHEMA_VERSION",
    "THRESHOLDS",
    "build_integration_canary_receipt",
    "candidate_verdict",
    "integration_canary_blockers",
    "integration_checks",
    "session_checkpoint_mismatches",
]
