"""Pure replay of family evidence at the canary/result boundary.

Native action delivery and native task measurements establish interpretability;
an arm-motion flag cannot represent all supported robot configurations.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .task_family_scoring import score_task_family_episode, validate_task_family_contract


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sealed(value: Mapping[str, Any], field: str) -> bool:
    try:
        return bool(value) and value.get(field) == canonical_digest(value, digest_field=field)
    except (ValueError, TypeError):
        return False


def _episode(task: Mapping[str, Any], value: Any) -> bool:
    row = _mapping(value)
    telemetry = _mapping(row.get("telemetry"))
    provenance = _mapping(row.get("native_provenance"))
    if (row.get("schema_version") != "task_family_native_episode.v1"
            or row.get("status") != "completed" or row.get("runtime_closed") is not True
            or row.get("failure") is not None or row.get("scoring_source") != "deterministic_native_state"
            or row.get("physical_validation") is not False or not _sealed(row, "receipt_digest")
            or not provenance or telemetry.get("native_provenance") != provenance
            or _mapping(row.get("visual_evidence")).get("status") != "complete"):
        return False
    samples = telemetry.get("samples")
    steps = row.get("maximum_steps")
    if (type(steps) is not int or steps < 1 or not isinstance(samples, list)
            or len(samples) != steps + 1
            or any(_mapping(sample).get("native_provenance") != provenance for sample in samples)):
        return False
    try:
        replayed = score_task_family_episode(contract=task, telemetry=telemetry)
    except (ValueError, TypeError, KeyError):
        return False
    return (row.get("scoring") == replayed
            and row.get("task_succeeded") is replayed["task_succeeded"])


def family_site_qualification_valid(task: Mapping[str, Any], value: Any) -> bool:
    """Require retained native robot-free, negative and positive controls."""
    site = _mapping(value)
    controls = _mapping(site.get("controls"))
    names = {"robot_free_probe", "zero_action_negative", "deterministic_scripted_positive"}
    if (site.get("schema_version") != "task_family_native_site_qualification.v1"
            or site.get("status") != "passed" or site.get("failure") is not None
            or site.get("candidate_policy_queried") is not False
            or site.get("scoring_source") != "deterministic_native_state"
            or site.get("physical_validation") is not False
            or site.get("task_success_contract_digest") != task.get("contract_digest")
            or not _sealed(site, "receipt_digest") or set(controls) != names):
        return False
    for name in names:
        control = _mapping(controls[name])
        native = _mapping(control.get("episode"))
        states = control.get("native_state_trace")
        if (control.get("passed") is not True or not _episode(task, native)
                or native.get("task_succeeded") is not (name == "deterministic_scripted_positive")
                or native.get("native_provenance") != site.get("native_provenance")
                or not isinstance(states, list) or len(states) != native["maximum_steps"] + 1
                or any(_mapping(state).get("safety_violation") is not False for state in states)
                or any(_mapping(state).get("robot_present") is not (name != "robot_free_probe") for state in states)):
            return False
        if name != "deterministic_scripted_positive" and any(
                _mapping(state).get("robot_action_applied") is not False for state in states):
            return False
        if name == "deterministic_scripted_positive" and any(
                _mapping(state).get("robot_action_applied") is not True for state in states[1:]):
            return False
    return True


def family_policy_episode_interpretable(row: Mapping[str, Any], contract: Mapping[str, Any]) -> bool:
    """Replay measurements and controls without inventing arm-motion evidence."""
    try:
        task = validate_task_family_contract(contract)
    except (ValueError, TypeError):
        return False
    episode = _mapping(row.get("episode"))
    native = _mapping(episode.get("native_family_receipt"))
    site = _mapping(episode.get("native_site_qualification"))
    motion = _mapping(episode.get("motion_evidence"))
    states = episode.get("state_trace")
    if (row.get("status") != "completed" or row.get("candidate_policy_queried") is not True
            or row.get("actions_reached_robot") is not True
            or row.get("scoring_authority") != "deterministic_simulator_state"
            or not _episode(task, native) or not family_site_qualification_valid(task, site)
            or native.get("native_provenance") != site.get("native_provenance")
            or motion.get("actions_reached_robot") is not True
            or motion.get("native_action_steps") != native.get("maximum_steps")
            or not isinstance(states, list) or len(states) != native["maximum_steps"] + 1
            or any(_mapping(state).get("robot_action_applied") is not True for state in states[1:])):
        return False
    expected = {**native["scoring"], "status": "scored"}
    return (episode.get("score") == expected
            and row.get("deterministic_score_digest") == canonical_digest({"value": expected}))


__all__ = ["family_site_qualification_valid", "family_policy_episode_interpretable"]
