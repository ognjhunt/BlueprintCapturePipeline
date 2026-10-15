"""Operator registry for the team-candidate Quick-10 runtime (beta L5).

Beta journey step: L5 team policy evaluation. As with the controlled native
queue, customer JSON never chooses files, budgets or launch commands: an
operator-owned registry (path in ``BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY``,
not group- or world-writable, owned by root or the service user, self-digest
intact) binds one capture to:

- the scene's frozen internal Quick-10 inputs and activation (ten cells);
- the directory the scene's robot-free probe receipt is retained in (read
  only through ``robot_free_probe_gate.retained_probe_receipt``);
- the open-loop horizon, hourly-rate ceiling and hard TTL of a run;
- the founder approvers whose free-beta funding approval may launch a run;
- one sealed rights record per admitted team checkpoint.

:func:`offer_eligibility` is the one rule both the offer publisher and the
launch apply: the Quick-10 cells resolve to the DROID Franka (Robotiq 2F-85)
embodiment, strict per-cell controls are not required, and the robot-free
probe passed at the inputs' current scene revision for the scene's task.
Anything else fails closed with a typed reason, and the team offer is not
listed.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID
from .team_candidate_quick10_staging import TeamCandidateStagingError, frozen_quick10_inputs

REGISTRY_ENV = "BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY"
REGISTRY_SCHEMA = "blueprint.team_candidate_quick10_registry.v1"
PROFILE = "policy_canary_quick10_team_candidate_v1"
EPISODES = 10


class TeamCandidateRegistryError(ValueError):
    """The operator registry or a profile it names is unusable."""


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TeamCandidateRegistryError("team_candidate_registry_record_invalid")
    return value


def load_registry() -> dict[str, Any] | None:
    raw = os.environ.get(REGISTRY_ENV)
    if not raw:
        return None
    path = Path(raw)
    try:
        status = path.lstat()
    except OSError as exc:
        raise TeamCandidateRegistryError("team_candidate_registry_missing") from exc
    if (not path.is_absolute() or stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode)
            or status.st_mode & 0o022 or status.st_uid not in {0, os.getuid()}):
        raise TeamCandidateRegistryError("team_candidate_registry_not_operator_owned")
    registry = _read(path)
    if (registry.get("schema_version") != REGISTRY_SCHEMA
            or registry.get("registry_digest") != canonical_digest(registry, digest_field="registry_digest")
            or not isinstance(registry.get("profiles"), list)):
        raise TeamCandidateRegistryError("team_candidate_registry_binding_invalid")
    return registry


def profile_for_capture(capture_root: str | Path) -> dict[str, Any] | None:
    registry = load_registry()
    if registry is None:
        return None
    root = str(Path(capture_root).resolve())
    matches = [dict(row) for row in registry["profiles"] if isinstance(row, Mapping) and row.get("capture_root") == root]
    if len(matches) > 1:
        raise TeamCandidateRegistryError("team_candidate_profile_ambiguous")
    return matches[0] if matches else None


def profile_for_request(request: Mapping[str, Any]) -> dict[str, Any] | None:
    """The capture's profile when the request names exactly its task and scenario."""

    tasks = request.get("requested_tasks") or []
    if len(tasks) != 1 or len(tasks[0].get("scenario_ids") or []) != 1:
        return None
    profile = profile_for_capture(str(request.get("capture_root") or ""))
    if (profile is None or profile.get("task_id") != tasks[0].get("task_id")
            or profile.get("scenario_id") != tasks[0]["scenario_ids"][0]):
        return None
    return profile


def rights_record(profile: Mapping[str, Any], *, team_id: str, checkpoint_id: str) -> dict[str, Any] | None:
    path = (profile.get("team_rights") or {}).get(f"{team_id}/{checkpoint_id}")
    return _read(Path(path)) if path else None


def scene_facts(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a team run needs from the scene's frozen Quick-10 inputs (no network)."""

    from .adp009d_policy_episode import maximum_policy_queries_for_task_spec
    from .native_task_arena_policy_canary_worker import _resolved_scene_plan

    try:
        inputs, activation = frozen_quick10_inputs(Path(profile["pair_runtime_inputs_path"]),
                                                   Path(profile["pair_activation_path"]))
    except (KeyError, OSError, ValueError, TeamCandidateStagingError) as exc:
        raise TeamCandidateRegistryError("team_candidate_frozen_quick10_inputs_unavailable") from exc
    packet_dir = Path(inputs["base_native_packet"]["path"]).parent
    plan = _read(packet_dir / "native_task_arena_scene_plan.v1.json")
    try:
        resolved = [_resolved_scene_plan(plan, cell, task_success_contract=inputs["task_success_contract"])
                    for cell in inputs["cells"]]
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TeamCandidateRegistryError("team_candidate_droid_franka_cells_unavailable") from exc
    if ((plan.get("robot") or {}).get("robot_id") != "franka_panda" or len(resolved) != EPISODES
            or any((row.get("policy_canary_embodiment_profile") or {}).get("robot_preset_id")
                   != DROID_POLICY_CANARY_PRESET_ID for row in resolved)):
        raise TeamCandidateRegistryError("team_candidate_droid_franka_cells_unavailable")
    horizon = profile.get("open_loop_horizon")
    if type(horizon) is not int or horizon <= 0:
        raise TeamCandidateRegistryError("team_candidate_open_loop_horizon_invalid")
    raw = Path(profile["pair_runtime_inputs_path"]).read_bytes()
    return {
        "inputs": inputs, "activation": activation, "packet_dir": packet_dir,
        "runtime_source_packet_receipt": Path(inputs["runtime_source"]["path"]),
        "scene": {"scene_revision_digest": inputs.get("scene_revision_digest"), "task_spec": plan.get("task_spec"),
                  "task_success_contract": inputs["task_success_contract"],
                  "embodiment_preset_id": DROID_POLICY_CANARY_PRESET_ID},
        "prompt": resolved[0]["task_spec"]["prompt"],
        "max_policy_queries": maximum_policy_queries_for_task_spec(resolved[0]["task_spec"],
                                                                   open_loop_horizon=horizon),
        "open_loop_horizon": horizon,
        "episode_specs_sha256": "sha256:" + hashlib.sha256(raw).hexdigest(),
    }


def offer_eligibility(profile: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, list[str]]:
    """The scene facts when the team runtime may be listed and launched, else every reason."""

    from .native_policy_canary_control_gate import controls_required
    from .robot_free_probe_gate import (
        RobotFreeProbeGateError, require_robot_free_probe, retained_probe_receipt, task_success_contract_binding)

    if profile is None:
        return None, ["team_candidate_profile_missing"]
    try:
        facts = scene_facts(profile)
        receipt = retained_probe_receipt(Path(profile["robot_free_probe_root"]))
    except (KeyError, OSError, ValueError) as exc:
        return None, [str(exc) or "team_candidate_scene_facts_unavailable"]
    blockers = []
    if controls_required(facts["inputs"]["task_success_contract"]):
        blockers.append("team_candidate_strict_controls_runtime_unavailable")
    task_spec = facts["scene"]["task_spec"] if isinstance(facts["scene"]["task_spec"], Mapping) else {}
    try:
        facts["probe"] = require_robot_free_probe(
            receipt, scene_revision_digest=facts["scene"]["scene_revision_digest"],
            task_spec_digest=canonical_digest(dict(task_spec)), task_kind=task_spec.get("task_kind"),
            task_success_contract_digest=task_success_contract_binding(
                task_spec.get("task_kind"), facts["inputs"]["task_success_contract"]))
        facts["probe_receipt"] = receipt
    except RobotFreeProbeGateError as exc:
        blockers.append(str(exc))
    return (None, blockers) if blockers else (facts, [])


__all__ = [
    "EPISODES",
    "PROFILE",
    "REGISTRY_ENV",
    "REGISTRY_SCHEMA",
    "TeamCandidateRegistryError",
    "load_registry",
    "offer_eligibility",
    "profile_for_capture",
    "profile_for_request",
    "rights_record",
    "scene_facts",
]
