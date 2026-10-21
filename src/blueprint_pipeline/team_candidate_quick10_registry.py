"""Operator registry for team evaluations (beta L5).

Beta journey step: L5 team policy evaluation. As with the controlled native
queue, customer JSON never chooses files, budgets or launch commands: an
operator-owned registry (path in ``BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY``,
not group- or world-writable, owned by root or the service user, self-digest
intact) binds one capture to:

- the scene's half of a run: the prepared scene's own compiled packet
  (``team_scene_packet``, no robot placement and no prior run), or the scene's
  frozen internal Quick-10 inputs and activation;
- the directory the scene's robot-free probe receipt is retained in (read
  only through ``robot_free_probe_gate.retained_probe_receipt``);
- the open-loop horizon, hourly-rate ceiling and hard TTL of a run;
- the founder approvers whose free-beta funding approval may launch a run;
- one sealed rights record per admitted team checkpoint.

A scene need not be written in by hand: when the registry names a
``derived_profiles`` root, a sealed row holding a prepared scene's own records
(capture, task, the scene half, probe root, revision) is read from there, and
the operator's ``defaults`` supply everything only the operator decides.
``write_derived_profile`` writes that row; ``team_scene_packet`` calls it once
it has compiled a prepared scene. A scored matrix's salt is an HMAC of the
scene revision under the operator's key file; without one, scored evaluations
stay unlisted. An operator row always wins.

:func:`offer_eligibility` is the one rule both the offer publisher and the
launch apply (``team_evaluation_plan``): the scene's packet resolves its
nominal cell, strict per-cell controls are not required, the scene's secret matrix
salt is present, and the robot-free probe passed at the inputs' current
scene revision for the scene's task before any paid team allocation. Missing,
unavailable, or failed probes keep the evaluation unlisted.

Anything else fails closed with a typed reason, and that offer is not listed.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .droid_policy_canary_embodiment import DROID_POLICY_CANARY_PRESET_ID
from .team_candidate_quick10_staging import TeamCandidateStagingError, scene_inputs_from_frozen_pair
from .team_evaluation_plan import PURPOSES, SCORED
from .team_robot_free_probe import scored_task_spec

REGISTRY_ENV = "BLUEPRINT_TEAM_CANDIDATE_QUICK10_REGISTRY"
REGISTRY_SCHEMA = "blueprint.team_candidate_quick10_registry.v1"
DERIVED_SCHEMA = "blueprint.team_candidate_derived_profile.v1"
# What a prepared scene's own records determine (the first six are required, plus one scene half:
# the scene's own compiled packet, or a frozen internal Quick-10 pair).
DERIVED_FIELDS = ("capture_root", "scene_id", "task_id", "scenario_id", "robot_free_probe_root",
                  "scene_revision_digest", "team_scene_inputs_path", "pair_runtime_inputs_path",
                  "pair_activation_path", "nominal_dynamic_friction")
REQUIRED_DERIVED_FIELDS = DERIVED_FIELDS[:6]
# What only the operator decides: who may fund a run, at what rate, and each team's rights.
OPERATOR_FIELDS = ("allowed_founder_approvers", "max_hourly_rate_usd", "open_loop_horizon", "team_rights",
                   "machine_avoidlist_path", "private_robot_models", "prepared_robot_models")
# The scene's nominal cell: what the prompt and query budget are read from.
NOMINAL_CELL = {"cell_id": "scene.nominal", "seed": 0, "family": "canonical_anchor",
                "resolved_scenario": {"family": "canonical_anchor", "ordinal": 0, "parameters": {}}}


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


def _operator_file(path: Path, *, code: str) -> Path:
    try:
        status = path.lstat()
    except OSError as exc:
        raise TeamCandidateRegistryError(code) from exc
    if (not path.is_absolute() or stat.S_ISLNK(status.st_mode) or not stat.S_ISREG(status.st_mode)
            or status.st_mode & 0o022 or status.st_uid not in {0, os.getuid()}):
        raise TeamCandidateRegistryError(code)
    return path


def _derived_settings(registry: Mapping[str, Any]) -> dict[str, Any] | None:
    settings = registry.get("derived_profiles")
    if settings is None:
        return None
    root = Path(str((settings or {}).get("root") or ""))
    defaults = (settings or {}).get("defaults")
    if (not isinstance(settings, Mapping) or not root.is_absolute() or not isinstance(defaults, Mapping)
            or set(defaults) - set(OPERATOR_FIELDS) or set(OPERATOR_FIELDS[:3]) - set(defaults)):
        raise TeamCandidateRegistryError("team_candidate_derived_profile_settings_invalid")
    return {"root": root, "defaults": dict(defaults), "salt_key_path": settings.get("scored_matrix_salt_key_path")}


def _derived_path(root: Path, capture_root: str) -> Path:
    return root / (hashlib.sha256(capture_root.encode()).hexdigest()[:32] + ".json")


def _scored_salt(key_path: Any, scene_revision_digest: Any) -> str | None:
    """A per-scene scored-matrix salt from the operator's key; None (scored unlisted) without one."""

    if not key_path:
        return None
    path = _operator_file(Path(str(key_path)), code="team_candidate_scored_salt_key_not_operator_owned")
    if path.stat().st_mode & 0o077:
        raise TeamCandidateRegistryError("team_candidate_scored_salt_key_not_operator_owned")
    key = path.read_bytes().strip()
    if len(key) < 32:
        raise TeamCandidateRegistryError("team_candidate_scored_salt_key_invalid")
    return hmac.new(key, str(scene_revision_digest).encode(), hashlib.sha256).hexdigest()


def _derived_profile(registry: Mapping[str, Any], capture_root: str) -> dict[str, Any] | None:
    """A scene's derived row (scene facts only) under the operator's defaults (approvers, rights, rate)."""

    settings = _derived_settings(registry)
    if settings is None:
        return None
    path = _derived_path(settings["root"], capture_root)
    if not path.exists():
        return None
    row = _read(_operator_file(path, code="team_candidate_derived_profile_not_operator_owned"))
    if (row.get("schema_version") != DERIVED_SCHEMA or set(row) - {*DERIVED_FIELDS, "schema_version", "profile_digest"}
            or row.get("capture_root") != capture_root
            or row.get("profile_digest") != canonical_digest(row, digest_field="profile_digest")):
        raise TeamCandidateRegistryError("team_candidate_derived_profile_invalid")
    profile = {key: value for key, value in row.items() if key in DERIVED_FIELDS}
    salt = _scored_salt(settings["salt_key_path"], row.get("scene_revision_digest"))
    return {**profile, **settings["defaults"], **({"scored_matrix_salt": salt} if salt else {})}


def profile_for_capture(capture_root: str | Path) -> dict[str, Any] | None:
    """The capture's team-evaluation profile: an operator row, else the scene's derived row."""

    registry = load_registry()
    if registry is None:
        return None
    root = str(Path(capture_root).resolve())
    matches = [dict(row) for row in registry["profiles"] if isinstance(row, Mapping) and row.get("capture_root") == root]
    if len(matches) > 1:
        raise TeamCandidateRegistryError("team_candidate_profile_ambiguous")
    return matches[0] if matches else _derived_profile(registry, root)


def write_derived_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    """Record a prepared scene's facts (never approvers, rights, rate or salt) where the registry reads them."""

    registry = load_registry()
    settings = _derived_settings(registry) if registry is not None else None
    if settings is None:
        return {"status": "derived_profiles_not_configured"}
    value = {key: row[key] for key in DERIVED_FIELDS if row.get(key) is not None}
    missing = sorted(set(REQUIRED_DERIVED_FIELDS) - set(value))
    if not _one_scene_half(value):
        missing.append("team_scene_inputs_path")
    if missing or set(row) - set(DERIVED_FIELDS):
        return {"status": "derived_profile_refused", "missing": missing,
                "unexpected": sorted(set(row) - set(DERIVED_FIELDS))}
    value["capture_root"] = str(Path(str(value["capture_root"])).resolve())
    value = {"schema_version": DERIVED_SCHEMA, **value, "profile_digest": ""}
    value["profile_digest"] = canonical_digest(value, digest_field="profile_digest")
    settings["root"].mkdir(parents=True, exist_ok=True)
    path = _derived_path(settings["root"], value["capture_root"])
    staging = path.with_suffix(".tmp")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    staging.chmod(0o644)
    os.replace(staging, path)
    return {"status": "written", "path": str(path), "profile_digest": value["profile_digest"]}


def _one_scene_half(profile: Mapping[str, Any]) -> bool:
    pair = [bool(profile.get(key)) for key in ("pair_runtime_inputs_path", "pair_activation_path")]
    return bool(profile.get("team_scene_inputs_path")) != (pair == [True, True]) and pair in ([True, True], [False, False])


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


def _scene_inputs(profile: Mapping[str, Any]) -> dict[str, Any]:
    if not _one_scene_half(profile):
        raise TeamCandidateRegistryError("team_candidate_scene_half_ambiguous")
    if profile.get("team_scene_inputs_path"):
        from .team_scene_packet import TeamScenePacketError, read_team_scene_inputs

        try:
            return read_team_scene_inputs(Path(profile["team_scene_inputs_path"]))
        except (OSError, ValueError, TeamScenePacketError) as exc:
            raise TeamCandidateRegistryError("team_scene_inputs_unavailable") from exc
    try:
        return scene_inputs_from_frozen_pair(Path(profile["pair_runtime_inputs_path"]),
                                             Path(profile["pair_activation_path"]))
    except (KeyError, OSError, ValueError, TeamCandidateStagingError) as exc:
        raise TeamCandidateRegistryError("team_candidate_frozen_quick10_inputs_unavailable") from exc


def scene_facts(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Everything a team run needs from the scene's half of the evaluation (no network)."""

    from .adp009d_policy_episode import maximum_policy_queries_for_task_spec
    from .native_task_arena_policy_canary_worker import _resolved_scene_plan

    inputs = _scene_inputs(profile)
    packet_dir = Path(inputs["base_native_packet"]["path"]).parent
    plan = _read(packet_dir / "native_task_arena_scene_plan.v1.json")
    try:
        nominal = _resolved_scene_plan(plan, {**NOMINAL_CELL, "control_diagnostic": inputs["control_diagnostic"]},
                                       task_success_contract=inputs["task_success_contract"])
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        raise TeamCandidateRegistryError("team_candidate_droid_franka_cells_unavailable") from exc
    if ((plan.get("robot") or {}).get("robot_id") != "franka_panda"
            or (nominal.get("policy_canary_embodiment_profile") or {}).get("robot_preset_id")
            != DROID_POLICY_CANARY_PRESET_ID):
        raise TeamCandidateRegistryError("team_candidate_droid_franka_cells_unavailable")
    horizon = profile.get("open_loop_horizon")
    if type(horizon) is not int or horizon <= 0:
        raise TeamCandidateRegistryError("team_candidate_open_loop_horizon_invalid")
    return {
        "inputs": inputs, "packet_dir": packet_dir,
        "runtime_source_packet_receipt": Path(inputs["runtime_source"]["path"]),
        # A rigid task's robot-free probe is planned and bound against the specification a team is
        # scored with: the scene's, under the job's success contract (team_robot_free_probe). The
        # articulated probe binds the scene's specification as compiled.
        "scene": {"scene_revision_digest": inputs.get("scene_revision_digest"),
                  "task_spec": (scored_task_spec(plan["task_spec"], inputs["task_success_contract"])
                                if (plan.get("task_spec") or {}).get("task_kind") == "rigid_pick_place"
                                else plan.get("task_spec")),
                  "task_success_contract": inputs["task_success_contract"],
                  "embodiment_preset_id": DROID_POLICY_CANARY_PRESET_ID},
        "prompt": nominal["task_spec"]["prompt"],
        "max_policy_queries": maximum_policy_queries_for_task_spec(nominal["task_spec"], open_loop_horizon=horizon),
        "open_loop_horizon": horizon,
        "episode_specs_sha256": canonical_digest(inputs),
    }


def offer_eligibility(profile: Mapping[str, Any] | None,
                      purpose: str = SCORED) -> tuple[dict[str, Any] | None, list[str]]:
    """The scene facts when this evaluation purpose may be listed and launched, else every reason."""

    from .native_policy_canary_control_gate import controls_required
    from .robot_free_probe_gate import (
        RobotFreeProbeGateError,
        require_robot_free_probe,
        retained_probe_receipt,
        task_success_contract_binding,
    )

    if profile is None:
        return None, ["team_candidate_profile_missing"]
    if purpose not in PURPOSES:
        return None, ["team_candidate_evaluation_purpose_invalid"]
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
        facts["probe"], facts["probe_receipt"] = None, receipt
        blockers.append(str(exc))
    salt = profile.get("scored_matrix_salt")
    if purpose == SCORED and (not isinstance(salt, str) or len(salt) < 32):
        blockers.append("team_scored_matrix_salt_missing")
    return (None, blockers) if blockers else (facts, [])


__all__ = [
    "DERIVED_FIELDS",
    "DERIVED_SCHEMA",
    "NOMINAL_CELL",
    "OPERATOR_FIELDS",
    "REGISTRY_ENV",
    "REGISTRY_SCHEMA",
    "REQUIRED_DERIVED_FIELDS",
    "TeamCandidateRegistryError",
    "load_registry",
    "offer_eligibility",
    "profile_for_capture",
    "profile_for_request",
    "rights_record",
    "scene_facts",
    "write_derived_profile",
]
