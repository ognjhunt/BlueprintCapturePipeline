"""The frozen success definition for one articulated open/close task.

Twin of the rigid task success contract: same envelope, same provenance rules,
same cross-runtime digest, same frozen safety predicate. Only the criteria
differ, because an open/close task has no destination, no lift and no
placement tolerance. What it has is one target joint, an interval on that
joint's own coordinate, and the requirement that the part stay open while the
assembly stays put.

The compatibility default is a pure translation of the already-frozen
articulated task spec, so sealing a default contract never changes what the
scorer would have done.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

try:  # flat provider-bundle layout
    from adp_task_scoring import (
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        TaskNeutralScoringError,
        cross_runtime_canonical_digest,
        validate_articulated_task_spec,
    )
except ModuleNotFoundError:  # repository package
    from .adp_task_scoring import (
        TASK_KIND_ARTICULATED_OPEN_CLOSE,
        TaskNeutralScoringError,
        cross_runtime_canonical_digest,
        validate_articulated_task_spec,
    )

SCHEMA_VERSION = "articulated_task_success_contract.v1"
AUTHOR_SOURCES = {"compatibility_default", "site_robot_team", "task_owner", "agent_proposal"}
CONFIRMATION_STATUSES = {"proposal_only", "confirmed"}
EVENT_LEDGER_SCHEMA_VERSION = "articulated_task_event_ledger_expectation.v1"
#: Frozen, exactly as the rigid contract freezes it: an author may not relax
#: the requirement that forbidden collisions and joint-limit violations fail.
SAFETY_PREDICATE = {"mode": "required"}


def _error(codes: Sequence[str]) -> TaskNeutralScoringError:
    return TaskNeutralScoringError([f"articulated_task_success_contract_{code}" for code in codes])


def _finite(value: Any, *, positive: bool = False, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if positive and number <= 0.0:
        return None
    if nonnegative and number < 0.0:
        return None
    return number


def _fields(value: Mapping[str, Any], *, allowed: set[str], label: str, errors: list[str]) -> None:
    for field in sorted(set(value) - allowed):
        errors.append(f"{label}_unknown_field:{field}")
    for field in sorted(allowed - set(value)):
        errors.append(f"{label}_missing_field:{field}")


def compatibility_articulated_success_criteria(task_spec: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the frozen articulated spec into explicit success predicates."""

    spec = validate_articulated_task_spec(task_spec)
    target = str(spec["target_joint_id"])
    interval = [float(value) for value in spec["target_success_interval_rad"]]
    locked = sorted(
        joint_id for joint_id, role in (spec.get("joint_roles") or {}).items()
        if role != "target"
    )
    return {
        "target_joint": {"joint_id": target, "joint_ids": list(spec["target_joint_ids"])},
        "opening": {"mode": "required", "success_interval": interval,
                    "joint_hard_limits": [float(v) for v in spec["joint_hard_limits_rad"][target]],
                    "reset_position": float(spec["joint_reset_positions_rad"][target])},
        "hold": {"mode": "required", "window_samples": int(spec["settle_window_samples"]),
                 "maximum_settled_target_speed": float(spec["maximum_settled_target_speed_rad_s"])},
        "locked_joints": {"mode": "required" if locked else "ignored",
                          "joint_ids": locked,
                          "motion_tolerance": float(spec["non_task_joint_motion_tolerance_rad"])},
        "reset": {"tolerance": float(spec["reset_tolerance_rad"])},
        "motion": {"movement_epsilon": float(spec["movement_epsilon_rad"])},
        "assembly_root": {"mode": "required"},
        "safety": dict(SAFETY_PREDICATE),
        "temporal_invariants": {
            "schema_version": EVENT_LEDGER_SCHEMA_VERSION,
            "rebound_below_threshold_allowed": False,
            "forbidden_collision_allowed": False,
            "joint_limit_violation_allowed": False,
            "assembly_root_excursion_allowed": False,
        },
    }


def _validate_criteria(criteria: Any, errors: list[str]) -> None:
    if not isinstance(criteria, Mapping):
        errors.append("criteria_invalid")
        return
    _fields(criteria, allowed={
        "target_joint", "opening", "hold", "locked_joints", "reset", "motion",
        "assembly_root", "safety", "temporal_invariants",
    }, label="criteria", errors=errors)
    target = criteria.get("target_joint")
    if not isinstance(target, Mapping) or not str(target.get("joint_id") or "").strip():
        errors.append("criteria_target_joint_invalid")
    opening = criteria.get("opening")
    interval: list[float] = []
    if not isinstance(opening, Mapping) or opening.get("mode") != "required":
        errors.append("criteria_opening_invalid")
    else:
        raw = opening.get("success_interval")
        limits = opening.get("joint_hard_limits")
        reset = _finite(opening.get("reset_position"))
        if (
            not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(list(raw)) != 2
            or not isinstance(limits, Sequence) or isinstance(limits, (str, bytes))
            or len(list(limits)) != 2 or reset is None
        ):
            errors.append("criteria_opening_invalid")
        else:
            interval = [_finite(value) for value in raw]  # type: ignore[misc]
            bounds = [_finite(value) for value in limits]  # type: ignore[misc]
            if (
                any(value is None for value in interval + bounds)
                or interval[0] >= interval[1]
                or bounds[0] >= bounds[1]
                or interval[0] < bounds[0] - 1e-9
                or interval[1] > bounds[1] + 1e-9
                # A success interval containing the closed reset would score an
                # untouched drawer as opened.
                or bounds[0] - 1e-9 <= reset and interval[0] <= reset <= interval[1]
            ):
                errors.append("criteria_opening_invalid")
    hold = criteria.get("hold")
    if (
        not isinstance(hold, Mapping) or hold.get("mode") not in {"required", "ignored"}
        or not isinstance(hold.get("window_samples"), int) or isinstance(hold.get("window_samples"), bool)
        or hold["window_samples"] < 1
        or _finite(hold.get("maximum_settled_target_speed"), positive=True) is None
    ):
        errors.append("criteria_hold_invalid")
    locked = criteria.get("locked_joints")
    if (
        not isinstance(locked, Mapping) or locked.get("mode") not in {"required", "ignored"}
        or not isinstance(locked.get("joint_ids"), list)
        or _finite(locked.get("motion_tolerance"), positive=True) is None
    ):
        errors.append("criteria_locked_joints_invalid")
    for name, key in (("reset", "tolerance"), ("motion", "movement_epsilon")):
        block = criteria.get(name)
        if not isinstance(block, Mapping) or _finite(block.get(key), positive=True) is None:
            errors.append(f"criteria_{name}_invalid")
    root = criteria.get("assembly_root")
    if not isinstance(root, Mapping) or root.get("mode") not in {"required", "ignored"}:
        errors.append("criteria_assembly_root_invalid")
    if criteria.get("safety") != SAFETY_PREDICATE:
        errors.append("criteria_safety_not_frozen")
    ledger = criteria.get("temporal_invariants")
    if not isinstance(ledger, Mapping):
        errors.append("criteria_temporal_invariants_invalid")
    else:
        _fields(ledger, allowed={
            "schema_version", "rebound_below_threshold_allowed", "forbidden_collision_allowed",
            "joint_limit_violation_allowed", "assembly_root_excursion_allowed",
        }, label="criteria_temporal_invariants", errors=errors)
        if ledger.get("schema_version") != EVENT_LEDGER_SCHEMA_VERSION or any(
            ledger.get(field) is not False for field in (
                "forbidden_collision_allowed", "joint_limit_violation_allowed")
        ):
            errors.append("criteria_temporal_invariants_invalid")


def validate_articulated_task_success_contract(
    value: Mapping[str, Any], *, require_confirmed: bool = True,
    expected_site_id: str | None = None, expected_task_id: str | None = None,
) -> dict[str, Any]:
    """Validate one frozen articulated task/team contract without consulting a model."""

    if not isinstance(value, Mapping):
        raise _error(["invalid"])
    try:
        contract = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise _error(["invalid"]) from exc
    errors: list[str] = []
    _fields(contract, allowed={"schema_version", "scope", "provenance", "criteria", "contract_digest"},
            label="root", errors=errors)
    if contract.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_invalid")
    scope = contract.get("scope")
    if not isinstance(scope, Mapping):
        errors.append("scope_invalid")
        scope = {}
    else:
        _fields(scope, allowed={"site_id", "task_id"}, label="scope", errors=errors)
    site_id, task_id = str(scope.get("site_id") or ""), str(scope.get("task_id") or "")
    if not site_id or not task_id:
        errors.append("scope_invalid")
    if expected_site_id is not None and site_id != str(expected_site_id):
        errors.append("site_binding_mismatch")
    if expected_task_id is not None and task_id != str(expected_task_id):
        errors.append("task_binding_mismatch")
    provenance = contract.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("provenance_invalid")
        provenance = {}
    else:
        _fields(provenance, allowed={
            "author_source", "author_id", "confirmation_status", "confirmed_by_team_id", "proposal_digest",
        }, label="provenance", errors=errors)
    author_source = str(provenance.get("author_source") or "")
    confirmation_status = str(provenance.get("confirmation_status") or "")
    confirmed_by = provenance.get("confirmed_by_team_id")
    proposal_digest = provenance.get("proposal_digest")
    if author_source not in AUTHOR_SOURCES or not str(provenance.get("author_id") or ""):
        errors.append("author_invalid")
    if confirmation_status not in CONFIRMATION_STATUSES:
        errors.append("confirmation_invalid")
    if require_confirmed and confirmation_status != "confirmed":
        errors.append("not_confirmed")
    if confirmation_status == "confirmed":
        if not isinstance(confirmed_by, str) or not confirmed_by.strip():
            errors.append("confirming_team_missing")
        # An agent may only originate a proposal; a confirmed agent contract
        # must name the exact proposal a human confirmed.
        if author_source == "agent_proposal" and (
            not isinstance(proposal_digest, str)
            or not proposal_digest.startswith("sha256:")
            or len(proposal_digest) != 71
        ):
            errors.append("agent_confirmation_requires_proposal_digest")
    else:
        if confirmed_by is not None:
            errors.append("unconfirmed_contract_names_a_team")
        if proposal_digest is not None:
            errors.append("unconfirmed_contract_names_a_proposal")
    _validate_criteria(contract.get("criteria"), errors)
    if contract.get("contract_digest") != cross_runtime_canonical_digest(
        contract, digest_field="contract_digest"
    ):
        errors.append("digest_mismatch")
    if errors:
        raise _error(errors)
    return contract


def seal_articulated_task_success_contract(
    *, task_spec: Mapping[str, Any], site_id: str, task_id: str,
    author_source: Literal["compatibility_default", "site_robot_team", "task_owner", "agent_proposal"],
    author_id: str,
    confirmation_status: Literal["proposal_only", "confirmed"],
    confirmed_by_team_id: str | None = None,
    criteria: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Seal team criteria or an explicitly proposal-only agent draft."""

    if author_source == "agent_proposal" and confirmation_status != "proposal_only":
        raise _error(["agent_must_originate_proposal"])
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scope": {"site_id": str(site_id), "task_id": str(task_id)},
        "provenance": {
            "author_source": author_source, "author_id": str(author_id),
            "confirmation_status": confirmation_status,
            "confirmed_by_team_id": confirmed_by_team_id, "proposal_digest": None,
        },
        "criteria": json.loads(json.dumps(
            dict(criteria) if criteria is not None
            else compatibility_articulated_success_criteria(task_spec), allow_nan=False)),
        "contract_digest": "",
    }
    document["contract_digest"] = cross_runtime_canonical_digest(document, digest_field="contract_digest")
    return validate_articulated_task_success_contract(
        document, require_confirmed=False, expected_site_id=site_id, expected_task_id=task_id)


def confirm_articulated_task_success_contract(
    proposal: Mapping[str, Any], *, confirmed_by_team_id: str
) -> dict[str, Any]:
    """Create a confirmed immutable successor to a proposal-only document."""

    validated = validate_articulated_task_success_contract(proposal, require_confirmed=False)
    if validated["provenance"]["confirmation_status"] != "proposal_only":
        raise _error(["not_a_proposal"])
    confirmed = json.loads(json.dumps(validated))
    confirmed["provenance"]["confirmation_status"] = "confirmed"
    confirmed["provenance"]["confirmed_by_team_id"] = str(confirmed_by_team_id)
    if confirmed["provenance"]["author_source"] == "agent_proposal":
        confirmed["provenance"]["proposal_digest"] = validated["contract_digest"]
    confirmed["contract_digest"] = cross_runtime_canonical_digest(confirmed, digest_field="contract_digest")
    return validate_articulated_task_success_contract(confirmed)


def confirmed_articulated_task_success_contract_matches_published(
    *, published: Mapping[str, Any], selected: Mapping[str, Any]
) -> bool:
    """Match an exact published contract or its one team-confirmed successor."""

    try:
        public_contract = validate_articulated_task_success_contract(published, require_confirmed=False)
        confirmed_contract = validate_articulated_task_success_contract(selected)
    except TaskNeutralScoringError:
        return False
    if public_contract["scope"] != confirmed_contract["scope"]:
        return False
    if public_contract["provenance"]["confirmation_status"] == "confirmed":
        return public_contract["contract_digest"] == confirmed_contract["contract_digest"]
    team_id = confirmed_contract["provenance"]["confirmed_by_team_id"]
    if not isinstance(team_id, str) or not team_id:
        return False
    try:
        successor = confirm_articulated_task_success_contract(
            public_contract, confirmed_by_team_id=team_id)
    except TaskNeutralScoringError:
        return False
    return successor["contract_digest"] == confirmed_contract["contract_digest"]


def _rigid(name: str) -> Any:
    """Resolve one rigid-lane symbol under either package layout."""
    try:  # flat provider-bundle layout
        module = __import__("adp_task_scoring")
    except ModuleNotFoundError:  # repository package
        from . import adp_task_scoring as module  # type: ignore[no-redef]
    return getattr(module, name)


# --- kind-dispatching facade -------------------------------------------------
# One entry point per operation, so a call site binds the task kind instead of
# assuming the rigid lane.

def task_success_contract_schema_version(task_kind: str) -> str:
    RIGID_TASK_SUCCESS_CONTRACT_SCHEMA_VERSION = _rigid("RIGID_TASK_SUCCESS_CONTRACT_SCHEMA_VERSION")
    return (SCHEMA_VERSION if task_kind == TASK_KIND_ARTICULATED_OPEN_CLOSE
            else RIGID_TASK_SUCCESS_CONTRACT_SCHEMA_VERSION)


def validate_task_success_contract(value: Mapping[str, Any], *, task_kind: str, **kwargs: Any) -> Any:
    validate_rigid_task_success_contract = _rigid("validate_rigid_task_success_contract")
    if task_kind == TASK_KIND_ARTICULATED_OPEN_CLOSE:
        return validate_articulated_task_success_contract(value, **kwargs)
    return validate_rigid_task_success_contract(value, **kwargs)


def seal_task_success_contract(*, task_kind: str, **kwargs: Any) -> Any:
    seal_rigid_task_success_contract = _rigid("seal_rigid_task_success_contract")
    if task_kind == TASK_KIND_ARTICULATED_OPEN_CLOSE:
        return seal_articulated_task_success_contract(**kwargs)
    return seal_rigid_task_success_contract(**kwargs)


def confirmed_task_success_contract_matches_published(
    *, task_kind: str, published: Mapping[str, Any], selected: Mapping[str, Any]
) -> bool:
    """Match a selection against the published contract of a named kind.

    Callers take the kind from the published contract, which is the authority.
    A selection that answers with the other kind is not validated as the kind it
    claims to be: the matcher's validator refuses it and the match is false.
    """
    confirmed_rigid_task_success_contract_matches_published = _rigid(
        "confirmed_rigid_task_success_contract_matches_published")
    if task_kind == TASK_KIND_ARTICULATED_OPEN_CLOSE:
        return confirmed_articulated_task_success_contract_matches_published(
            published=published, selected=selected)
    return confirmed_rigid_task_success_contract_matches_published(
        published=published, selected=selected)


def task_kind_of_contract(value: Mapping[str, Any]) -> str:
    """Read the task kind a contract belongs to, from its own schema."""
    TASK_KIND_RIGID_PICK_PLACE = _rigid("TASK_KIND_RIGID_PICK_PLACE")
    return (TASK_KIND_ARTICULATED_OPEN_CLOSE
            if isinstance(value, Mapping) and value.get("schema_version") == SCHEMA_VERSION
            else TASK_KIND_RIGID_PICK_PLACE)


__all__ = [
    "EVENT_LEDGER_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "compatibility_articulated_success_criteria",
    "confirm_articulated_task_success_contract",
    "confirmed_articulated_task_success_contract_matches_published",
    "confirmed_task_success_contract_matches_published",
    "seal_articulated_task_success_contract",
    "seal_task_success_contract",
    "task_kind_of_contract",
    "task_success_contract_schema_version",
    "validate_articulated_task_success_contract",
    "validate_task_success_contract",
]
