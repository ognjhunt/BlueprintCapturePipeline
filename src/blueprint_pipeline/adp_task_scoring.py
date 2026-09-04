"""Task-neutral deterministic scoring for ADP rigid and articulated tasks.

The original ADP-009D scorer remains the authority for its sealed rigid
pick/place fixture.  This module adds a stable discriminator and an articulated
joint-state scorer without copying or weakening that legacy path.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, TypedDict

try:  # flat provider-bundle layout
    from adp009d_task_scoring import TaskScoringError, score_task_episode
except ModuleNotFoundError:  # repository package
    from .adp009d_task_scoring import TaskScoringError, score_task_episode
try:  # flat provider-bundle layout
    from decision_evidence_contracts import (
        canonical_digest,
        cross_runtime_canonical_digest,
    )
except ModuleNotFoundError:  # repository package
    from .decision_evidence_contracts import (
        canonical_digest,
        cross_runtime_canonical_digest,
    )
try:  # flat provider-bundle layout
    from articulation_graph_contract import (
        ArticulationGraphContractError,
        validate_articulation_graph,
    )
except ModuleNotFoundError:  # repository package
    from .articulation_graph_contract import (
        ArticulationGraphContractError,
        validate_articulation_graph,
    )


TASK_SPEC_SCHEMA_VERSION = "adp_task_spec.v1"
TASK_SPEC_GRAPH_SCHEMA_VERSION = "adp_task_spec.v2"
ARTICULATED_REPORT_SCHEMA_VERSION = "adp_articulated_task_scoring.v1"
RIGID_REPORT_SCHEMA_VERSION = "adp_rigid_task_scoring.v2"
TASK_KIND_RIGID_PICK_PLACE = "rigid_pick_place"
TASK_KIND_ARTICULATED_OPEN_CLOSE = "articulated_open_close"
RIGID_MANIPULATION_STRATEGIES = {"pick_and_place", "planar_push"}
RIGID_TASK_SUCCESS_CONTRACT_SCHEMA_VERSION = "rigid_task_success_contract.v1"


class RigidTaskSuccessContractScope(TypedDict):
    """Immutable identity boundary for one task/site success definition."""

    site_id: str
    task_id: str


class RigidTaskSuccessContractProvenance(TypedDict):
    """Authorship and confirmation facts; agents can only originate proposals."""

    author_source: Literal[
        "compatibility_default", "site_robot_team", "task_owner", "agent_proposal"
    ]
    author_id: str
    confirmation_status: Literal["proposal_only", "confirmed"]
    confirmed_by_team_id: str | None
    proposal_digest: str | None


class RigidTaskEventLedgerExpectation(TypedDict):
    """Whole-episode event limits, disabled only through explicit null/ignore."""

    schema_version: Literal["rigid_task_event_ledger_expectation.v1"]
    no_drop: dict[str, Any]
    maximum_task_contact_force_n: float | None
    forbidden_contact_classes: list[str]
    containment_excursions: Literal["forbidden", "ignored"]
    workspace_excursions: Literal["forbidden", "ignored"]
    maximum_retries: int | None
    maximum_regrasps: int | None


class RigidTaskSuccessContractCriteria(TypedDict):
    """JSON-shaped deterministic predicates consumed by the rigid scorer."""

    destination_containment: dict[str, Any]
    orientation: dict[str, Any]
    support: dict[str, Any]
    terminal_task_contact: dict[str, Any]
    gripper_state: dict[str, Any]
    settling: dict[str, Any]
    safety: dict[str, Any]
    motion: dict[str, Any]
    temporal_invariants: RigidTaskEventLedgerExpectation


class RigidTaskSuccessContract(TypedDict):
    """Digest-sealed success definition; confirmation creates a new document."""

    schema_version: Literal["rigid_task_success_contract.v1"]
    scope: RigidTaskSuccessContractScope
    provenance: RigidTaskSuccessContractProvenance
    criteria: RigidTaskSuccessContractCriteria
    contract_digest: str

# How far past a sealed hard limit a joint may read before this recomputation
# calls it a violation.  This is a solver-residual allowance, not a task
# tolerance: a joint resting on its own stop reports a tiny excursion in any
# physics engine, and C29 measured -5.7e-8 rad on the closed washer door.  At
# 1e-5 rad the allowance is ~175x that residual and still ~0.0006 degrees, far
# below any mechanically meaningful excursion, while the simulator's own
# per-sample violation flag remains authoritative and unqualified.
JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD = 1.0e-5

OUTCOME_NEVER_MOVED = "never_moved"
OUTCOME_MOVED_BELOW_THRESHOLD = "moved_below_threshold"
OUTCOME_OPENED_THEN_REBOUNDED = "opened_then_rebounded"
OUTCOME_NON_TASK_JOINT_MOVED = "non_task_joint_moved"
OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION = "joint_limit_or_containment_violation"
OUTCOME_COLLISION_FAILURE = "robot_or_scene_collision_failure"
OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE = "release_or_retreat_incomplete"
OUTCOME_OPENED_AND_SETTLED = "opened_and_settled"
OUTCOME_PUSHED_AND_SETTLED = "pushed_and_settled"

_ARTICULATED_OUTCOME_RANK = {
    OUTCOME_NEVER_MOVED: 0,
    OUTCOME_NON_TASK_JOINT_MOVED: 0,
    OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION: 0,
    OUTCOME_COLLISION_FAILURE: 0,
    OUTCOME_MOVED_BELOW_THRESHOLD: 1,
    OUTCOME_OPENED_THEN_REBOUNDED: 2,
    OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE: 3,
    OUTCOME_OPENED_AND_SETTLED: 4,
}


class TaskNeutralScoringError(ValueError):
    """Stable, sorted task-neutral scoring failures."""

    def __init__(self, errors: Sequence[str]):
        self.errors = tuple(sorted(set(str(error) for error in errors if str(error))))
        super().__init__(";".join(self.errors))


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _compatibility_rigid_success_criteria(
    task_spec: Mapping[str, Any],
) -> RigidTaskSuccessContractCriteria:
    """Translate the historical strategy branch into explicit predicates."""

    strategy = str(task_spec.get("manipulation_strategy") or "pick_and_place")
    planar_push = strategy == "planar_push"
    return {
        "destination_containment": {
            "mode": "required",
            "position_bounds_world_m": json.loads(
                json.dumps(task_spec.get("destination_position_bounds_world_m"))
            ),
        },
        "orientation": {
            "mode": "ignored" if planar_push else "required",
            "reference_xyzw": list(task_spec.get("destination_orientation_xyzw") or []),
            "tolerance_rad": task_spec.get(
                "destination_orientation_tolerance_rad"
            ),
        },
        "support": {
            "height_mode": "required",
            "height_interval_m": list(task_spec.get("support_height_interval_m") or []),
            "contact_mode": "required",
        },
        "terminal_task_contact": {
            # Historical pick/place release already couples an open gripper to
            # cleared task contact.  Keeping this separate predicate ignored
            # preserves that exact behavior while allowing authored tasks to
            # require cleared, maintained, or irrelevant terminal contact.
            "mode": "cleared" if planar_push else "ignored",
        },
        "gripper_state": {
            "mode": "ignored" if planar_push else "released",
            "threshold_m": (
                None
                if planar_push
                else task_spec.get("release_gripper_width_min_m")
            ),
        },
        "settling": {
            "mode": "required",
            "window_samples": task_spec.get("settle_window_samples"),
            "position_tolerance_m": task_spec.get("settle_position_tolerance_m"),
            "orientation_tolerance_rad": task_spec.get(
                "settle_orientation_tolerance_rad"
            ),
        },
        # Safety is deliberately not author-overridable.  A task team may
        # define success, but it may not turn an unsafe rollout into success.
        "safety": {"mode": "required"},
        "motion": {
            "movement_epsilon_m": task_spec.get("movement_epsilon_m"),
            "minimum_translation_m": task_spec.get("minimum_translation_m"),
            "minimum_lift_m": (
                None if planar_push else task_spec.get("minimum_lift_m")
            ),
        },
        "temporal_invariants": {
            "schema_version": "rigid_task_event_ledger_expectation.v1",
            # Compatibility defaults preserve the historical eventual-state
            # result.  A task/site team may explicitly require no-drop, in
            # which case a later recovery into the target does not erase the
            # observed drop event.
            "no_drop": {"mode": "ignored", "minimum_fall_m": 0.02},
            "maximum_task_contact_force_n": None,
            "forbidden_contact_classes": [],
            "containment_excursions": "forbidden",
            "workspace_excursions": "ignored",
            "maximum_retries": None,
            "maximum_regrasps": None,
        },
    }


def _validate_contract_fields(
    value: Mapping[str, Any],
    *,
    allowed: set[str],
    label: str,
    errors: list[str],
) -> None:
    for field in sorted(set(value) - allowed):
        errors.append(f"rigid_task_success_contract_{label}_unknown_field:{field}")
    for field in sorted(allowed - set(value)):
        errors.append(f"rigid_task_success_contract_{label}_missing_field:{field}")


def validate_rigid_task_success_contract(
    value: Mapping[str, Any],
    *,
    require_confirmed: bool = True,
    expected_site_id: str | None = None,
    expected_task_id: str | None = None,
) -> RigidTaskSuccessContract:
    """Validate one frozen task/team contract without consulting a model.

    A proposal may be inspected with ``require_confirmed=False``.  Deterministic
    scoring and pre-execution admission use the default and therefore refuse an
    unconfirmed proposal.  Confirmation never mutates a proposal in place; it
    produces a new digest-bound document through
    :func:`confirm_rigid_task_success_contract`.
    """

    if not isinstance(value, Mapping):
        raise TaskNeutralScoringError(["rigid_task_success_contract_invalid"])
    try:
        contract = json.loads(json.dumps(dict(value), allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise TaskNeutralScoringError(
            ["rigid_task_success_contract_invalid"]
        ) from exc
    errors: list[str] = []
    _validate_contract_fields(
        contract,
        allowed={"schema_version", "scope", "provenance", "criteria", "contract_digest"},
        label="root",
        errors=errors,
    )
    if contract.get("schema_version") != RIGID_TASK_SUCCESS_CONTRACT_SCHEMA_VERSION:
        errors.append("rigid_task_success_contract_schema_invalid")

    scope = contract.get("scope")
    if not isinstance(scope, Mapping):
        errors.append("rigid_task_success_contract_scope_invalid")
        scope = {}
    else:
        _validate_contract_fields(
            scope,
            allowed={"site_id", "task_id"},
            label="scope",
            errors=errors,
        )
    site_id = str(scope.get("site_id") or "")
    task_id = str(scope.get("task_id") or "")
    if not site_id or not task_id:
        errors.append("rigid_task_success_contract_scope_invalid")
    if expected_site_id is not None and site_id != str(expected_site_id):
        errors.append("rigid_task_success_contract_site_binding_mismatch")
    if expected_task_id is not None and task_id != str(expected_task_id):
        errors.append("rigid_task_success_contract_task_binding_mismatch")

    provenance = contract.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append("rigid_task_success_contract_provenance_invalid")
        provenance = {}
    else:
        _validate_contract_fields(
            provenance,
            allowed={
                "author_source",
                "author_id",
                "confirmation_status",
                "confirmed_by_team_id",
                "proposal_digest",
            },
            label="provenance",
            errors=errors,
        )
    author_source = str(provenance.get("author_source") or "")
    author_id = str(provenance.get("author_id") or "")
    confirmation_status = str(provenance.get("confirmation_status") or "")
    confirmed_by = provenance.get("confirmed_by_team_id")
    proposal_digest = provenance.get("proposal_digest")
    if author_source not in {
        "compatibility_default",
        "site_robot_team",
        "task_owner",
        "agent_proposal",
    } or not author_id:
        errors.append("rigid_task_success_contract_author_invalid")
    if confirmation_status not in {"proposal_only", "confirmed"}:
        errors.append("rigid_task_success_contract_confirmation_invalid")
    elif confirmation_status == "proposal_only":
        if confirmed_by is not None or proposal_digest is not None:
            errors.append("rigid_task_success_contract_proposal_state_invalid")
        if require_confirmed:
            errors.append("rigid_task_success_contract_unconfirmed")
    else:
        if author_source == "compatibility_default":
            if confirmed_by is not None or proposal_digest is not None:
                errors.append("rigid_task_success_contract_default_provenance_invalid")
        elif not isinstance(confirmed_by, str) or not confirmed_by.strip():
            errors.append("rigid_task_success_contract_team_confirmation_missing")
        if author_source == "agent_proposal" and (
            not isinstance(proposal_digest, str)
            or not proposal_digest.startswith("sha256:")
            or len(proposal_digest) != 71
        ):
            errors.append("rigid_task_success_contract_agent_proposal_digest_missing")

    criteria = contract.get("criteria")
    if not isinstance(criteria, Mapping):
        errors.append("rigid_task_success_contract_criteria_invalid")
        criteria = {}
    else:
        _validate_contract_fields(
            criteria,
            allowed={
                "destination_containment",
                "orientation",
                "support",
                "terminal_task_contact",
                "gripper_state",
                "settling",
                "safety",
                "motion",
                "temporal_invariants",
            },
            label="criteria",
            errors=errors,
        )

    destination = criteria.get("destination_containment")
    if not isinstance(destination, Mapping):
        errors.append("rigid_task_success_contract_destination_invalid")
        destination = {}
    else:
        _validate_contract_fields(
            destination,
            allowed={"mode", "position_bounds_world_m"},
            label="destination",
            errors=errors,
        )
    if destination.get("mode") not in {"required", "ignored"}:
        errors.append("rigid_task_success_contract_destination_mode_invalid")
    bounds = destination.get("position_bounds_world_m")
    try:
        lower = _vector(bounds["minimum"], 3, error="destination")
        upper = _vector(bounds["maximum"], 3, error="destination")
        if any(low >= high for low, high in zip(lower, upper, strict=True)):
            raise ValueError
    except (KeyError, TypeError, ValueError, TaskNeutralScoringError):
        errors.append("rigid_task_success_contract_destination_invalid")

    orientation = criteria.get("orientation")
    if not isinstance(orientation, Mapping):
        errors.append("rigid_task_success_contract_orientation_invalid")
        orientation = {}
    else:
        _validate_contract_fields(
            orientation,
            allowed={"mode", "reference_xyzw", "tolerance_rad"},
            label="orientation",
            errors=errors,
        )
    if orientation.get("mode") not in {"required", "ignored"}:
        errors.append("rigid_task_success_contract_orientation_mode_invalid")
    try:
        reference = _vector(
            orientation.get("reference_xyzw"), 4, error="orientation"
        )
        tolerance = _finite(orientation.get("tolerance_rad"))
        if (
            abs(sum(item * item for item in reference) - 1.0) > 1.0e-6
            or tolerance is None
            or tolerance < 0.0
        ):
            raise ValueError
    except (ValueError, TaskNeutralScoringError):
        errors.append("rigid_task_success_contract_orientation_invalid")

    support = criteria.get("support")
    if not isinstance(support, Mapping):
        errors.append("rigid_task_success_contract_support_invalid")
        support = {}
    else:
        _validate_contract_fields(
            support,
            allowed={"height_mode", "height_interval_m", "contact_mode"},
            label="support",
            errors=errors,
        )
    if support.get("height_mode") not in {"required", "ignored"}:
        errors.append("rigid_task_success_contract_support_height_mode_invalid")
    if support.get("contact_mode") not in {"required", "ignored"}:
        errors.append("rigid_task_success_contract_support_contact_mode_invalid")
    try:
        support_interval = _vector(
            support.get("height_interval_m"), 2, error="support"
        )
        if support_interval[0] >= support_interval[1]:
            raise ValueError
    except (ValueError, TaskNeutralScoringError):
        errors.append("rigid_task_success_contract_support_invalid")

    task_contact = criteria.get("terminal_task_contact")
    if not isinstance(task_contact, Mapping) or set(task_contact) != {"mode"}:
        errors.append("rigid_task_success_contract_task_contact_invalid")
        task_contact = {}
    if task_contact.get("mode") not in {"cleared", "maintained", "ignored"}:
        errors.append("rigid_task_success_contract_task_contact_mode_invalid")

    gripper = criteria.get("gripper_state")
    if not isinstance(gripper, Mapping):
        errors.append("rigid_task_success_contract_gripper_invalid")
        gripper = {}
    else:
        _validate_contract_fields(
            gripper,
            allowed={"mode", "threshold_m"},
            label="gripper",
            errors=errors,
        )
    gripper_mode = gripper.get("mode")
    threshold = _finite(gripper.get("threshold_m"))
    if gripper_mode not in {"released", "closed_at_most", "ignored"}:
        errors.append("rigid_task_success_contract_gripper_mode_invalid")
    elif gripper_mode == "ignored":
        if gripper.get("threshold_m") is not None:
            errors.append("rigid_task_success_contract_gripper_threshold_invalid")
    elif threshold is None or threshold < 0.0:
        errors.append("rigid_task_success_contract_gripper_threshold_invalid")

    settling = criteria.get("settling")
    if not isinstance(settling, Mapping):
        errors.append("rigid_task_success_contract_settling_invalid")
        settling = {}
    else:
        _validate_contract_fields(
            settling,
            allowed={
                "mode",
                "window_samples",
                "position_tolerance_m",
                "orientation_tolerance_rad",
            },
            label="settling",
            errors=errors,
        )
    window = settling.get("window_samples")
    if settling.get("mode") not in {"required", "ignored"}:
        errors.append("rigid_task_success_contract_settling_mode_invalid")
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        errors.append("rigid_task_success_contract_settling_window_invalid")
    for field in ("position_tolerance_m", "orientation_tolerance_rad"):
        number = _finite(settling.get(field))
        if number is None or number < 0.0:
            errors.append(f"rigid_task_success_contract_settling_{field}_invalid")

    safety = criteria.get("safety")
    if not isinstance(safety, Mapping) or safety != {"mode": "required"}:
        errors.append("rigid_task_success_contract_safety_invalid")

    motion = criteria.get("motion")
    if not isinstance(motion, Mapping):
        errors.append("rigid_task_success_contract_motion_invalid")
        motion = {}
    else:
        _validate_contract_fields(
            motion,
            allowed={
                "movement_epsilon_m",
                "minimum_translation_m",
                "minimum_lift_m",
            },
            label="motion",
            errors=errors,
        )
    epsilon = _finite(motion.get("movement_epsilon_m"))
    if epsilon is None or epsilon <= 0.0:
        errors.append("rigid_task_success_contract_movement_epsilon_invalid")
    for field in ("minimum_translation_m", "minimum_lift_m"):
        raw = motion.get(field)
        if raw is not None and (_finite(raw) is None or float(raw) < 0.0):
            errors.append(f"rigid_task_success_contract_{field}_invalid")

    temporal = criteria.get("temporal_invariants")
    if not isinstance(temporal, Mapping):
        errors.append("rigid_task_success_contract_temporal_invariants_invalid")
        temporal = {}
    else:
        _validate_contract_fields(
            temporal,
            allowed={
                "schema_version",
                "no_drop",
                "maximum_task_contact_force_n",
                "forbidden_contact_classes",
                "containment_excursions",
                "workspace_excursions",
                "maximum_retries",
                "maximum_regrasps",
            },
            label="temporal_invariants",
            errors=errors,
        )
    if temporal.get("schema_version") != "rigid_task_event_ledger_expectation.v1":
        errors.append("rigid_task_success_contract_event_ledger_schema_invalid")
    no_drop = temporal.get("no_drop")
    if not isinstance(no_drop, Mapping) or set(no_drop) != {
        "mode",
        "minimum_fall_m",
    }:
        errors.append("rigid_task_success_contract_no_drop_invalid")
        no_drop = {}
    if no_drop.get("mode") not in {"required", "ignored"}:
        errors.append("rigid_task_success_contract_no_drop_mode_invalid")
    minimum_fall = _finite(no_drop.get("minimum_fall_m"))
    if minimum_fall is None or minimum_fall <= 0.0:
        errors.append("rigid_task_success_contract_no_drop_threshold_invalid")
    maximum_force = temporal.get("maximum_task_contact_force_n")
    if maximum_force is not None and (
        _finite(maximum_force) is None or float(maximum_force) <= 0.0
    ):
        errors.append("rigid_task_success_contract_maximum_force_invalid")
    forbidden_classes = temporal.get("forbidden_contact_classes")
    if (
        not isinstance(forbidden_classes, list)
        or any(not isinstance(item, str) or not item.strip() for item in forbidden_classes)
        or len(forbidden_classes) != len(set(forbidden_classes))
    ):
        errors.append("rigid_task_success_contract_forbidden_contacts_invalid")
    for field in ("containment_excursions", "workspace_excursions"):
        if temporal.get(field) not in {"forbidden", "ignored"}:
            errors.append(f"rigid_task_success_contract_{field}_invalid")
    for field in ("maximum_retries", "maximum_regrasps"):
        raw = temporal.get(field)
        if raw is not None and (
            isinstance(raw, bool) or not isinstance(raw, int) or raw < 0
        ):
            errors.append(f"rigid_task_success_contract_{field}_invalid")

    if contract.get("contract_digest") != cross_runtime_canonical_digest(
        contract, digest_field="contract_digest"
    ):
        errors.append("rigid_task_success_contract_digest_mismatch")
    if errors:
        raise TaskNeutralScoringError(errors)
    return contract


def seal_rigid_task_success_contract(
    *,
    task_spec: Mapping[str, Any],
    site_id: str,
    task_id: str,
    author_source: Literal[
        "compatibility_default", "site_robot_team", "task_owner", "agent_proposal"
    ],
    author_id: str,
    confirmation_status: Literal["proposal_only", "confirmed"],
    confirmed_by_team_id: str | None = None,
    criteria: Mapping[str, Any] | None = None,
) -> RigidTaskSuccessContract:
    """Seal human/team criteria or an explicitly proposal-only agent draft."""

    if author_source == "agent_proposal" and confirmation_status != "proposal_only":
        raise TaskNeutralScoringError(
            ["rigid_task_success_contract_agent_must_originate_proposal"]
        )
    document: dict[str, Any] = {
        "schema_version": RIGID_TASK_SUCCESS_CONTRACT_SCHEMA_VERSION,
        "scope": {"site_id": str(site_id), "task_id": str(task_id)},
        "provenance": {
            "author_source": author_source,
            "author_id": str(author_id),
            "confirmation_status": confirmation_status,
            "confirmed_by_team_id": confirmed_by_team_id,
            "proposal_digest": None,
        },
        "criteria": json.loads(
            json.dumps(
                dict(criteria)
                if criteria is not None
                else _compatibility_rigid_success_criteria(task_spec),
                allow_nan=False,
            )
        ),
        "contract_digest": "",
    }
    document["contract_digest"] = cross_runtime_canonical_digest(
        document, digest_field="contract_digest"
    )
    return validate_rigid_task_success_contract(
        document,
        require_confirmed=False,
        expected_site_id=site_id,
        expected_task_id=task_id,
    )


def confirm_rigid_task_success_contract(
    proposal: Mapping[str, Any], *, confirmed_by_team_id: str
) -> RigidTaskSuccessContract:
    """Create a confirmed immutable successor to a proposal-only document."""

    validated = validate_rigid_task_success_contract(
        proposal, require_confirmed=False
    )
    if validated["provenance"]["confirmation_status"] != "proposal_only":
        raise TaskNeutralScoringError(
            ["rigid_task_success_contract_not_a_proposal"]
        )
    confirmed = json.loads(json.dumps(validated))
    confirmed["provenance"]["confirmation_status"] = "confirmed"
    confirmed["provenance"]["confirmed_by_team_id"] = str(confirmed_by_team_id)
    if confirmed["provenance"]["author_source"] == "agent_proposal":
        confirmed["provenance"]["proposal_digest"] = validated["contract_digest"]
    confirmed["contract_digest"] = cross_runtime_canonical_digest(
        confirmed, digest_field="contract_digest"
    )
    return validate_rigid_task_success_contract(confirmed)


def confirmed_rigid_task_success_contract_matches_published(
    *, published: Mapping[str, Any], selected: Mapping[str, Any]
) -> bool:
    """Match an exact published contract or its one team-confirmed successor."""

    try:
        public_contract = validate_rigid_task_success_contract(
            published, require_confirmed=False
        )
        confirmed_contract = validate_rigid_task_success_contract(selected)
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
        expected = confirm_rigid_task_success_contract(
            public_contract, confirmed_by_team_id=team_id
        )
    except TaskNeutralScoringError:
        return False
    return expected["contract_digest"] == confirmed_contract["contract_digest"]


def _default_rigid_task_success_contract(
    task_spec: Mapping[str, Any],
) -> RigidTaskSuccessContract:
    subject = str(task_spec.get("subject_asset_id") or "legacy_rigid_task")
    return seal_rigid_task_success_contract(
        task_spec=task_spec,
        site_id=str(task_spec.get("site_id") or "compatibility_unspecified_site"),
        task_id=str(task_spec.get("task_id") or subject),
        author_source="compatibility_default",
        author_id="blueprint:manipulation_strategy_defaults.v1",
        confirmation_status="confirmed",
    )


def _normalize_legacy_articulated_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if spec.get("schema_version") != TASK_SPEC_SCHEMA_VERSION:
        errors.append("task_spec_schema_invalid")
    if spec.get("task_kind") != TASK_KIND_ARTICULATED_OPEN_CLOSE:
        errors.append("articulated_task_kind_invalid")
    target = str(spec.get("target_joint_id") or "")
    if not target:
        errors.append("articulated_target_joint_missing")
    joint_resets = spec.get("joint_reset_positions_rad")
    if not isinstance(joint_resets, Mapping) or target not in joint_resets:
        errors.append("articulated_joint_resets_invalid")
        joint_resets = {}
    normalized_resets: dict[str, float] = {}
    for joint_id, raw in joint_resets.items():
        value = _finite(raw)
        if not str(joint_id) or value is None:
            errors.append("articulated_joint_resets_invalid")
        else:
            normalized_resets[str(joint_id)] = value
    if not normalized_resets:
        errors.append("articulated_joint_resets_invalid")
    interval = spec.get("target_success_interval_rad")
    if (
        not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) != 2
        or _finite(interval[0]) is None
        or _finite(interval[1]) is None
        or float(interval[0]) >= float(interval[1])
    ):
        errors.append("articulated_success_interval_invalid")
        normalized_interval = [0.0, 0.0]
    else:
        normalized_interval = [float(interval[0]), float(interval[1])]
    hard_limits = spec.get("joint_hard_limits_rad")
    normalized_limits: dict[str, list[float]] = {}
    if not isinstance(hard_limits, Mapping) or set(hard_limits) != set(normalized_resets):
        errors.append("articulated_joint_limits_invalid")
    else:
        for joint_id, raw in hard_limits.items():
            if (
                not isinstance(raw, Sequence)
                or isinstance(raw, (str, bytes))
                or len(raw) != 2
                or _finite(raw[0]) is None
                or _finite(raw[1]) is None
                or float(raw[0]) >= float(raw[1])
            ):
                errors.append("articulated_joint_limits_invalid")
            else:
                normalized_limits[str(joint_id)] = [float(raw[0]), float(raw[1])]
    fields = {
        "settle_window_samples": spec.get("settle_window_samples"),
        "maximum_settled_target_speed_rad_s": spec.get(
            "maximum_settled_target_speed_rad_s"
        ),
        "non_task_joint_motion_tolerance_rad": spec.get(
            "non_task_joint_motion_tolerance_rad"
        ),
        "movement_epsilon_rad": spec.get("movement_epsilon_rad"),
        "reset_tolerance_rad": spec.get("reset_tolerance_rad"),
    }
    normalized_fields: dict[str, float | int] = {}
    for field, raw in fields.items():
        value = _finite(raw)
        if value is None or value <= 0:
            errors.append(f"articulated_{field}_invalid")
        elif field == "settle_window_samples" and (
            isinstance(raw, bool) or not isinstance(raw, int)
        ):
            errors.append("articulated_settle_window_samples_invalid")
        else:
            normalized_fields[field] = int(raw) if field == "settle_window_samples" else value
    if target in normalized_resets and normalized_interval[0] <= normalized_resets[target] <= normalized_interval[1]:
        errors.append("articulated_reset_inside_success_interval")
    if errors:
        raise TaskNeutralScoringError(errors)
    return {
        "schema_version": TASK_SPEC_SCHEMA_VERSION,
        "target_joint_id": target,
        "target_joint_ids": [target],
        "joint_reset_positions_rad": normalized_resets,
        "joint_reset_positions": normalized_resets,
        "target_success_interval_rad": normalized_interval,
        "target_success_intervals": {target: normalized_interval},
        "joint_hard_limits_rad": normalized_limits,
        "joint_hard_limits": normalized_limits,
        "joint_roles": {
            joint_id: "target" if joint_id == target else "locked"
            for joint_id in normalized_resets
        },
        "dependent_joints": {},
        **normalized_fields,
    }


def _normalize_graph_articulated_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if spec.get("schema_version") != TASK_SPEC_GRAPH_SCHEMA_VERSION:
        errors.append("task_spec_schema_invalid")
    if spec.get("task_kind") != TASK_KIND_ARTICULATED_OPEN_CLOSE:
        errors.append("articulated_task_kind_invalid")
    graph = spec.get("articulation_graph")
    if not isinstance(graph, Mapping):
        errors.append("articulated_graph_missing")
        normalized_graph: dict[str, Any] = {}
    else:
        try:
            normalized_graph = validate_articulation_graph(graph)
        except ArticulationGraphContractError as exc:
            errors.extend(exc.errors)
            normalized_graph = {}
    fields = {
        "settle_window_samples": spec.get("settle_window_samples"),
        "maximum_settled_target_speed": spec.get(
            "maximum_settled_target_speed"
        ),
        "locked_joint_motion_tolerance": spec.get(
            "locked_joint_motion_tolerance"
        ),
        "movement_epsilon": spec.get("movement_epsilon"),
    }
    normalized_fields: dict[str, float | int] = {}
    for field, raw in fields.items():
        value = _finite(raw)
        if value is None or value <= 0:
            errors.append(f"articulated_{field}_invalid")
        elif field == "settle_window_samples" and (
            isinstance(raw, bool) or not isinstance(raw, int)
        ):
            errors.append("articulated_settle_window_samples_invalid")
        else:
            normalized_fields[field] = (
                int(raw) if field == "settle_window_samples" else value
            )
    joints = normalized_graph.get("joints") or []
    resets = {str(row["joint_id"]): float(row["reset_position"]) for row in joints}
    limits = {str(row["joint_id"]): list(row["limits"]) for row in joints}
    roles = {str(row["joint_id"]): str(row["role"]) for row in joints}
    reset_tolerances = {
        str(row["joint_id"]): float(row["reset_tolerance"]) for row in joints
    }
    dependent = {
        str(row["joint_id"]): dict(row["dependency"])
        for row in joints
        if row["role"] == "dependent"
    }
    success = (
        normalized_graph.get("success_predicate", {}).get("joint_intervals") or {}
    )
    targets = sorted(success)
    if errors:
        raise TaskNeutralScoringError(errors)
    return {
        "schema_version": TASK_SPEC_GRAPH_SCHEMA_VERSION,
        "articulation_graph": normalized_graph,
        "target_joint_id": targets[0],
        "target_joint_ids": targets,
        "joint_reset_positions": resets,
        "joint_reset_positions_rad": resets,
        "joint_reset_tolerances": reset_tolerances,
        "reset_tolerance_rad": max(reset_tolerances.values()),
        "target_success_intervals": {
            str(joint_id): list(interval) for joint_id, interval in success.items()
        },
        "target_success_interval_rad": list(success[targets[0]]),
        "joint_hard_limits": limits,
        "joint_hard_limits_rad": limits,
        "joint_roles": roles,
        "dependent_joints": dependent,
        "maximum_settled_target_speed_rad_s": normalized_fields[
            "maximum_settled_target_speed"
        ],
        "non_task_joint_motion_tolerance_rad": normalized_fields[
            "locked_joint_motion_tolerance"
        ],
        "movement_epsilon_rad": normalized_fields["movement_epsilon"],
        **normalized_fields,
    }


def _normalize_articulated_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    if spec.get("schema_version") == TASK_SPEC_GRAPH_SCHEMA_VERSION:
        return _normalize_graph_articulated_spec(spec)
    return _normalize_legacy_articulated_spec(spec)


def validate_articulated_task_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Public fail-closed validator for a frozen articulated scorer contract."""

    return _normalize_articulated_spec(spec)


def _normalize_articulated_samples(
    samples: Sequence[Mapping[str, Any]], *, joint_ids: set[str], generic_units: bool = False
) -> list[dict[str, Any]]:
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence) or not samples:
        raise TaskNeutralScoringError(["articulated_samples_invalid"])
    errors: list[str] = []
    normalized: list[dict[str, Any]] = []
    previous_step: int | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            errors.append(f"articulated_sample_{index}_not_mapping")
            continue
        raw_step = sample.get("step_index")
        if isinstance(raw_step, bool) or not isinstance(raw_step, int):
            errors.append(f"articulated_sample_{index}_step_invalid")
            step = index
        else:
            step = raw_step
            if previous_step is not None and step <= previous_step:
                errors.append(f"articulated_sample_{index}_step_not_increasing")
            previous_step = step
        position_field = "joint_positions" if generic_units else "joint_positions_rad"
        velocity_field = (
            "joint_velocities_per_s" if generic_units else "joint_velocities_rad_s"
        )
        positions = sample.get(position_field)
        velocities = sample.get(velocity_field)
        if not isinstance(positions, Mapping) or set(positions) != joint_ids:
            errors.append(f"articulated_sample_{index}_joint_positions_invalid")
            continue
        if not isinstance(velocities, Mapping) or set(velocities) != joint_ids:
            errors.append(f"articulated_sample_{index}_joint_velocities_invalid")
            continue
        normalized_positions: dict[str, float] = {}
        normalized_velocities: dict[str, float] = {}
        for joint_id in sorted(joint_ids):
            position = _finite(positions[joint_id])
            velocity = _finite(velocities[joint_id])
            if position is None:
                errors.append(f"articulated_sample_{index}_position_nonfinite:{joint_id}")
            else:
                normalized_positions[joint_id] = position
            if velocity is None:
                errors.append(f"articulated_sample_{index}_velocity_nonfinite:{joint_id}")
            else:
                normalized_velocities[joint_id] = velocity
        boolean_fields = (
            "task_contact_active",
            "joint_limit_violation",
            "containment_violation",
            "robot_collision_failure",
            "scene_collision_failure",
            "retreat_completed",
        )
        booleans: dict[str, bool] = {}
        for field in boolean_fields:
            raw = sample.get(field)
            if not isinstance(raw, bool):
                errors.append(f"articulated_sample_{index}_{field}_invalid")
            else:
                booleans[field] = raw
        normalized.append(
            {
                "step_index": step,
                "joint_positions_rad": normalized_positions,
                "joint_velocities_rad_s": normalized_velocities,
                **booleans,
            }
        )
    if errors:
        raise TaskNeutralScoringError(errors)
    return normalized


def score_articulated_task_episode(
    *, task_spec: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score an open/close episode only from native deterministic state."""

    spec = _normalize_articulated_spec(task_spec)
    resets = spec["joint_reset_positions_rad"]
    target = spec["target_joint_id"]
    targets = list(spec["target_joint_ids"])
    normalized = _normalize_articulated_samples(
        samples,
        joint_ids=set(resets),
        generic_units=spec["schema_version"] == TASK_SPEC_GRAPH_SCHEMA_VERSION,
    )
    reset_tolerance = float(spec["reset_tolerance_rad"])
    reset_tolerances = spec.get("joint_reset_tolerances") or {
        joint_id: reset_tolerance for joint_id in resets
    }
    reset_errors = {
        joint_id: abs(normalized[0]["joint_positions_rad"][joint_id] - reset)
        for joint_id, reset in resets.items()
    }
    if any(
        value > float(reset_tolerances[joint_id])
        for joint_id, value in reset_errors.items()
    ):
        raise TaskNeutralScoringError(["articulated_episode_reset_readback_mismatch"])

    success_intervals = spec["target_success_intervals"]
    lower, upper = success_intervals[target]
    target_positions_by_joint = {
        joint_id: [
            sample["joint_positions_rad"][joint_id] for sample in normalized
        ]
        for joint_id in targets
    }
    target_positions = target_positions_by_joint[target]
    target_displacements_by_joint = {
        joint_id: [
            abs(value - resets[joint_id])
            for value in target_positions_by_joint[joint_id]
        ]
        for joint_id in targets
    }
    maximum_displacement_by_joint = {
        joint_id: max(values)
        for joint_id, values in target_displacements_by_joint.items()
    }
    maximum_displacement = max(maximum_displacement_by_joint.values())
    reached_success_interval = any(
        all(
            success_intervals[joint_id][0]
            <= sample["joint_positions_rad"][joint_id]
            <= success_intervals[joint_id][1]
            for joint_id in targets
        )
        for sample in normalized
    )
    window_count = int(spec["settle_window_samples"])
    settle_available = len(normalized) >= window_count
    settle = normalized[-window_count:] if settle_available else normalized
    settle_target_positions = [sample["joint_positions_rad"][target] for sample in settle]
    settle_target_velocities = [sample["joint_velocities_rad_s"][target] for sample in settle]
    settle_in_interval = settle_available and all(
        all(
            success_intervals[joint_id][0]
            <= sample["joint_positions_rad"][joint_id]
            <= success_intervals[joint_id][1]
            for joint_id in targets
        )
        for sample in settle
    )
    settle_speed_ok = settle_available and all(
        abs(sample["joint_velocities_rad_s"][joint_id])
        <= float(spec["maximum_settled_target_speed_rad_s"])
        for sample in settle
        for joint_id in targets
    )
    roles = spec["joint_roles"]
    locked_joint_ids = sorted(
        joint_id for joint_id, role in roles.items() if role == "locked"
    )
    locked_max_delta = {
        joint_id: max(
            abs(sample["joint_positions_rad"][joint_id] - resets[joint_id])
            for sample in normalized
        )
        for joint_id in locked_joint_ids
    }
    locked_joints_stable = all(
        value <= float(spec["non_task_joint_motion_tolerance_rad"])
        for value in locked_max_delta.values()
    )
    dependent_max_error: dict[str, float] = {}
    for joint_id, dependency in spec["dependent_joints"].items():
        driver = dependency["driver_joint_id"]
        multiplier = float(dependency["multiplier"])
        offset = float(dependency["offset"])
        dependent_max_error[joint_id] = max(
            abs(
                sample["joint_positions_rad"][joint_id]
                - (
                    multiplier * sample["joint_positions_rad"][driver]
                    + offset
                )
            )
            for sample in normalized
        )
    dependent_joints_consistent = all(
        dependent_max_error[joint_id]
        <= float(spec["dependent_joints"][joint_id]["tolerance"])
        for joint_id in dependent_max_error
    )
    non_task_locked = locked_joints_stable and dependent_joints_consistent
    # A joint resting against its own hard stop sits a solver residual past
    # it, and this recomputation compared that residual with exact arithmetic.
    # C29 measured the consequence: the washer door, closed and merely nudged,
    # read -5.7e-8 rad, so `joint_hard_limits_respected` went false and the
    # positive control failed on 57 nanoradians -- about 34 nanometres at the
    # handle.  That would have failed a run whose grasp succeeded.  The
    # simulator's own per-sample `joint_limit_violation` flag reported no
    # violation in the same trace, which is the tell: the native readback
    # knows the articulation's real limits, and only this redundant check
    # disagreed.  So the flag stays authoritative and unchanged, and the
    # recomputation admits a declared solver residual -- still four orders of
    # magnitude below any mechanically meaningful excursion.  The worst
    # observed excursion is sealed either way, so a real violation creeping
    # up on the tolerance stays visible rather than silently absorbed.
    hard_limit_excursion_rad = 0.0
    for sample in normalized:
        for joint_id, position in sample["joint_positions_rad"].items():
            lower, upper = spec["joint_hard_limits_rad"][joint_id]
            hard_limit_excursion_rad = max(
                hard_limit_excursion_rad,
                float(lower) - float(position),
                float(position) - float(upper),
            )
    hard_limit_violation = any(
        sample["joint_limit_violation"] for sample in normalized
    ) or hard_limit_excursion_rad > JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD
    containment_violation = any(sample["containment_violation"] for sample in normalized)
    collision_failure = any(
        sample["robot_collision_failure"] or sample["scene_collision_failure"]
        for sample in normalized
    )
    released_in_settle = settle_available and all(
        not sample["task_contact_active"] for sample in settle
    )
    retreat_completed = bool(normalized[-1]["retreat_completed"])
    task_succeeded = bool(
        settle_in_interval
        and settle_speed_ok
        and non_task_locked
        and not hard_limit_violation
        and not containment_violation
        and not collision_failure
        and released_in_settle
        and retreat_completed
    )
    if task_succeeded:
        outcome = OUTCOME_OPENED_AND_SETTLED
    elif hard_limit_violation or containment_violation:
        outcome = OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION
    elif collision_failure:
        outcome = OUTCOME_COLLISION_FAILURE
    elif not non_task_locked:
        outcome = OUTCOME_NON_TASK_JOINT_MOVED
    elif settle_in_interval and settle_speed_ok and (
        not released_in_settle or not retreat_completed
    ):
        # Reaching and stably holding the requested angle is materially farther
        # than a rebound, but the task contract still requires release and
        # retreat.  Keep that failure rung distinct so neither a policy nor a
        # human reviewer can promote an assisted/unfinished open to success.
        outcome = OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE
    elif reached_success_interval:
        outcome = OUTCOME_OPENED_THEN_REBOUNDED
    elif maximum_displacement > float(spec["movement_epsilon_rad"]):
        outcome = OUTCOME_MOVED_BELOW_THRESHOLD
    else:
        outcome = OUTCOME_NEVER_MOVED

    report: dict[str, Any] = {
        "schema_version": ARTICULATED_REPORT_SCHEMA_VERSION,
        "status": "scored" if settle_available else "undetermined",
        "task_kind": TASK_KIND_ARTICULATED_OPEN_CLOSE,
        "task_succeeded": task_succeeded,
        "outcome": outcome,
        "outcome_rank": _ARTICULATED_OUTCOME_RANK[outcome],
        "measurements": {
            "sample_count": len(normalized),
            "first_step_index": normalized[0]["step_index"],
            "final_step_index": normalized[-1]["step_index"],
            "reset_readback_error_rad": reset_errors,
            "target_start_position_rad": target_positions[0],
            "target_final_position_rad": target_positions[-1],
            "target_maximum_displacement_rad": maximum_displacement,
            "target_positions_by_joint": {
                joint_id: {
                    "start": values[0],
                    "final": values[-1],
                    "maximum_displacement": maximum_displacement_by_joint[joint_id],
                }
                for joint_id, values in target_positions_by_joint.items()
            },
            "target_reached_success_interval": reached_success_interval,
            "settle_window_available": settle_available,
            "settle_target_min_position_rad": min(settle_target_positions),
            "settle_target_max_position_rad": max(settle_target_positions),
            "settle_target_max_abs_velocity_rad_s": max(
                abs(value) for value in settle_target_velocities
            ),
            "non_target_max_delta_rad": locked_max_delta,
            "locked_joint_max_delta": locked_max_delta,
            "dependent_joint_max_error": dependent_max_error,
            "joint_hard_limit_max_excursion_rad": hard_limit_excursion_rad,
            "joint_hard_limit_solver_residual_rad": (
                JOINT_HARD_LIMIT_SOLVER_RESIDUAL_RAD
            ),
            "released_in_settle": released_in_settle,
            "retreat_completed": retreat_completed,
        },
        "predicates": {
            "settle_in_success_interval": settle_in_interval,
            "settle_speed_within_limit": settle_speed_ok,
            "non_task_joints_locked": non_task_locked,
            "locked_joints_stable": locked_joints_stable,
            "dependent_joints_consistent": dependent_joints_consistent,
            "joint_hard_limits_respected": not hard_limit_violation,
            "containment_respected": not containment_violation,
            "collision_failure_absent": not collision_failure,
            "task_contact_released": released_in_settle,
            "retreat_completed": retreat_completed,
        },
        "thresholds": {
            "target_success_interval_rad": [lower, upper],
            "target_success_intervals": success_intervals,
            "maximum_settled_target_speed_rad_s": spec[
                "maximum_settled_target_speed_rad_s"
            ],
            "non_task_joint_motion_tolerance_rad": spec[
                "non_task_joint_motion_tolerance_rad"
            ],
            "movement_epsilon_rad": spec["movement_epsilon_rad"],
            "reset_tolerance_rad": reset_tolerance,
            "settle_window_samples": window_count,
            "joint_hard_limits_rad": spec["joint_hard_limits_rad"],
        },
        "judgement_source": "deterministic_native_simulator_joint_state",
        "rendered_image_consulted": False,
        "learned_judge_consulted": False,
        "candidate_policy_queried_by_scorer": False,
        "caller_asserted_success_accepted": False,
        "report_digest": "",
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


def _vector(value: Any, length: int, *, error: str) -> list[float]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        raise TaskNeutralScoringError([error])
    result = [_finite(item) for item in value]
    if any(item is None for item in result):
        raise TaskNeutralScoringError([error])
    return [float(item) for item in result]


def _quaternion_angle(a: Sequence[float], b: Sequence[float]) -> float:
    qa = _vector(a, 4, error="rigid_task_quaternion_invalid")
    qb = _vector(b, 4, error="rigid_task_quaternion_invalid")
    na = math.sqrt(sum(item * item for item in qa))
    nb = math.sqrt(sum(item * item for item in qb))
    if min(na, nb) <= 0.0:
        raise TaskNeutralScoringError(["rigid_task_quaternion_invalid"])
    dot = abs(sum(x * y for x, y in zip(qa, qb, strict=True)) / (na * nb))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def _rotate_xyzw(
    vector: Sequence[float], quaternion: Sequence[float]
) -> list[float]:
    x, y, z, w = quaternion
    qx, qy, qz, qw = (x, y, z, w)
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + (qy * tz - qz * ty),
        vy + qw * ty + (qz * tx - qx * tz),
        vz + qw * tz + (qx * ty - qy * tx),
    ]


def _rotate_inverse_xyzw(
    vector: Sequence[float], quaternion: Sequence[float]
) -> list[float]:
    return _rotate_xyzw(
        vector,
        (-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]),
    )


def _normalize_rigid_task_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if (
        spec.get("schema_version") != TASK_SPEC_GRAPH_SCHEMA_VERSION
        or spec.get("task_kind") != TASK_KIND_RIGID_PICK_PLACE
    ):
        errors.append("rigid_task_spec_schema_invalid")
    subject = str(spec.get("subject_asset_id") or "")
    if not subject:
        errors.append("rigid_task_subject_asset_id_missing")
    manipulation_strategy = str(
        spec.get("manipulation_strategy") or "pick_and_place"
    )
    if manipulation_strategy not in RIGID_MANIPULATION_STRATEGIES:
        errors.append("rigid_task_manipulation_strategy_invalid")
    try:
        start_pose = _vector(
            spec.get("start_pose_world"), 7, error="rigid_task_start_pose_invalid"
        )
        orientation_reference = _vector(
            spec.get("destination_orientation_xyzw"),
            4,
            error="rigid_task_destination_orientation_invalid",
        )
        raw_bounds = spec["destination_position_bounds_world_m"]
        lower = _vector(raw_bounds["minimum"], 3, error="rigid_task_destination_invalid")
        upper = _vector(raw_bounds["maximum"], 3, error="rigid_task_destination_invalid")
        support_interval = _vector(
            spec["support_height_interval_m"],
            2,
            error="rigid_task_support_interval_invalid",
        )
    except (KeyError, TypeError, TaskNeutralScoringError) as exc:
        if isinstance(exc, TaskNeutralScoringError):
            errors.extend(exc.errors)
        else:
            errors.append("rigid_task_spec_invalid")
        start_pose = [0.0] * 7
        orientation_reference = [0.0, 0.0, 0.0, 1.0]
        lower = [0.0] * 3
        upper = [0.0] * 3
        support_interval = [0.0, 0.0]
    numeric_fields = (
        "destination_orientation_tolerance_rad",
        "minimum_translation_m",
        "minimum_lift_m",
        "movement_epsilon_m",
        "reset_translation_tolerance_m",
        "reset_orientation_tolerance_rad",
        "settle_position_tolerance_m",
        "settle_orientation_tolerance_rad",
        "release_gripper_width_min_m",
        "task_contact_minimum_force_n",
    )
    numbers: dict[str, float] = {}
    for field in numeric_fields:
        value = _finite(spec.get(field))
        if value is None or value < 0.0 or (
            value == 0.0 and field not in {"minimum_lift_m"}
        ):
            errors.append(f"rigid_task_{field}_invalid")
        else:
            numbers[field] = value
    settle = spec.get("settle_window_samples")
    if isinstance(settle, bool) or not isinstance(settle, int) or settle <= 0:
        errors.append("rigid_task_settle_window_samples_invalid")
    if spec.get("release_required") is not True:
        errors.append("rigid_task_release_contract_invalid")
    if any(low >= high for low, high in zip(lower, upper, strict=True)):
        errors.append("rigid_task_destination_invalid")
    if support_interval[0] >= support_interval[1]:
        errors.append("rigid_task_support_interval_invalid")
    for quaternion, error in (
        (start_pose[3:], "rigid_task_start_pose_invalid"),
        (orientation_reference, "rigid_task_destination_orientation_invalid"),
    ):
        if abs(sum(item * item for item in quaternion) - 1.0) > 1.0e-6:
            errors.append(error)
    if errors:
        raise TaskNeutralScoringError(errors)
    normalized: dict[str, Any] = {
        "subject_asset_id": subject,
        "manipulation_strategy": manipulation_strategy,
        "start_pose_world": start_pose,
        "destination_position_bounds_world_m": {"minimum": lower, "maximum": upper},
        "destination_orientation_xyzw": orientation_reference,
        "support_height_interval_m": support_interval,
        "settle_window_samples": settle,
        "release_required": True,
        **numbers,
    }
    destination_relation = spec.get("destination_relation")
    if destination_relation is not None:
        if destination_relation not in {"inside", "on"}:
            raise TaskNeutralScoringError(
                ["rigid_task_destination_relation_invalid"]
            )
        destination_pose = _vector(
            spec.get("destination_pose_world"),
            7,
            error="rigid_task_destination_pose_invalid",
        )
        local_bounds = spec.get(
            "destination_position_bounds_destination_frame_m"
        )
        if not isinstance(local_bounds, Mapping):
            raise TaskNeutralScoringError(
                ["rigid_task_destination_local_bounds_invalid"]
            )
        local_lower = _vector(
            local_bounds.get("minimum"),
            3,
            error="rigid_task_destination_local_bounds_invalid",
        )
        local_upper = _vector(
            local_bounds.get("maximum"),
            3,
            error="rigid_task_destination_local_bounds_invalid",
        )
        subject_bounds = spec.get("subject_collision_bounds_scoring_frame_m")
        interior_bounds = spec.get("destination_interior_bounds_body_frame_m")
        if not isinstance(subject_bounds, Mapping) or not isinstance(
            interior_bounds, Mapping
        ):
            raise TaskNeutralScoringError(
                ["rigid_task_destination_collision_geometry_invalid"]
            )
        subject_lower = _vector(
            subject_bounds.get("minimum"),
            3,
            error="rigid_task_destination_collision_geometry_invalid",
        )
        subject_upper = _vector(
            subject_bounds.get("maximum"),
            3,
            error="rigid_task_destination_collision_geometry_invalid",
        )
        interior_lower = _vector(
            interior_bounds.get("minimum"),
            3,
            error="rigid_task_destination_collision_geometry_invalid",
        )
        interior_upper = _vector(
            interior_bounds.get("maximum"),
            3,
            error="rigid_task_destination_collision_geometry_invalid",
        )
        translation_tolerance = _finite(
            spec.get("destination_reset_translation_tolerance_m")
        )
        rotation_tolerance = _finite(
            spec.get("destination_reset_rotation_tolerance_rad")
        )
        if (
            abs(sum(item * item for item in destination_pose[3:]) - 1.0)
            > 1.0e-6
            or any(
                low >= high
                for low, high in zip(local_lower, local_upper, strict=True)
            )
            or any(
                low >= high
                for low, high in zip(subject_lower, subject_upper, strict=True)
            )
            or any(
                low >= high
                for low, high in zip(interior_lower, interior_upper, strict=True)
            )
            or translation_tolerance is None
            or translation_tolerance <= 0.0
            or rotation_tolerance is None
            or rotation_tolerance <= 0.0
        ):
            raise TaskNeutralScoringError(
                ["rigid_task_destination_local_contract_invalid"]
            )
        normalized.update(
            destination_relation=destination_relation,
            destination_pose_world=destination_pose,
            destination_position_bounds_destination_frame_m={
                "minimum": local_lower,
                "maximum": local_upper,
            },
            subject_collision_bounds_scoring_frame_m={
                "minimum": subject_lower,
                "maximum": subject_upper,
            },
            destination_interior_bounds_body_frame_m={
                "minimum": interior_lower,
                "maximum": interior_upper,
            },
            destination_reset_translation_tolerance_m=translation_tolerance,
            destination_reset_rotation_tolerance_rad=rotation_tolerance,
        )
    control_frequency_hz = _finite(spec.get("control_frequency_hz"))
    normalized["control_frequency_hz"] = (
        control_frequency_hz
        if control_frequency_hz is not None and control_frequency_hz > 0.0
        else None
    )
    raw_success_contract = spec.get("task_success_contract")
    if raw_success_contract is None:
        normalized["task_success_contract"] = _default_rigid_task_success_contract(
            normalized
        )
    else:
        validated_success_contract = validate_rigid_task_success_contract(
            raw_success_contract,
            expected_site_id=(
                str(spec["site_id"]) if spec.get("site_id") is not None else None
            ),
            expected_task_id=(
                str(spec["task_id"]) if spec.get("task_id") is not None else None
            ),
        )
        if (
            validated_success_contract["provenance"]["author_source"]
            == "compatibility_default"
            and validated_success_contract["criteria"]
            != _compatibility_rigid_success_criteria(normalized)
        ):
            raise TaskNeutralScoringError(
                ["rigid_task_success_contract_default_criteria_mismatch"]
            )
        normalized["task_success_contract"] = validated_success_contract
    return normalized


def score_rigid_task_episode(
    *, task_spec: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Score a scene-neutral rigid relocation from deterministic native state."""

    spec = _normalize_rigid_task_spec(task_spec)
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)) or not samples:
        raise TaskNeutralScoringError(["rigid_task_samples_invalid"])
    normalized: list[dict[str, Any]] = []
    previous_step: int | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TaskNeutralScoringError([f"rigid_task_sample_invalid:{index}"])
        step = sample.get("step_index")
        if (
            isinstance(step, bool)
            or not isinstance(step, int)
            or (previous_step is not None and step <= previous_step)
        ):
            raise TaskNeutralScoringError([f"rigid_task_sample_step_invalid:{index}"])
        previous_step = step
        pose = _vector(
            sample.get("task_object_pose_world"),
            7,
            error=f"rigid_task_sample_pose_invalid:{index}",
        )
        if abs(sum(item * item for item in pose[3:]) - 1.0) > 1.0e-3:
            raise TaskNeutralScoringError([f"rigid_task_sample_pose_invalid:{index}"])
        destination_pose = None
        if spec.get("destination_relation") is not None:
            raw_destination_pose = sample.get("destination_pose_world")
            if raw_destination_pose is not None:
                destination_pose = _vector(
                    raw_destination_pose,
                    7,
                    error=f"rigid_task_sample_destination_pose_invalid:{index}",
                )
                if (
                    abs(sum(item * item for item in destination_pose[3:]) - 1.0)
                    > 1.0e-3
                ):
                    raise TaskNeutralScoringError(
                        [f"rigid_task_sample_destination_pose_invalid:{index}"]
                    )
        native_readback = sample.get("native_readback")
        if not isinstance(native_readback, Mapping):
            native_readback = {}
        task_contact_force_n = _finite(
            sample.get("task_robot_contact_peak_force_n")
        )
        task_contact_force_source = (
            "task_robot_contact_peak_force_n"
            if task_contact_force_n is not None
            else None
        )
        if task_contact_force_n is None:
            task_contact_force_n = _finite(sample.get("task_contact_force_n"))
            if task_contact_force_n is not None:
                task_contact_force_source = "task_contact_force_n"
        if task_contact_force_n is None:
            task_contact_force_n = _finite(
                native_readback.get("task_robot_contact_peak_force_n")
            )
            if task_contact_force_n is not None:
                task_contact_force_source = (
                    "native_readback.task_robot_contact_peak_force_n"
                )
        if task_contact_force_n is not None and task_contact_force_n < 0.0:
            task_contact_force_n = None
            task_contact_force_source = None
        normalized.append(
            {
                "step_index": step,
                "pose": pose,
                "destination_pose": destination_pose,
                "gripper_width_m": _finite(sample.get("gripper_width_m")),
                "task_contact_active": sample.get("task_contact_active"),
                "support_contact_active": sample.get("support_contact_active"),
                "robot_collision_failure": sample.get("robot_collision_failure"),
                "scene_collision_failure": sample.get("scene_collision_failure"),
                "containment_violation": sample.get("containment_violation"),
                "forbidden_robot_task_collision_failure": sample.get(
                    "forbidden_robot_task_collision_failure"
                ),
                "locked_joint_containment_violation": sample.get(
                    "locked_joint_containment_violation"
                ),
                "robot_task_forbidden_collision_peak_force_n": _finite(
                    sample.get("robot_task_forbidden_collision_peak_force_n")
                    if sample.get("robot_task_forbidden_collision_peak_force_n")
                    is not None
                    else native_readback.get(
                        "robot_task_forbidden_collision_peak_force_n"
                    )
                ),
                "robot_scene_contact_peak_force_n": _finite(
                    sample.get("robot_scene_contact_peak_force_n")
                    if sample.get("robot_scene_contact_peak_force_n") is not None
                    else native_readback.get("robot_scene_contact_peak_force_n")
                ),
                "task_scene_collision_peak_force_n": _finite(
                    sample.get("task_scene_collision_peak_force_n")
                    if sample.get("task_scene_collision_peak_force_n") is not None
                    else native_readback.get("task_scene_collision_peak_force_n")
                ),
                "collision_failure_minimum_force_n": _finite(
                    sample.get("collision_failure_minimum_force_n")
                ),
                "robot_task_forbidden_contact_pairs": sample.get(
                    "robot_task_forbidden_contact_pairs"
                ),
                "workspace_excursion": sample.get("workspace_excursion"),
                "task_contact_force_n": task_contact_force_n,
                "task_contact_force_source": task_contact_force_source,
                "contact_classes_active": sample.get("contact_classes_active"),
                "retry_count": sample.get("retry_count"),
                "regrasp_count": sample.get("regrasp_count"),
            }
        )

    start = normalized[0]["pose"]
    reset_translation_error = math.dist(start[:3], spec["start_pose_world"][:3])
    reset_orientation_error = _quaternion_angle(
        start[3:], spec["start_pose_world"][3:]
    )
    if (
        reset_translation_error > spec["reset_translation_tolerance_m"]
        or reset_orientation_error > spec["reset_orientation_tolerance_rad"]
    ):
        raise TaskNeutralScoringError(["rigid_task_reset_readback_mismatch"])
    positions = [row["pose"][:3] for row in normalized]
    translation = [math.dist(position[:2], start[:2]) for position in positions]
    lift = [position[2] - start[2] for position in positions]
    success_contract = spec["task_success_contract"]
    criteria = success_contract["criteria"]
    settling_criteria = criteria["settling"]
    settling_required = settling_criteria["mode"] == "required"
    settle_window_samples = int(settling_criteria["window_samples"])
    settle_available = (
        len(normalized) >= settle_window_samples if settling_required else True
    )
    settle = (
        normalized[-settle_window_samples:]
        if settling_required
        else normalized[-1:]
    )
    destination_criteria = criteria["destination_containment"]
    destination_pose_readback_complete = (
        spec.get("destination_relation") is None
        or all(row["destination_pose"] is not None for row in normalized)
    )
    destination_pose_stable = destination_pose_readback_complete
    if spec.get("destination_relation") is not None:
        expected_destination_pose = spec["destination_pose_world"]
        destination_pose_stable = destination_pose_readback_complete and all(
            math.dist(row["destination_pose"][:3], expected_destination_pose[:3])
            <= spec["destination_reset_translation_tolerance_m"]
            and _quaternion_angle(
                row["destination_pose"][3:], expected_destination_pose[3:]
            )
            <= spec["destination_reset_rotation_tolerance_rad"]
            for row in normalized
        )
        subject_bounds = spec["subject_collision_bounds_scoring_frame_m"]
        subject_lower = subject_bounds["minimum"]
        subject_upper = subject_bounds["maximum"]
        interior_bounds = spec["destination_interior_bounds_body_frame_m"]
        lower = interior_bounds["minimum"]
        upper = interior_bounds["maximum"]

        def subject_inside_destination(row: Mapping[str, Any]) -> bool:
            destination_pose = row["destination_pose"]
            subject_corners_world = [
                [
                    row["pose"][axis] + rotated[axis]
                    for axis in range(3)
                ]
                for rotated in (
                    _rotate_xyzw([x, y, z], row["pose"][3:])
                    for x in (subject_lower[0], subject_upper[0])
                    for y in (subject_lower[1], subject_upper[1])
                    for z in (subject_lower[2], subject_upper[2])
                )
            ]
            corners_destination = [
                _rotate_inverse_xyzw(
                    [
                        point[axis] - destination_pose[axis]
                        for axis in range(3)
                    ],
                    destination_pose[3:],
                )
                for point in subject_corners_world
            ]
            return all(
                low <= value <= high
                for point in corners_destination
                for low, value, high in zip(lower, point, upper, strict=True)
            )

        destination_inside = (
            settle_available
            and destination_pose_stable
            and all(subject_inside_destination(row) for row in settle)
        )
    else:
        lower = destination_criteria["position_bounds_world_m"]["minimum"]
        upper = destination_criteria["position_bounds_world_m"]["maximum"]
        destination_inside = settle_available and all(
            all(
                low <= value <= high
                for low, value, high in zip(
                    lower, row["pose"][:3], upper, strict=True
                )
            )
            for row in settle
        )

        def subject_inside_destination(row: Mapping[str, Any]) -> bool:
            return all(
                low <= value <= high
                for low, value, high in zip(
                    lower, row["pose"][:3], upper, strict=True
                )
            )
    orientation_criteria = criteria["orientation"]
    orientation_errors = [
        _quaternion_angle(row["pose"][3:], orientation_criteria["reference_xyzw"])
        for row in settle
    ]
    orientation_ok = settle_available and all(
        error <= orientation_criteria["tolerance_rad"]
        for error in orientation_errors
    )
    support_criteria = criteria["support"]
    support_ok = settle_available and all(
        support_criteria["height_interval_m"][0]
        <= row["pose"][2]
        <= support_criteria["height_interval_m"][1]
        for row in settle
    )
    support_contact_complete = settle_available and all(
        isinstance(row["support_contact_active"], bool) for row in settle
    )
    support_contact_ok = support_contact_complete and all(
        row["support_contact_active"] is True for row in settle
    )
    anchor = settle[-1]["pose"] if settle else start
    settled = settle_available and all(
        math.dist(row["pose"][:3], anchor[:3])
        <= settling_criteria["position_tolerance_m"]
        and _quaternion_angle(row["pose"][3:], anchor[3:])
        <= settling_criteria["orientation_tolerance_rad"]
        for row in settle
    )
    gripper_criteria = criteria["gripper_state"]
    gripper_threshold = gripper_criteria["threshold_m"]
    released = settle_available and all(
        row["gripper_width_m"] is not None
        and row["gripper_width_m"] >= gripper_threshold
        and row["task_contact_active"] is False
        for row in settle
    ) if gripper_criteria["mode"] == "released" else False
    gripper_closed = settle_available and all(
        row["gripper_width_m"] is not None
        and row["gripper_width_m"] <= gripper_threshold
        for row in settle
    ) if gripper_criteria["mode"] == "closed_at_most" else False
    task_contact_complete = settle_available and all(
        isinstance(row["task_contact_active"], bool) for row in settle
    )
    task_contact_cleared = task_contact_complete and all(
        row["task_contact_active"] is False for row in settle
    )
    task_contact_maintained = task_contact_complete and all(
        row["task_contact_active"] is True for row in settle
    )
    safety_fields = (
        "robot_collision_failure",
        "scene_collision_failure",
        "containment_violation",
        "forbidden_robot_task_collision_failure",
        "locked_joint_containment_violation",
    )
    safety_complete = all(
        isinstance(row[field], bool) for row in normalized for field in safety_fields
    )
    safety_ok = safety_complete and not any(
        row[field] for row in normalized for field in safety_fields
    )
    safety_events: list[dict[str, Any]] = []
    for row in normalized:
        event_type = None
        measured_force_n = None
        if row["forbidden_robot_task_collision_failure"] is True:
            event_type = "forbidden_robot_object_contact_force_exceeded"
            measured_force_n = row[
                "robot_task_forbidden_collision_peak_force_n"
            ]
        elif row["robot_collision_failure"] is True:
            event_type = "robot_scene_contact_force_exceeded"
            measured_force_n = row["robot_scene_contact_peak_force_n"]
        elif row["scene_collision_failure"] is True:
            event_type = "task_scene_contact_force_exceeded"
            measured_force_n = row["task_scene_collision_peak_force_n"]
        elif row["containment_violation"] is True:
            event_type = "workspace_containment_excursion"
        elif row["locked_joint_containment_violation"] is True:
            event_type = "locked_joint_containment_excursion"
        if event_type is None:
            continue
        raw_pairs = row["robot_task_forbidden_contact_pairs"]
        contact_pair_identities = (
            [
                json.loads(json.dumps(item, allow_nan=False))
                for item in raw_pairs
                if isinstance(item, (str, Mapping))
            ]
            if event_type == "forbidden_robot_object_contact_force_exceeded"
            and isinstance(raw_pairs, Sequence)
            and not isinstance(raw_pairs, (str, bytes))
            else []
        )
        event = {
            "event_type": event_type,
            "step_index": row["step_index"],
            "simulation_time_seconds": (
                row["step_index"] / spec["control_frequency_hz"]
                if spec["control_frequency_hz"] is not None
                else None
            ),
            "measured_force_n": measured_force_n,
            "threshold_n": row["collision_failure_minimum_force_n"],
            "contact_pair_identities": contact_pair_identities,
            "contact_pair_identity_status": (
                "observed"
                if contact_pair_identities
                else "contact_pair_identity_missing"
            ),
        }
        safety_events.append(event)
    temporal_criteria = criteria["temporal_invariants"]
    no_drop_required = temporal_criteria["no_drop"]["mode"] == "required"
    contact_trace_complete = all(
        isinstance(row["task_contact_active"], bool) for row in normalized
    )
    support_trace_complete = all(
        isinstance(row["support_contact_active"], bool) for row in normalized
    )
    drop_events: list[dict[str, Any]] = []
    if contact_trace_complete and support_trace_complete:
        loss: dict[str, Any] | None = None

        def retain_drop_event(
            candidate: Mapping[str, Any],
            *,
            support_recontact: Mapping[str, Any] | None,
            task_contact_recovered_step: int | None = None,
        ) -> None:
            fall_m = max(
                0.0,
                float(candidate["reference_height_m"])
                - float(candidate["minimum_height_m"]),
            )
            if (
                candidate.get("unsupported_started_step") is None
                or fall_m < temporal_criteria["no_drop"]["minimum_fall_m"]
            ):
                return
            destination_inside_at_recontact = None
            support_recontact_step = None
            if support_recontact is not None:
                support_recontact_step = support_recontact["step_index"]
                destination_inside_at_recontact = (
                    support_recontact["destination_pose"] is not None
                    or spec.get("destination_relation") is None
                ) and subject_inside_destination(support_recontact)
            drop_events.append(
                {
                    "contact_lost_step": candidate["contact_lost_step"],
                    "unsupported_started_step": candidate[
                        "unsupported_started_step"
                    ],
                    "reference_height_m": candidate["reference_height_m"],
                    "minimum_height_m": candidate["minimum_height_m"],
                    "minimum_height_step": candidate["minimum_height_step"],
                    "support_recontact_step": support_recontact_step,
                    "task_contact_recovered_step": task_contact_recovered_step,
                    "fall_m": fall_m,
                    "destination_inside_at_recontact": (
                        destination_inside_at_recontact
                    ),
                }
            )

        for previous, row in zip(normalized, normalized[1:], strict=False):
            if previous["task_contact_active"] is True and row["task_contact_active"] is False:
                loss = {
                    "contact_lost_step": row["step_index"],
                    "reference_height_m": previous["pose"][2],
                    "minimum_height_m": previous["pose"][2],
                    "minimum_height_step": previous["step_index"],
                    "unsupported_started_step": None,
                }
            if loss is not None:
                if row["support_contact_active"] is False:
                    if loss["unsupported_started_step"] is None:
                        loss["unsupported_started_step"] = row["step_index"]
                    if row["pose"][2] < loss["minimum_height_m"]:
                        loss["minimum_height_m"] = row["pose"][2]
                        loss["minimum_height_step"] = row["step_index"]
                elif loss["unsupported_started_step"] is not None:
                    if row["pose"][2] < loss["minimum_height_m"]:
                        loss["minimum_height_m"] = row["pose"][2]
                        loss["minimum_height_step"] = row["step_index"]
                    retain_drop_event(loss, support_recontact=row)
                    loss = None
                    continue
                if loss is not None and row["task_contact_active"] is True:
                    retain_drop_event(
                        loss,
                        support_recontact=None,
                        task_contact_recovered_step=row["step_index"],
                    )
                    loss = None
        if loss is not None:
            retain_drop_event(loss, support_recontact=None)
    maximum_force_limit = temporal_criteria["maximum_task_contact_force_n"]
    force_trace_complete = all(
        row["task_contact_force_n"] is not None for row in normalized
    )
    peak_task_contact_force_n = (
        max(float(row["task_contact_force_n"]) for row in normalized)
        if force_trace_complete
        else None
    )
    task_contact_force_sources = sorted(
        {
            str(row["task_contact_force_source"])
            for row in normalized
            if row["task_contact_force_source"] is not None
        }
    )
    contact_classes_complete = all(
        isinstance(row["contact_classes_active"], Sequence)
        and not isinstance(row["contact_classes_active"], (str, bytes))
        and all(isinstance(item, str) for item in row["contact_classes_active"])
        for row in normalized
    )
    observed_contact_classes = sorted(
        {
            item
            for row in normalized
            for item in (
                row["contact_classes_active"] if contact_classes_complete else []
            )
        }
    )
    forbidden_contact_classes = set(temporal_criteria["forbidden_contact_classes"])
    observed_forbidden_contact_classes = sorted(
        forbidden_contact_classes.intersection(observed_contact_classes)
    )
    containment_trace_complete = all(
        isinstance(row["containment_violation"], bool) for row in normalized
    )
    containment_excursion_steps = [
        row["step_index"]
        for row in normalized
        if row["containment_violation"] is True
    ]
    workspace_trace_complete = all(
        isinstance(row["workspace_excursion"], bool) for row in normalized
    )
    workspace_excursion_steps = [
        row["step_index"]
        for row in normalized
        if row["workspace_excursion"] is True
    ]
    retry_counts = [row["retry_count"] for row in normalized]
    regrasp_counts = [row["regrasp_count"] for row in normalized]
    retry_trace_complete = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in retry_counts
    ) and all(
        previous <= current
        for previous, current in zip(retry_counts, retry_counts[1:], strict=False)
    )
    regrasp_trace_complete = all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in regrasp_counts
    ) and all(
        previous <= current
        for previous, current in zip(regrasp_counts, regrasp_counts[1:], strict=False)
    )
    maximum_retries_observed = (
        max(row["retry_count"] for row in normalized) if retry_trace_complete else None
    )
    maximum_regrasps_observed = (
        max(row["regrasp_count"] for row in normalized)
        if regrasp_trace_complete
        else None
    )
    temporal_readback_gaps: list[str] = []
    if no_drop_required and not (contact_trace_complete and support_trace_complete):
        temporal_readback_gaps.append("no_drop_contact_support_trace")
    if maximum_force_limit is not None and not force_trace_complete:
        temporal_readback_gaps.append("task_contact_force_trace")
    if forbidden_contact_classes and not contact_classes_complete:
        temporal_readback_gaps.append("contact_class_trace")
    if (
        temporal_criteria["containment_excursions"] == "forbidden"
        and not containment_trace_complete
    ):
        temporal_readback_gaps.append("containment_excursion_trace")
    if (
        temporal_criteria["workspace_excursions"] == "forbidden"
        and not workspace_trace_complete
    ):
        temporal_readback_gaps.append("workspace_excursion_trace")
    if temporal_criteria["maximum_retries"] is not None and not retry_trace_complete:
        temporal_readback_gaps.append("retry_event_ledger")
    if temporal_criteria["maximum_regrasps"] is not None and not regrasp_trace_complete:
        temporal_readback_gaps.append("regrasp_event_ledger")
    motion_criteria = criteria["motion"]
    moved = max(translation) > motion_criteria["movement_epsilon_m"]
    # Simulator state is floating-point readback. Treat a value that differs
    # from an exact preregistered boundary only by round-off as on the
    # boundary; this is not a task tolerance and must remain far below any
    # physical/scoring tolerance in the task contract.
    minimum_translation = motion_criteria["minimum_translation_m"]
    translated = minimum_translation is None or (
        max(translation) >= minimum_translation
        or math.isclose(
            max(translation), minimum_translation, rel_tol=0.0, abs_tol=1.0e-12
        )
    )
    minimum_lift = motion_criteria["minimum_lift_m"]
    lifted = minimum_lift is None or (
        max(lift) >= minimum_lift
        or math.isclose(
            max(lift), minimum_lift, rel_tol=0.0, abs_tol=1.0e-12
        )
    )
    task_contact_mode = criteria["terminal_task_contact"]["mode"]
    task_contact_satisfied = (
        True
        if task_contact_mode == "ignored"
        else task_contact_cleared
        if task_contact_mode == "cleared"
        else task_contact_maintained
    )
    gripper_mode = gripper_criteria["mode"]
    gripper_satisfied = (
        True
        if gripper_mode == "ignored"
        else released
        if gripper_mode == "released"
        else gripper_closed
    )
    criterion_results = {
        "destination_containment": (
            destination_criteria["mode"] == "ignored" or destination_inside
        ),
        **(
            {"destination_pose_stability": destination_pose_stable}
            if spec.get("destination_relation") is not None
            else {}
        ),
        "orientation": (
            orientation_criteria["mode"] == "ignored" or orientation_ok
        ),
        "support_height": (
            support_criteria["height_mode"] == "ignored" or support_ok
        ),
        "support_contact": (
            support_criteria["contact_mode"] == "ignored" or support_contact_ok
        ),
        "terminal_task_contact": task_contact_satisfied,
        "gripper_state": gripper_satisfied,
        "settling": not settling_required or settled,
        "safety": safety_ok,
        "minimum_translation": translated,
        "minimum_lift": lifted,
        "no_drop": not no_drop_required
        or (
            contact_trace_complete
            and support_trace_complete
            and not drop_events
        ),
        "maximum_task_contact_force": (
            maximum_force_limit is None
            or (
                peak_task_contact_force_n is not None
                and peak_task_contact_force_n <= maximum_force_limit
            )
        ),
        "forbidden_contact_classes": (
            not forbidden_contact_classes
            or (
                contact_classes_complete
                and not observed_forbidden_contact_classes
            )
        ),
        "containment_excursions": (
            temporal_criteria["containment_excursions"] == "ignored"
            or (
                containment_trace_complete and not containment_excursion_steps
            )
        ),
        "workspace_excursions": (
            temporal_criteria["workspace_excursions"] == "ignored"
            or (workspace_trace_complete and not workspace_excursion_steps)
        ),
        "maximum_retries": (
            temporal_criteria["maximum_retries"] is None
            or (
                maximum_retries_observed is not None
                and maximum_retries_observed <= temporal_criteria["maximum_retries"]
            )
        ),
        "maximum_regrasps": (
            temporal_criteria["maximum_regrasps"] is None
            or (
                maximum_regrasps_observed is not None
                and maximum_regrasps_observed <= temporal_criteria["maximum_regrasps"]
            )
        ),
    }
    succeeded = all(criterion_results.values())
    planar_push = spec["manipulation_strategy"] == "planar_push"
    required_task_contact_readback = task_contact_mode != "ignored"
    required_support_contact_readback = support_criteria["contact_mode"] == "required"
    if (
        not destination_pose_readback_complete
        or not safety_complete
        or (required_support_contact_readback and not support_contact_complete)
        or (required_task_contact_readback and not task_contact_complete)
        or temporal_readback_gaps
    ):
        status = "undetermined"
        if not destination_pose_readback_complete:
            outcome = "native_destination_pose_readback_missing"
        elif not safety_complete:
            outcome = "native_safety_readback_missing"
        elif required_support_contact_readback and not support_contact_complete:
            outcome = "native_support_contact_readback_missing"
        elif required_task_contact_readback and not task_contact_complete:
            outcome = "native_task_contact_readback_missing"
        else:
            outcome = "native_temporal_event_readback_missing"
    elif not safety_ok:
        status = "scored"
        outcome = "collision_or_containment_failure"
    elif succeeded:
        status = "scored"
        outcome = (
            OUTCOME_PUSHED_AND_SETTLED if planar_push else "placed_and_settled"
        )
    elif not moved:
        status = "scored"
        outcome = OUTCOME_NEVER_MOVED
    elif (
        planar_push
        and all(
            value
            for key, value in criterion_results.items()
            if key != "terminal_task_contact"
        )
        and not task_contact_satisfied
    ):
        status = "scored"
        outcome = "push_contact_not_cleared"
    elif (
        destination_inside
        and gripper_mode == "released"
        and not gripper_satisfied
    ):
        status = "scored"
        outcome = "release_incomplete"
    else:
        status = "scored" if settle_available else "undetermined"
        outcome = "moved_below_success_contract"
    failed_criteria = [
        name for name, satisfied in criterion_results.items() if not satisfied
    ]
    plain_reasons = {
        "destination_containment": "The object did not remain inside the authored destination.",
        "destination_pose_stability": "The destination moved beyond its qualified pose tolerance.",
        "orientation": "The object did not finish in the required orientation.",
        "support_height": "The object did not finish at the required support height.",
        "support_contact": "The object was not supported at the end of the episode.",
        "terminal_task_contact": "Robot contact with the task object did not match the required terminal state.",
        "gripper_state": "The gripper did not match the required terminal state.",
        "settling": "The object did not remain still for the required settling window.",
        "safety": "A collision or containment safety check failed or was unavailable.",
        "minimum_translation": "The object did not move the required minimum distance.",
        "minimum_lift": "The object was not lifted by the required minimum height.",
        "no_drop": "The object was dropped during the episode, even though it may later have reached the destination.",
        "maximum_task_contact_force": "Task contact exceeded the authored maximum force.",
        "forbidden_contact_classes": "A forbidden contact occurred during the episode.",
        "containment_excursions": "The object left its allowed containment region during the episode.",
        "workspace_excursions": "The object or robot left the authored workspace during the episode.",
        "maximum_retries": "The episode exceeded the authored retry limit.",
        "maximum_regrasps": "The episode exceeded the authored regrasp limit.",
    }
    specific_failure_reason = None
    if safety_events:
        first_safety_event = safety_events[0]
        if (
            first_safety_event["event_type"]
            == "forbidden_robot_object_contact_force_exceeded"
        ):
            force = first_safety_event["measured_force_n"]
            threshold = first_safety_event["threshold_n"]
            force_text = f"{force:g} N" if force is not None else "an unknown force"
            threshold_text = (
                f"{threshold:g} N" if threshold is not None else "the safety threshold"
            )
            specific_failure_reason = (
                "Forbidden robot-object contact reached "
                f"{force_text}, exceeding {threshold_text} at step "
                f"{first_safety_event['step_index']}."
            )
    report: dict[str, Any] = {
        "schema_version": RIGID_REPORT_SCHEMA_VERSION,
        "status": status,
        "task_kind": TASK_KIND_RIGID_PICK_PLACE,
        "subject_asset_id": spec["subject_asset_id"],
        "manipulation_strategy": spec["manipulation_strategy"],
        "task_success_contract": success_contract,
        "task_success_contract_digest": success_contract["contract_digest"],
        "task_succeeded": succeeded,
        "outcome": outcome,
        "criteria_satisfied": criterion_results,
        "failed_criteria": failed_criteria,
        "failure_reason_plain_english": (
            None
            if succeeded
            else specific_failure_reason
            if specific_failure_reason is not None
            else plain_reasons[failed_criteria[0]]
            if failed_criteria
            else "Required deterministic readback was unavailable."
        ),
        "event_ledger": {
            "schema_version": "rigid_task_event_ledger.v1",
            "safety_events": safety_events,
            "drop_events": drop_events,
            "peak_task_contact_force_n": peak_task_contact_force_n,
            "task_contact_force_sources": task_contact_force_sources,
            "observed_contact_classes": observed_contact_classes,
            "observed_forbidden_contact_classes": observed_forbidden_contact_classes,
            "containment_excursion_steps": containment_excursion_steps,
            "workspace_excursion_steps": workspace_excursion_steps,
            "maximum_retries_observed": maximum_retries_observed,
            "maximum_regrasps_observed": maximum_regrasps_observed,
            "required_readback_gaps": temporal_readback_gaps,
            "derived_only_from_episode_samples": True,
        },
        "measurements": {
            "sample_count": len(normalized),
            "reset_translation_error_m": reset_translation_error,
            "reset_orientation_error_rad": reset_orientation_error,
            "maximum_translation_m": max(translation),
            "maximum_lift_m": max(lift),
            "settle_window_available": settle_available,
            "settle_destination_inside": destination_inside,
            "destination_pose_readback_complete": destination_pose_readback_complete,
            "destination_pose_stable": destination_pose_stable,
            "settle_orientation_ok": orientation_ok,
            "settle_support_height_ok": support_ok,
            "settle_support_contact_readback_complete": support_contact_complete,
            "settle_support_contact_ok": support_contact_ok,
            "settled": settled,
            "released": released,
            "gripper_closed": gripper_closed,
            "settle_task_contact_readback_complete": task_contact_complete,
            "settle_task_contact_cleared": task_contact_cleared,
            "settle_task_contact_maintained": task_contact_maintained,
            "native_safety_readback_complete": safety_complete,
            "native_safety_ok": safety_ok,
        },
        "learned_judge_consulted": False,
        "candidate_policy_queried_by_scorer": False,
        "report_digest": "",
    }
    report["report_digest"] = canonical_digest(report, digest_field="report_digest")
    return report


def score_task_episode_from_spec(
    *, task_spec: Mapping[str, Any], samples: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Dispatch a frozen task spec without scene- or object-name conditionals."""

    kind = task_spec.get("task_kind")
    if kind == TASK_KIND_ARTICULATED_OPEN_CLOSE:
        return score_articulated_task_episode(task_spec=task_spec, samples=samples)
    if kind == TASK_KIND_RIGID_PICK_PLACE:
        if task_spec.get("schema_version") == TASK_SPEC_GRAPH_SCHEMA_VERSION:
            return score_rigid_task_episode(task_spec=task_spec, samples=samples)
        if task_spec.get("schema_version") != TASK_SPEC_SCHEMA_VERSION:
            raise TaskNeutralScoringError(["task_spec_schema_invalid"])
        try:
            return score_task_episode(
                samples=samples,
                destination_position_world_m=task_spec["destination_position_world_m"],
                support_plane_z_m=float(task_spec["support_plane_z_m"]),
                settle_window_samples=int(task_spec["settle_window_samples"]),
                require_sealed_start_pose=bool(task_spec.get("require_sealed_start_pose", True)),
            )
        except (KeyError, TypeError, ValueError, TaskScoringError) as exc:
            if isinstance(exc, TaskScoringError):
                raise TaskNeutralScoringError(exc.errors) from exc
            raise TaskNeutralScoringError(["rigid_task_spec_invalid"]) from exc
    raise TaskNeutralScoringError(["task_kind_unsupported"])


__all__ = [
    "ARTICULATED_REPORT_SCHEMA_VERSION",
    "RIGID_REPORT_SCHEMA_VERSION",
    "RIGID_TASK_SUCCESS_CONTRACT_SCHEMA_VERSION",
    "RigidTaskEventLedgerExpectation",
    "RigidTaskSuccessContract",
    "OUTCOME_COLLISION_FAILURE",
    "OUTCOME_LIMIT_OR_CONTAINMENT_VIOLATION",
    "OUTCOME_MOVED_BELOW_THRESHOLD",
    "OUTCOME_NEVER_MOVED",
    "OUTCOME_NON_TASK_JOINT_MOVED",
    "OUTCOME_OPENED_AND_SETTLED",
    "OUTCOME_OPENED_THEN_REBOUNDED",
    "OUTCOME_PUSHED_AND_SETTLED",
    "OUTCOME_RELEASE_OR_RETREAT_INCOMPLETE",
    "TASK_KIND_ARTICULATED_OPEN_CLOSE",
    "TASK_KIND_RIGID_PICK_PLACE",
    "TASK_SPEC_SCHEMA_VERSION",
    "TASK_SPEC_GRAPH_SCHEMA_VERSION",
    "TaskNeutralScoringError",
    "confirm_rigid_task_success_contract",
    "confirmed_rigid_task_success_contract_matches_published",
    "score_articulated_task_episode",
    "score_rigid_task_episode",
    "score_task_episode_from_spec",
    "seal_rigid_task_success_contract",
    "validate_articulated_task_spec",
    "validate_rigid_task_success_contract",
]
