"""The scripted reference check: one row run FIRST on every website screening campaign (owner, 2026-11-12).

Beta journey: a site's paid candidate screening. A result must be about the robot, not about our set-up:
"every candidate failed" has to be told apart from "this simulated set-up cannot succeed". So every
campaign, a new one and every re-plan, plans one row first: Blueprint's scripted pick/place controller
(``scripted_robot_baseline``), which reads exact object and arm poses from the simulator, runs ``EPISODES``
nominal (canonical anchor) tries in its own session, through the same harness and scorer as the candidates.

What the row is, and is not:

- It is ``reference_only``: never a candidate, never part of the estimate, not Arm Decision Proof evidence,
  and it does not lift the standing controls-stage pause (``task_evaluation_control_stage_policy``). A pass
  shows only that the simulated set-up can register a success, never physical performance or safety.
- It never gates. Whatever its outcome the AI robot rows run (the owner's rule: flagged and keeps going).
  Its outcome is one typed flag on its own row's ``setup_quality_flags`` (``control_flags``) with its count:
  ``known_good_control_passed:<k>_of_3`` at ``PASS_AT_SUCCESSES`` (2) or more, ``known_good_control_partly``
  at ``PARTLY_AT_SUCCESSES`` (1), ``known_good_control_failed`` at 0 of 3 scored, else ``interrupted``. What
  the site is told changes with it (the WebApp's set-up notice); what runs does not.
- It runs first and is funded first: the plan takes its cap out of the budget before any AI robot row is
  funded (``beta_policy_screening_plan.build_job_ceiling_screening_plan``), it is first in plan order, and
  the coordinator reserves it like any funded row. Its cap is its body's quoted hard cap lowered to
  ``REFERENCE_CAP_USD`` ($3.00, owner, 2026-11-11): the hard TTL is shortened to fit
  (``_site_fitted_quote``), never below ``MIN_HARD_TTL_SECONDS``. Its 3 tries and time limits are unchanged.
- Its body is Blueprint's choice, never the site's fit table: the registered scripted body of the robot
  under test when it has one (``chosen_by: candidate_robot``), else the registered scripted body of that
  robot's arm family (``ARM_FAMILY_BODIES``, ``chosen_by: arm_family``), the tested robots in the plan's
  order. A body not bound to its own station stands at the scene packet's, flagged ``scene_packet_station``.
- When no row can run, the row says why, per job: ``known_good_control_unavailable:<reason>`` with a
  reason of ``REASONS`` (``task_kind``, ``backend``, ``embodiment``: no registered scripted body for the
  tested robots' arm family or it was refused on this scene, ``budget``: its cap does not fit the job).
- A pass, partly or fail is reused, never paid for twice, for the same site, scene revision, task contract
  and harness revision (``reuse_key``); while the same check runs in another campaign, this one waits.

A version-1 row (planned last, on leftover money, passing at 1 of 3) still validates and reads by its own
marker; none was ever planned on the host (0 rows in 46 campaigns, 2026-11-12).
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from decimal import Decimal
import hashlib
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .website_candidate_feasibility import OWN_STATION, PACKET_STATION

CONTROL_FIELD = "known_good_control"
SCHEMA = "beta_screening_known_good_control.v2"
SCHEMA_V1 = "beta_screening_known_good_control.v1"
EPISODES = 3
#: Passed at 2 of 3 or better; 1 of 3 is ``PARTLY``; 0 of 3 scored is ``FAILED`` (owner, 2026-11-12).
PASS_AT_SUCCESSES = 2
PARTLY_AT_SUCCESSES = 1
#: A version-1 row passed at 1 of 3 and had no partly.
PASS_AT_SUCCESSES_V1 = 1
#: The scripted reference row's cap (owner approval, 2026-11-11): its quote is lowered to this when
#: higher, so it fits the $3.00 a typical rigid job leaves after three learned policies at $4.00.
#: Only this row; no learned-policy cap, plan budget, site limit, job ceiling or account limit.
REFERENCE_CAP_USD = Decimal("3.00")
#: The shortest hard TTL a screening run takes (``beta_policy_screening_launch._check_quote``).
MIN_HARD_TTL_SECONDS = 1800
DISPLAY_NAME = "Blueprint scripted reference check"
#: The row's id when no scripted profile could be matched: no catalog row stands behind it.
UNAVAILABLE_ID = "blueprint_known_good_control"
SCRIPTED = "scripted_simulator_state_baseline"
SCRIPTED_INTERFACE = "blueprint_scripted_pick_place.v1"
RIGID_TASK_KIND = "rigid_pick_place"

PENDING = "known_good_control_pending"
PASSED = "known_good_control_passed"
FAILED = "known_good_control_failed"
PARTLY = "known_good_control_partly"
UNAVAILABLE = "known_good_control_unavailable"
INTERRUPTED = "known_good_control_interrupted"
REASONS = ("task_kind", "embodiment", "backend", "budget", "reach")
REUSED_QUALIFIER = "reused"
#: The check's whole money per scene revision and task contract, across every attempt (owner, 2026-11-12): a
#: later attempt is only funded from what the earlier ones left unspent, and a classed result is final.
CLAIMED_FIELD = "known_good_control_claimed"
#: The planning capability: what the check may still hold for the scene (``reference_remaining_usd``).
REMAINING_CAPABILITY = "reference_remaining_usd"
#: The one blocker a control row that did not run here publishes, whatever the reason: the plan's own
#: "switched off" code, which a WebApp released before this change already reads as a row that was not
#: for running (as it read scripted rows before 2026-11-05). The reason is the row's outcome flag.
NOT_RUN_BLOCKER = "candidate_disabled"
REUSED_BLOCKER = NOT_RUN_BLOCKER
REUSED_FIELD = "known_good_control_reused"
KEY_FIELD = "known_good_control_key"
ALERT_CODE = "beta_screening_known_good_control_failed"
ALERT_STATE_FILENAME = "known_good_control_alerts.json"

#: The planning capabilities a version-1 row's body was chosen from; read by nothing here any more.
FIT_CAPABILITY = "robot_fit_rows"
STATION_CAPABILITY = "candidate_station_basis"
PREFERENCE_CAPABILITY = "preferred_robot_model_ids"
PREFERENCE_LIMIT = 3
#: A tested robot's arm family, as the registered scripted bodies that stand for it, best first. Closed: a
#: robot not listed has only its own body, when one is registered. The DROID Franka (Panda with a Robotiq
#: 2F-85) has no scripted body of its own; the Panda body is listed first in case one is registered, then
#: the FR3 with the same Robotiq 2F-85: same maker, seven joints, same gripper, the nearest, not the same arm.
ARM_FAMILY_BODIES: Mapping[str, tuple[str, ...]] = {
    "droid_franka_panda_robotiq_2f85_v1": ("franka_emika_panda_robotiq_2f85_sim", "franka_research_3_robotiq_2f85_sim"),
}
CHOSEN_BY = ("candidate_robot", "arm_family")
#: A version-1 row's rules for what it stood for (``validate_control_row``).
CHOSEN_BY_V1 = ("candidate_robot", "nearest_body", "preferred", "fit_table")
#: The coordinator's "nothing can fund it" close (``beta_policy_screening_coordinator.BUDGET_EXHAUSTED_BLOCKER``).
BUDGET_EXHAUSTED_BLOCKER = "beta_screening_site_budget_exhausted"
#: The learned rows' states in which they may still need money or run: the control waits on these alone.
_LEARNED_OPEN = frozenset({"pending", "waitlisted", "preparing", "launching", "running", "reconciliation_required"})
#: Logged, never raised, when the website sent a preference this planner cannot use.
PREFERENCE_INVALID = "known_good_control_preference_invalid"
#: The feasibility check's own station code for a body bound to the scene packet's station, not its own.
PACKET_STATION_FLAG = PACKET_STATION

_MARKER = {"schema_version": SCHEMA, "role": "reference_only", "episodes": EPISODES,
           "pass_at_successes": PASS_AT_SUCCESSES, "partly_at_successes": PARTLY_AT_SUCCESSES}
_MARKER_V1 = {"schema_version": SCHEMA_V1, "role": "reference_only", "episodes": EPISODES,
              "pass_at_successes": PASS_AT_SUCCESSES_V1}
_TERMINAL = frozenset({"completed", "blocked", "unavailable", "deferred"})
_ACTIVE = frozenset({"preparing", "launching", "running", "reconciliation_required"})
_SELECTED_FIELDS = frozenset({"candidate_id", "display_name", "configuration_digest", "status", "blockers",
                              "admission_digest", "execution_spec", "cost_quote", "max_cost_usd", "episode_count",
                              "scenario_cells", "execution_evidence_kind", CONTROL_FIELD})
_UNAVAILABLE_FIELDS = frozenset({"candidate_id", "display_name", "configuration_digest", "status", "blockers",
                                 "execution_evidence_kind", CONTROL_FIELD})


def is_control(row: Mapping[str, Any]) -> bool:
    """A plan row (its marker) or ledger row (``True``) that is the campaign's known-good control."""
    return bool(row.get(CONTROL_FIELD))


def _budget_blockers() -> frozenset[str]:
    from .account_monthly_spend import LIMIT_REACHED
    from .beta_policy_screening_plan import JOB_CEILING_DEFERRED, WAITLIST_BLOCKER

    from .beta_policy_screening_job_ceiling import EXCEEDED

    return frozenset({WAITLIST_BLOCKER, JOB_CEILING_DEFERRED, BUDGET_EXHAUSTED_BLOCKER, LIMIT_REACHED, EXCEEDED})


def control_cells(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    """``EPISODES`` nominal cells: the canonical anchor, repeated under seeds of the control's own domain."""
    from .beta_policy_screening_plan import _native_scene_plan
    from .policy_canary_scene_resolution import constrain_diagnostic_scenario

    native = _native_scene_plan(scene)
    material = "beta_policy_screening_known_good_control.v1:" + scene["scene_revision_digest"] + ":" + scene["task_context_digest"]
    salt = hashlib.sha256(material.encode()).hexdigest()
    cells = []
    for ordinal in range(EPISODES):
        scenario = constrain_diagnostic_scenario(native, {"family": "canonical_anchor", "ordinal": ordinal,
                                                          "parameters": {}})
        digest = canonical_digest(scenario)
        seed = int(hashlib.sha256(f"{salt}:{ordinal}".encode()).hexdigest()[:8], 16) % 2**31
        cells.append({"cell_id": f"screening.{scene['scene_id']}.control.{ordinal:03d}", "seed": seed,
                      "family": "canonical_anchor", "partition": "screening", "resolved_scenario": scenario,
                      "cell_spec_digest": digest, "resolved_scenario_digest": digest})
    return cells


def _unavailable(reason: str, candidate: Mapping[str, Any] | None = None) -> dict[str, Any]:
    marker = {**_MARKER, "unavailable": reason}
    return {"candidate_id": candidate["candidate_id"] if candidate else UNAVAILABLE_ID, "display_name": DISPLAY_NAME,
            # With no scripted profile behind the row, its configuration is this marker and nothing else.
            "configuration_digest": candidate["configuration_digest"] if candidate else canonical_digest(marker),
            "status": "unavailable", "blockers": [NOT_RUN_BLOCKER], "execution_evidence_kind": SCRIPTED,
            CONTROL_FIELD: marker}


def preferred_model_ids(value: Any, known: Iterable[str]) -> tuple[list[str], str | None]:
    """The website's preference as the planner uses it, and a typed code when it was sent but unusable.

    Absent (``None``) is no preference. Anything but a list of 1 to ``PREFERENCE_LIMIT`` distinct
    spec-catalog model ids is ignored whole, with ``PREFERENCE_INVALID``: it never stops a plan.
    """
    if value is None:
        return [], None
    known = set(known)
    if (not isinstance(value, list) or not 1 <= len(value) <= PREFERENCE_LIMIT
            or any(not isinstance(item, str) or item not in known for item in value) or len(set(value)) != len(value)):
        return [], PREFERENCE_INVALID
    return list(value), None


def tested_robot_bodies(catalog: Mapping[str, Any], tested: Iterable[str]) -> list[tuple[Mapping[str, Any], dict[str, Any]]]:
    """The registered scripted rows that may carry the check, best first, each with what it stands for.

    For each tested candidate in order: its robot's own scripted body, then its arm family's
    (``ARM_FAMILY_BODIES``). Never the site's fit table.
    """
    by_id = {row["candidate_id"]: row for row in catalog["candidates"]}
    scripted: dict[str, Mapping[str, Any]] = {}
    for row in catalog["candidates"]:
        if (row["policy"]["kind"] == "scripted"
                and (row.get("interface") or {}).get("policy_interface_id") == SCRIPTED_INTERFACE):
            scripted.setdefault(row["interface"].get("robot_profile_id"), row)
    bodies: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    seen: set[str] = set()
    for candidate_id in tested:
        candidate = by_id.get(candidate_id) or {}
        robot = (candidate.get("interface") or {}).get("robot_profile_id")
        if not robot or candidate.get("policy", {}).get("kind") == "scripted":
            continue
        for profile, rule in ((robot, "candidate_robot"), *((body, "arm_family") for body in ARM_FAMILY_BODIES.get(robot, ()))):
            row = scripted.get(profile)
            if row is not None and row["candidate_id"] not in seen:
                seen.add(row["candidate_id"])
                bodies.append((row, {"for_candidate_id": candidate_id, "chosen_by": rule}))
    return bodies


def plan_control_row(catalog: Mapping[str, Any], scene: Mapping[str, Any], *, tested: Iterable[str],
                     capabilities: Mapping[str, Any] | None, cost_quotes: Mapping[str, Any] | None,
                     budget_usd: Decimal, now: Any) -> dict[str, Any]:
    """The reference row a campaign's plan puts first, funded from ``budget_usd`` before any candidate."""
    from .task_family_policy_execution import FAMILY_BACKEND

    if scene.get("evaluation_backend") == FAMILY_BACKEND:
        return _unavailable("backend")
    if scene.get("task_kind") != RIGID_TASK_KIND:
        return _unavailable("task_kind")
    bodies = tested_robot_bodies(catalog, tested)
    if not bodies:
        return _unavailable("embodiment")
    first = None
    for match, stands_for in bodies:
        row = _body_row(match, stands_for, scene, capabilities=capabilities or {}, cost_quotes=cost_quotes,
                        budget_usd=budget_usd, now=now)
        if row["status"] == "selected":
            return row
        first = first or row
    return first


def _body_row(match: Mapping[str, Any], stands_for: dict[str, Any], scene: Mapping[str, Any], *,
              capabilities: Mapping[str, Any], cost_quotes: Mapping[str, Any] | None, budget_usd: Decimal,
              now: Any) -> dict[str, Any]:
    from .beta_policy_screening_plan import (
        _HOSTED_COST_FIELDS, BetaPolicyScreeningPlanError, _candidate_blockers, _scene, _site_fitted_quote,
        execution_cost_floor, fitted_headroom_usd, money, validate_cost_quote,
    )

    scene = _scene(scene)
    if (capabilities.get(STATION_CAPABILITY) or {}).get(match["candidate_id"]) != OWN_STATION:
        stands_for = {**stands_for, "flags": [PACKET_STATION_FLAG]}
    binding = (capabilities.get("candidate_bindings") or {}).get(match["candidate_id"])
    refused = _candidate_blockers(match, scene, binding, set(capabilities.get("credentials_available") or []))
    refused += list((capabilities.get("candidate_blockers") or {}).get(match["candidate_id"]) or [])
    refused += list((capabilities.get("candidate_infeasible") or {}).get(match["candidate_id"]) or [])
    if refused:
        return _unavailable("embodiment", match)
    # A reference must show the job can be done in this scene: a body measured unable to reach the pick-up and
    # the put-down from its station is not one (``reach_cannot_span``); not launched, not paid.
    rank = (capabilities.get("candidate_reach_rank") or {}).get(match["candidate_id"]) or {}
    if rank.get("basis") == "reach_cannot_span":
        return _unavailable("reach", match)
    spec = binding["execution_spec"]
    try:
        quote = validate_cost_quote((cost_quotes or {}).get(match["candidate_id"], {}),
                                    configuration_digest=match["configuration_digest"],
                                    scene_revision_digest=scene["scene_revision_digest"],
                                    episode_count=((cost_quotes or {}).get(match["candidate_id"]) or {}).get("episode_count"),
                                    now=now)
    except BetaPolicyScreeningPlanError:
        return _unavailable("budget", match)
    # The quote is the run's hard cap (hourly rate x hard TTL plus headroom), whatever its episode count,
    # re-sealed for this row's episodes. Above REFERENCE_CAP_USD it is fitted to that cap by the plan's own
    # rule (``_site_fitted_quote``): the hard TTL is shortened (whole seconds, rounded down) to the longest
    # that keeps half the original headroom inside the cap. Never raised.
    admission = binding["admission_digest"]
    limit = min(REFERENCE_CAP_USD, money(str(capabilities.get(REMAINING_CAPABILITY, REFERENCE_CAP_USD))))
    if money(quote["maximum_cost_usd"]) > limit:
        if _HOSTED_COST_FIELDS & set(spec):
            return _unavailable("budget", match)
        original_floor = execution_cost_floor(spec)
        bound = limit - fitted_headroom_usd(quote, floor=original_floor)
        ttl = min(spec["hard_ttl_seconds"], int(bound * 3600 / money(spec["max_hourly_rate_usd"]))) if bound > 0 else 0
        fitted = None
        if ttl >= MIN_HARD_TTL_SECONDS:
            spec = {**spec, "hard_ttl_seconds": ttl}
            fitted = _site_fitted_quote(quote, limit, floor=execution_cost_floor(spec),
                                        original_floor=original_floor, reason="fitted_to_scripted_reference_cap",
                                        suffix="-reference-cap")
        if fitted is None:
            return _unavailable("budget", match)
        admission, quote = canonical_digest(spec), fitted
    quote = {**quote, "episode_count": EPISODES, "quote_digest": ""}
    quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
    cap = money(quote["maximum_cost_usd"])
    # Funded first: its cap comes out of the plan's budget before any candidate row is funded.
    if cap < execution_cost_floor(spec) or cap > budget_usd:
        return _unavailable("budget", match)
    return {"candidate_id": match["candidate_id"], "display_name": DISPLAY_NAME,
            "configuration_digest": match["configuration_digest"], "status": "selected", "blockers": [],
            "admission_digest": admission, "execution_spec": deepcopy(spec),
            "cost_quote": quote, "max_cost_usd": quote["maximum_cost_usd"], "episode_count": EPISODES,
            "scenario_cells": control_cells(scene), "execution_evidence_kind": SCRIPTED,
            CONTROL_FIELD: {**_MARKER, **stands_for}}


def fund_from_leftover(control: dict[str, Any], plan: Mapping[str, Any], *,
                       waitlist_blockers: list[str]) -> dict[str, Any]:
    """The reference row funded only from what the plan's AI robot rows leave; it never takes their money.

    ``selected`` when its cap fits what the selected AI rows leave of the budget; ``waitlisted`` (with the
    plan's own waitlist reason) when it fits the budget but not that; ``unavailable:budget`` when no AI row is
    selected, so it is never paid for a job whose robots are not tested.
    """
    from .beta_policy_screening_plan import money

    if control["status"] != "selected":
        return control
    learned = [row for row in plan["candidates"] if not is_control(row)]
    selected = sum((money(row["max_cost_usd"]) for row in learned if row["status"] == "selected"), Decimal(0))
    if not any(row["status"] == "selected" for row in learned):
        return _unavailable("budget", control)
    # Selected only when every AI row that can be funded is, and its cap still fits what they leave.
    if (not any(row["status"] == "waitlisted" for row in learned)
            and money(control["max_cost_usd"]) <= money(plan["budget_usd"]) - selected):
        return control
    return {**control, "status": "waitlisted", "blockers": list(waitlist_blockers)}


def validate_control_row(row: Mapping[str, Any], plan: Mapping[str, Any]) -> Decimal:
    """A retained control row; returns what it commits from the plan's budget."""
    from .beta_policy_screening_plan import (
        BetaPolicyScreeningPlanError, _CANDIDATE_ID, _digest, execution_cost_floor, money, validate_cost_quote,
        validate_execution_descriptor,
    )
    from .task_family_policy_execution import FAMILY_BACKEND

    def require(condition: bool) -> None:
        if not condition:
            raise BetaPolicyScreeningPlanError("beta_screening_known_good_control_invalid")

    marker = row.get(CONTROL_FIELD)
    require(isinstance(marker, dict) and row.get("execution_evidence_kind") == SCRIPTED
            and row.get("display_name") == DISPLAY_NAME and isinstance(row.get("candidate_id"), str)
            and _CANDIDATE_ID.fullmatch(row["candidate_id"]) is not None)
    _digest(row["configuration_digest"], "beta_screening_known_good_control_invalid")
    base = _marker_base(marker)
    require(base is not None)
    if row["status"] == "unavailable":
        require(set(row) == _UNAVAILABLE_FIELDS and set(marker) == {*base, "unavailable"}
                and marker["unavailable"] in REASONS and row["blockers"] == [NOT_RUN_BLOCKER])
        return Decimal(0)
    stands_for = set(marker) - set(base)
    from .beta_policy_screening_plan import JOB_CEILING_DEFERRED, WAITLIST_BLOCKER
    waitlisted = row["status"] == "waitlisted" and base is _MARKER
    require((row["status"] == "selected" and row["blockers"] == []
             or waitlisted and row["blockers"] in ([WAITLIST_BLOCKER], [JOB_CEILING_DEFERRED]))
            and set(row) == _SELECTED_FIELDS)
    learned = [item for item in plan["candidates"] if not is_control(item)]
    if base is _MARKER:
        # A tested candidate of this plan (any status: the row is funded before any is), its own body or
        # its arm family's, with the station flag when the body stands at the scene packet's station.
        require(stands_for <= {"for_candidate_id", "chosen_by", "flags"} and {"for_candidate_id", "chosen_by"} <= stands_for
                and marker["chosen_by"] in CHOSEN_BY and marker.get("flags", [PACKET_STATION_FLAG]) == [PACKET_STATION_FLAG]
                and any(item["candidate_id"] == marker["for_candidate_id"] for item in learned))
    elif "for_candidate_id" in stands_for:
        require(stands_for <= {"for_candidate_id", "chosen_by"}
                and marker.get("chosen_by", "candidate_robot") == "candidate_robot"
                and isinstance(marker["for_candidate_id"], str)
                and any(item["candidate_id"] == marker["for_candidate_id"] and item["status"] == "selected"
                        for item in learned))
    else:
        require("for_model_id" in stands_for and stands_for <= {"for_model_id", "flags", "preferred", "chosen_by"}
                and marker.get("chosen_by", "fit_table") in CHOSEN_BY_V1[1:]
                and isinstance(marker["for_model_id"], str) and _CANDIDATE_ID.fullmatch(marker["for_model_id"]) is not None
                and marker.get("flags", [PACKET_STATION_FLAG]) == [PACKET_STATION_FLAG]
                and type(marker.get("preferred", False)) is bool)
    require(row["episode_count"] == EPISODES and isinstance(row["scenario_cells"], list)
            and len(row["scenario_cells"]) == EPISODES)
    ids, seeds = set(), set()
    for ordinal, cell in enumerate(row["scenario_cells"]):
        scenario = cell.get("resolved_scenario") if isinstance(cell, dict) else None
        require(isinstance(scenario, dict) and set(cell) == {"cell_id", "seed", "family", "partition",
                "resolved_scenario", "cell_spec_digest", "resolved_scenario_digest"}
                and cell["family"] == "canonical_anchor" and cell["partition"] == "screening"
                and cell["cell_id"] == f"screening.{plan['scene_id']}.control.{ordinal:03d}"
                and type(cell["seed"]) is int and 0 <= cell["seed"] < 2**31
                and scenario.get("family") == "canonical_anchor" and scenario.get("ordinal") == ordinal
                and scenario.get("parameters") == {} and set(scenario) <= {"family", "ordinal", "parameters"}
                and cell["cell_spec_digest"] == canonical_digest(scenario) == cell["resolved_scenario_digest"])
        ids.add(cell["cell_id"])
        seeds.add(cell["seed"])
    require(len(ids) == len(seeds) == EPISODES)
    spec = validate_execution_descriptor(row["execution_spec"], candidate_id=row["candidate_id"],
                                         configuration_digest=row["configuration_digest"])
    require(row["admission_digest"] == canonical_digest(spec)
            and spec["task_success_contract_digest"] == plan["task_success_contract_digest"]
            and spec["backend"] != FAMILY_BACKEND)
    quote = validate_cost_quote(row["cost_quote"], configuration_digest=row["configuration_digest"],
                                scene_revision_digest=plan["scene_revision_digest"], episode_count=EPISODES)
    require(row["max_cost_usd"] == quote["maximum_cost_usd"] and money(row["max_cost_usd"]) >= execution_cost_floor(spec))
    # A waitlisted reference row commits nothing until the coordinator funds it from what the rows give back.
    return Decimal(0) if waitlisted else money(row["max_cost_usd"])


def _marker_base(marker: Any) -> Mapping[str, Any] | None:
    """The fixed part of a row's marker, by its version: ``_MARKER`` (first) or ``_MARKER_V1`` (last)."""
    if not isinstance(marker, Mapping):
        return None
    for base in (_MARKER, _MARKER_V1):
        if set(base) <= set(marker) and all(marker[key] == value for key, value in base.items()):
            return base
    return None


def first_in_plan(row: Mapping[str, Any]) -> bool:
    """Whether a control row's version puts it first in its plan (version 2) rather than last (version 1)."""
    return _marker_base(row.get(CONTROL_FIELD)) is _MARKER


def control_outcome(configured: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """The control's typed outcome from its plan row and ledger row; ``PENDING`` until the row ends.

    A completed run's outcome carries its count: ``passed:<k>_of_3``, ``partly:1_of_3``, ``failed:0_of_3``.
    """
    if row.get("status") not in _TERMINAL:
        return PENDING
    reused = row.get(REUSED_FIELD)
    if isinstance(reused, Mapping) and reused.get("outcome") in {PASSED, PARTLY, FAILED}:
        return f"{reused['outcome']}:{REUSED_QUALIFIER}"
    marker = configured.get(CONTROL_FIELD) or {}
    reason = marker.get("unavailable") if isinstance(marker, Mapping) else None
    if reason in REASONS:
        return f"{UNAVAILABLE}:{reason}"
    if row["status"] == "completed":
        summary = row.get("summary") or {}
        succeeded, scored = summary.get("episodes_succeeded", 0), summary.get("episodes_scored", 0)
        count = f"{succeeded}_of_{configured.get('episode_count', EPISODES)}"
        pass_at = marker.get("pass_at_successes", PASS_AT_SUCCESSES) if isinstance(marker, Mapping) else PASS_AT_SUCCESSES
        partly_at = marker.get("partly_at_successes") if isinstance(marker, Mapping) else None
        if succeeded >= pass_at:
            return f"{PASSED}:{count}"
        # The rest needs every planned try scored; anything less did not finish.
        if scored != configured.get("episode_count"):
            return INTERRUPTED
        return f"{PARTLY}:{count}" if partly_at is not None and succeeded >= partly_at else f"{FAILED}:{count}"
    if set(row.get("blockers") or []) & _budget_blockers():
        return f"{UNAVAILABLE}:budget"
    return INTERRUPTED


def control_flags(configured: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, list[str]]:
    """The control row's ``setup_quality_flags``: its one outcome code. Nothing for any other row."""
    return {"setup_quality_flags": [control_outcome(configured, row)]} if is_control(configured) else {}


def reuse_key(plan: Mapping[str, Any]) -> str:
    """What makes two controls the same check: site, scene revision and task contract (review 2026-11-12).

    Not the harness revision: a re-plan or a deploy reuses a finished count rather than pay for the same scene.
    """
    return canonical_digest({"schema_version": SCHEMA, "site_id": plan["site_id"],
                             "scene_revision_digest": plan["scene_revision_digest"],
                             "task_success_contract_digest": plan["task_success_contract_digest"]})


def control_gate(ledger: Mapping[str, Any], campaign_id: str, plan: Mapping[str, Any], *,
                 plans: Callable[[str, str], Mapping[str, Any]]) -> str:
    """Whether a pending control may go to the ordinary reservation now: ``proceed``, ``wait`` or ``reused``.

    Called under the ledger lock; may mark the ledger row (its key, a reused result) in place. A finished
    pass, partly or fail of the same check (``reuse_key``) is reused, never paid for twice; while the same
    check runs in another campaign, this one waits. Otherwise it proceeds: it is funded like any row, and
    the reservation's own money rules end it for budget. A version-1 row also waits while a learned row of
    its own campaign may still need money or run (it was planned to run last).
    """
    state = ledger["campaigns"][campaign_id]
    configured = next(row for row in plan["candidates"] if is_control(row))
    row = state["candidates"][configured["candidate_id"]]
    learned = [state["candidates"][item["candidate_id"]] for item in plan["candidates"] if not is_control(item)]
    if not first_in_plan(configured) and any(item["status"] in _LEARNED_OPEN for item in learned):
        return "wait"
    if first_in_plan(configured):
        # Never ahead of a robot: it launches only after an AI row has launched an attempt, and never while an
        # AI row still waits for money; if no AI row ever launches it ends without running or paying.
        launched = any("launch_intent_at" in attempt for item in learned
                       for attempt in [*(item.get("attempt_history") or []), *([item["attempt"]] if item.get("attempt") else [])])
        waiting = any(item["status"] == "waitlisted" for item in learned)
        if not launched or waiting:
            if any(item["status"] in _LEARNED_OPEN for item in learned):
                return "wait"
            from .beta_policy_screening_site_limit import budget_reason
            row.update(status="deferred", blockers=[budget_reason(plan, BUDGET_EXHAUSTED_BLOCKER)])
            return "closed"
    key = row.setdefault(KEY_FIELD, reuse_key(plan))
    for other_id, other in sorted(ledger["campaigns"].items()):
        if other["site_id"] != state["site_id"]:
            continue
        for other_key, other_row in other["candidates"].items():
            if other_id == campaign_id or not is_control(other_row) or other_row.get(KEY_FIELD) != key:
                continue
            if other_row["status"] in _ACTIVE or other_row["status"] == "pending" and other_row.get(CLAIMED_FIELD):
                # The same check runs, or was let through to its reservation, elsewhere: two workers never both
                # start an attempt (this gate runs under the ledger lock and claims before it returns).
                return "wait"
            other_plan = plans(other_id, other["plan_digest"])
            other_configured = next((item for item in other_plan["candidates"] if item["candidate_id"] == other_key), None)
            outcome = control_outcome(other_configured or {}, other_row).split(":")[0]
            if outcome in {PASSED, PARTLY, FAILED}:
                row.update(status="unavailable", blockers=[REUSED_BLOCKER])
                row[REUSED_FIELD] = {"campaign_id": other_id, "outcome": outcome}
                return "reused"
    # No classed result yet: an attempt is funded only from what the earlier ones left of the $3.00.
    from .beta_policy_screening_plan import money

    left = REFERENCE_CAP_USD - reference_spent(ledger, key, site_id=state["site_id"])
    if money(configured["max_cost_usd"]) > left:
        from .beta_policy_screening_site_limit import budget_reason
        row.update(status="deferred", blockers=[budget_reason(plan, BUDGET_EXHAUSTED_BLOCKER)])
        return "closed"
    row[CLAIMED_FIELD] = True
    return "proceed"


def reference_spent(ledger: Mapping[str, Any], key: str, *, site_id: str) -> Decimal:
    """What the check has spent for ``key`` across every attempt: settled cost where settled, else the reservation."""
    from .beta_policy_screening_plan import money
    from .beta_policy_screening_settlement import settled_cost

    spent = Decimal(0)
    for state in ledger["campaigns"].values():
        if state.get("site_id") != site_id:
            continue
        for row in state["candidates"].values():
            if not is_control(row) or row.get(KEY_FIELD) != key:
                continue
            for attempt in [*(row.get("attempt_history") or []), *([row["attempt"]] if row.get("attempt") else [])]:
                status = attempt.get("reservation_status")
                if status == "released" and not attempt.get("provider_allocation_observed"):
                    continue
                settled = settled_cost(attempt) if status == "committed" else None
                spent += money(attempt["maximum_cost_usd"]) if settled is None else settled
    return spent


def reference_remaining_usd(queue_root: str | Path, scene: Mapping[str, Any]) -> Decimal:
    """What a new plan's reference row may still hold for this scene revision and task contract (at planning)."""
    from .beta_policy_screening_coordinator import _ledger, _lock, _root

    root = _root(queue_root)
    try:
        with _lock(root):
            ledger = _ledger(root)
    except Exception:  # noqa: BLE001 - no queue yet: nothing spent
        return REFERENCE_CAP_USD
    key = reuse_key({"site_id": scene["site_id"], "scene_revision_digest": scene["scene_revision_digest"],
                     "task_success_contract_digest": scene["task_success_contract_digest"]})
    return max(Decimal(0), REFERENCE_CAP_USD - reference_spent(ledger, key, site_id=scene["site_id"]))


def failed_controls(campaigns: Iterable[tuple[str, Mapping[str, Any], Mapping[str, Any]]]) -> list[str]:
    """Campaigns whose own control ran and failed (a reused result is not a new failure)."""
    failed = []
    for campaign_id, state, plan in campaigns:
        for configured in plan["candidates"]:
            outcome = control_outcome(configured, state["candidates"][configured["candidate_id"]]) if is_control(configured) else ""
            if outcome.split(":")[0] == FAILED and not outcome.endswith(":" + REUSED_QUALIFIER):
                failed.append(campaign_id)
    return sorted(failed)


def alert_failed_controls(root: Path, failed: Iterable[str], *, webhook_url: str,
                          poster: Callable[[str, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """Page the owner once per campaign whose control failed, through the capacity controller's route.

    A failed post is tried again on the next pass; a delivered one is never sent again.
    """
    import json

    from .common import write_json

    path = root / ALERT_STATE_FILENAME
    try:
        previous = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    posted = set(previous.get("posted_campaign_ids") or []) if isinstance(previous, dict) else set()
    new = sorted(set(failed) - posted)
    record = {"posted_campaign_ids": sorted(posted), "last_error": None, "posted_now": []}
    if not new:
        return record
    if not webhook_url:
        record["last_error"] = "operator_alert_route_unconfigured"
    else:
        from . import control_plane_capacity_controller as capacity

        alerts = [{"severity": "page", "code": ALERT_CODE, "mount": f"beta screening {campaign_id}",
                   "campaign_id": campaign_id} for campaign_id in new]
        try:
            (poster or capacity.post_alert)(webhook_url, {"level": "critical", "mounts": [], "alerts": alerts})
        except Exception as error:  # noqa: BLE001 - retried next pass; the message may carry the URL
            record["last_error"] = type(error).__name__
        else:
            record.update(posted_campaign_ids=sorted(posted | set(new)), posted_now=new)
    write_json(path, record)
    return record


__all__ = ["ALERT_CODE", "ARM_FAMILY_BODIES", "BUDGET_EXHAUSTED_BLOCKER", "CHOSEN_BY", "CONTROL_FIELD", "DISPLAY_NAME",
           "EPISODES", "FAILED", "FIT_CAPABILITY", "INTERRUPTED", "PACKET_STATION_FLAG", "PARTLY", "PARTLY_AT_SUCCESSES",
           "PASSED", "PASS_AT_SUCCESSES", "PENDING", "PREFERENCE_CAPABILITY", "PREFERENCE_INVALID", "REASONS",
           "REFERENCE_CAP_USD", "REUSED_BLOCKER", "STATION_CAPABILITY", "UNAVAILABLE", "alert_failed_controls",
           "control_cells", "control_flags", "control_gate", "control_outcome", "failed_controls", "first_in_plan",
           "is_control", "plan_control_row", "preferred_model_ids", "reuse_key", "tested_robot_bodies",
           "validate_control_row"]
