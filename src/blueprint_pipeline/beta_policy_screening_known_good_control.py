"""One scripted known-good control row per new website screening campaign (owner approval 2026-11-10).

Beta journey: a site's paid candidate screening. Before this, nothing showed that a newly built
scene, its task contract and the evaluation harness could register a success at all, so "every
candidate failed" could not be told apart from "this simulated set-up cannot succeed". A new
website campaign now plans one extra row: Blueprint's scripted pick/place controller
(``scripted_robot_baseline``), which reads exact object and arm poses from the simulator, runs
``EPISODES`` nominal (canonical anchor) episodes in its own session, through the same harness and
scorer as the candidates.

What the row is, and is not:

- It is ``reference_only``. It is never a candidate, never part of the estimate (the WebApp's
  estimate already excludes scripted rows) and its name says it is a check. It is not Arm Decision
  Proof evidence (that program's "positive control" is a separate thing), it does not lift the
  standing controls-stage pause (``task_evaluation_control_stage_policy``) and a pass shows only
  that the simulated set-up can register a success, never physical performance or safety.
- It never gates. Pass, fail, unavailable or interrupted, the candidates run and are reported as
  before; the outcome is a typed flag on the control row's own ``setup_quality_flags``
  (``control_flags``). A failed control alerts the owner once (``alert_failed_controls``).
- Its cap is its body's quoted hard cap, lowered to ``REFERENCE_CAP_USD`` ($3.00, owner,
  2026-11-11) when higher. The lowered quote is fitted by the plan's own rule for a smaller amount
  (``beta_policy_screening_plan._site_fitted_quote``): the row's hard TTL is shortened so it still
  keeps at least half the headroom the original quote kept above hourly rate x TTL, for startup,
  transfer, storage and teardown.
- It runs strictly last and only on money that is left over. At planning it is selected when its
  cap fits what the selected candidates leave of the plan budget and of the site's remaining
  allowance; otherwise it is listed ``unavailable`` (``known_good_control_unavailable:budget``).
  At run time nothing that funds, waitlists, refills, defers or closes a learned candidate counts
  it while it is unreserved (``beta_policy_screening_coordinator._refill_waitlist``, the job's
  local holds): every learned row is decided exactly as if the row did not exist. It is reserved
  only when its campaign, read as if the row did not exist, has ended ``completed`` for good
  (``beta_policy_screening_coordinator._final_without_control``: ``campaign_status`` of the learned
  rows, not superseded, and ``_replan_scope`` would plan nothing again by itself or after any
  deploy, apart from rows closed because the plan budget, the site allowance or the job ceiling could
  not fund them: those count as final, owner decision 2026-11-11), the job's source is still current (the read every paid attempt makes; an owner's closed
  run or ended consent is not), and the sibling-harness hold does not apply. Otherwise it closes at
  once without running or holding money (``known_good_control_interrupted``), so a re-plan and any
  successor see exactly what they would without it. When the same check is active in another
  campaign it closes the same way; a finished pass or fail of it is reused. It never waits on money
  or on another campaign: it waits only while a learned row of its own campaign may still run. Then
  it goes through the coordinator's ordinary reservation, so the plan budget, the site limit, the
  job ceiling and the account limit bound it; if its cap does not fit, it ends at once with
  ``known_good_control_unavailable:budget``. Once reserved its money is real against the site
  limit, like any row's: the owner approved up to $3.00 per job for it. What this costs: if a later
  deploy plans the rows closed for money again in a new campaign, the site's remaining allowance is
  lower by what the row actually spent (at most $3.00). A re-plan and a successor campaign carry no
  reference row, so a job whose first campaign did not finish cleanly for any other reason gets no
  check.
- It is planned only for a rigid pick/place task on the rigid lane, with a registered scripted
  profile for its body. The body is the robot of a selected candidate when one has a scripted
  profile (the marker's ``for_candidate_id``). Otherwise it is the body of a model the job's own
  robot fit table (``website_robot_fit``, passed as the ``robot_fit_rows`` capability) says fits
  this scene: ``fits`` before ``fits_spec``, ``unknown`` only when neither has a registered body,
  never ``doesnt_fit``, ties in the table's order. The body only carries the check: the row tests
  this simulated set-up, never the robot, whichever body it runs on. The website's own picks
  (``preferred_robot_model_ids``, read with the job's ceiling and passed as the
  ``preferred_robot_model_ids`` capability) come first, in their order, among the fitting models
  (``fits`` and ``fits_spec`` together) and among the ``unknown`` ones, never ahead of a fitting
  model; the marker then says whether the body was a preferred one (``preferred``). A model's
  body is ``<recipe>_robotiq_2f85_sim``
  (or its recipe, when that is itself a body). The marker then names the spec-catalog model
  (``for_model_id``) and no candidate. When that body's run is not bound to its own station on this
  scene it stands at the scene packet's (another robot's) station; it still runs, and the marker
  says so with the feasibility check's own station code (``flags``: ``scene_packet_station``).
  Without a table, or with no registered body for any fitting model, there is no fallback. Otherwise
  its row says which of ``task_kind``, ``backend`` or ``embodiment`` was missing. Nothing is
  invented: no placement, no body, no verdict and no result.
- A pass or a fail is reused, never paid for twice, for the same site, scene revision, task
  contract and harness revision (``reuse_key``); while another such control is running, this one
  waits for it.

A plan made before this change, a re-plan and a successor campaign carry no control row and run
exactly as before.
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
SCHEMA = "beta_screening_known_good_control.v1"
EPISODES = 3
PASS_AT_SUCCESSES = 1
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
UNAVAILABLE = "known_good_control_unavailable"
INTERRUPTED = "known_good_control_interrupted"
REASONS = ("task_kind", "embodiment", "backend", "budget")
REUSED_QUALIFIER = "reused"
#: The one blocker a control row that did not run here publishes, whatever the reason: the plan's own
#: "switched off" code, which a WebApp released before this change already reads as a row that was not
#: for running (as it read scripted rows before 2026-11-05). The reason is the row's outcome flag.
NOT_RUN_BLOCKER = "candidate_disabled"
REUSED_BLOCKER = NOT_RUN_BLOCKER
REUSED_FIELD = "known_good_control_reused"
KEY_FIELD = "known_good_control_key"
ALERT_CODE = "beta_screening_known_good_control_failed"
ALERT_STATE_FILENAME = "known_good_control_alerts.json"

#: The planning capabilities the fallback body reads (``beta_policy_screening_capabilities``).
FIT_CAPABILITY = "robot_fit_rows"
STATION_CAPABILITY = "candidate_station_basis"
#: A fitting model's verdicts in the order a body is chosen; ``doesnt_fit`` never is.
FALLBACK_VERDICTS = ("fits", "fits_spec", "unknown")
SCRIPTED_BODY_SUFFIX = "_robotiq_2f85_sim"
#: The website's picks for the job, best first (its answer's pick, then its runner-up).
PREFERENCE_CAPABILITY = "preferred_robot_model_ids"
PREFERENCE_LIMIT = 3
#: The spec-catalog models of the registered scripted body nearest a learned candidate's robot, best first,
#: used ahead of the website's picks and the table's order. Closed: a robot not listed has none.
#: The DROID Franka (Panda with a Robotiq 2F-85, recipe ``droid_franka_robotiq_2f85``) has no scripted
#: body; its Panda model is listed first in case one is registered, then the FR3 with the same Robotiq
#: 2F-85 (``franka_research_3`` -> ``franka_research_3_robotiq_2f85_sim``): same maker, seven joints, same
#: gripper. Its recipe is a distinct asset from the DROID one; it is the nearest, not the same, arm.
NEAREST_BODY_MODELS: Mapping[str, tuple[str, ...]] = {
    "droid_franka_panda_robotiq_2f85_v1": ("franka_emika_panda", "franka_research_3"),
}
CHOSEN_BY = ("candidate_robot", "nearest_body", "preferred", "fit_table")
#: The coordinator's "nothing can fund it" close (``beta_policy_screening_coordinator.BUDGET_EXHAUSTED_BLOCKER``).
BUDGET_EXHAUSTED_BLOCKER = "beta_screening_site_budget_exhausted"
#: The learned rows' states in which they may still need money or run: the control waits on these alone.
_LEARNED_OPEN = frozenset({"pending", "waitlisted", "preparing", "launching", "running", "reconciliation_required"})
#: Logged, never raised, when the website sent a preference this planner cannot use.
PREFERENCE_INVALID = "known_good_control_preference_invalid"
#: The feasibility check's own station code for a body bound to the scene packet's station, not its own.
PACKET_STATION_FLAG = PACKET_STATION

_MARKER = {"schema_version": SCHEMA, "role": "reference_only", "episodes": EPISODES,
           "pass_at_successes": PASS_AT_SUCCESSES}
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


def _fallback_body(scripted: list[Mapping[str, Any]], fit: Any,
                   preferred: Any = None) -> tuple[str, Mapping[str, Any], bool] | None:
    """The one registered scripted row for a model the job's fit table says fits; its model id with it,
    and whether that model was one the website preferred."""
    if not isinstance(fit, list):
        return None
    bodies: dict[str, Mapping[str, Any]] = {}
    for row in scripted:
        bodies.setdefault(row["interface"].get("robot_profile_id"), row)
    ranked = [entry for entry in fit if isinstance(entry, Mapping)]
    order = list(preferred) if isinstance(preferred, list) else []
    # fits and fits_spec both fit on what was checked: one tier, where the website's pick goes first.
    for tier in (("fits", "fits_spec"), ("unknown",)):
        rows = sorted((entry for entry in ranked if entry.get("verdict") in tier),
                      key=lambda entry: FALLBACK_VERDICTS.index(entry["verdict"]))  # stable: table order within
        rows = ([entry for model in order for entry in rows if entry.get("model_id") == model]
                + [entry for entry in rows if entry.get("model_id") not in order])
        for entry in rows:
            model, recipe = entry.get("model_id"), entry.get("recipe_id")
            if not isinstance(model, str) or not isinstance(recipe, str) or not recipe:
                continue
            for body in dict.fromkeys((recipe if recipe.endswith(SCRIPTED_BODY_SUFFIX) else recipe + SCRIPTED_BODY_SUFFIX,
                                       recipe)):
                if body in bodies:
                    return model, bodies[body], model in order
    return None


def plan_control_row(plan: Mapping[str, Any], catalog: Mapping[str, Any], scene: Mapping[str, Any], *,
                     capabilities: Mapping[str, Any] | None, cost_quotes: Mapping[str, Any] | None,
                     site_remaining_usd: str | None, now: Any) -> dict[str, Any]:
    """The control row a new website campaign's plan appends after every candidate row."""
    from .beta_policy_screening_plan import (
        _HOSTED_COST_FIELDS, BetaPolicyScreeningPlanError, _candidate_blockers, _scene, _site_fitted_quote,
        execution_cost_floor, fitted_headroom_usd, money, validate_cost_quote,
    )
    from .task_family_policy_execution import FAMILY_BACKEND

    if scene.get("evaluation_backend") == FAMILY_BACKEND:
        return _unavailable("backend")
    if scene.get("task_kind") != RIGID_TASK_KIND:
        return _unavailable("task_kind")
    scene = _scene(scene)
    by_id = {row["candidate_id"]: row for row in catalog["candidates"]}
    planned = {row["candidate_id"] for row in plan["candidates"]}
    scripted = [row for row in catalog["candidates"] if row["policy"]["kind"] == "scripted"
                and (row.get("interface") or {}).get("policy_interface_id") == SCRIPTED_INTERFACE
                and row["candidate_id"] not in planned]
    capabilities = capabilities or {}
    match, stands_for, robots = None, {}, []
    for row in plan["candidates"]:
        robot = ((by_id.get(row["candidate_id"]) or {}).get("interface") or {}).get("robot_profile_id")
        if row["status"] != "selected" or not robot:
            continue
        robots.append(robot)
        match = next((item for item in scripted if item["interface"].get("robot_profile_id") == robot), None)
        if match is not None:
            stands_for = {"for_candidate_id": row["candidate_id"], "chosen_by": "candidate_robot"}
            break
    if match is None:
        # No selected candidate's robot has a scripted row: a body the site's fit table says fits, the one
        # nearest the candidates' own robot first, then the website's picks, then the table's order.
        preference = capabilities.get(PREFERENCE_CAPABILITY)
        nearest = list(dict.fromkeys(model for robot in robots for model in NEAREST_BODY_MODELS.get(robot, ())))
        chosen = _fallback_body(scripted, capabilities.get(FIT_CAPABILITY),
                                nearest + [model for model in preference or [] if model not in nearest]
                                if isinstance(preference, list) else nearest)
        if chosen is None:
            return _unavailable("embodiment")
        model, match, _ = chosen
        rule = ("nearest_body" if model in nearest else "preferred"
                if isinstance(preference, list) and model in preference else "fit_table")
        stands_for = {"for_model_id": model, "chosen_by": rule}
        if isinstance(preference, list) and preference:
            # Only a job the website sent a preference for says whether its pick was followed.
            stands_for["preferred"] = model in preference
        if (capabilities.get(STATION_CAPABILITY) or {}).get(match["candidate_id"]) != OWN_STATION:
            stands_for["flags"] = [PACKET_STATION_FLAG]
    binding = (capabilities.get("candidate_bindings") or {}).get(match["candidate_id"])
    refused = _candidate_blockers(match, scene, binding, set(capabilities.get("credentials_available") or []))
    refused += list((capabilities.get("candidate_blockers") or {}).get(match["candidate_id"]) or [])
    refused += list((capabilities.get("candidate_infeasible") or {}).get(match["candidate_id"]) or [])
    if refused:
        return _unavailable("embodiment", match)
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
    if money(quote["maximum_cost_usd"]) > REFERENCE_CAP_USD:
        if _HOSTED_COST_FIELDS & set(spec):
            return _unavailable("budget", match)
        original_floor = execution_cost_floor(spec)
        bound = REFERENCE_CAP_USD - fitted_headroom_usd(quote, floor=original_floor)
        ttl = min(spec["hard_ttl_seconds"], int(bound * 3600 / money(spec["max_hourly_rate_usd"]))) if bound > 0 else 0
        fitted = None
        if ttl >= MIN_HARD_TTL_SECONDS:
            spec = {**spec, "hard_ttl_seconds": ttl}
            fitted = _site_fitted_quote(quote, REFERENCE_CAP_USD, floor=execution_cost_floor(spec),
                                        original_floor=original_floor, reason="fitted_to_scripted_reference_cap",
                                        suffix="-reference-cap")
        if fitted is None:
            return _unavailable("budget", match)
        admission, quote = canonical_digest(spec), fitted
    quote = {**quote, "episode_count": EPISODES, "quote_digest": ""}
    quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
    cap = money(quote["maximum_cost_usd"])
    selected = sum((money(row["max_cost_usd"]) for row in plan["candidates"] if row["status"] == "selected"), Decimal(0))
    left = money(plan["budget_usd"]) - selected
    if site_remaining_usd is not None:
        left = min(left, money(site_remaining_usd) - selected)
    # Planned on what the selected candidates leave; at run time it is reserved only after every learned
    # row has ended, from what is left then (``control_gate``), so it never takes a candidate's money.
    if cap < execution_cost_floor(spec) or cap > left:
        return _unavailable("budget", match)
    return {"candidate_id": match["candidate_id"], "display_name": DISPLAY_NAME,
            "configuration_digest": match["configuration_digest"], "status": "selected", "blockers": [],
            "admission_digest": admission, "execution_spec": deepcopy(spec),
            "cost_quote": quote, "max_cost_usd": quote["maximum_cost_usd"], "episode_count": EPISODES,
            "scenario_cells": control_cells(scene), "execution_evidence_kind": SCRIPTED,
            CONTROL_FIELD: {**_MARKER, **stands_for}}


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
    if row["status"] == "unavailable":
        require(set(row) == _UNAVAILABLE_FIELDS and set(marker) == {*_MARKER, "unavailable"}
                and all(marker[key] == value for key, value in _MARKER.items())
                and marker["unavailable"] in REASONS and row["blockers"] == [NOT_RUN_BLOCKER])
        return Decimal(0)
    # What the row stands for: a selected candidate's robot, or (since the fit-table fallback) a
    # spec-catalog model, with the station flag only then. Never both, never neither.
    stands_for = set(marker) - set(_MARKER)
    require(row["status"] == "selected" and set(row) == _SELECTED_FIELDS and row["blockers"] == []
            and set(_MARKER) <= set(marker) and all(marker[key] == value for key, value in _MARKER.items()))
    if "for_candidate_id" in stands_for:
        require(stands_for <= {"for_candidate_id", "chosen_by"}
                and marker.get("chosen_by", "candidate_robot") == "candidate_robot"
                and isinstance(marker["for_candidate_id"], str)
                and any(item["candidate_id"] == marker["for_candidate_id"] and item["status"] == "selected"
                        for item in plan["candidates"] if not is_control(item)))
    else:
        require("for_model_id" in stands_for and stands_for <= {"for_model_id", "flags", "preferred", "chosen_by"}
                and marker.get("chosen_by", "fit_table") in CHOSEN_BY[1:]
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
    return money(row["max_cost_usd"])


def append_control_row(plan: dict[str, Any], catalog: Mapping[str, Any], scene: Mapping[str, Any],
                       options: Mapping[str, Any]) -> dict[str, Any]:
    """A new website campaign's plan with its control row last; any other plan unchanged."""
    from .beta_policy_screening_plan import money

    if options.get("candidate_ids") is not None or any(is_control(row) for row in plan["candidates"]):
        return plan
    row = plan_control_row(plan, catalog, scene, capabilities=options.get("capabilities"),
                           cost_quotes=options.get("cost_quotes"), site_remaining_usd=options.get("site_remaining_usd"),
                           now=options.get("now"))
    plan["candidates"].append(row)
    if row["status"] == "selected":
        plan["maximum_planned_cost_usd"] = f"{money(plan['maximum_planned_cost_usd']) + money(row['max_cost_usd']):.2f}"
    return plan


def control_outcome(configured: Mapping[str, Any], row: Mapping[str, Any]) -> str:
    """The control's typed outcome from its plan row and ledger row; ``PENDING`` until the row ends."""
    if row.get("status") not in _TERMINAL:
        return PENDING
    reused = row.get(REUSED_FIELD)
    if isinstance(reused, Mapping) and reused.get("outcome") in {PASSED, FAILED}:
        return f"{reused['outcome']}:{REUSED_QUALIFIER}"
    reason = (configured.get(CONTROL_FIELD) or {}).get("unavailable")
    if reason in REASONS:
        return f"{UNAVAILABLE}:{reason}"
    if row["status"] == "completed":
        summary = row.get("summary") or {}
        if summary.get("episodes_succeeded", 0) >= PASS_AT_SUCCESSES:
            return PASSED
        # A fail is every planned episode scored, none a success; anything less did not finish.
        return FAILED if summary.get("episodes_scored", 0) == configured.get("episode_count") else INTERRUPTED
    if set(row.get("blockers") or []) & _budget_blockers():
        return f"{UNAVAILABLE}:budget"
    return INTERRUPTED


def control_flags(configured: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, list[str]]:
    """The control row's ``setup_quality_flags``: its one outcome code. Nothing for any other row."""
    return {"setup_quality_flags": [control_outcome(configured, row)]} if is_control(configured) else {}


def reuse_key(plan: Mapping[str, Any]) -> str:
    """What makes two controls the same check: site, scene revision, task contract and harness revision."""
    from .beta_policy_screening_plan import plan_harness_revision

    return canonical_digest({"schema_version": SCHEMA, "site_id": plan["site_id"],
                             "scene_revision_digest": plan["scene_revision_digest"],
                             "task_success_contract_digest": plan["task_success_contract_digest"],
                             "screening_harness_revision": plan_harness_revision(plan)})


def control_gate(ledger: Mapping[str, Any], campaign_id: str, plan: Mapping[str, Any], *,
                 plans: Callable[[str, str], Mapping[str, Any]], committed: Callable[..., Decimal],
                 site_limit: Decimal, final: Callable[[], bool]) -> str:
    """Whether a pending control may be reserved now: ``proceed``, ``wait``, ``reused`` or ``closed``.

    Called under the ledger lock; may mark the ledger row (its key, a reused result, or its close) in place.
    ``committed(site_id, campaign_id=None)`` is what funding already counts. ``final()``: whether the
    campaign, as if the row did not exist, has ended ``completed`` for good, with nothing a re-plan or a
    successor would take up (``beta_policy_screening_coordinator._final_without_control``).

    It waits only while a learned row of its own campaign may still need money or run. Then, once, it
    reuses a finished pass or fail of the same check, or is closed without running (as not run:
    ``known_good_control_interrupted``) when the same check is active in another campaign or the campaign
    without it did not end ``completed`` for good, or closes for budget when its cap does not fit, or
    proceeds to the ordinary reservation. It never waits on money or on another campaign.
    """
    from .beta_policy_screening_plan import money

    state = ledger["campaigns"][campaign_id]
    configured = next(row for row in plan["candidates"] if is_control(row))
    row = state["candidates"][configured["candidate_id"]]
    if any(state["candidates"][item["candidate_id"]]["status"] in _LEARNED_OPEN
           for item in plan["candidates"] if not is_control(item)):
        return "wait"
    key = row.setdefault(KEY_FIELD, reuse_key(plan))
    for other_id, other in sorted(ledger["campaigns"].items()):
        if other["site_id"] != state["site_id"]:
            continue
        for other_key, other_row in other["candidates"].items():
            if other_id == campaign_id or not is_control(other_row) or other_row.get(KEY_FIELD) != key:
                continue
            if other_row["status"] in _ACTIVE:
                # The same check runs elsewhere: this one is not paid for twice, and never waits for it.
                row.update(status="unavailable", blockers=[NOT_RUN_BLOCKER])
                return "closed"
            other_plan = plans(other_id, other["plan_digest"])
            other_configured = next((item for item in other_plan["candidates"] if item["candidate_id"] == other_key), None)
            outcome = control_outcome(other_configured or {}, other_row).split(":")[0]
            if outcome in {PASSED, FAILED}:
                row.update(status="unavailable", blockers=[REUSED_BLOCKER])
                row[REUSED_FIELD] = {"campaign_id": other_id, "outcome": outcome}
                return "reused"
    if not final():
        # Something of the campaign may still be retried, held for a deploy or planned again: the row never
        # runs or holds money beside it, so the re-plan and any successor see exactly what they would without it.
        row.update(status="unavailable", blockers=[NOT_RUN_BLOCKER])
        return "closed"
    cap = money(configured["max_cost_usd"])
    if (committed(state["site_id"]) + cap <= site_limit
            and committed(state["site_id"], campaign_id=campaign_id) + cap <= money(plan["budget_usd"])):
        return "proceed"
    row.update(status="deferred", blockers=[BUDGET_EXHAUSTED_BLOCKER])
    return "closed"


def failed_controls(campaigns: Iterable[tuple[str, Mapping[str, Any], Mapping[str, Any]]]) -> list[str]:
    """Campaigns whose own control ran and failed (a reused result is not a new failure)."""
    failed = []
    for campaign_id, state, plan in campaigns:
        for configured in plan["candidates"]:
            if is_control(configured) and control_outcome(configured, state["candidates"][configured["candidate_id"]]) == FAILED:
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


__all__ = ["ALERT_CODE", "CONTROL_FIELD", "DISPLAY_NAME", "EPISODES", "FAILED", "FIT_CAPABILITY", "INTERRUPTED",
           "PACKET_STATION_FLAG", "PASSED", "PREFERENCE_CAPABILITY", "PREFERENCE_INVALID", "STATION_CAPABILITY",
           "PASS_AT_SUCCESSES", "PENDING", "REFERENCE_CAP_USD", "REASONS", "REUSED_BLOCKER", "UNAVAILABLE", "alert_failed_controls",
           "append_control_row", "control_cells", "control_flags", "control_gate", "control_outcome",
           "failed_controls", "is_control", "BUDGET_EXHAUSTED_BLOCKER", "NEAREST_BODY_MODELS", "CHOSEN_BY", "plan_control_row", "preferred_model_ids", "reuse_key", "validate_control_row"]
