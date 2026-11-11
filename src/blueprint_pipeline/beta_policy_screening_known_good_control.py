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
- It never displaces a candidate. It is planned after every candidate row, only when every
  runnable candidate is already funded and the plan budget and the site's allowance still hold
  its quoted cap; otherwise it is listed ``unavailable`` with ``known_good_control_unavailable:budget``.
  At run time it is launched first when a slot is free, but only while its cap fits beside every
  queued candidate's; otherwise it waits for them (``control_gate``). It is reserved through the
  coordinator's unchanged reservation, so the site limit, the plan budget and the account limit
  bound it like any row, with no retry beyond what any row gets.
- It is planned only for a rigid pick/place task on the rigid lane, with a scripted profile
  registered for the robot of a selected candidate. Otherwise its row says which of
  ``task_kind``, ``backend`` or ``embodiment`` was missing. Nothing is invented: no placement, no
  body and no result.
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

CONTROL_FIELD = "known_good_control"
SCHEMA = "beta_screening_known_good_control.v1"
EPISODES = 3
PASS_AT_SUCCESSES = 1
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

_MARKER = {"schema_version": SCHEMA, "role": "reference_only", "episodes": EPISODES,
           "pass_at_successes": PASS_AT_SUCCESSES}
_TERMINAL = frozenset({"completed", "blocked", "unavailable", "deferred"})
_ACTIVE = frozenset({"preparing", "launching", "running", "reconciliation_required"})
_QUEUED = frozenset({"pending", "waitlisted"})
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

    return frozenset({WAITLIST_BLOCKER, JOB_CEILING_DEFERRED, "beta_screening_site_budget_exhausted", LIMIT_REACHED})


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


def plan_control_row(plan: Mapping[str, Any], catalog: Mapping[str, Any], scene: Mapping[str, Any], *,
                     capabilities: Mapping[str, Any] | None, cost_quotes: Mapping[str, Any] | None,
                     site_remaining_usd: str | None, now: Any) -> dict[str, Any]:
    """The control row a new website campaign's plan appends after every candidate row."""
    from .beta_policy_screening_plan import (
        WAITLIST_BLOCKER, WAITLISTED, BetaPolicyScreeningPlanError, JOB_CEILING_DEFERRED, _candidate_blockers,
        _scene, execution_cost_floor, money, validate_cost_quote,
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
    match = served = None
    for row in plan["candidates"]:
        robot = ((by_id.get(row["candidate_id"]) or {}).get("interface") or {}).get("robot_profile_id")
        if row["status"] != "selected" or not robot:
            continue
        match = next((item for item in scripted if item["interface"].get("robot_profile_id") == robot), None)
        if match is not None:
            served = row["candidate_id"]
            break
    if match is None:
        return _unavailable("embodiment")
    capabilities = capabilities or {}
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
    # The quote is the run's hard cap (hourly rate x hard TTL), whatever its episode count: the same
    # cap re-sealed for this row's episodes, never lowered or raised.
    quote = {**quote, "episode_count": EPISODES, "quote_digest": ""}
    quote["quote_digest"] = canonical_digest(quote, digest_field="quote_digest")
    cap = money(quote["maximum_cost_usd"])
    selected = sum((money(row["max_cost_usd"]) for row in plan["candidates"] if row["status"] == "selected"), Decimal(0))
    unfunded = any(row["status"] == WAITLISTED or row["status"] == "deferred"
                   and row["blockers"] in ([WAITLIST_BLOCKER], [JOB_CEILING_DEFERRED]) for row in plan["candidates"])
    site_left = None if site_remaining_usd is None else money(site_remaining_usd) - selected
    if (unfunded or cap < execution_cost_floor(spec) or selected + cap > money(plan["budget_usd"])
            or site_left is not None and cap > site_left):
        return _unavailable("budget", match)
    return {"candidate_id": match["candidate_id"], "display_name": DISPLAY_NAME,
            "configuration_digest": match["configuration_digest"], "status": "selected", "blockers": [],
            "admission_digest": binding["admission_digest"], "execution_spec": deepcopy(spec),
            "cost_quote": quote, "max_cost_usd": quote["maximum_cost_usd"], "episode_count": EPISODES,
            "scenario_cells": control_cells(scene), "execution_evidence_kind": SCRIPTED,
            CONTROL_FIELD: {**_MARKER, "for_candidate_id": served}}


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
    require(row["status"] == "selected" and set(row) == _SELECTED_FIELDS and row["blockers"] == []
            and set(marker) == {*_MARKER, "for_candidate_id"} and all(marker[key] == value for key, value in _MARKER.items())
            and isinstance(marker["for_candidate_id"], str)
            and any(item["candidate_id"] == marker["for_candidate_id"] and item["status"] == "selected"
                    for item in plan["candidates"] if not is_control(item))
            and row["episode_count"] == EPISODES and isinstance(row["scenario_cells"], list)
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
                 site_limit: Decimal) -> str:
    """Whether a pending control may be reserved now: ``proceed``, ``wait`` or ``reused``.

    Called under the ledger lock; may mark the ledger row (its key, or a reused result) in place.
    ``committed(site_id, campaign_id=None)`` is what funding already counts.
    """
    from .beta_policy_screening_plan import money

    state = ledger["campaigns"][campaign_id]
    configured = next(row for row in plan["candidates"] if is_control(row))
    row = state["candidates"][configured["candidate_id"]]
    key = row.setdefault(KEY_FIELD, reuse_key(plan))
    for other_id, other in sorted(ledger["campaigns"].items()):
        if other["site_id"] != state["site_id"]:
            continue
        for other_key, other_row in other["candidates"].items():
            if other_id == campaign_id or not is_control(other_row) or other_row.get(KEY_FIELD) != key:
                continue
            if other_row["status"] in _ACTIVE:
                return "wait"  # The same check is running; its result is reused, never paid twice.
            other_plan = plans(other_id, other["plan_digest"])
            other_configured = next((item for item in other_plan["candidates"] if item["candidate_id"] == other_key), None)
            outcome = control_outcome(other_configured or {}, other_row).split(":")[0]
            if outcome in {PASSED, FAILED}:
                row.update(status="unavailable", blockers=[REUSED_BLOCKER])
                row[REUSED_FIELD] = {"campaign_id": other_id, "outcome": outcome}
                return "reused"
    queued = sum((money(item["max_cost_usd"]) for item in plan["candidates"] if not is_control(item)
                  and state["candidates"][item["candidate_id"]]["status"] in _QUEUED and "max_cost_usd" in item), Decimal(0))
    cap = money(configured["max_cost_usd"])
    fits = (committed(state["site_id"]) + queued + cap <= site_limit
            and committed(state["site_id"], campaign_id=campaign_id) + queued + cap <= money(plan["budget_usd"]))
    # Never ahead of a candidate the money cannot also hold; with none queued, the reservation decides.
    return "proceed" if fits or not queued else "wait"


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


__all__ = ["ALERT_CODE", "CONTROL_FIELD", "DISPLAY_NAME", "EPISODES", "FAILED", "INTERRUPTED", "PASSED",
           "PASS_AT_SUCCESSES", "PENDING", "REASONS", "REUSED_BLOCKER", "UNAVAILABLE", "alert_failed_controls",
           "append_control_row", "control_cells", "control_flags", "control_gate", "control_outcome",
           "failed_controls", "is_control", "plan_control_row", "reuse_key", "validate_control_row"]
