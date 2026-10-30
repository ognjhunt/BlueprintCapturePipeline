"""Stage three of autonomous candidate selection: choose from the published result.

Beta journey: once a site's screening result is published and terminal, this
deterministic rule picks the best evaluated candidate from the result's own
independent scoring fields, records the catalog configuration and policy route
it names and the deployment companies its row lists, and keeps every
candidate's result or reason beside it. No person chooses, and nothing here
re-scores an episode.

The rule:

1. Eligible: published ``completed`` with at least one scored episode, from a
   learned policy. A scripted simulator-state baseline reads state no deployed
   robot has, so it stays a listed reference and is never the choice.
2. Order: higher success rate, then more scored episodes, then fewer void
   episodes, then the plan's order (the intake ranking), then identifier.
3. The first is chosen when it succeeded at least once; candidates equal to it
   on success rate, scored and void episodes are recorded as tied.
4. No eligible candidate: ``no_evaluated_candidate``, with every reason. Eligible
   candidates without a success: ``no_successful_candidate``. Nothing is chosen.

The record is a simulation estimate, never physical or ADP evidence.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest

SCHEMA_VERSION = "website_candidate_selection.v1"
RULE_ID = "success_rate_then_scored_then_void_then_priority_then_identifier.v1"
SELECTION_DIRECTORY = "selections"
CATALOG_FILENAME = "effective_catalog.json"
SCRIPTED = "scripted_simulator_state_baseline"
TERMINAL = frozenset({"completed", "blocked"})


def _partner(catalog_row: Mapping[str, Any] | None, published: Mapping[str, Any],
             bodies: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    deployment = (catalog_row or {}).get("deployment") or {}
    blockers = []
    if catalog_row is None:
        blockers.append("catalog_row_unavailable")
    elif catalog_row["configuration_digest"] != published["configuration_digest"]:
        blockers.append("catalog_configuration_changed_since_plan")
    # A partner is never inferred: no sourced deployment block or company for
    # this exact configuration means none is named.
    if catalog_row is not None and not deployment.get("companies"):
        blockers.append("no_recorded_deployment_partner_for_configuration")
    body = bodies.get(deployment.get("configuration_id")) or {}
    return {"configuration_id": deployment.get("configuration_id"),
            "policy_route_id": deployment.get("policy_route_id"),
            "deployment_companies": list(deployment.get("companies") or []),
            "relationship": deployment.get("relationship"), "evidence_url": deployment.get("evidence_url"),
            "related_teams": list(body.get("related_teams") or []), "partner_blockers": blockers}


def select_candidate(*, publication: Mapping[str, Any], plan: Mapping[str, Any],
                     catalog: Mapping[str, Any] | None, catalog_basis: str,
                     support: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The deterministic choice and every candidate's result or reason; pure."""
    planned = {row["candidate_id"]: (index, row) for index, row in enumerate(plan["candidates"])}
    by_id = {row["candidate_id"]: row for row in (catalog or {}).get("candidates") or []}
    bodies = {row["embodiment_id"]: row for row in (support or {}).get("embodiments") or []}
    rows, eligible = [], []
    for published in publication["candidates"]:
        key = published["candidate_id"]
        index, plan_row = planned[key]
        kind = published.get("execution_evidence_kind") or plan_row.get("execution_evidence_kind")
        blocker = (None if published["status"] == "completed" and published["episodes_scored"] > 0 and kind != SCRIPTED
                   else "scripted_baseline_reference_only" if published["status"] == "completed" and kind == SCRIPTED
                   else "no_scored_episodes" if published["status"] == "completed" else "not_evaluated")
        row = {"candidate_id": key, "display_name": published["display_name"], "priority_rank": index + 1,
               "plan_status": plan_row["status"], "result_status": published["status"],
               "blockers": list(published["blockers"]) or list(plan_row["blockers"]),
               **{field: published[field] for field in (
                   "episodes_planned", "episodes_scored", "episodes_succeeded", "episodes_failed", "episodes_void",
                   "success_rate", "session_result_digest")},
               "choice_blocker": blocker}
        rows.append(row)
        if blocker is None:
            eligible.append(row)
    def order(row):
        return (-row["success_rate"], -row["episodes_scored"], row["episodes_void"], row["priority_rank"],
                row["candidate_id"])
    eligible.sort(key=order)
    for rank, row in enumerate(eligible, start=1):
        row["result_rank"] = rank
    best = eligible[0] if eligible else None
    chosen, tied = None, []
    if best is not None and best["episodes_succeeded"] > 0:
        tied = [row["candidate_id"] for row in eligible[1:] if order(row)[:3] == order(best)[:3]]
        published = next(row for row in publication["candidates"] if row["candidate_id"] == best["candidate_id"])
        chosen = {"candidate_id": best["candidate_id"], "configuration_digest": published["configuration_digest"],
                  "display_name": best["display_name"], "success_rate": best["success_rate"],
                  "episodes_scored": best["episodes_scored"], "episodes_succeeded": best["episodes_succeeded"],
                  **_partner(by_id.get(best["candidate_id"]), published, bodies)}
    value = {"schema_version": SCHEMA_VERSION, "campaign_id": publication["campaign_id"],
             "site_id": publication["site_id"], "plan_digest": plan["plan_digest"],
             "publication_digest": publication["publication_digest"], "publication_revision": publication["revision"],
             "result_status": publication["status"], "catalog_digest": plan["catalog_digest"],
             "catalog_basis": catalog_basis, "rule": RULE_ID, "decided_by": "deterministic_rule_no_human_input",
             "status": ("chosen" if chosen else "no_successful_candidate" if eligible else "no_evaluated_candidate"),
             "chosen": chosen, "tied_with": tied, "candidates": rows,
             "claim_ceiling": publication["claim_ceiling"], "selection_digest": ""}
    value["selection_digest"] = canonical_digest(value, digest_field="selection_digest")
    return value


def retain_campaign_catalog(*, queue_root: Path, campaign_id: str, catalog: Mapping[str, Any]) -> None:
    """Keep the exact effective catalog a plan was frozen from, for the later partner match."""
    from .beta_policy_screening_coordinator import _read, _write

    path = Path(queue_root) / "inputs" / campaign_id / CATALOG_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.exists() or _read(path) != catalog:
        _write(path, catalog)


def _campaign_catalog(root: Path, plan: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    from .beta_policy_candidate_catalog import BetaPolicyCatalogError, load_default_catalog, validate_catalog
    from .beta_policy_screening_coordinator import _read

    path = root / "inputs" / plan["campaign_id"] / CATALOG_FILENAME
    try:
        catalog = validate_catalog(_read(path))
        if catalog["catalog_digest"] == plan["catalog_digest"]:
            return catalog, "campaign_catalog"
    except (OSError, ValueError, BetaPolicyCatalogError):
        pass
    try:
        return load_default_catalog(), "current_default_catalog_not_campaign_catalog"
    except (OSError, ValueError, BetaPolicyCatalogError):
        return None, "catalog_unavailable"


def retain_selections(*, queue_root: str | Path | None = None) -> dict[str, Any]:
    """Write a selection for every acknowledged, terminal campaign; idempotent and unpaid."""
    from .beta_policy_screening_coordinator import _budget, _ledger, _lock, _plan, _read, _root, _write
    from .beta_policy_screening_result import screening_publication
    from .deployment_embodiment_support import support_catalog

    root = _root(queue_root)
    published = []
    with _lock(root):
        ledger = _ledger(root)
        for campaign_id, state in sorted(ledger["campaigns"].items()):
            plan = _plan(root, campaign_id, state["plan_digest"])
            publication = screening_publication(plan=plan, state=state, budget=_budget(ledger, state["site_id"]))
            if publication["status"] in TERMINAL and state["published_digest"] == publication["publication_digest"]:
                published.append((plan, publication))
    if not published:
        return {"written": [], "terminal_published": 0}
    try:
        support = support_catalog()
    except (OSError, ValueError):
        support = None
    directory = root / SELECTION_DIRECTORY
    directory.mkdir(exist_ok=True, mode=0o700)
    written = []
    for plan, publication in published:
        catalog, basis = _campaign_catalog(root, plan)
        value = select_candidate(publication=publication, plan=plan, catalog=catalog, catalog_basis=basis,
                                 support=support)
        path = directory / f"{plan['campaign_id']}.json"
        if not path.exists() or _read(path) != value:
            _write(path, value)
            written.append(plan["campaign_id"])
    return {"written": written, "terminal_published": len(published)}


__all__ = ["RULE_ID", "SCHEMA_VERSION", "retain_campaign_catalog", "retain_selections", "select_candidate"]
