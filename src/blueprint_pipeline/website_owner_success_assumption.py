"""A website job's success thresholds, scored as Blueprint's standard ones until its site confirms them.

A website job's task template carries Blueprint's fixed development controls as
its success thresholds, marked ``proposal_only`` (``website_native_submission``).
The team packet used to refuse such a job until the site owner confirmed them
(``task_evaluation_owner_success_confirmation``): the ask came after the scene
was paid for, with no deadline, and protected none of privacy, rights or money.
Owner direction, launch tracker 2026-11-07: submitting is the confirmation, and
inferences are disclosed as assumptions.

So the website team packet (``team_scene_packet_worker``, the only caller that
asks for it) compiles such a job at once, on exactly those thresholds, and says
what it did:

- The success contract seals the same criteria the owner's confirmed contract
  would (``task_evaluation_rigid_owner_contract.configured_owner_success_criteria``;
  an open/close task's own). Only its provenance differs. The contract schema
  is shared with Arm Decision Proof and is unchanged: it can score only a
  ``confirmed`` contract and has no field for an assumption, so the contract
  names Blueprint where a confirmed one names the site: ``author_source``
  ``agent_proposal``, ``author_id`` ``AUTHOR_ID`` and ``confirmed_by_team_id``
  ``ASSUMED_BY``. ``contract_assumed`` reads that back; every scored episode is
  bound to its contract's digest, so a result scored under the assumption stays
  one whatever happens later.
- The half record, the compile result and the request's status carry ``FIELD``:
  ``status`` ``ASSUMED``, the digests of the thresholds and the proposal, and
  (when the request's intent is bound) the digest of everything a confirmation
  of them is bound to. A half or status without ``FIELD`` assumed nothing.

The site can still confirm, on the job page or through the operator door. The
packet queue's timer pass keeps sending the thresholds of a half compiled this
way to the WebApp (``website_owner_success_confirmation``), and a sealed
confirmation of exactly them returns the completed request to pending once
(``untried_confirmations``); the half is then compiled again as the site's
(``site_confirmation``), with no ``FIELD``, through the path that already
recompiles a half (``beta_policy_screening_scene_half`` plans what has not run
against the new one). Nothing waits for any of it.

An operator's or Arm Decision Proof compile never passes the flag, and refuses a
``proposal_only`` authority exactly as before. No threshold changes here, and
nothing starts compute, calls a provider or moves money.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

#: The half record, compile result and request status field that says the thresholds were assumed.
FIELD = "owner_success_assumption"
ASSUMED = "blueprint_standard_thresholds_assumed_not_confirmed_by_site"
#: The assumed contract's author and the party that stands behind it, in place of the site.
AUTHOR_ID = "blueprint:website_standard_success_thresholds.v1"
ASSUMED_BY = "blueprint:assumed_not_confirmed_by_site"
ARTICULATED = "articulated_open_close"


def assumable(task_spec: Mapping[str, Any]) -> bool:
    """A ``proposal_only`` authority over thresholds that require an owner contract: what a site can confirm."""

    authority, criteria = task_spec.get("configured_owner_authority"), task_spec.get("configured_success_criteria")
    return (isinstance(authority, Mapping) and authority.get("confirmation_status") == "proposal_only"
            and isinstance(criteria, Mapping) and criteria.get("owner_success_contract_required") is True)


def assumed_contract(task_spec: Mapping[str, Any], *, task_kind: str, site_id: str, task_id: str) -> dict[str, Any]:
    """The task's success contract on its thresholds as built, with Blueprint's assumption as its provenance.

    The criteria are the confirmed contract's. Raises ``ValueError`` (a rigid
    task's ``TaskNeutralScoringError``) when the thresholds cannot be sealed,
    as a confirmed compile would.
    """

    from .adp_articulated_task_success_contract import (
        confirm_articulated_task_success_contract,
        seal_task_success_contract,
    )
    from .adp_task_scoring import confirm_rigid_task_success_contract

    articulated = task_kind == ARTICULATED
    if articulated:
        # The same task conditions the owner's confirmed drawer contract requires
        # (``task_evaluation_scene_control_omission.confirmed_articulated_contract``).
        if (task_spec.get("task_kind") != ARTICULATED
                or (task_spec.get("configured_success_criteria") or {}).get("task_joint_drive_forbidden") is not True):
            raise ValueError("owner_success_assumption_articulated_task_invalid")
        criteria = None
    else:
        from .task_evaluation_rigid_owner_contract import configured_owner_success_criteria

        criteria = configured_owner_success_criteria(task_spec)
    proposed = seal_task_success_contract(
        task_kind=task_kind, task_spec=task_spec, site_id=site_id, task_id=task_id, author_source="agent_proposal",
        author_id=AUTHOR_ID, confirmation_status="proposal_only", criteria=criteria)
    confirm = confirm_articulated_task_success_contract if articulated else confirm_rigid_task_success_contract
    contract = confirm(proposed, confirmed_by_team_id=ASSUMED_BY)
    if articulated and contract["criteria"]["opening"]["success_interval"] != (
            task_spec.get("executable_opening_threshold") or {}).get("success_interval"):
        raise ValueError("owner_success_assumption_articulated_threshold_changed")
    return contract


def contract_assumed(contract: Any) -> bool:
    """Whether a sealed success contract was scored on Blueprint's standard thresholds, not confirmed by the site."""

    provenance = contract.get("provenance") if isinstance(contract, Mapping) else None
    return (isinstance(provenance, Mapping) and provenance.get("author_id") == AUTHOR_ID
            and provenance.get("confirmed_by_team_id") == ASSUMED_BY)


def assumption_record(*, task_spec: Mapping[str, Any], revision: Mapping[str, Any], scope: Mapping[str, Any] | None,
                      contract: Mapping[str, Any]) -> dict[str, Any]:
    """What a half compiled on assumed thresholds records under ``FIELD``.

    ``binding_digest`` is the digest of everything the site's confirmation of
    these thresholds is bound to (``task_evaluation_owner_success_confirmation.binding``,
    as the WebApp's proposal carries it); absent when the request's intent is
    not bound, so nothing can be looked up.
    """

    from .task_evaluation_owner_success_confirmation import binding
    from .task_evaluation_scene_intent_contracts import canonical_digest

    authority, criteria = dict(task_spec["configured_owner_authority"]), dict(task_spec["configured_success_criteria"])
    value = {"status": ASSUMED, "task_success_contract_digest": contract["contract_digest"],
             "configured_success_criteria_digest": canonical_digest(json.loads(json.dumps(criteria))),
             "proposal_authority_digest": canonical_digest(json.loads(json.dumps(authority)))}
    if scope is not None:
        try:
            value["binding_digest"] = canonical_digest(binding(intent=scope["intent"], revision=revision,
                                                               criteria=criteria, proposal=authority))
        except (KeyError, TypeError, ValueError):
            pass  # an intent or revision a confirmation cannot be bound to: recorded without one
    return value


def half_assumption(path: Any) -> dict[str, Any] | None:
    """The assumption the compiled half at ``path`` records; None for any other half, or one that does not read."""

    from .team_scene_packet import read_team_scene_inputs

    try:
        if not path:
            return None
        read_team_scene_inputs(str(path))
        value = json.loads(Path(str(path)).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    marker = value.get(FIELD) if isinstance(value, Mapping) else None
    return dict(marker) if isinstance(marker, Mapping) and marker.get("status") == ASSUMED else None


def status_assumed(status: Mapping[str, Any]) -> bool:
    """A completed request whose status says its half was compiled on assumed thresholds."""

    marker = status.get(FIELD)
    return (status.get("status") == "completed" and bool(status.get("team_scene_inputs_path"))
            and isinstance(marker, Mapping) and marker.get("status") == ASSUMED)


def site_confirmation(path: Any, request: Mapping[str, Any]) -> dict[str, Any] | None:
    """The site's sealed confirmation of the thresholds the half at ``path`` assumed (it is then compiled again).

    Only a record under the request's own intent bound to exactly what the half
    recorded counts. A half compiled while its intent was unbound recorded no
    binding: any confirmation of the intent's owner then sends it to the compile,
    which decides.
    """

    from .task_evaluation_owner_success_confirmation import _BINDING_FIELDS, confirmations
    from .task_evaluation_scene_intent_contracts import canonical_digest
    from .team_scene_packet_worker import owner_success_confirmation_scope

    try:
        scope, _unbound = owner_success_confirmation_scope(request)
        rows = [] if scope is None else confirmations(scope["directory"], scope["intent"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    # The intent's few records are listed first: most jobs have none, and the half is then never opened.
    marker = half_assumption(path) if rows else None
    if marker is None:
        return None
    bound = marker.get("binding_digest")
    return next((row for row in rows if bound is None
                 or canonical_digest({field: row[field] for field in _BINDING_FIELDS}) == bound), None)


def untried_confirmations(status: Mapping[str, Any], request: Mapping[str, Any], *, tried: Any) -> list[str]:
    """The site's confirmation of a completed half's assumed thresholds that no compile has tried yet."""

    if not status_assumed(status):
        return []
    row = site_confirmation(status.get("team_scene_inputs_path"), request)
    if row is None or row["confirmation_digest"] in (tried if isinstance(tried, list) else []):
        return []
    return [row["confirmation_digest"]]


__all__ = [
    "ASSUMED",
    "ASSUMED_BY",
    "AUTHOR_ID",
    "FIELD",
    "assumable",
    "assumed_contract",
    "assumption_record",
    "contract_assumed",
    "half_assumption",
    "site_confirmation",
    "status_assumed",
    "untried_confirmations",
]
