"""Refuse OpenAI spend that could never be officially attributed.

``OpenAIProjectCandidateCostAuthority`` can only produce an official per-run
cost if it reserves *before* the money moves, against an exclusively-scoped
project and API key whose attribution window is still zero::

    if float(baseline["total_cost_usd"]) != 0.0:
        raise OpenAICostAuthorityError("openai_cost_scope_baseline_not_zero")

Once a run has spent without that reservation, no later query can isolate its
charge, and the only honest closure left is the conservative full-cap reserve
in ``openai_unattributable_spend``.  That closure exists because this gate did
not.  This module is the forward fix: a lane asks first, and does not spend if
the answer is no.

Two entry points:

``preflight_official_cost_attribution``
    Costs nothing and moves no money.  Answers "would a reservation succeed
    right now?" so a lane's readiness can be checked before it is launched.

``require_official_cost_reservation``
    The fail-closed gate.  Returns a reservation or raises.  A production lane
    that calls this cannot spend unattributably, because there is no path
    through it that both fails and returns.

The scope attestation is deliberately an operator artifact -- the upstream
validator requires ``issued_by_agent`` to be exactly ``False`` -- so an agent
cannot self-authorize attribution.  Provisioning the dedicated project, key,
and admin key file is a human step this module can check but never perform.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import utc_now_iso
from .task_evaluation_supervisor.openai_cost_authority import (
    OpenAICostAuthorityError,
    OpenAIOrganizationCostsClient,
    OpenAIProjectCandidateCostAuthority,
)


PREFLIGHT_SCHEMA_VERSION = "openai_official_cost_preflight.v1"
DEFAULT_PROVIDER_ID = "pigey_external_candidate"
DEFAULT_PAID_RESOURCE_CLASS = "openai_api_candidate"

# A baseline probe needs a nonzero ceiling to be admitted by the authority, but
# the probe itself never spends: it only reads the organization costs endpoint.
_PROBE_CEILING_USD = 0.01


class OpenAIOfficialCostGateError(RuntimeError):
    """Fail-closed refusal to spend OpenAI money that cannot be attributed."""


def _read_attestation(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise OpenAIOfficialCostGateError("openai_cost_scope_attestation_missing")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpenAIOfficialCostGateError(
            "openai_cost_scope_attestation_unreadable"
        ) from exc
    if not isinstance(value, dict):
        raise OpenAIOfficialCostGateError("openai_cost_scope_attestation_unreadable")
    return value


def _authority(
    *,
    scope_attestation_path: str | Path,
    admin_api_key_file: str | Path,
    project_id: str,
    api_key_id: str,
    provider_id: str,
    paid_resource_class: str,
    transport: Callable[..., Mapping[str, Any]] | None,
    wall_clock: Callable[[], datetime],
) -> OpenAIProjectCandidateCostAuthority:
    client = OpenAIOrganizationCostsClient(
        project_id=project_id,
        api_key_id=api_key_id,
        admin_api_key_file=Path(admin_api_key_file),
        transport=transport,
        wall_clock=wall_clock,
    )
    return OpenAIProjectCandidateCostAuthority(
        client=client,
        scope_attestation=_read_attestation(scope_attestation_path),
        provider_id=provider_id,
        paid_resource_class=paid_resource_class,
        wall_clock=wall_clock,
    )


def preflight_official_cost_attribution(
    *,
    scope_attestation_path: str | Path,
    admin_api_key_file: str | Path,
    project_id: str,
    api_key_id: str,
    lane_id: str,
    provider_id: str = DEFAULT_PROVIDER_ID,
    paid_resource_class: str = DEFAULT_PAID_RESOURCE_CLASS,
    transport: Callable[..., Mapping[str, Any]] | None = None,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Answer whether official attribution is possible, for ``$0``.

    Never raises for an unmet precondition: an unmet precondition is the
    answer, and is returned as a blocker so a caller can report every problem
    at once rather than discovering them one launch at a time.
    """

    blockers: list[str] = []
    baseline_cost_usd: float | None = None
    try:
        authority = _authority(
            scope_attestation_path=scope_attestation_path,
            admin_api_key_file=admin_api_key_file,
            project_id=project_id,
            api_key_id=api_key_id,
            provider_id=provider_id,
            paid_resource_class=paid_resource_class,
            transport=transport,
            wall_clock=wall_clock,
        )
    except (OpenAIOfficialCostGateError, OpenAICostAuthorityError) as exc:
        blockers.append(str(exc))
    else:
        # A reserve() dry run is the only faithful probe: it applies the exact
        # window, exclusivity, and zero-baseline rules the real reservation
        # will apply. It reads costs and writes nothing.
        try:
            authority.reserve(
                candidate_id=f"preflight-{lane_id}",
                candidate_evaluation_suite_digest="sha256:" + "0" * 64,
                authorization_receipt_digest="sha256:" + "0" * 64,
                max_cost_usd=_PROBE_CEILING_USD,
            )
        except OpenAICostAuthorityError as exc:
            code = str(exc)
            if code == "openai_cost_scope_baseline_not_zero":
                blockers.append("openai_attribution_window_not_zero")
            else:
                blockers.append(code)
        if baseline_cost_usd is None:
            try:
                snapshot = authority.client.snapshot(
                    start_time=int(
                        datetime(
                            wall_clock().year,
                            wall_clock().month,
                            wall_clock().day,
                            tzinfo=timezone.utc,
                        ).timestamp()
                    ),
                    end_time=int(wall_clock().timestamp()) + 86_400,
                )
            except OpenAICostAuthorityError as exc:
                blockers.append(str(exc))
            else:
                baseline_cost_usd = float(snapshot["total_cost_usd"])

    receipt: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "ready" if not blockers else "blocked",
        "lane_id": str(lane_id),
        "provider_id": provider_id,
        "paid_resource_class": paid_resource_class,
        "project_id": str(project_id),
        "api_key_id": str(api_key_id),
        "baseline_cost_usd": baseline_cost_usd,
        "blockers": sorted(set(blockers)),
        "official_attribution_possible": not blockers,
        "provider_mutation_performed": False,
        "spend_incurred_usd": 0.0,
        "checked_at": utc_now_iso(),
        "proof_effect": "none",
    }
    return receipt


def require_official_cost_reservation(
    *,
    scope_attestation_path: str | Path,
    admin_api_key_file: str | Path,
    project_id: str,
    api_key_id: str,
    lane_id: str,
    candidate_id: str,
    max_cost_usd: float,
    candidate_evaluation_suite_digest: str,
    authorization_receipt_digest: str,
    provider_id: str = DEFAULT_PROVIDER_ID,
    paid_resource_class: str = DEFAULT_PAID_RESOURCE_CLASS,
    transport: Callable[..., Mapping[str, Any]] | None = None,
    wall_clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> Mapping[str, Any]:
    """Reserve official attribution, or refuse to let the lane spend.

    There is no return path that does not carry a reservation, so a caller
    cannot proceed to spend on a failure by ignoring a status field.
    """

    try:
        authority = _authority(
            scope_attestation_path=scope_attestation_path,
            admin_api_key_file=admin_api_key_file,
            project_id=project_id,
            api_key_id=api_key_id,
            provider_id=provider_id,
            paid_resource_class=paid_resource_class,
            transport=transport,
            wall_clock=wall_clock,
        )
        return authority.reserve(
            candidate_id=candidate_id,
            candidate_evaluation_suite_digest=candidate_evaluation_suite_digest,
            authorization_receipt_digest=authorization_receipt_digest,
            max_cost_usd=max_cost_usd,
        )
    except (OpenAIOfficialCostGateError, OpenAICostAuthorityError) as exc:
        raise OpenAIOfficialCostGateError(
            "openai_official_cost_attribution_unavailable:"
            f"{lane_id}:{exc}"
        ) from exc


__all__ = [
    "DEFAULT_PAID_RESOURCE_CLASS",
    "DEFAULT_PROVIDER_ID",
    "PREFLIGHT_SCHEMA_VERSION",
    "OpenAIOfficialCostGateError",
    "preflight_official_cost_attribution",
    "require_official_cost_reservation",
]
