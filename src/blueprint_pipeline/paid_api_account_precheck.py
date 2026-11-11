"""The account-wide monthly limit, asked before an API-priced call that no paid lane admitted (WP-23, G48).

Every such call reachable from an automatic job asks here, at the start of the step that would
make it, never between the requests of work already paid for. A refusal sends nothing and
reserves nothing; the caller records it by name and the next pass asks again
(``scripts/paid_api_call_sites.json`` names each caller and what it does with a refusal).
"""

from __future__ import annotations


def account_limit_blockers(resource_class: str = "openai_api_candidate") -> list[str]:
    """``[]`` when new spend may start; else the chokepoint's refusal codes."""

    from .paid_resource_admission import PaidResourceAdmissionBlocked, require_paid_resource_budget

    try:
        require_paid_resource_budget(resource_class=resource_class)
    except PaidResourceAdmissionBlocked as exc:
        return list(exc.blockers)
    return []


__all__ = ["account_limit_blockers"]
