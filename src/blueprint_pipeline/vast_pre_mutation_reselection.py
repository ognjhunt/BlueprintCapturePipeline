"""Bounded re-selection of a Vast offer the provider refused before any mutation.

Every single-attempt paid lane seals its environment with one paid attempt, zero
automatic retries and a hard per-attempt cap, and the adapter enforces that on
the instance it creates.  Until 2026-09-05 the lanes also pinned
``BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS`` to ``0``, so when Vast
refused the create itself -- HTTP 400 ``no_such_ask`` because the selected offer
had just been rented by someone else, or 404/409/410 -- the attempt ended with
``vast_api_http_error`` although nothing had been created or spent.  Submission
#9 of scene 841757 lost a whole submission cycle to one stale RTX 4090 offer.

A refused create is not a paid attempt: no provider mutation happened, no
instance exists, no scientific attempt was consumed.  The adapter's re-selection
path excludes the refused machine, searches again and tries the next offer,
bounded by this constant and recorded per refusal in ``create_retry_attempts``;
the paid authority is still consumed once and at most one instance ever exists.
Every lane takes the bound from here so the doctrine has one home.
"""

from __future__ import annotations

RESELECTION_ENV = "BLUEPRINT_VAST_CREATE_STALE_OFFER_RETRY_ATTEMPTS"
PRE_MUTATION_OFFER_RESELECTION_ATTEMPTS = 2


def pre_mutation_offer_reselection_attempts() -> str:
    """The sealed-environment value: how many refused creates may re-select an offer."""

    return str(PRE_MUTATION_OFFER_RESELECTION_ATTEMPTS)


__all__ = ["PRE_MUTATION_OFFER_RESELECTION_ATTEMPTS", "RESELECTION_ENV", "pre_mutation_offer_reselection_attempts"]
