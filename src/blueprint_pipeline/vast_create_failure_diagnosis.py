"""Decide whether a Vast create that failed may safely be retried elsewhere.

Vast answers an unusable ask with an empty HTTP 400 and no body. Creation names
one ask -- ``PUT /asks/{id}/`` -- so an offer must be selected first, and
selection and creation are separate calls against a marketplace that moves
between them. An adapter that cannot get past one bad ask throws away its whole
search: scene 839873 stopped on its first failed offer with 100 qualifying
offers left unused.

Two independent questions decide what happened, and the answers must be
recorded either way -- the failure that motivated this module recorded only
"empty 400", which cannot distinguish a vanished offer from a rejected payload.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

ApiJson = Callable[..., tuple[int, Mapping[str, Any]]]


def diagnose_empty_create_400(
    *,
    api_json: ApiJson,
    api_key: str,
    search_request: Mapping[str, Any],
    offer_id: int,
    attempted_label: str,
    offers_from_response: Callable[[Mapping[str, Any]], list[Any]],
    active_instance_rows: Callable[[Mapping[str, Any]], list[Mapping[str, Any]]],
    instance_label: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    """Ask the provider what actually happened, and record both answers.

    Returns the diagnosis. ``selected_offer_absent_from_fresh_search`` proves
    the documented stale-offer race. ``create_produced_no_instance`` proves the
    create mutated nothing, which is what makes trying the next offer free of
    any double-allocation risk. Either probe failing proves nothing and leaves
    its flag false, so the caller still fails closed.
    """

    diagnosis: dict[str, Any] = {
        "selected_offer_absent_from_fresh_search": False,
        "catalog_readback_http_status_code": None,
        "catalog_readback_offer_count": None,
        "catalog_readback_error": None,
        "create_produced_no_instance": False,
        "create_inventory_http_status_code": None,
        "create_inventory_error": None,
    }

    try:
        status_code, response = api_json(
            method="POST",
            path="/bundles/",
            api_key=api_key,
            payload={**search_request, "limit": 1, "id": {"eq": int(offer_id)}},
            timeout_seconds=45,
        )
        offer_count = len(offers_from_response(response))
        diagnosis["catalog_readback_http_status_code"] = status_code
        diagnosis["catalog_readback_offer_count"] = offer_count
        diagnosis["selected_offer_absent_from_fresh_search"] = bool(
            200 <= int(status_code or 0) < 300 and offer_count == 0
        )
    except Exception as exc:  # noqa: BLE001 - a failed readback is not evidence
        diagnosis["catalog_readback_error"] = type(exc).__name__

    try:
        status_code, response = api_json(
            method="GET",
            path="/instances/",
            api_key=api_key,
            timeout_seconds=30,
        )
        diagnosis["create_inventory_http_status_code"] = status_code
        diagnosis["create_produced_no_instance"] = bool(
            attempted_label
            and 200 <= int(status_code or 0) < 300
            and not any(
                instance_label(row) == attempted_label
                for row in active_instance_rows(response)
            )
        )
    except Exception as exc:  # noqa: BLE001 - an unreadable listing proves nothing
        diagnosis["create_inventory_error"] = type(exc).__name__

    return diagnosis


__all__ = ["diagnose_empty_create_400"]
