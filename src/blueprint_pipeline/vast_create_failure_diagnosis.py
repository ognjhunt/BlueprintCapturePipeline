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

from typing import Any, Callable, Mapping, Sequence

ApiJson = Callable[..., tuple[int, Mapping[str, Any]]]


def diagnose_empty_create_400(
    *,
    api_json: ApiJson,
    api_key: str,
    search_request: Mapping[str, Any],
    offer_id: int,
    attempted_label: str,
    offers_from_response: Callable[[Mapping[str, Any]], list[Any]],
    instance_rows: Callable[[Mapping[str, Any]], list[Mapping[str, Any]]],
    attempted_labels: Sequence[str] = (),
    instance_label: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    """Retain catalog and inventory observations; neither turns an empty 400 into a refusal.

    The legacy create_produced_no_instance field means verified inventory
    absence only. Reselection additionally requires an explicit definite
    refusal. All attempted labels, including earlier retries, remain in scope.
    """
    labels = sorted({label for label in (*attempted_labels, attempted_label) if label})

    diagnosis: dict[str, Any] = {
        "selected_offer_absent_from_fresh_search": False,
        "catalog_readback_http_status_code": None,
        "catalog_readback_offer_count": None,
        "catalog_readback_error": None,
        "create_produced_no_instance": False,
        "create_inventory_verified": False,
        "attempted_labels": labels,
        "matching_attempt_instance_ids": [],
        "inventory_absence_is_not_definitive_create_refusal": True,
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
        raw = response.get("instances")
        if not 200 <= int(status_code or 0) < 300 or not isinstance(raw, (list, Mapping)):
            raise ValueError("vast_create_inventory_unrecognized")
        rows = instance_rows(response)
        if isinstance(raw, list):
            raw_rows = raw
        elif any(key in raw for key in ("id", "instance_id", "contract_id", "actual_status", "cur_state", "status", "intended_status")):
            raw_rows = [raw]
        else:
            raw_rows = list(raw.values())
        malformed = len(rows) != len(raw_rows) or not all(isinstance(row, Mapping) for row in raw_rows)
        matches = set()
        for row in rows:
            identifier = row.get("id") or row.get("instance_id") or row.get("contract_id")
            valid_id = (not isinstance(identifier, bool) and str(identifier).isdigit() and int(identifier) > 0)
            label = instance_label(row)
            if not valid_id or not isinstance(label, str) or not label.strip():
                malformed = True
                continue
            if label in labels:
                matches.add(int(identifier))
        diagnosis["matching_attempt_instance_ids"] = sorted(matches)
        if malformed:
            raise ValueError("vast_create_inventory_identity_incomplete")
        diagnosis["create_inventory_verified"] = True
        diagnosis["create_produced_no_instance"] = bool(labels and not matches)

    except Exception as exc:  # noqa: BLE001 - an unreadable listing proves nothing
        diagnosis["create_inventory_error"] = (str(exc) if isinstance(exc, ValueError) else type(exc).__name__)

    return diagnosis


__all__ = ["diagnose_empty_create_400"]
