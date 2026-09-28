"""Producer evidence for an inventory throttle before any provider launch.

This only classifies a terminal no-allocation outcome. The scene controller
still reopens fresh global zero, ownership, retry count and aggregate spend
before reserving a distinct successor; consumed authority is never released.
"""
from collections.abc import Mapping

from .decision_evidence_contracts import canonical_digest


def _snapshots(value):
    if not isinstance(value, list) or len(value) != 2:
        return None
    rows = []
    for row, label, prefix in zip(value, ("scoped", "global"), ("blueprint-sam31-source-tracks-", ""), strict=True):
        if (not isinstance(row, Mapping) or row.get("label") != label or row.get("name_prefix") != prefix
                or not isinstance(row.get("inventory"), Mapping)):
            return None
        rows.append(row["inventory"])
    return rows


def _zero(row):
    return (row.get("api_confirmed") is True and type(row.get("live_resource_count")) is int
            and row["live_resource_count"] == 0 and row.get("resources") == [])


def proven_prelaunch_inventory_throttle(*, receipt, bound_request, result, cleanup, watchdog,
                                       launch_evidence_present):
    """Recognize only exact, newly produced prelaunch proof and closed cleanup."""
    if (launch_evidence_present or receipt.get("schema_version") != "semantic_sam31_prelaunch_inventory_block.v1"
            or receipt.get("receipt_digest") != canonical_digest(receipt, digest_field="receipt_digest")
            or receipt.get("status") != "blocked" or receipt.get("provider") != "vast"
            or receipt.get("blocker") != "sam31_provider_not_zero_before_launch"
            or receipt.get("provider_launch_invoked") is not False
            or receipt.get("failure_phase") != "prelaunch_inventory_read"
            or type(receipt.get("provider_mutations_performed")) is not int
            or receipt.get("provider_mutations_performed") != 0
            or not bound_request.get("request_digest")
            or bound_request.get("bound_request_digest") != canonical_digest(bound_request, digest_field="bound_request_digest")
            or any(receipt.get(key) != bound_request.get(key) for key in ("request_digest", "bound_request_digest"))
            or result.get("status") != "failed" or result.get("instance_id") is not None
            or type(result.get("provider_mutations_performed")) is not int
            or result.get("provider_mutations_performed") != 0
            or result.get("allocation_outcome_ambiguous") is True
            or result.get("provider_mutation_outcome_ambiguous") is True
            or result.get("provider_zero_verified") is not True
            or cleanup.get("all_objects_absent") is not True
            or watchdog.get("status") != "cancelled_no_allocation"):
        return False
    before = _snapshots(receipt.get("inventory_snapshots"))
    after = _snapshots(receipt.get("postfailure_inventory_snapshots"))
    if before is None or after is None or not all(_zero(row) for row in after):
        return False
    if receipt.get("postfailure_inventory_digest") != canonical_digest(
            {"inventory_snapshots": receipt["postfailure_inventory_snapshots"]},
            digest_field="postfailure_inventory_digest"):
        return False
    throttled = [row.get("http") == 429 and row.get("api_confirmed") is False
                 and row.get("live_resource_count") is None and row.get("resources") == [] for row in before]
    return any(throttled) and all(limited or _zero(row) for row, limited in zip(before, throttled, strict=True))
