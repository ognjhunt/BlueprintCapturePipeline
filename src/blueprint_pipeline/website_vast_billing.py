"""Reconcile completed website GPU rentals from Vast's per-instance charges."""

from __future__ import annotations

import json
import math
from pathlib import Path
from urllib.parse import urlencode

from .common import write_json
from .decision_evidence_contracts import canonical_digest


def _verified_attempt(root: Path, task_context: dict) -> tuple[dict, dict, dict]:
    request = json.loads((root / "request.json").read_text())
    admission = json.loads((root / "controller_admission.json").read_text())
    operation = root / "reconstruction_vast_operation"
    execution = json.loads((operation / "reconstruction_vast_operation_execution.json").read_text())
    teardown = json.loads((operation / "teardown_receipt.json").read_text())
    provider_zero = json.loads((operation / "provider_zero_verification.json").read_text())
    if (execution.get("status") not in {"failed", "completed"}
            or execution.get("provider") != "vast"
            or execution.get("instance_id") is None
            or execution.get("request_digest") != request.get("request_digest")
            or teardown.get("request_digest") != request.get("request_digest")
            or provider_zero.get("request_digest") != request.get("request_digest")
            or execution.get("execution_result_digest") != canonical_digest(execution, digest_field="execution_result_digest")
            or execution.get("teardown_receipt_digest") != teardown.get("teardown_receipt_digest")
            or teardown.get("teardown_receipt_digest") != canonical_digest(teardown, digest_field="teardown_receipt_digest")
            or execution.get("provider_zero_digest") != provider_zero.get("provider_zero_digest")
            or provider_zero.get("provider_zero_digest") != canonical_digest(provider_zero, digest_field="provider_zero_digest")
            or teardown.get("status") != "PASS" or provider_zero.get("status") != "PASS"
            or execution.get("provider_zero_verified") is not True
            or admission.get("provider") != "vast" or admission.get("resource_class") != "gpu_render"
            or admission.get("task_context_digest") != task_context.get("context_digest")
            or not str(admission.get("allocation_binding_digest") or "").startswith("sha256:")):
        raise ValueError("website_vast_billing_attempt_binding_invalid")
    instance_id = str(execution["instance_id"])
    if not instance_id.isdecimal() or int(instance_id) <= 0:
        raise ValueError("website_vast_billing_instance_invalid")
    return admission, execution, teardown


def _provider_charge(instance_id: str, *, preflight_epoch: float, teardown_epoch: float) -> dict:
    from .gpu_render_providers import _read_secret
    from .vast_provider_adapter import _api_json

    key = _read_secret("vast_api_key")
    if not key:
        raise ValueError("website_vast_billing_key_unavailable")
    start = int(preflight_epoch // 86400) * 86400 - 86400
    end = int(teardown_epoch // 86400) * 86400 + 86400
    filters = json.dumps({"day": {"gte": start, "lte": end}, "type": {"in": ["instance"]}},
                         separators=(",", ":"))
    rows: list[dict] = []
    cursor = None
    seen_cursors = set()
    for _ in range(20):
        params = {"select_filters": filters, "limit": "500"}
        if cursor:
            params["after_token"] = cursor
        status, response = _api_json(method="GET", path="/charges?" + urlencode(params),
                                     api_key=key, timeout_seconds=30)
        if status != 200 or response.get("success") is not True or not isinstance(response.get("results"), list):
            raise ValueError("website_vast_billing_provider_unavailable")
        for row in response["results"]:
            if row.get("source") == f"instance-{instance_id}":
                rows.append(row)
        cursor = response.get("next_token")
        if not cursor:
            break
        if cursor in seen_cursors:
            raise ValueError("website_vast_billing_pagination_invalid")
        seen_cursors.add(cursor)
    else:
        raise ValueError("website_vast_billing_pagination_exhausted")
    if not rows or len({(row.get("source"), row.get("start"), row.get("end")) for row in rows}) != len(rows):
        raise ValueError("website_vast_billing_charge_missing_or_duplicate")
    total = 0.0
    safe_rows = []
    for row in rows:
        amount = row.get("amount")
        if (row.get("type") != "instance" or isinstance(amount, bool)
                or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0
                or not isinstance(row.get("start"), (int, float))
                or not isinstance(row.get("end"), (int, float))):
            raise ValueError("website_vast_billing_charge_invalid")
        total += amount
        safe_rows.append({"source": row["source"], "start": row["start"],
                          "end": row["end"], "amount_usd": amount,
                          "items": [{"type": item.get("type"), "amount_usd": item.get("amount")}
                                    for item in row.get("items", [])]})
    safe_rows.sort(key=lambda row: (row["start"], row["end"]))
    receipt = {"schema_version": "website_vast_provider_charge.v1", "provider": "vast",
               "instance_id": instance_id, "source": f"instance-{instance_id}",
               "amount_usd": round(total, 6), "rows": safe_rows}
    receipt["provider_charge_receipt_digest"] = canonical_digest(receipt, digest_field="provider_charge_receipt_digest")
    return receipt


def settle_prior_website_vast_attempts(*, output_root: Path, attempted_count: int, task_context: dict) -> None:
    """Release unused quotes only after closed rental and provider charge proof."""
    from datetime import datetime
    from .website_task_context import website_webapp_request

    for index in range(attempted_count):
        root = output_root / ("controller_geometry" if index == 0 else f"controller_geometry_retry_{index}")
        admission, execution, teardown = _verified_attempt(root, task_context)
        preflight = json.loads((root / "preflight.json").read_text())
        if (preflight.get("status") != "verified" or
                preflight.get("preflight_digest") != canonical_digest(preflight, digest_field="preflight_digest")):
            raise ValueError("website_vast_billing_preflight_invalid")
        preflight_epoch = float(preflight["observed_at_epoch"])
        teardown_epoch = datetime.fromisoformat(teardown["timestamp"].replace("Z", "+00:00")).timestamp()
        charge = _provider_charge(str(execution["instance_id"]),
                                  preflight_epoch=preflight_epoch, teardown_epoch=teardown_epoch)
        if charge["amount_usd"] > admission["maximum_cost_usd"]:
            raise ValueError("website_vast_billing_exceeds_reservation")
        write_json(root / "provider_charge_receipt.json", charge)
        command = {"task_context_digest": task_context["context_digest"],
                   "allocation_binding_digest": admission["allocation_binding_digest"],
                   "provider": "vast", "instance_id": charge["instance_id"],
                   "provider_charge_source": charge["source"],
                   "provider_charge_amount_usd": charge["amount_usd"],
                   "provider_charge_receipt_digest": charge["provider_charge_receipt_digest"],
                   "execution_result_digest": execution["execution_result_digest"],
                   "teardown_receipt_digest": teardown["teardown_receipt_digest"],
                   "provider_zero_digest": execution["provider_zero_digest"]}
        settled = website_webapp_request(capture_id=task_context["capture_id"],
            operation="preparation-settlement", payload={"request_id": task_context["request_id"],
                "scene_id": task_context["scene_id"], "settlement": command})
        if (any(settled.get(key) != value for key, value in command.items())
                or settled.get("status") != "settled"
                or settled.get("actual_cost_usd") != charge["amount_usd"]):
            raise ValueError("website_vast_billing_settlement_invalid")
        write_json(root / "provider_charge_settlement.json", settled)
