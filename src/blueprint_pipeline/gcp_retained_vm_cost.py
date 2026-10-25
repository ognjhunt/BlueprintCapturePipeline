"""Read-only list-price bound for supported stopped-VM attachments.

Only the observed zonal pd-balanced and attached IPv4 cases are supported.
Unknown disks, incomplete inventory, changed VM state, or pricing failure stay
unknown. No resource is created, modified, detached, or deleted.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Callable, Mapping

# Official Compute Engine catalog: Balanced PD Capacity (zonal, not regional).
BALANCED_SKU = "6AE1-525F-8B80"
BALANCED_REGIONS = {
    "us-central1",
    "us-central2",
    "us-east1",
    "us-west1",
    "asia-east1",
    "europe-west1",
}
STANDARD_IP_SKU = "C054-7F72-A02E"
SPOT_IP_SKU = "4AF8-7C1F-39C4"
# Google's published hourly disk price normalizes the monthly price to 730h.
# https://cloud.google.com/compute/disks-image-pricing
HOURS_PER_PRICE_MONTH = 730


def _number(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("gcp_attachment_price_number_invalid")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("gcp_attachment_price_number_invalid")
    return result


def _price(quote: Mapping, sku: str, unit: str) -> float:
    if (
        quote.get("name") != f"skus/{sku}/price"
        or quote.get("currencyCode") != "USD"
        or quote.get("valueType") != "rate"
    ):
        raise ValueError("gcp_attachment_price_identity_invalid")
    rate = quote["rate"]
    if rate["unitInfo"]["unit"] != unit or _number(rate["unitInfo"]["unitQuantity"]["value"]) != 1:
        raise ValueError("gcp_attachment_price_unit_invalid")
    tiers = rate["tiers"]
    if not tiers or _number(tiers[0]["startAmount"]["value"]) != 0:
        raise ValueError("gcp_attachment_price_tiers_invalid")
    amounts, starts = [], []
    for tier in tiers:
        starts.append(_number(tier["startAmount"]["value"]))
        money = tier["listPrice"]
        nanos = _number(money.get("nanos", 0))
        if money.get("currencyCode") != "USD" or nanos >= 1e9:
            raise ValueError("gcp_attachment_price_currency_invalid")
        amounts.append(_number(money.get("units", 0)) + nanos / 1e9)
    if starts != sorted(set(starts)) or max(amounts) <= 0:
        raise ValueError("gcp_attachment_price_tiers_invalid")
    # Do not assume account-wide credits/free tiers remain available to this VM.
    return max(amounts)


def stopped_vm_cost(
    row: Mapping,
    *,
    project: str,
    zone: str,
    compute: Callable,
    quote: Callable[[str], Mapping],
) -> dict:
    evidence: dict[str, Any] = {"kind": "retained_attachment_list_price_bound", "components": []}
    try:
        if row.get("status") != "TERMINATED" or not row.get("id"):
            raise ValueError("gcp_stopped_vm_identity_unverified")
        name = str(row["name"])
        if not all(re.fullmatch(r"[a-z0-9-]+", value) for value in (project, zone, name)):
            raise ValueError("gcp_stopped_vm_scope_invalid")
        vm_suffix = f"/projects/{project}/zones/{zone}/instances/{name}"
        disks = row.get("disks")
        if not isinstance(disks, list) or not disks:
            raise ValueError("gcp_stopped_vm_disks_unverified")
        total, sources = 0.0, set()
        for attachment in disks:
            source = attachment["source"]
            prefix = f"https://www.googleapis.com/compute/v1/projects/{project}/zones/{zone}/disks/"
            disk_name = source.removeprefix(prefix)
            if (
                attachment.get("type") != "PERSISTENT"
                or not source.startswith(prefix)
                or not re.fullmatch(r"[a-z0-9-]+", disk_name)
                or source in sources
            ):
                raise ValueError("gcp_stopped_vm_disk_scope_invalid")
            sources.add(source)
            status, disk = compute(
                "GET", f"/projects/{project}/zones/{zone}/disks/{disk_name}", timeout=45
            )
            if (
                status != 200
                or not disk.get("id")
                or disk.get("selfLink") != source
                or disk.get("type") != prefix.rsplit("/disks/", 1)[0] + "/diskTypes/pd-balanced"
                or zone.rsplit("-", 1)[0] not in BALANCED_REGIONS
                or disk.get("status") != "READY"
                or not any(str(user).endswith(vm_suffix) for user in disk.get("users", []))
            ):
                raise ValueError("gcp_stopped_vm_disk_unpriced")
            size = _number(disk["sizeGb"])
            if size <= 0 or size != int(size) or size != _number(attachment["diskSizeGb"]):
                raise ValueError("gcp_stopped_vm_disk_size_invalid")
            price = quote(BALANCED_SKU)
            hourly = size * _price(price, BALANCED_SKU, "GiBy.mo") / HOURS_PER_PRICE_MONTH
            total += hourly
            evidence["components"].append(
                {
                    "resource": source,
                    "resource_id": disk["id"],
                    "size_gib": size,
                    "sku": BALANCED_SKU,
                    "price": price,
                    "price_month_hours": HOURS_PER_PRICE_MONTH,
                    "hourly_bound_usd": hourly,
                }
            )
        nics = row.get("networkInterfaces")
        if not isinstance(nics, list):
            raise ValueError("gcp_stopped_vm_network_unverified")
        ips = set()
        for nic in nics:
            for config in nic.get("accessConfigs", []):
                if config.get("type") != "ONE_TO_ONE_NAT" or not config.get("natIP"):
                    raise ValueError("gcp_stopped_vm_external_ip_unverified")
                ips.add(config["natIP"])
        if ips:
            addresses, token, seen = [], "", set()
            for _ in range(100):
                path = f"/projects/{project}/aggregated/addresses"
                if token:
                    from urllib.parse import quote as urlquote

                    path += "?pageToken=" + urlquote(token, safe="")
                status, body = compute("GET", path, timeout=45)
                if (
                    status != 200
                    or body.get("unreachables")
                    or not isinstance(body.get("items"), dict)
                ):
                    raise ValueError("gcp_stopped_vm_address_inventory_incomplete")
                for group in body["items"].values():
                    warning = group.get("warning", {})
                    if warning and warning.get("code") != "NO_RESULTS_ON_PAGE":
                        raise ValueError("gcp_stopped_vm_address_inventory_incomplete")
                    values = group.get("addresses", [])
                    if not isinstance(values, list):
                        raise ValueError("gcp_stopped_vm_address_inventory_incomplete")
                    addresses.extend(values)
                token = body.get("nextPageToken")
                if not token:
                    break
                if not isinstance(token, str) or token in seen:
                    raise ValueError("gcp_stopped_vm_address_inventory_incomplete")
                seen.add(token)
            else:
                raise ValueError("gcp_stopped_vm_address_inventory_incomplete")
            for ip in ips:
                matches = [a for a in addresses if a.get("address") == ip]
                if (
                    len(matches) != 1
                    or matches[0].get("status") != "IN_USE"
                    or matches[0].get("addressType") != "EXTERNAL"
                    or not any(
                        str(user).endswith(vm_suffix) for user in matches[0].get("users", [])
                    )
                ):
                    raise ValueError("gcp_stopped_vm_external_ip_unverified")
                scheduling = row.get("scheduling", {})
                sku = (
                    SPOT_IP_SKU
                    if scheduling.get("preemptible") is True
                    or scheduling.get("provisioningModel") == "SPOT"
                    else STANDARD_IP_SKU
                )
                price = quote(sku)
                hourly = _price(price, sku, "h")
                total += hourly
                evidence["components"].append(
                    {
                        "resource": matches[0].get("selfLink"),
                        "resource_id": matches[0].get("id"),
                        "sku": sku,
                        "price": price,
                        "hourly_bound_usd": hourly,
                    }
                )
        status, after = compute("GET", vm_suffix, timeout=45)
        fields = ("id", "status", "disks", "networkInterfaces", "scheduling")
        if status != 200 or any(after.get(key) != row.get(key) for key in fields):
            raise ValueError("gcp_stopped_vm_changed_during_pricing")
        evidence["observed_at"] = datetime.now(timezone.utc).isoformat()
        evidence["evidence_sha256"] = hashlib.sha256(
            json.dumps(evidence, sort_keys=True).encode()
        ).hexdigest()
        return {"cost_per_hour": total, "cost_rate_evidence": evidence}
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        return {"cost_per_hour": None, "cost_rate_blocker": str(exc)}
