"""No-network retained-resource pricing and stopped-state race tests."""

import copy

import pytest

from blueprint_pipeline.cloud_vm_render_providers import GCPRenderProvider
from blueprint_pipeline.gcp_retained_vm_cost import BALANCED_SKU, STANDARD_IP_SKU, stopped_vm_cost

BASE = "https://www.googleapis.com/compute/v1/projects/test-project/zones/us-central1-a"
VM = {
    "id": "123",
    "name": "stopped",
    "status": "TERMINATED",
    "zone": BASE,
    "machineType": BASE + "/machineTypes/t2a-standard-2",
    "scheduling": {"automaticRestart": False},
    "disks": [
        {
            "type": "PERSISTENT",
            "source": BASE + "/disks/boot",
            "diskSizeGb": "30",
            "autoDelete": False,
        }
    ],
    "networkInterfaces": [{"accessConfigs": [{"type": "ONE_TO_ONE_NAT", "natIP": "192.0.2.1"}]}],
}
DISK = {
    "id": "456",
    "selfLink": BASE + "/disks/boot",
    "type": BASE + "/diskTypes/pd-balanced",
    "sizeGb": "30",
    "status": "READY",
    "users": [BASE + "/instances/stopped"],
}
ADDRESS = {
    "id": "789",
    "address": "192.0.2.1",
    "status": "IN_USE",
    "addressType": "EXTERNAL",
    "users": [BASE + "/instances/stopped"],
    "selfLink": "https://www.googleapis.com/compute/v1/projects/test-project/regions/us-central1/addresses/ip",
}


def _quote(sku):
    return {
        "name": f"skus/{sku}/price",
        "currencyCode": "USD",
        "valueType": "rate",
        "rate": {
            "tiers": [
                {
                    "startAmount": {"value": "0"},
                    "listPrice": {
                        "currencyCode": "USD",
                        "nanos": 100000000 if sku == BALANCED_SKU else 5000000,
                    },
                }
            ],
            "unitInfo": {
                "unit": "GiBy.mo" if sku == BALANCED_SKU else "h",
                "unitQuantity": {"value": "1"},
            },
        },
    }


class Provider:
    def __init__(self):
        self.vm, self.disk, self.address = copy.deepcopy((VM, DISK, ADDRESS))
        self.calls = []
        self.quotes = {}
        self.address_http = 200
        self.address_extra = {}
        self.after = None

    def compute(self, method, path, **kwargs):
        assert method == "GET"
        self.calls.append(path)
        if path.endswith("/disks/boot"):
            return 200, self.disk
        if path.endswith("/aggregated/addresses"):
            return self.address_http, {
                "items": {"regions/us-central1": {"addresses": [self.address]}},
                **self.address_extra,
            }
        if path.endswith("/instances/stopped"):
            return 200, self.after or self.vm
        if path.endswith("/aggregated/instances"):
            return 200, {"items": {"zones/us-central1-a": {"instances": [self.vm]}}}
        raise AssertionError(path)

    def quote(self, sku):
        return self.quotes.get(sku, _quote(sku))

    def cost(self):
        return stopped_vm_cost(
            self.vm,
            project="test-project",
            zone="us-central1-a",
            compute=self.compute,
            quote=self.quote,
        )


def test_both_disk_and_in_use_static_ip_are_counted_without_mutation():
    p = Provider()
    result = p.cost()
    assert result["cost_per_hour"] == pytest.approx(3 / 730 + 0.005)
    assert len(result["cost_rate_evidence"]["components"]) == 2
    assert p.vm == VM
    assert p.disk == DISK


def test_free_tier_is_not_assumed_available_to_this_vm():
    p = Provider()
    quote = _quote(STANDARD_IP_SKU)
    quote["rate"]["tiers"].insert(
        0, {"startAmount": {"value": "0"}, "listPrice": {"currencyCode": "USD"}}
    )
    quote["rate"]["tiers"][1]["startAmount"]["value"] = "744"
    p.quotes[STANDARD_IP_SKU] = quote
    assert p.cost()["cost_per_hour"] == pytest.approx(3 / 730 + 0.005)


def test_no_external_ip_needs_only_disk_price():
    p = Provider()
    p.vm["networkInterfaces"] = [{}]
    assert p.cost()["cost_per_hour"] == pytest.approx(3 / 730)
    assert not any("addresses" in path for path in p.calls)


@pytest.mark.parametrize(
    "change",
    [
        "running",
        "no_identity",
        "unknown_disk",
        "regional_disk",
        "wrong_disk_identity",
        "size_changed",
        "disk_missing_user",
        "disk_nan",
        "address_403",
        "address_partial",
        "address_wrong_user",
        "address_reserved",
        "restarted",
        "new_disk",
        "wrong_currency",
        "wrong_unit",
        "nonfinite_price",
        "wrong_sku",
        "no_tiers",
        "zero_price",
        "missing_network",
    ],
)
def test_unverified_cost_remains_unknown(change):
    p = Provider()
    if change == "running":
        p.vm["status"] = "RUNNING"
    elif change == "no_identity":
        p.vm.pop("id")
    elif change == "unknown_disk":
        p.disk["type"] = BASE + "/diskTypes/hyperdisk-balanced"
    elif change == "regional_disk":
        p.vm["disks"][0]["source"] = (
            BASE.replace("zones/us-central1-a", "regions/us-central1") + "/disks/boot"
        )
    elif change == "wrong_disk_identity":
        p.disk["selfLink"] += "-wrong"
    elif change == "size_changed":
        p.disk["sizeGb"] = "40"
    elif change == "disk_missing_user":
        p.disk["users"] = []
    elif change == "disk_nan":
        p.disk["sizeGb"] = "NaN"
    elif change == "address_403":
        p.address_http = 403
    elif change == "address_partial":
        p.address_extra = {"unreachables": ["regions/us-east1"]}
    elif change == "address_wrong_user":
        p.address["users"] = []
    elif change == "address_reserved":
        p.address["status"] = "RESERVED"
    elif change in {"restarted", "new_disk"}:
        p.after = copy.deepcopy(p.vm)
        if change == "restarted":
            p.after["status"] = "RUNNING"
        else:
            p.after["disks"].append({"source": "unknown"})
    elif change == "missing_network":
        p.vm.pop("networkInterfaces")
    else:
        quote = _quote(BALANCED_SKU)
        if change == "wrong_currency":
            quote["currencyCode"] = "EUR"
        elif change == "wrong_unit":
            quote["rate"]["unitInfo"]["unit"] = "GiBy.h"
        elif change == "nonfinite_price":
            quote["rate"]["tiers"][0]["listPrice"]["nanos"] = float("nan")
        elif change == "wrong_sku":
            quote["name"] = "skus/regional-disk/price"
        elif change == "no_tiers":
            quote["rate"]["tiers"] = []
        elif change == "zero_price":
            quote["rate"]["tiers"][0]["listPrice"]["nanos"] = 0
        p.quotes[BALANCED_SKU] = quote
    result = p.cost()
    assert result["cost_per_hour"] is None
    assert result["cost_rate_blocker"]


def test_project_inventory_keeps_stopped_vm_visible_and_preserves_cleanup_ownership(monkeypatch):
    p = Provider()
    monkeypatch.setenv("BLUEPRINT_GCP_PROJECT", "test-project")
    monkeypatch.setenv("BLUEPRINT_GCP_ZONE", "us-central1-a")
    provider = GCPRenderProvider()
    monkeypatch.setattr(provider, "_call", p.compute)
    monkeypatch.setattr(provider, "_public_sku_price", p.quote)
    result = provider.project_inventory(projects=["test-project"])
    assert result["api_confirmed"] is True
    assert result["live_resource_count"] == 1
    row = result["resources"][0]
    assert row["cost_per_hour"] == pytest.approx(3 / 730 + 0.005)
    assert row["status"] == "TERMINATED"
    assert row["cleanup_authorized"] is False
