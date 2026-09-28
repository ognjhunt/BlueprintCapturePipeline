"""Vast VM contracts in the existing adapter; no API calls or paid allocation."""

import pytest

from blueprint_pipeline import vast_provider_adapter as adapter

IMAGE = "docker.io/vastai/kvm@sha256:" + "a" * 64


def _offer(**changes):
    return {"id": 101, "machine_id": 201, "gpu_name": "RTX 4090", "gpu_ram": 24576,
            "compute_cap": 890, "dph_total": 0.4, "storage_cost": 0,
            "disk_space": 200, "vms_enabled": True, "num_gpus": 1,
            "verified": True, "rentable": True, **changes}


def _create(**changes):
    return adapter._create_payload(image=IMAGE, label="blueprint-g1-team-test", launch_mode="ssh_direct",
                                   probe_script="echo reviewed-bootstrap", disk_gb=120,
                                   virtual_machine=True, selected_offer=adapter._offer_summary(_offer()), **changes)


def test_vm_offer_search_filters_server_side_and_leaves_normal_search_unchanged():
    args = {"limit": 20, "max_hourly_rate": 0.5, "required_provider_disk_gb": 120,
            "min_gpu_ram_mb": 24000, "allowed_machine_ids": [201]}
    normal = adapter._search_payload(**args)
    vm = adapter._search_payload(**args, require_virtual_machine=True)
    assert vm.pop("vms_enabled") == {"eq": True}
    assert vm == normal


@pytest.mark.parametrize("value", [True, 1])
def test_vm_selection_requires_explicit_capability(value):
    chosen = adapter._select_offer([_offer(vms_enabled=value)], max_hourly_rate=0.5,
                                   require_virtual_machine=True)
    assert chosen["vms_enabled"] is True
    assert adapter._offer_artifact_summary(chosen)["vms_enabled"] is True


@pytest.mark.parametrize("value", [False, 0, None, "true", "1", "yes"])
def test_vm_selection_rejects_unknown_or_string_capability(value):
    expected = False if value is False or type(value) is int and value == 0 else None
    assert adapter._offer_summary(_offer(vms_enabled=value))["vms_enabled"] is expected
    assert adapter._select_offer([_offer(vms_enabled=value)], max_hourly_rate=0.5,
                                 require_virtual_machine=True) is None


def test_vm_selection_does_not_replace_existing_cost_or_disk_gates():
    assert adapter._select_offer([_offer(dph_total=0.6)], max_hourly_rate=0.5,
                                 require_virtual_machine=True) is None
    assert adapter._select_offer([_offer(disk_space=100)], max_hourly_rate=0.5,
                                 require_virtual_machine=True, required_provider_disk_gb=120) is None
    normal = _offer(id=102, vms_enabled=False, dph_total=0.3)
    assert adapter._select_offer([normal, _offer()], max_hourly_rate=0.5)["ask_contract_id"] == 102
    assert adapter._select_offer([normal, _offer()], max_hourly_rate=0.5,
                                 require_virtual_machine=True)["ask_contract_id"] == 101


def test_vm_create_is_explicit_pinned_ssh_and_receipt_stays_redacted():
    payload = _create()
    assert payload["vm"] is True
    assert payload["image"] == IMAGE and payload["runtype"] == "ssh_direct"
    assert "onstart" in payload and "args" not in payload and "args_str" not in payload
    summary = adapter._create_request_summary(payload, secret_values=())
    assert summary["virtual_machine"] is True
    assert "reviewed-bootstrap" not in str(summary)
    assert summary["raw_payload_redacted"]["onstart"] != payload["onstart"]


@pytest.mark.parametrize("field,value", [
    ("image", "docker.io/vastai/kvm:ubuntu_terminal"),
    ("image", "registry.example.org/other@sha256:" + "a" * 64),
    ("image", None), ("launch_mode", "args"), ("launch_mode", "jupyter_direct"),
    ("selected_offer", None), ("selected_offer", {}),
    ("selected_offer", {"vms_enabled": "true", "ask_contract_id": 101}),
    ("template_hash_id", "hidden-defaults"),
])
def test_vm_create_refuses_incompatible_contract(field, value):
    args = {"image": IMAGE, "label": "test", "launch_mode": "ssh_direct", "probe_script": "true",
            "disk_gb": 120, "virtual_machine": True, "selected_offer": adapter._offer_summary(_offer())}
    args[field] = value
    with pytest.raises(ValueError, match="virtual_machine"):
        adapter._create_payload(**args)


def test_normal_create_remains_container_without_vm_flag():
    payload = adapter._create_payload(image="existing-image", label="test", launch_mode="ssh_direct",
                                      probe_script="true", disk_gb=20)
    assert "vm" not in payload
    assert adapter._create_request_summary(payload, secret_values=())["virtual_machine"] is False
