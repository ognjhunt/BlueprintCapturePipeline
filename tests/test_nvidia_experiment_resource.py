from __future__ import annotations

import json
from pathlib import Path

import jsonschema

from blueprint_pipeline.external_tool_runtime import canonical_sha256
from blueprint_pipeline.nvidia_experiment_resource import (
    build_resource_closeout,
    load_resource_closeout,
    load_resource_context,
    local_unpaid_resource_context,
    resource_stop_evidence,
    validate_resource_context,
)


def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _digest(char: str) -> str:
    return char * 64


def test_local_resource_context_is_explicitly_unpaid() -> None:
    context = local_unpaid_resource_context()
    assert validate_resource_context(context) == []
    assert resource_stop_evidence(context) == {
        "paid_resource_admission_enforced": True,
        "provider_teardown_provable": True,
    }
    assert context["claim_boundary"]["local_unpaid_is_not_provider_execution_proof"] is True


def test_paid_resource_requires_shared_admission_and_exact_global_zero(tmp_path: Path) -> None:
    context = {
        "schema_version": "nvidia_experiment_resource_context.v1",
        "resource_origin": "paid_provider",
        "paid_resource": True,
        "allocation": {
            "provider_id": "runpod",
            "allocation_id": "pod-123",
            "attempt_id": "attempt-edge-1",
        },
        "admission": {
            "status": "PASS",
            "spend_allowed": True,
            "allocator_module": "blueprint_pipeline.paid_resource_allocator",
            "allocation_kind": "gpu-canary",
            "spend_cap_usd": 2.5,
            "pre_spend_preflight_sha256": _digest("a"),
            "allocation_receipt_sha256": _digest("b"),
        },
        "claim_boundary": {},
    }
    context["context_fingerprint"] = canonical_sha256(context)
    teardown = {
        "schema_version": "nvidia_experiment_paid_teardown.v1",
        "provider_id": "runpod",
        "attempt_id": "attempt-edge-1",
        "exact_attempt_allocation_ids": ["pod-123"],
        "exact_attempt_active_resource_count": 0,
        "global_provider_inventory": [
            {
                "provider_id": "runpod",
                "active_resource_count": 0,
                "hourly_allocation_burn_usd": 0.0,
                "inventory_report_sha256": _digest("c"),
            }
        ],
        "teardown_report_sha256": _digest("d"),
        "billing_reconciliation": {
            "status": "reconciled",
            "billing_export_sha256": _digest("e"),
            "total_spend_usd": 0.42,
        },
        "observed_at": "2026-07-21T18:00:00Z",
    }
    context_path = tmp_path / "context.json"
    teardown_path = tmp_path / "teardown.json"
    output_path = tmp_path / "closeout.json"
    _write(context_path, context)
    _write(teardown_path, teardown)
    closeout = build_resource_closeout(
        resource_context_path=context_path,
        teardown_evidence_path=teardown_path,
        output_path=output_path,
    )
    assert closeout["status"] == "proven_zero"
    assert closeout["total_spend_usd"] == 0.42
    assert resource_stop_evidence(context, closeout)["provider_teardown_provable"] is True
    loaded_context, context_blockers = load_resource_context(context_path)
    assert context_blockers == []
    loaded_closeout, closeout_blockers = load_resource_closeout(loaded_context, output_path)
    assert closeout_blockers == []
    assert loaded_closeout["allocation_id"] == "pod-123"
    schema = json.loads(
        (
            Path(__file__).parents[1]
            / "docs/schemas/nvidia_experiment_resource_closeout.schema.json"
        ).read_text()
    )
    jsonschema.Draft202012Validator(schema).validate(closeout)


def test_paid_resource_rejects_provider_specific_launcher() -> None:
    context = {
        "schema_version": "nvidia_experiment_resource_context.v1",
        "resource_origin": "paid_provider",
        "paid_resource": True,
        "allocation": {"provider_id": "x", "allocation_id": "y", "attempt_id": "z"},
        "admission": {"allocator_module": "provider.launcher", "allocation_kind": "gpu-canary"},
    }
    assert "paid_resource_must_use_shared_paid_resource_allocator" in validate_resource_context(
        context
    )


def test_paid_resource_requires_closeout_before_adapter_can_pass() -> None:
    context = {
        "schema_version": "nvidia_experiment_resource_context.v1",
        "resource_origin": "paid_provider",
        "paid_resource": True,
        "allocation": {
            "provider_id": "runpod",
            "allocation_id": "pod-a",
            "attempt_id": "attempt-a",
        },
    }
    closeout, blockers = load_resource_closeout(context, None)
    assert closeout == {}
    assert blockers == ["nvidia_paid_resource_closeout_required"]

    local_closeout, local_blockers = load_resource_closeout(local_unpaid_resource_context(), None)
    assert local_closeout == {}
    assert local_blockers == []


def test_closeout_loader_rejects_unbound_allocation(tmp_path: Path) -> None:
    context = {
        "schema_version": "nvidia_experiment_resource_context.v1",
        "resource_origin": "paid_provider",
        "paid_resource": True,
        "allocation": {
            "provider_id": "runpod",
            "allocation_id": "pod-a",
            "attempt_id": "attempt-a",
        },
        "admission": {},
        "source_sha256": _digest("a"),
    }
    closeout = {
        "schema_version": "nvidia_experiment_resource_closeout.v1",
        "status": "proven_zero",
        "provider_id": "runpod",
        "allocation_id": "pod-b",
        "attempt_id": "attempt-a",
        "resource_context_sha256": _digest("a"),
        "blockers": [],
    }
    path = tmp_path / "closeout.json"
    _write(path, closeout)
    _, blockers = load_resource_closeout(context, path)
    assert "nvidia_resource_closeout_identity_mismatch:allocation_id" in blockers
