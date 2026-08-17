from __future__ import annotations

import json
import math
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.provider_billing_reconciler import (
    BILLING_SOURCE_SCHEMA_VERSION,
    VAST_CHARGES_URL,
)
from blueprint_pipeline.vast_official_billing_extractor import (
    ENTRY_SCHEMA_VERSION,
    RECONCILIATION_SCHEMA_VERSION,
    VastOfficialBillingExtractionError,
    main,
    materialize_vast_official_same_goal_reconciliation,
    validate_vast_official_same_goal_reconciliation,
)


INSTANCE_A = 47_912_530
INSTANCE_B = 47_913_976
LABEL_A = "blueprint-groot-oscar-canary-adp-artifixer3d-1786935680"
LABEL_B = "blueprint-groot-oscar-canary-adp-artifixer3d-1786937589"
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLI_SCRIPT = REPOSITORY_ROOT / "scripts/materialize_vast_official_same_goal_reconciliation.py"


def _sha256(path: Path) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _charge(
    *,
    instance_id: int,
    label: str,
    total: float,
    gpu: float,
    disk: float,
) -> dict:
    return {
        "source": f"instance-{instance_id}",
        "amount": total,
        "description": f"Instance {instance_id} Charges - 1 day",
        "type": "instance",
        "start": 1_786_924_800,
        "end": 1_786_924_800,
        "items": [
            {
                "start": 1_786_924_800,
                "end": 1_786_924_800,
                "type": "gpu",
                "source": None,
                "description": "production-shaped GPU line",
                "amount": gpu,
                "metadata": {},
                "items": [],
            },
            {
                "start": 1_786_924_800,
                "end": 1_786_924_800,
                "type": "disk",
                "source": None,
                "description": "production-shaped disk line",
                "amount": disk,
                "metadata": {},
                "items": [],
            },
            {
                "start": 1_786_924_800,
                "end": 1_786_924_800,
                "type": "bwd",
                "source": None,
                "description": "0.0 GB Downloaded",
                "amount": 0.0,
                "metadata": {},
                "items": [],
            },
            {
                "start": 1_786_924_800,
                "end": 1_786_924_800,
                "type": "bwu",
                "source": None,
                "description": "0.0 GB Uploaded",
                "amount": 0.0,
                "metadata": {},
                "items": [],
            },
        ],
        "metadata": {"label": label},
    }


def _bound(path: Path) -> dict:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _terminal_fixture(
    tmp_path: Path, *, instance_id: int, status: str
) -> Path:
    run_id = f"adp-artifixer3d-fixture-{instance_id}"
    profile_id = f"adp-artifixer3d-live-fixture-{instance_id}"
    run_root = tmp_path / "task-evaluation-launch-runs" / run_id
    allocator = run_root / "allocator"
    provider_run = allocator / "artifixer3d-job" / "vast_provider_run"

    profile = {
        "schema_version": "task_evaluation_launch_profile.v1",
        "profile_id": profile_id,
        "execution_admission": {"live_enabled": True, "blockers": []},
        "reconciliation": {"required_providers": ["vast"]},
        "profile_digest": "",
    }
    profile["profile_digest"] = canonical_digest(
        profile, digest_field="profile_digest"
    )
    profile_path = _write(run_root / "launch_profile.json", profile)
    request = {
        "schema_version": "task_evaluation_launch_request.v1",
        "launch_id": run_id,
        "run_id": run_id,
        "launch_profile_id": profile_id,
        "launch_profile_digest": profile["profile_digest"],
        "idempotency_key": run_id,
        "request_digest": "",
    }
    request["request_digest"] = canonical_digest(
        request, digest_field="request_digest"
    )
    request_path = _write(run_root / "launch_request.json", request)
    binding = {
        "schema_version": "task_evaluation_launch_binding.v1",
        "launch_id": run_id,
        "run_id": run_id,
        "request_digest": request["request_digest"],
        "profile_digest": profile["profile_digest"],
        "execute_requested": True,
        "binding_digest": "",
    }
    binding["binding_digest"] = canonical_digest(
        binding, digest_field="binding_digest"
    )
    binding_path = _write(run_root / "launch_binding.json", binding)
    started = {
        "schema_version": "task_evaluation_launch_started.v1",
        "launch_id": run_id,
        "run_id": run_id,
        "request_digest": request["request_digest"],
        "binding_digest": binding["binding_digest"],
        "automatic_retry_authorized": False,
        "started_digest": "",
    }
    started["started_digest"] = canonical_digest(
        started, digest_field="started_digest"
    )
    started_path = _write(run_root / "launch_started.json", started)

    adapter = {
        "schema_version": "vast_provider_adapter_result.v1",
        "status": status,
        "vast_instance_ids": [instance_id],
        "continuing_spend_from_this_run": False,
        "final_validation_status": "passed",
        "retained_owned": False,
        "raw_api_key_stored": False,
        "secret_values_in_artifact": False,
        "blockers": [] if status == "completed" else ["runtime_result_missing"],
    }
    adapter_path = _write(provider_run / "vast_provider_adapter_result.json", adapter)
    teardown = {
        "schema_version": "vast_teardown_manifest.v1",
        "status": "completed",
        "vast_instance_ids": [instance_id],
        "continuing_spend_from_this_run": False,
        "runner_gpu_teardown_completed": True,
        "retention_authorized": False,
        "raw_secret_values_recorded": False,
        "zero_continuing_spend_scope": "all Vast instances created were destroyed",
    }
    teardown_path = _write(provider_run / "vast_teardown_manifest.json", teardown)
    result = {
        "schema_version": "public_scene_artifixer3d_vast_run.v1",
        "status": status,
        "retry_cap": 0,
        "continuing_spend_from_this_run": False,
        "raw_secret_values_recorded": False,
        "adapter_result_path": str(adapter_path),
        "teardown_manifest_path": str(teardown_path),
        "provider_closeout": {
            "adapter_result": _bound(adapter_path),
            "teardown_manifest": _bound(teardown_path),
            "provider_zero_confirmed": True,
            "all_staged_objects_absent": True,
        },
        "independent_watchdog": {
            "schema_version": "vast_independent_watchdog_handoff.v1",
            "status": "provider_terminal",
            "instance_ids": [instance_id],
            "provider_absence_confirmed": True,
            "provider_mutations_performed": 0,
            "raw_secret_values_recorded": False,
        },
        "blockers": [] if status == "completed" else ["runtime_result_missing"],
    }
    result_path = _write(
        allocator
        / "artifixer3d-job"
        / "public_scene_artifixer3d_vast_result.json",
        result,
    )
    allocator_result_path = _write(allocator / "result.json", result)
    terminal_status = "passed" if status == "completed" else "blocked"
    receipt = {
        "schema_version": "task_evaluation_launch_receipt.v1",
        "status": status,
        "launch_id": run_id,
        "run_id": run_id,
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "binding_digest": binding["binding_digest"],
        "execute_requested": True,
        "raw_secret_values_recorded": False,
        "terminal_evidence": {
            "status": terminal_status,
            "result": {
                "path": str(allocator_result_path),
                "digest": _sha256(allocator_result_path),
                "exists": True,
            },
            "artifacts": {
                "teardown_manifest_path": {
                    "path": str(teardown_path),
                    "digest": _sha256(teardown_path),
                    "exists": True,
                }
            },
            "blockers": [] if status == "completed" else ["terminal_blocked"],
        },
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = _write(run_root / "launch_receipt.json", receipt)
    zero = {
        "schema_version": "task_evaluation_post_teardown_provider_zero.v1",
        "status": "provider_zero_confirmed",
        "launch_id": run_id,
        "run_id": run_id,
        "request_digest": request["request_digest"],
        "launch_profile_digest": profile["profile_digest"],
        "receipt_digest": receipt["receipt_digest"],
        "provider_zero_verified": True,
        "continuing_spend_from_this_run": False,
        "automatic_retry_performed": False,
        "provider_mutation_performed": False,
        "required_providers": ["vast"],
        "blockers": [],
        "provider_zero_receipt_digest": "",
    }
    zero["provider_zero_receipt_digest"] = canonical_digest(
        zero, digest_field="provider_zero_receipt_digest"
    )
    zero_path = _write(run_root / "post_teardown_provider_zero_receipt.json", zero)
    assert all(
        path.is_file()
        for path in (
            profile_path,
            request_path,
            binding_path,
            started_path,
            receipt_path,
            zero_path,
        )
    )
    return result_path


def _refresh_terminal_bindings(result_path: Path) -> None:
    run_root = result_path.parents[2]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    adapter_path = Path(result["adapter_result_path"])
    teardown_path = Path(result["teardown_manifest_path"])
    result["provider_closeout"]["adapter_result"] = _bound(adapter_path)
    result["provider_closeout"]["teardown_manifest"] = _bound(teardown_path)
    _write(result_path, result)
    allocator_result_path = _write(run_root / "allocator" / "result.json", result)
    receipt_path = run_root / "launch_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["terminal_evidence"]["result"]["digest"] = _sha256(
        allocator_result_path
    )
    receipt["terminal_evidence"]["artifacts"]["teardown_manifest_path"][
        "digest"
    ] = _sha256(teardown_path)
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write(receipt_path, receipt)
    zero_path = run_root / "post_teardown_provider_zero_receipt.json"
    zero = json.loads(zero_path.read_text(encoding="utf-8"))
    zero["receipt_digest"] = receipt["receipt_digest"]
    zero["provider_zero_receipt_digest"] = canonical_digest(
        zero, digest_field="provider_zero_receipt_digest"
    )
    _write(zero_path, zero)


def _fixture(tmp_path: Path) -> dict[str, object]:
    audit = tmp_path / "billing-audit" / "20260817T055421.193507Z"
    responses = [
        _write(
            audit / "response-004-vast.json",
            {
                "success": True,
                "next_token": "page-two",
                "results": [
                    _charge(
                        instance_id=INSTANCE_A,
                        label=LABEL_A,
                        total=0.123,
                        gpu=0.112,
                        disk=0.011,
                    ),
                    {
                        "source": "instance-1",
                        "amount": 9.0,
                        "type": "instance",
                        "items": [],
                        "metadata": {"label": "blueprint-unrelated-private-label"},
                        "unretained_secret": "raw-response-secret",
                    },
                ],
            },
        ),
        _write(
            audit / "response-005-vast.json",
            {
                "success": True,
                "next_token": None,
                "results": [
                    _charge(
                        instance_id=INSTANCE_B,
                        label=LABEL_B,
                        total=2.183,
                        gpu=2.056,
                        disk=0.127,
                    )
                ],
            },
        ),
    ]
    receipt = {
        "schema_version": BILLING_SOURCE_SCHEMA_VERSION,
        "status": "reconciled",
        "generated_at": "2026-08-17T05:54:21.193507+00:00",
        "cohort_start_at": "2026-07-01T00:00:00+00:00",
        "cohort_end_at": "2026-08-17T05:54:21.193507+00:00",
        "provider_totals_usd": {
            "runpod": 98.563962,
            "vast": 281.889,
            "digitalocean": 152.25,
        },
        "sources": [
            {
                "provider": "vast",
                "endpoint": VAST_CHARGES_URL,
                "request_query_digest": "sha256:" + "1" * 64,
                "response_digest": _sha256(path),
                "response_size_bytes": path.stat().st_size,
                "retained_path": str(path),
            }
            for path in responses
        ],
        "provider_mutation_performed": False,
        "raw_secret_values_recorded": False,
        "receipt_digest": "",
    }
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path = _write(audit / "provider_billing_source_receipt.json", receipt)
    terminals = {
        INSTANCE_A: _terminal_fixture(tmp_path, instance_id=INSTANCE_A, status="blocked"),
        INSTANCE_B: _terminal_fixture(
            tmp_path, instance_id=INSTANCE_B, status="completed"
        ),
    }
    return {
        "audit": audit,
        "responses": responses,
        "receipt": receipt_path,
        "receipt_value": receipt,
        "terminals": terminals,
    }


def _refresh_response_binding(fixture: dict[str, object], index: int) -> None:
    path = fixture["responses"][index]
    assert isinstance(path, Path)
    receipt_path = fixture["receipt"]
    assert isinstance(receipt_path, Path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["sources"][index]["response_digest"] = _sha256(path)
    receipt["sources"][index]["response_size_bytes"] = path.stat().st_size
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    _write(receipt_path, receipt)


def _materialize(
    fixture: dict[str, object],
    output: Path,
    *,
    expected: list[tuple[int, str, Path]] | None = None,
    prior: Path | None = None,
) -> dict:
    return materialize_vast_official_same_goal_reconciliation(
        provider_billing_source_receipt_path=fixture["receipt"],
        expected_instances=expected
        or [
            _spec(fixture, INSTANCE_A, LABEL_A),
            _spec(fixture, INSTANCE_B, LABEL_B),
        ],
        output_path=output,
        prior_reconciliation_path=prior,
    )


def _spec(
    fixture: dict[str, object], instance_id: int, label: str
) -> tuple[int, str, Path]:
    terminals = fixture["terminals"]
    assert isinstance(terminals, dict)
    terminal = terminals[instance_id]
    assert isinstance(terminal, Path)
    return instance_id, label, terminal


def test_extracts_production_shaped_posted_instance_charges_exactly(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "official" / "same-goal.json"
    value = _materialize(fixture, output)

    assert value["schema_version"] == RECONCILIATION_SCHEMA_VERSION
    assert value["entry_count"] == 2
    assert value["provider_instance_ids"] == [INSTANCE_A, INSTANCE_B]
    assert value["official_total_usd"] == pytest.approx(2.306)
    assert value["provider_mutation_performed"] is False
    assert value["paid_resource_allocated"] is False
    assert value["raw_secret_values_recorded"] is False
    assert value["receipt_digest"] == canonical_digest(
        value, digest_field="receipt_digest"
    )
    assert value["current_provider_billing_source_receipt"]["receipt_digest"] == (
        fixture["receipt_value"]["receipt_digest"]
    )
    first, second = value["entries"]
    assert first["schema_version"] == ENTRY_SCHEMA_VERSION
    assert first["provider_instance_id"] == INSTANCE_A
    assert first["official_charge_usd"] == pytest.approx(0.123)
    assert first["official_line_items_usd"] == {
        "gpu": 0.112,
        "disk": 0.011,
        "bandwidth_download": 0.0,
        "bandwidth_upload": 0.0,
    }
    assert second["provider_instance_id"] == INSTANCE_B
    assert second["official_charge_usd"] == pytest.approx(2.183)
    assert second["official_line_items_usd"] == {
        "gpu": 2.056,
        "disk": 0.127,
        "bandwidth_download": 0.0,
        "bandwidth_upload": 0.0,
    }
    assert first["entry_digest"] == canonical_digest(
        first, digest_field="entry_digest"
    )
    first_terminal = first["terminal_execution_evidence"]
    second_terminal = second["terminal_execution_evidence"]
    assert first_terminal["terminal_status"] == "blocked"
    assert second_terminal["terminal_status"] == "completed"
    for terminal, instance_id in (
        (first_terminal, INSTANCE_A),
        (second_terminal, INSTANCE_B),
    ):
        assert terminal["retry_cap"] == 0
        assert terminal["continuing_spend_from_this_run"] is False
        assert terminal["provider_absence_confirmed"] is True
        assert terminal["provider_zero_verified"] is True
        assert terminal["launch_id"] == terminal["run_id"]
        assert terminal["request_digest"].startswith("sha256:")
        assert terminal["profile_id"].startswith("adp-artifixer3d-live-")
        assert terminal["profile_digest"].startswith("sha256:")
        assert terminal["provider_adapter_result"]["status"] in {
            "blocked",
            "completed",
        }
        assert terminal["teardown_manifest"]["status"] == "completed"
        assert str(instance_id) in terminal["terminal_result"]["path"]
        terminal_path = Path(terminal["terminal_result"]["path"])
        assert terminal_path.name == "public_scene_artifixer3d_vast_result.json"
        assert terminal_path.parent.name == "artifixer3d-job"
        assert terminal_path.parent.parent.name == "allocator"
    assert validate_vast_official_same_goal_reconciliation(output) == value
    assert stat.S_IMODE(output.stat().st_mode) == 0o440
    serialized = output.read_text(encoding="utf-8")
    assert "raw-response-secret" not in serialized
    assert "production-shaped GPU line" not in serialized


def test_prior_reconciliation_extends_without_repricing_prior_entry(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prior_path = tmp_path / "prior.json"
    prior = _materialize(
        fixture, prior_path, expected=[_spec(fixture, INSTANCE_A, LABEL_A)]
    )
    output = tmp_path / "extended.json"
    value = _materialize(
        fixture,
        output,
        expected=[_spec(fixture, INSTANCE_B, LABEL_B)],
        prior=prior_path,
    )

    assert value["entry_count"] == 2
    assert value["new_entry_count"] == 1
    assert value["prior_entry_count"] == 1
    assert value["official_total_usd"] == pytest.approx(2.306)
    assert value["entries"][0] == prior["entries"][0]
    assert value["predecessor_reconciliation"]["receipt_digest"] == prior[
        "receipt_digest"
    ]


def test_rejects_tampered_or_overlapping_prior_reconciliation(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    prior_path = tmp_path / "prior.json"
    _materialize(
        fixture, prior_path, expected=[_spec(fixture, INSTANCE_A, LABEL_A)]
    )
    prior_path.chmod(0o600)
    tampered = json.loads(prior_path.read_text(encoding="utf-8"))
    tampered["official_total_usd"] = 0.0
    _write(prior_path, tampered)
    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(
            fixture,
            tmp_path / "tampered-output.json",
            expected=[_spec(fixture, INSTANCE_B, LABEL_B)],
            prior=prior_path,
        )

    prior_path.unlink()
    _materialize(
        fixture, prior_path, expected=[_spec(fixture, INSTANCE_A, LABEL_A)]
    )
    with pytest.raises(
        VastOfficialBillingExtractionError, match="vast_official_prior_overlap"
    ):
        _materialize(
            fixture,
            tmp_path / "overlap-output.json",
            expected=[_spec(fixture, INSTANCE_A, LABEL_A)],
            prior=prior_path,
        )


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        ("duplicate", "vast_official_charge_duplicate"),
        ("wrong_label", "vast_official_charge_identity_invalid"),
        ("non_instance", "vast_official_charge_identity_invalid"),
        ("negative_amount", "vast_official_charge_amount_invalid"),
        ("nonfinite_amount", "vast_official_charge_amount_invalid"),
        ("negative_item", "vast_official_charge_item_amount_invalid"),
        ("nonfinite_item", "vast_official_charge_item_amount_invalid"),
        ("missing_item", "vast_official_charge_items_invalid"),
        ("duplicate_item", "vast_official_charge_items_invalid"),
        ("contradictory_total", "vast_official_charge_total_contradiction"),
    ],
)
def test_rejects_ambiguous_or_invalid_official_rows(
    tmp_path: Path, mutation: str, blocker: str
) -> None:
    fixture = _fixture(tmp_path)
    response = fixture["responses"][0]
    assert isinstance(response, Path)
    payload = json.loads(response.read_text(encoding="utf-8"))
    row = payload["results"][0]
    if mutation == "duplicate":
        payload["results"].append(dict(row))
    elif mutation == "wrong_label":
        row["metadata"]["label"] = "blueprint-wrong-label"
    elif mutation == "non_instance":
        row["type"] = "storage"
    elif mutation == "negative_amount":
        row["amount"] = -0.123
    elif mutation == "nonfinite_amount":
        row["amount"] = math.inf
    elif mutation == "negative_item":
        row["items"][0]["amount"] = -0.112
    elif mutation == "nonfinite_item":
        row["items"][0]["amount"] = math.inf
    elif mutation == "missing_item":
        row["items"].pop()
    elif mutation == "duplicate_item":
        row["items"][-1]["type"] = "gpu"
    elif mutation == "contradictory_total":
        row["items"][0]["amount"] = 0.111
    _write(response, payload)
    _refresh_response_binding(fixture, 0)

    with pytest.raises(VastOfficialBillingExtractionError, match=blocker):
        _materialize(
            fixture,
            tmp_path / "output.json",
            expected=[_spec(fixture, INSTANCE_A, LABEL_A)],
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "unsupported_terminal_schema",
        "retry_cap",
        "continuing_spend",
        "adapter_instance",
        "teardown_incomplete",
        "provider_zero_false",
        "launch_identity",
        "wrong_result_path",
        "terminal_symlink",
        "alternate_depth",
    ],
)
def test_rejects_unbound_or_incomplete_terminal_execution_evidence(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    result_path = _spec(fixture, INSTANCE_A, LABEL_A)[2]
    expected = [_spec(fixture, INSTANCE_A, LABEL_A)]
    run_root = result_path.parents[2]
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if mutation == "unsupported_terminal_schema":
        result["schema_version"] = "unknown_vast_run.v1"
        _write(result_path, result)
        _refresh_terminal_bindings(result_path)
    elif mutation == "retry_cap":
        result["retry_cap"] = 1
        _write(result_path, result)
        _refresh_terminal_bindings(result_path)
    elif mutation == "continuing_spend":
        result["continuing_spend_from_this_run"] = True
        _write(result_path, result)
        _refresh_terminal_bindings(result_path)
    elif mutation == "adapter_instance":
        adapter_path = Path(result["adapter_result_path"])
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        adapter["vast_instance_ids"] = [INSTANCE_B]
        _write(adapter_path, adapter)
        _refresh_terminal_bindings(result_path)
    elif mutation == "teardown_incomplete":
        teardown_path = Path(result["teardown_manifest_path"])
        teardown = json.loads(teardown_path.read_text(encoding="utf-8"))
        teardown["runner_gpu_teardown_completed"] = False
        _write(teardown_path, teardown)
        _refresh_terminal_bindings(result_path)
    elif mutation == "provider_zero_false":
        zero_path = run_root / "post_teardown_provider_zero_receipt.json"
        zero = json.loads(zero_path.read_text(encoding="utf-8"))
        zero["provider_zero_verified"] = False
        zero["provider_zero_receipt_digest"] = canonical_digest(
            zero, digest_field="provider_zero_receipt_digest"
        )
        _write(zero_path, zero)
    elif mutation == "launch_identity":
        request_path = run_root / "launch_request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        request["run_id"] = "different-run"
        request["request_digest"] = canonical_digest(
            request, digest_field="request_digest"
        )
        _write(request_path, request)
    elif mutation == "wrong_result_path":
        expected = [
            (INSTANCE_A, LABEL_A, _spec(fixture, INSTANCE_B, LABEL_B)[2])
        ]
    elif mutation == "terminal_symlink":
        real_result = result_path.with_suffix(".real.json")
        result_path.rename(real_result)
        result_path.symlink_to(real_result)
    elif mutation == "alternate_depth":
        alternate = run_root / "allocator" / result_path.name
        alternate.write_bytes(result_path.read_bytes())
        expected = [(INSTANCE_A, LABEL_A, alternate)]
    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(fixture, tmp_path / "output.json", expected=expected)


@pytest.mark.parametrize("mutation", ["digest", "size", "path", "receipt_digest"])
def test_rejects_response_or_source_receipt_binding_mismatch(
    tmp_path: Path, mutation: str
) -> None:
    fixture = _fixture(tmp_path)
    receipt_path = fixture["receipt"]
    assert isinstance(receipt_path, Path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if mutation == "digest":
        receipt["sources"][0]["response_digest"] = "sha256:" + "0" * 64
    elif mutation == "size":
        receipt["sources"][0]["response_size_bytes"] += 1
    elif mutation == "path":
        receipt["sources"][0]["retained_path"] = receipt["sources"][1][
            "retained_path"
        ]
    elif mutation == "receipt_digest":
        receipt["provider_totals_usd"]["vast"] += 1
    if mutation != "receipt_digest":
        receipt["receipt_digest"] = canonical_digest(
            receipt, digest_field="receipt_digest"
        )
    _write(receipt_path, receipt)

    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(fixture, tmp_path / "output.json")


def test_rejects_symlinked_source_receipt_and_response(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path / "receipt-case")
    receipt = fixture["receipt"]
    assert isinstance(receipt, Path)
    receipt_link = tmp_path / "receipt-link.json"
    receipt_link.symlink_to(receipt)
    with pytest.raises(VastOfficialBillingExtractionError):
        materialize_vast_official_same_goal_reconciliation(
            provider_billing_source_receipt_path=receipt_link,
            expected_instances=[_spec(fixture, INSTANCE_A, LABEL_A)],
            output_path=tmp_path / "receipt-output.json",
        )

    fixture = _fixture(tmp_path / "response-case")
    response = fixture["responses"][0]
    assert isinstance(response, Path)
    real = response.with_suffix(".real.json")
    response.rename(real)
    response.symlink_to(real)
    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(
            fixture,
            tmp_path / "response-output.json",
            expected=[_spec(fixture, INSTANCE_A, LABEL_A)],
        )


def test_rejects_unposted_duplicate_expectation_and_existing_output(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    with pytest.raises(
        VastOfficialBillingExtractionError, match="vast_official_charge_unposted"
    ):
        _materialize(
            fixture,
            tmp_path / "missing.json",
            expected=[
                (
                    99_999_999,
                    "blueprint-missing-instance",
                    _spec(fixture, INSTANCE_A, LABEL_A)[2],
                )
            ],
        )
    with pytest.raises(
        VastOfficialBillingExtractionError,
        match="vast_official_expected_instances_duplicate",
    ):
        _materialize(
            fixture,
            tmp_path / "duplicate.json",
            expected=[
                _spec(fixture, INSTANCE_A, LABEL_A),
                (INSTANCE_A, LABEL_B, _spec(fixture, INSTANCE_B, LABEL_B)[2]),
            ],
        )
    output = tmp_path / "exists.json"
    output.write_text("user-owned", encoding="utf-8")
    with pytest.raises(
        VastOfficialBillingExtractionError, match="vast_official_output_invalid"
    ):
        _materialize(
            fixture,
            output,
            expected=[_spec(fixture, INSTANCE_A, LABEL_A)],
        )
    assert output.read_text(encoding="utf-8") == "user-owned"


def test_cli_materializes_without_provider_mutation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "cli.json"
    assert (
        main(
            [
                "--provider-billing-source-receipt",
                str(fixture["receipt"]),
                "--expected-instance",
                f"{INSTANCE_A}={LABEL_A}={_spec(fixture, INSTANCE_A, LABEL_A)[2]}",
                "--expected-instance",
                f"{INSTANCE_B}={LABEL_B}={_spec(fixture, INSTANCE_B, LABEL_B)[2]}",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["status"] == "materialized"
    assert summary["official_total_usd"] == pytest.approx(2.306)
    assert summary["provider_mutation_performed"] is False


def test_script_entrypoint_is_reachable_without_provider_access() -> None:
    completed = subprocess.run(
        [sys.executable, str(CLI_SCRIPT), "--help"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(REPOSITORY_ROOT / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert "--provider-billing-source-receipt" in completed.stdout
    assert "--expected-instance" in completed.stdout
