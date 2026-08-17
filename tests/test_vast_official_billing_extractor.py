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
    return {
        "audit": audit,
        "responses": responses,
        "receipt": receipt_path,
        "receipt_value": receipt,
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
    expected: list[tuple[int, str]] | None = None,
    prior: Path | None = None,
) -> dict:
    return materialize_vast_official_same_goal_reconciliation(
        provider_billing_source_receipt_path=fixture["receipt"],
        expected_instances=expected
        or [(INSTANCE_A, LABEL_A), (INSTANCE_B, LABEL_B)],
        output_path=output,
        prior_reconciliation_path=prior,
    )


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
        fixture, prior_path, expected=[(INSTANCE_A, LABEL_A)]
    )
    output = tmp_path / "extended.json"
    value = _materialize(
        fixture,
        output,
        expected=[(INSTANCE_B, LABEL_B)],
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
    _materialize(fixture, prior_path, expected=[(INSTANCE_A, LABEL_A)])
    prior_path.chmod(0o600)
    tampered = json.loads(prior_path.read_text(encoding="utf-8"))
    tampered["official_total_usd"] = 0.0
    _write(prior_path, tampered)
    with pytest.raises(VastOfficialBillingExtractionError):
        _materialize(
            fixture,
            tmp_path / "tampered-output.json",
            expected=[(INSTANCE_B, LABEL_B)],
            prior=prior_path,
        )

    prior_path.unlink()
    _materialize(fixture, prior_path, expected=[(INSTANCE_A, LABEL_A)])
    with pytest.raises(
        VastOfficialBillingExtractionError, match="vast_official_prior_overlap"
    ):
        _materialize(
            fixture,
            tmp_path / "overlap-output.json",
            expected=[(INSTANCE_A, LABEL_A)],
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
            expected=[(INSTANCE_A, LABEL_A)],
        )


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
            expected_instances=[(INSTANCE_A, LABEL_A)],
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
            expected=[(INSTANCE_A, LABEL_A)],
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
            expected=[(99_999_999, "blueprint-missing-instance")],
        )
    with pytest.raises(
        VastOfficialBillingExtractionError,
        match="vast_official_expected_instances_duplicate",
    ):
        _materialize(
            fixture,
            tmp_path / "duplicate.json",
            expected=[(INSTANCE_A, LABEL_A), (INSTANCE_A, LABEL_B)],
        )
    output = tmp_path / "exists.json"
    output.write_text("user-owned", encoding="utf-8")
    with pytest.raises(
        VastOfficialBillingExtractionError, match="vast_official_output_invalid"
    ):
        _materialize(
            fixture,
            output,
            expected=[(INSTANCE_A, LABEL_A)],
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
                f"{INSTANCE_A}={LABEL_A}",
                "--expected-instance",
                f"{INSTANCE_B}={LABEL_B}",
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
