import hashlib
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from botocore.credentials import Credentials

from blueprint_pipeline.provider_billing_reconciler import (
    AWS_BILLING_UNAVAILABLE_REASON,
    ProviderBillingReconciliationError,
    reconcile_provider_billing,
)
from scripts import gpu_spend_guard as guard


NOW = datetime(2026, 8, 10, 17, 0, tzinfo=timezone.utc)


def _secrets(tmp_path: Path) -> Path:
    root = tmp_path / "secrets"
    root.mkdir()
    for name in ("runpod_api_key", "vast_api_key", "digitalocean_api_token"):
        path = root / name
        path.write_text(f"{name}-value\n", encoding="utf-8")
        path.chmod(0o600)
    aws = root / "aws_agent_credentials"
    aws.write_text("[test]\naws_access_key_id = test\naws_secret_access_key = test\n")
    aws.chmod(0o600)
    return root


def _aws_kwargs(secret_root: Path) -> dict:
    return {
        "aws_account_id": "111710313013",
        "aws_credentials_file": secret_root / "aws_agent_credentials",
        "aws_profile": "test",
        "aws_credentials": Credentials("AKIAIOSFODNN7EXAMPLE", "test-secret"),
    }


class _Transport:
    def __init__(self, *, digitalocean_generated_at: str = "2026-08-10T16:30:00Z") -> None:
        self.requests: list[tuple[str, str]] = []
        self.digitalocean_generated_at = digitalocean_generated_at

    def __call__(self, request, _timeout: float) -> bytes:
        self.requests.append((request.full_url, request.headers.get("Authorization", "")))
        parsed = urlsplit(request.full_url)
        query = parse_qs(parsed.query)
        if parsed.netloc == "rest.runpod.io":
            resource = parsed.path.rsplit("/", 1)[-1]
            rows = {
                "pods": [{"amount": 3.25}],
                "endpoints": [],
                "networkvolumes": [{"amount": 0.5, "highPerformanceStorageAmount": 0.25}],
            }[resource]
            return json.dumps(rows).encode()
        if parsed.netloc == "console.vast.ai":
            if "after_token" not in query:
                payload = {
                    "success": True,
                    "results": [{"amount": 4.0}],
                    "next_token": "page-two",
                }
            else:
                payload = {
                    "success": True,
                    "results": [{"amount": 1.5}],
                    "next_token": None,
                }
            return json.dumps(payload).encode()
        if parsed.netloc.endswith("amazonaws.com"):
            raise AssertionError("automatic AWS billing request forbidden")
        if parsed.path.endswith("/balance"):
            return json.dumps(
                {
                    "generated_at": self.digitalocean_generated_at,
                    "month_to_date_usage": "2.00",
                }
            ).encode()
        if parsed.path.endswith("/invoices"):
            return json.dumps(
                {
                    "invoices": [
                        {
                            "invoice_uuid": "july",
                            "invoice_period": "2026-07",
                            "amount": "6.00",
                        }
                    ],
                    "invoice_preview": {
                        "invoice_period": "2026-08",
                        "amount": "2.00",
                    },
                    "links": {"pages": {}},
                }
            ).encode()
        raise AssertionError(request.full_url)


def test_reconciles_exact_provider_responses_into_atomic_guard_export(
    tmp_path: Path,
) -> None:
    transport = _Transport()
    export = tmp_path / "guard" / "provider_billing_export.json"

    secrets = _secrets(tmp_path)
    result = reconcile_provider_billing(
        secrets_dir=secrets,
        billing_export_path=export,
        audit_root=tmp_path / "audit",
        start_at="2026-01-01T00:00:00Z",
        now=NOW,
        transport=transport,
        **_aws_kwargs(secrets),
    )

    assert result["status"] == "reconciled"
    assert result["provider_totals_usd"] == {
        "runpod": 4.0,
        "vast": 5.5,
        "digitalocean": 8.0,
    }
    assert result["provider_mutation_performed"] is False
    payload = json.loads(export.read_text())
    assert payload == {
        "schema_version": "blueprint.provider_billing_export.v1",
        "generated_at": "2026-08-10T17:00:00+00:00",
        "currency": "USD",
        "scope": "blueprint_beta_100_user_cohort",
        "provider_totals_usd": result["provider_totals_usd"],
    }
    source = json.loads(Path(result["source_receipt_path"]).read_text())
    assert source["status"] == "reconciled"
    assert len(source["sources"]) == 7
    assert all(Path(row["retained_path"]).is_file() for row in source["sources"])
    assert all(row["response_digest"].startswith("sha256:") for row in source["sources"])
    assert all(
        header.endswith("-value")
        for url, header in transport.requests
        if "amazonaws.com" not in url
    )
    assert not any("amazonaws.com" in url for url, _header in transport.requests)
    assert all("-value" not in json.dumps(row) for row in source["sources"])


def test_repeated_provider_responses_keep_distinct_paths_on_one_digest_inode(
    tmp_path: Path,
) -> None:
    audit_root = tmp_path / "audit"
    secrets = _secrets(tmp_path)
    first = reconcile_provider_billing(
        secrets_dir=secrets,
        billing_export_path=tmp_path / "guard" / "billing.json",
        audit_root=audit_root,
        start_at="2026-01-01T00:00:00Z",
        now=NOW,
        transport=_Transport(),
        **_aws_kwargs(secrets),
    )
    second = reconcile_provider_billing(
        secrets_dir=secrets,
        billing_export_path=tmp_path / "guard" / "billing.json",
        audit_root=audit_root,
        start_at="2026-01-01T00:00:00Z",
        now=NOW + timedelta(minutes=1),
        transport=_Transport(),
        **_aws_kwargs(secrets),
    )

    first_source = json.loads(Path(first["source_receipt_path"]).read_text())
    second_source = json.loads(Path(second["source_receipt_path"]).read_text())
    first_path = Path(first_source["sources"][0]["retained_path"])
    second_path = Path(second_source["sources"][0]["retained_path"])

    assert first_path != second_path
    assert first_path.stat().st_ino == second_path.stat().st_ino
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first_path.stat().st_mode & 0o777 == 0o600
    assert first_source["sources"][0]["response_digest"] == second_source["sources"][0][
        "response_digest"
    ]


def test_existing_digest_object_with_wrong_bytes_blocks_reconciliation(
    tmp_path: Path,
) -> None:
    audit_root = tmp_path / "audit"
    audit_root.mkdir(mode=0o700)
    payload = json.dumps([{"amount": 3.25}]).encode()
    hexadecimal = hashlib.sha256(payload).hexdigest()
    object_parent = audit_root / "objects" / "sha256" / hexadecimal[:2]
    object_parent.mkdir(parents=True, mode=0o700)
    for path in (
        audit_root / "objects",
        audit_root / "objects" / "sha256",
        object_parent,
    ):
        path.chmod(0o700)
    object_path = object_parent / hexadecimal
    object_path.write_bytes(b"not the bound response")
    object_path.chmod(0o600)
    secrets = _secrets(tmp_path)

    with pytest.raises(
        ProviderBillingReconciliationError,
        match="provider_billing_audit_response_metadata_invalid|"
        "provider_billing_audit_response_digest_mismatch",
    ):
        reconcile_provider_billing(
            secrets_dir=secrets,
            billing_export_path=tmp_path / "guard" / "billing.json",
            audit_root=audit_root,
            start_at="2026-01-01T00:00:00Z",
            now=NOW,
            transport=_Transport(),
            **_aws_kwargs(secrets),
        )
    assert list(audit_root.glob("20*Z")) == []


def test_atomic_service_owned_refresh_is_trusted_by_root_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror the production blueprint producer followed by a root guard."""

    export = tmp_path / "guard" / "provider_billing_export.json"
    export.parent.mkdir()
    export.write_text("stale\n", encoding="utf-8")
    stale_inode = export.stat().st_ino

    secrets = _secrets(tmp_path)
    reconcile_provider_billing(
        secrets_dir=secrets,
        billing_export_path=export,
        audit_root=tmp_path / "audit",
        start_at="2026-01-01T00:00:00Z",
        now=NOW,
        transport=_Transport(),
        **_aws_kwargs(secrets),
    )

    refreshed = export.stat()
    assert refreshed.st_ino != stale_inode
    assert refreshed.st_uid == os.getuid()
    assert refreshed.st_mode & 0o777 == 0o600

    # The test process represents the sandboxed ``blueprint`` producer.  Make
    # only the consumer root-like, exactly matching the production mismatch.
    monkeypatch.setattr(guard.os, "geteuid", lambda: 0)
    monkeypatch.setattr(
        guard.pwd,
        "getpwnam",
        lambda account: SimpleNamespace(
            pw_uid=os.getuid() if account == guard.BILLING_EXPORT_PRODUCER_ACCOUNT else 8675309
        ),
    )

    result = guard.reconcile_billing_export(
        billing_export_path=export,
        instances=[],
        now=NOW.timestamp(),
        required=True,
    )

    assert result["status"] == "reconciled"
    assert result["blockers"] == []

    # Trusting the exact service owner must not weaken the write boundary.
    export.chmod(0o620)
    writable = guard.reconcile_billing_export(
        billing_export_path=export,
        instances=[],
        now=NOW.timestamp(),
        required=True,
    )
    assert writable["status"] == "blocked"
    assert "provider_billing_export_writable_by_group_or_world" in writable["blockers"]


def test_failed_refresh_preserves_prior_export(tmp_path: Path) -> None:
    export = tmp_path / "provider_billing_export.json"
    export.write_text("sentinel\n", encoding="utf-8")

    def fail(_request, _timeout: float) -> bytes:
        raise ProviderBillingReconciliationError("provider_billing_request_failed")

    secrets = _secrets(tmp_path)
    with pytest.raises(
        ProviderBillingReconciliationError,
        match="provider_billing_no_provider_covered",
    ):
        reconcile_provider_billing(
            secrets_dir=secrets,
            billing_export_path=export,
            audit_root=tmp_path / "audit",
            start_at="2026-01-01T00:00:00Z",
            now=NOW,
            transport=fail,
            **_aws_kwargs(secrets),
        )

    assert export.read_text(encoding="utf-8") == "sentinel\n"


def test_automatic_aws_billing_removed_without_starving_other_providers(tmp_path: Path) -> None:
    secrets = _secrets(tmp_path)
    export = tmp_path / "provider_billing_export.json"
    transport = _Transport()
    result = reconcile_provider_billing(
        secrets_dir=secrets,
        billing_export_path=export,
        audit_root=tmp_path / "audit",
        start_at="2026-01-01T00:00:00Z",
        now=NOW,
        transport=transport,
        **_aws_kwargs(secrets),
    )
    assert result["status"] == "reconciled"
    assert result["covered_provider_ids"] == ["digitalocean", "runpod", "vast"]
    assert result["uncovered_provider_ids"] == ["aws"]
    assert result["optional_provider_failures"] == {"aws": AWS_BILLING_UNAVAILABLE_REASON}
    assert "aws" not in json.loads(export.read_text())["provider_totals_usd"]
    assert not any("amazonaws.com" in url for url, _header in transport.requests)
    receipt = json.loads(Path(result["source_receipt_path"]).read_text())
    assert all(row["provider"] != "aws" for row in receipt["sources"])

    # A current export for the other providers cannot admit a live AWS resource.
    live_aws = guard.GpuInstance(provider="aws", id="i-offline", name="offline",
                                 state="active", booted=True, live=True)
    admission = guard.reconcile_billing_export(
        billing_export_path=export, instances=[live_aws], now=NOW.timestamp(), required=True,
    )
    assert admission["status"] == "blocked"
    assert "provider_billing_export_missing:aws" in admission["blockers"]


def test_required_aws_billing_fails_without_requests_or_replacing_prior_export(tmp_path: Path) -> None:
    secrets = _secrets(tmp_path)
    export = tmp_path / "provider_billing_export.json"
    export.write_text("sentinel\n", encoding="utf-8")
    transport = _Transport()
    with pytest.raises(ProviderBillingReconciliationError, match=AWS_BILLING_UNAVAILABLE_REASON):
        reconcile_provider_billing(
            secrets_dir=secrets, billing_export_path=export, audit_root=tmp_path / "audit",
            start_at="2026-01-01T00:00:00Z", now=NOW, transport=transport,
            required_providers=("aws",), **_aws_kwargs(secrets),
        )
    assert transport.requests == []
    assert export.read_text(encoding="utf-8") == "sentinel\n"


def test_optional_digitalocean_failure_does_not_block_vast_export(
    tmp_path: Path,
) -> None:
    class DigitalOceanUnavailableTransport(_Transport):
        def __call__(self, request, timeout: float) -> bytes:
            if urlsplit(request.full_url).netloc == "api.digitalocean.com":
                raise ProviderBillingReconciliationError(
                    "provider_billing_request_failed"
                )
            return super().__call__(request, timeout)

    secrets = _secrets(tmp_path)
    export = tmp_path / "provider_billing_export.json"
    result = reconcile_provider_billing(
        secrets_dir=secrets,
        billing_export_path=export,
        audit_root=tmp_path / "audit",
        start_at="2026-01-01T00:00:00Z",
        now=NOW,
        transport=DigitalOceanUnavailableTransport(),
        required_providers=("vast",),
        **_aws_kwargs(secrets),
    )

    assert result["status"] == "reconciled"
    assert result["covered_provider_ids"] == ["runpod", "vast"]
    assert result["uncovered_provider_ids"] == ["aws", "digitalocean"]
    assert result["optional_provider_failures"] == {
        "digitalocean": "provider_billing_request_failed",
        "aws": AWS_BILLING_UNAVAILABLE_REASON,
    }
    assert "vast" in json.loads(export.read_text())["provider_totals_usd"]


def test_required_digitalocean_failure_preserves_prior_export(tmp_path: Path) -> None:
    class DigitalOceanUnavailableTransport(_Transport):
        def __call__(self, request, timeout: float) -> bytes:
            if urlsplit(request.full_url).netloc == "api.digitalocean.com":
                raise ProviderBillingReconciliationError(
                    "provider_billing_request_failed"
                )
            return super().__call__(request, timeout)

    secrets = _secrets(tmp_path)
    export = tmp_path / "provider_billing_export.json"
    export.write_text("sentinel\n", encoding="utf-8")
    with pytest.raises(
        ProviderBillingReconciliationError,
        match="provider_billing_request_failed",
    ):
        reconcile_provider_billing(
            secrets_dir=secrets,
            billing_export_path=export,
            audit_root=tmp_path / "audit",
            start_at="2026-01-01T00:00:00Z",
            now=NOW,
            transport=DigitalOceanUnavailableTransport(),
            required_providers=("digitalocean",),
            **_aws_kwargs(secrets),
        )

    assert export.read_text(encoding="utf-8") == "sentinel\n"


def test_accepts_digitalocean_daily_balance_after_24_hour_boundary(
    tmp_path: Path,
) -> None:
    secrets = _secrets(tmp_path)
    result = reconcile_provider_billing(
        secrets_dir=secrets,
        billing_export_path=tmp_path / "export.json",
        audit_root=tmp_path / "audit",
        start_at="2026-01-01T00:00:00Z",
        now=datetime(2026, 8, 16, 3, 31, tzinfo=timezone.utc),
        transport=_Transport(digitalocean_generated_at="2026-08-15T03:16:38Z"),
        **_aws_kwargs(secrets),
    )

    assert result["status"] == "reconciled"
    assert result["provider_totals_usd"]["digitalocean"] == 8.0


def test_rejects_digitalocean_balance_older_than_two_daily_intervals(
    tmp_path: Path,
) -> None:
    secrets = _secrets(tmp_path)
    with pytest.raises(ProviderBillingReconciliationError, match="digitalocean_balance_stale"):
        reconcile_provider_billing(
            secrets_dir=secrets,
            billing_export_path=tmp_path / "export.json",
            audit_root=tmp_path / "audit",
            start_at="2026-01-01T00:00:00Z",
            now=datetime(2026, 8, 16, 3, 31, tzinfo=timezone.utc),
            transport=_Transport(digitalocean_generated_at="2026-08-14T03:16:38Z"),
            required_providers=("digitalocean",),
            **_aws_kwargs(secrets),
        )


def test_secret_symlink_is_rejected_before_network_access(tmp_path: Path) -> None:
    root = _secrets(tmp_path)
    target = root / "real-vast-key"
    target.write_text("value\n", encoding="utf-8")
    target.chmod(0o600)
    (root / "vast_api_key").unlink()
    (root / "vast_api_key").symlink_to(target)

    with pytest.raises(
        ProviderBillingReconciliationError, match="secret_symlink_forbidden:vast_api_key"
    ):
        reconcile_provider_billing(
            secrets_dir=root,
            billing_export_path=tmp_path / "export.json",
            audit_root=tmp_path / "audit",
            start_at="2026-01-01T00:00:00Z",
            now=NOW,
            transport=_Transport(),
        )


def test_legacy_aws_configuration_is_ignored_without_reading_credentials(tmp_path: Path) -> None:
    secrets = _secrets(tmp_path)
    transport = _Transport()
    result = reconcile_provider_billing(
        secrets_dir=secrets, billing_export_path=tmp_path / "export.json",
        audit_root=tmp_path / "audit", start_at="2026-01-01T00:00:00Z", now=NOW,
        transport=transport, aws_account_id="999999999999",
        aws_credentials_file=tmp_path / "does-not-exist", aws_profile="missing",
    )
    assert result["optional_provider_failures"]["aws"] == AWS_BILLING_UNAVAILABLE_REASON
    assert "aws" not in result["provider_totals_usd"]
    assert not any("amazonaws.com" in url for url, _header in transport.requests)


def test_repeated_page_cursor_is_bounded_and_preserves_prior_accounting(tmp_path):
    secrets = _secrets(tmp_path)
    export = tmp_path / "export.json"
    kwargs = dict(secrets_dir=secrets, billing_export_path=export, audit_root=tmp_path / "audit",
                  start_at="2026-01-01T00:00:00Z", now=NOW, required_providers=("vast",), **_aws_kwargs(secrets))
    reconcile_provider_billing(**kwargs, transport=_Transport())
    prior = export.read_bytes()
    class Loop(_Transport):
        vast_reads = 0
        def __call__(self, request, timeout):
            data = super().__call__(request, timeout)
            if urlsplit(request.full_url).netloc == "console.vast.ai":
                self.vast_reads += 1
                value = json.loads(data)
                value["next_token"] = "page-two"
                return json.dumps(value).encode()
            return data
    transport = Loop()
    with pytest.raises(ProviderBillingReconciliationError, match="vast_billing_cursor_invalid"):
        reconcile_provider_billing(**kwargs, transport=transport)
    assert transport.vast_reads == 2
    assert export.read_bytes() == prior
