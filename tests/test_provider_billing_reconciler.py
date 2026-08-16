import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest

from blueprint_pipeline.provider_billing_reconciler import (
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
    return root


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

    result = reconcile_provider_billing(
        secrets_dir=_secrets(tmp_path),
        billing_export_path=export,
        audit_root=tmp_path / "audit",
        start_at="2026-01-01T00:00:00Z",
        now=NOW,
        transport=transport,
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
    assert all(header.endswith("-value") for _, header in transport.requests)
    assert all("-value" not in json.dumps(row) for row in source["sources"])


def test_atomic_service_owned_refresh_is_trusted_by_root_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mirror the production blueprint producer followed by a root guard."""

    export = tmp_path / "guard" / "provider_billing_export.json"
    export.parent.mkdir()
    export.write_text("stale\n", encoding="utf-8")
    stale_inode = export.stat().st_ino

    reconcile_provider_billing(
        secrets_dir=_secrets(tmp_path),
        billing_export_path=export,
        audit_root=tmp_path / "audit",
        start_at="2026-01-01T00:00:00Z",
        now=NOW,
        transport=_Transport(),
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
            pw_uid=os.getuid()
            if account == guard.BILLING_EXPORT_PRODUCER_ACCOUNT
            else 8675309
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
    assert (
        "provider_billing_export_writable_by_group_or_world"
        in writable["blockers"]
    )


def test_failed_refresh_preserves_prior_export(tmp_path: Path) -> None:
    export = tmp_path / "provider_billing_export.json"
    export.write_text("sentinel\n", encoding="utf-8")

    def fail(_request, _timeout: float) -> bytes:
        raise ProviderBillingReconciliationError("provider_billing_request_failed")

    with pytest.raises(ProviderBillingReconciliationError, match="provider_billing_request_failed"):
        reconcile_provider_billing(
            secrets_dir=_secrets(tmp_path),
            billing_export_path=export,
            audit_root=tmp_path / "audit",
            start_at="2026-01-01T00:00:00Z",
            now=NOW,
            transport=fail,
        )

    assert export.read_text(encoding="utf-8") == "sentinel\n"


def test_accepts_digitalocean_daily_balance_after_24_hour_boundary(
    tmp_path: Path,
) -> None:
    result = reconcile_provider_billing(
        secrets_dir=_secrets(tmp_path),
        billing_export_path=tmp_path / "export.json",
        audit_root=tmp_path / "audit",
        start_at="2026-01-01T00:00:00Z",
        now=datetime(2026, 8, 16, 3, 31, tzinfo=timezone.utc),
        transport=_Transport(digitalocean_generated_at="2026-08-15T03:16:38Z"),
    )

    assert result["status"] == "reconciled"
    assert result["provider_totals_usd"]["digitalocean"] == 8.0


def test_rejects_digitalocean_balance_older_than_two_daily_intervals(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ProviderBillingReconciliationError, match="digitalocean_balance_stale"
    ):
        reconcile_provider_billing(
            secrets_dir=_secrets(tmp_path),
            billing_export_path=tmp_path / "export.json",
            audit_root=tmp_path / "audit",
            start_at="2026-01-01T00:00:00Z",
            now=datetime(2026, 8, 16, 3, 31, tzinfo=timezone.utc),
            transport=_Transport(digitalocean_generated_at="2026-08-14T03:16:38Z"),
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
