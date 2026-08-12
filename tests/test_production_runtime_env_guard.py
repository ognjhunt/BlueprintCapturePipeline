from blueprint_pipeline.production_runtime_env_guard import (
    build_production_runtime_env_guard,
)
from blueprint_pipeline.spend_authority_ledger_migration import (
    SpendAuthorityLedgerError,
)


def test_production_runtime_env_guard_blocks_ambiguous_runtime_env():
    report = build_production_runtime_env_guard(env={})

    assert report["status"] == "blocked"
    assert "missing_BLUEPRINT_LAUNCH_PROOF_MODE_production" in report["blockers"]
    assert "missing_or_false_PRIVACY_PIPELINE_ENABLED" in report["blockers"]
    assert "missing_or_false_PIPELINE_SYNC_REQUIRED" in report["blockers"]


def test_production_runtime_env_guard_accepts_fail_closed_production_env():
    report = build_production_runtime_env_guard(
        env={
            "BLUEPRINT_LAUNCH_PROOF_MODE": "production",
            "PRIVACY_PIPELINE_ENABLED": "true",
            "PRIVACY_FAIL_CLOSED": "true",
            "PIPELINE_SYNC_REQUIRED": "true",
            "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO": "true",
        }
    )

    assert report["status"] == "ready"
    assert report["blockers"] == []


def _production_env() -> dict[str, str]:
    return {
        "BLUEPRINT_LAUNCH_PROOF_MODE": "production",
        "PRIVACY_PIPELINE_ENABLED": "true",
        "PRIVACY_FAIL_CLOSED": "true",
        "PIPELINE_SYNC_REQUIRED": "true",
        "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO": "true",
    }


def test_guard_blocks_a_ledger_stranded_at_a_previous_root():
    """A unit that starts with an unadopted ledger has no single-use enforcement.

    Observed in production: binding BLUEPRINT_SPEND_AUTHORITY_ROOT moved the
    ledger and left its consumption records behind, so the new root read empty
    and every already-spent authorization looked unspent.
    """

    def _stranded() -> dict[str, object]:
        raise SpendAuthorityLedgerError("legacy_ledger_record_conflict:authz.json")

    report = build_production_runtime_env_guard(
        env=_production_env(), reconcile_spend_authority=_stranded
    )

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("spend_authority_ledger_not_reconciled")
        for blocker in report["blockers"]
    )
    assert report["spend_authority_ledger"]["status"] == "blocked"


def test_guard_blocks_an_unwritable_ledger_root():
    def _unwritable() -> dict[str, object]:
        raise PermissionError(13, "Permission denied")

    report = build_production_runtime_env_guard(
        env=_production_env(), reconcile_spend_authority=_unwritable
    )

    assert report["status"] == "blocked"
    assert (
        "spend_authority_ledger_not_reconciled:unwritable_root" in report["blockers"]
    )


def test_guard_retains_the_reconciliation_receipt_when_ready():
    receipt = {
        "schema_version": "spend_authority_ledger_reconciliation.v1",
        "status": "reconciled",
        "records_adopted": 2,
    }

    report = build_production_runtime_env_guard(
        env=_production_env(), reconcile_spend_authority=lambda: receipt
    )

    assert report["status"] == "ready"
    assert report["spend_authority_ledger"] == receipt


def test_guard_reconciles_the_ledger_on_every_start_by_default():
    """Not only at install: a host restored from an image never runs installers."""
    calls: list[int] = []

    build_production_runtime_env_guard(
        env=_production_env(),
        reconcile_spend_authority=lambda: calls.append(1) or {"status": "reconciled"},
    )

    assert calls == [1]
