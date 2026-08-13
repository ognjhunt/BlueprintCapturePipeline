from blueprint_pipeline.production_runtime_env_guard import (
    build_production_runtime_env_guard,
)
from blueprint_pipeline.spend_authority_ledger_migration import (
    SpendAuthorityLedgerError,
)
from blueprint_pipeline.task_evaluation_launch_catalog import (
    LaunchCatalogError,
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


def test_guard_repairs_a_catalog_that_drifted_from_the_directory():
    """Observed live: 7 profiles published, 4 served, 3 unreachable."""
    receipt = {
        "schema_version": "task_evaluation_launch_catalog_reconciliation.v1",
        "status": "repaired",
        "profile_ids_added": ["profile-x"],
    }

    report = build_production_runtime_env_guard(
        env=_production_env(), reconcile_launch_catalog=lambda: receipt
    )

    assert report["status"] == "ready"
    assert report["launch_profile_catalog"] == receipt


def test_guard_blocks_when_the_published_directory_cannot_be_projected():
    """A catalog that silently omits a published profile is the defect itself."""

    def _invalid() -> dict[str, object]:
        raise LaunchCatalogError("published_profile_invalid:broken.json:not_an_object")

    report = build_production_runtime_env_guard(
        env=_production_env(), reconcile_launch_catalog=_invalid
    )

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("launch_profile_catalog_not_reconciled")
        for blocker in report["blockers"]
    )


def test_guard_skips_catalog_reconciliation_when_no_profiles_are_configured():
    """A host that publishes no launch profiles has no catalog to keep."""
    report = build_production_runtime_env_guard(env=_production_env())

    assert report["launch_profile_catalog"]["status"] == "not_configured"
    assert report["status"] == "ready"


def test_guard_reconciles_the_configured_catalog_from_the_environment(tmp_path):
    """The guard must read the same env the intake resolves the catalog from."""
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    catalog = tmp_path / "catalog.json"

    report = build_production_runtime_env_guard(
        env={
            **_production_env(),
            "BLUEPRINT_TASK_EVALUATION_LAUNCH_PROFILE_DIR": str(profile_dir),
            "BLUEPRINT_TASK_EVALUATION_LAUNCH_PUBLIC_CATALOG_PATH": str(catalog),
        }
    )

    assert report["launch_profile_catalog"]["status"] == "repaired"
    assert catalog.read_text(encoding="utf-8") == "[]\n"
