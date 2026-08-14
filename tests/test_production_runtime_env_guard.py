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


def _lock_env(base) -> dict[str, str]:
    env = _production_env()
    env["VAST_LAUNCH_LOCK_FILE"] = str(base)
    return env


def test_guard_blocks_a_launch_lock_slot_the_service_account_cannot_use(tmp_path):
    """A root-created slot silently halves the fleet's authorized concurrency.

    Production had `slot1`/`slot2` owned `root:root` at 0644 while the adapter
    runs as `blueprint`, so the N=3 semaphore was really N=1 and the overflow
    path raised instead of refusing. Nothing detected it: the lane that held
    slot 0 succeeded, so every signal looked healthy.
    """

    from blueprint_pipeline.vast_provider_adapter import vast_launch_lock_paths

    base = tmp_path / "provider-locks" / "vast_paid_launch.lock"
    base.parent.mkdir(parents=True, exist_ok=True)
    slots = vast_launch_lock_paths(base)
    slots[0].touch()
    slots[1].mkdir()  # exists, cannot be opened, for any uid including root

    report = build_production_runtime_env_guard(env=_lock_env(base))

    assert report["status"] == "blocked"
    assert any(
        blocker.startswith("paid_launch_lock_slot_unusable")
        for blocker in report["blockers"]
    ), report["blockers"]
    assert slots[1].name in " ".join(report["blockers"])


def test_guard_is_ready_when_every_existing_launch_lock_slot_is_usable(tmp_path):
    """An absent slot is fine -- the adapter creates it as the service account."""

    from blueprint_pipeline.vast_provider_adapter import vast_launch_lock_paths

    base = tmp_path / "provider-locks" / "vast_paid_launch.lock"
    base.parent.mkdir(parents=True, exist_ok=True)
    vast_launch_lock_paths(base)[0].touch()

    report = build_production_runtime_env_guard(env=_lock_env(base))

    assert report["status"] == "ready", report["blockers"]
    assert report["paid_launch_lock_slots"]["status"] == "usable"


def test_guard_probes_every_slot_the_adapter_would_use(tmp_path):
    """The checked set is rediscovered from the adapter, never hand-listed.

    Raising or lowering the concurrency ceiling must not be able to leave a
    slot unchecked, which is how a list drifts out of date without failing.
    """

    from blueprint_pipeline.vast_provider_adapter import vast_launch_lock_paths

    base = tmp_path / "provider-locks" / "vast_paid_launch.lock"
    base.parent.mkdir(parents=True, exist_ok=True)
    expected = vast_launch_lock_paths(base)
    for slot in expected:
        slot.touch()

    report = build_production_runtime_env_guard(env=_lock_env(base))

    assert report["paid_launch_lock_slots"]["slots_probed"] == [
        str(slot) for slot in expected
    ]


def test_guard_precreates_absent_slots_as_the_service_account(tmp_path):
    """Pre-creating is what actually stops the fault from recurring.

    `open("a+")` never changes the owner of a file that already exists, so a
    slot created correctly at first start survives any later tool that is run
    as root. Creation happens as the service account under the guard's own
    umask-independent 0600, which is the mode every paid lane demands.
    """

    from blueprint_pipeline.vast_provider_adapter import vast_launch_lock_paths

    base = tmp_path / "provider-locks" / "vast_paid_launch.lock"
    base.parent.mkdir(parents=True, exist_ok=True)

    report = build_production_runtime_env_guard(env=_lock_env(base))

    assert report["status"] == "ready", report["blockers"]
    for slot in vast_launch_lock_paths(base):
        assert slot.is_file(), f"{slot.name} was not provisioned"
        assert slot.stat().st_mode & 0o777 == 0o600, oct(slot.stat().st_mode)
    assert report["paid_launch_lock_slots"]["created_slots"] == [
        str(slot) for slot in vast_launch_lock_paths(base)
    ]


def test_guard_does_not_create_a_lock_directory_that_was_never_provisioned(tmp_path):
    """A guard must not scatter state into a home the deployment never set up.

    On a developer machine the default path resolves under `~/.blueprint-secrets`,
    which the installer alone owns. No parent directory means this host has no
    provider-lock tree, which is reported rather than invented.
    """

    base = tmp_path / "never-installed" / "vast_paid_launch.lock"

    report = build_production_runtime_env_guard(env=_lock_env(base))

    assert report["status"] == "ready", report["blockers"]
    assert report["paid_launch_lock_slots"]["status"] == "not_provisioned"
    assert not base.parent.exists()
