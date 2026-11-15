"""A rebuilt control-plane host gets its single-use spend-authority ledger from the installer."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTALLER = REPO_ROOT / "scripts" / "install_live_pipeline_control_plane.sh"


def _installer() -> str:
    return INSTALLER.read_text(encoding="utf-8")


def test_binds_the_spend_authority_ledger_off_the_service_home() -> None:
    """Single-use spend enforcement must not depend on a home directory.

    The hardened units set ``ProtectHome=true`` and the service account's home
    is ``/nonexistent``, so a ledger under ``Path.home()`` is unwritable and
    every paid run fails after its authority validates. A rebuilt host must get
    this binding from the installer, not from someone editing the env file.
    """
    text = _installer()
    assert "BLUEPRINT_SPEND_AUTHORITY_ROOT" in text, (
        "the installer must bind the spend-authority ledger; without it the "
        "single-use consumption record cannot be written on a hardened host"
    )
    assert "/var/lib/blueprint" in text.split("BLUEPRINT_SPEND_AUTHORITY_ROOT")[0][-400:] or (
        re.search(r'SPEND_AUTHORITY_ROOT="\$\{SPEND_AUTHORITY_ROOT:-/var/lib/blueprint', text)
    ), "the ledger must live inside the ReadWritePaths the units already grant"


def test_spend_authority_ledger_is_not_group_or_world_accessible() -> None:
    """A second writer could forge or delete a record and re-fund an allocation."""
    text = _installer()
    assert re.search(r'chmod 0700 "\$\{SPEND_AUTHORITY_ROOT\}"', text), (
        "the consumption check refuses a group- or world-accessible tree"
    )


def test_installer_reconciles_a_ledger_stranded_at_a_previous_root() -> None:
    """Binding the root moves the ledger; the records do not follow.

    Observed in production after PR #453 deployed: two consumption records from
    real paid runs stayed at the previous root while the newly bound root
    reported zero, so every authorization spent there looked unspent again.
    """
    text = _installer()
    assert "blueprint_pipeline.spend_authority_ledger_migration" in text, (
        "the installer must adopt a ledger left at a previous root; an empty "
        "ledger and a ledger with no matching record are indistinguishable at "
        "the point of use"
    )


def test_installer_refuses_to_continue_when_reconciliation_fails() -> None:
    """Installing over an unadopted ledger ships a host with no single-use gate."""
    text = _installer()
    block = text.split("spend_authority_ledger_migration", 1)[1][:600]
    assert "exit 1" in block, (
        "a ledger that cannot be proven adopted must stop the install"
    )


def test_ledger_reconciliation_runs_as_the_service_account() -> None:
    """Root-owned records are refused by the consumption check that reads them."""
    text = _installer()
    block = text.split("spend_authority_ledger_migration", 1)[0][-500:]
    assert 'runuser -u "${SERVICE_USER}"' in block, (
        "adopted records must carry the ownership the consumption check requires"
    )
