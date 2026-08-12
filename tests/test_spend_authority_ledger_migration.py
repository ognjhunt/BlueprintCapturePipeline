"""Moving the ledger root must not silently discard single-use history.

PR #453 let a deployment place the consumption ledger inside its unit's
``ReadWritePaths`` instead of an unwritable ``$HOME``. Binding that root on an
existing host moves the ledger, and the records written under the previous root
stay where they are: the new root reads empty, so every authorization already
spent there looks unspent again.

That is the single security property the ledger exists to provide, so an
unadopted legacy ledger must fail closed rather than be ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.spend_authority_consumption_root import (
    SPEND_AUTHORITY_ROOT_ENV,
    consumption_root,
)
from blueprint_pipeline.spend_authority_ledger_migration import (
    LEGACY_ROOTS_ENV,
    SpendAuthorityLedgerError,
    discover_legacy_ledgers,
    reconcile_spend_authority_ledger,
)

LEGACY_DIRECTORY_NAME = ".blueprint-spend-authority"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let discovery reach the developer's own ledger.

    ``$HOME/.blueprint-spend-authority`` is the pre-#453 default root, so it is
    a legitimate discovery target -- which means an un-isolated test adopts the
    real one. That is exactly what happened while writing these.
    """
    monkeypatch.setenv("HOME", str(tmp_path / "isolated-home"))
    monkeypatch.delenv(LEGACY_ROOTS_ENV, raising=False)


def _write_record(root: Path, name: str, payload: dict | None = None) -> Path:
    consumed = root / "consumed"
    consumed.mkdir(parents=True, exist_ok=True)
    path = consumed / f"{name}.json"
    path.write_text(
        json.dumps(payload or {"authorization": name, "consumed": True}, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _bind(monkeypatch: pytest.MonkeyPatch, root: Path) -> Path:
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(root))
    return root


def test_no_legacy_ledger_is_a_clean_reconciliation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _bind(monkeypatch, tmp_path / "spend-authority")

    receipt = reconcile_spend_authority_ledger(search_bases=[tmp_path])

    assert receipt["status"] == "no_legacy_ledger"
    assert receipt["records_adopted"] == 0
    assert receipt["blockers"] == []
    assert receipt["root"] == str(root)


def test_adopts_records_left_behind_at_the_previous_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact production condition: the root moved, the records did not."""
    legacy = tmp_path / "spend-authority-home" / LEGACY_DIRECTORY_NAME
    _write_record(legacy, "retained-scene-render-aaa")
    _write_record(legacy, "retained-scene-render-bbb")
    _bind(monkeypatch, tmp_path / "spend-authority")

    receipt = reconcile_spend_authority_ledger(search_bases=[tmp_path])

    assert receipt["status"] == "reconciled"
    assert receipt["records_adopted"] == 2
    adopted = {path.name for path in consumption_root().glob("*.json")}
    assert adopted == {
        "retained-scene-render-aaa.json",
        "retained-scene-render-bbb.json",
    }


def test_adopted_records_keep_their_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "old" / LEGACY_DIRECTORY_NAME
    original = _write_record(legacy, "authz", {"instance_id": 47574163, "spend_usd": 0.12})
    _bind(monkeypatch, tmp_path / "new")

    reconcile_spend_authority_ledger(search_bases=[tmp_path])

    assert (consumption_root() / "authz.json").read_bytes() == original.read_bytes()


def test_a_consumed_authorization_stays_consumed_after_the_move(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property under test: exclusive creation still refuses a replay."""
    legacy = tmp_path / "old" / LEGACY_DIRECTORY_NAME
    _write_record(legacy, "already-spent")
    _bind(monkeypatch, tmp_path / "new")

    reconcile_spend_authority_ledger(search_bases=[tmp_path])

    with pytest.raises(FileExistsError):
        # This is how every paid lane claims an authorization.
        (consumption_root() / "already-spent.json").open("x")


def test_leaves_the_legacy_ledger_in_place_as_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copy, never move: a failed migration must not destroy the only ledger."""
    legacy = tmp_path / "old" / LEGACY_DIRECTORY_NAME
    record = _write_record(legacy, "authz")
    _bind(monkeypatch, tmp_path / "new")

    reconcile_spend_authority_ledger(search_bases=[tmp_path])

    assert record.is_file()


def test_reconciliation_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = tmp_path / "old" / LEGACY_DIRECTORY_NAME
    _write_record(legacy, "authz")
    _bind(monkeypatch, tmp_path / "new")

    first = reconcile_spend_authority_ledger(search_bases=[tmp_path])
    second = reconcile_spend_authority_ledger(search_bases=[tmp_path])

    assert first["records_adopted"] == 1
    assert second["records_adopted"] == 0
    assert second["records_already_present"] == 1
    assert second["status"] == "reconciled"


def test_divergent_record_under_the_same_name_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two different records claiming one authorization is unresolvable here."""
    legacy = tmp_path / "old" / LEGACY_DIRECTORY_NAME
    _write_record(legacy, "authz", {"instance_id": 1})
    root = _bind(monkeypatch, tmp_path / "new")
    _write_record(root, "authz", {"instance_id": 2})

    with pytest.raises(SpendAuthorityLedgerError, match="legacy_ledger_record_conflict"):
        reconcile_spend_authority_ledger(search_bases=[tmp_path])


def test_unreadable_legacy_ledger_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refusing beats serving with a ledger that cannot be proven adopted."""
    legacy = tmp_path / "old" / LEGACY_DIRECTORY_NAME
    _write_record(legacy, "authz")
    (legacy / "consumed").chmod(0o000)
    _bind(monkeypatch, tmp_path / "new")

    try:
        with pytest.raises(SpendAuthorityLedgerError, match="legacy_ledger_unreadable"):
            reconcile_spend_authority_ledger(search_bases=[tmp_path])
    finally:
        (legacy / "consumed").chmod(0o700)


def test_discovers_the_home_ledger_even_outside_the_search_bases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-#453 default root is ``$HOME/.blueprint-spend-authority``."""
    home = tmp_path / "home"
    _write_record(home / LEGACY_DIRECTORY_NAME, "authz")
    monkeypatch.setenv("HOME", str(home))
    _bind(monkeypatch, tmp_path / "new")

    assert discover_legacy_ledgers(search_bases=[]) == [home / LEGACY_DIRECTORY_NAME]


def test_explicitly_named_legacy_roots_are_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment that moved the root twice can name the intermediate one."""
    named = tmp_path / "elsewhere" / "ledger"
    _write_record(named, "authz")
    _bind(monkeypatch, tmp_path / "new")
    monkeypatch.setenv(LEGACY_ROOTS_ENV, str(named))

    receipt = reconcile_spend_authority_ledger(search_bases=[])

    assert receipt["records_adopted"] == 1
    assert str(named) in receipt["legacy_roots_discovered"]


def test_the_configured_root_is_never_its_own_legacy_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root that happens to sit under a search base must not self-adopt."""
    root = tmp_path / LEGACY_DIRECTORY_NAME
    _bind(monkeypatch, root)
    _write_record(root, "authz")

    receipt = reconcile_spend_authority_ledger(search_bases=[tmp_path])

    assert receipt["status"] == "no_legacy_ledger"
    assert receipt["legacy_roots_discovered"] == []


def test_relative_root_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, "relative/ledger")

    with pytest.raises(SpendAuthorityLedgerError, match="spend_authority_root"):
        reconcile_spend_authority_ledger(search_bases=[tmp_path])


def test_search_bases_default_to_the_configured_root_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Production leaves the legacy ledger beside the new root, so look there."""
    base = tmp_path / "var-lib-blueprint"
    _write_record(base / "spend-authority-home" / LEGACY_DIRECTORY_NAME, "authz")
    _bind(monkeypatch, base / "spend-authority")

    receipt = reconcile_spend_authority_ledger()

    assert receipt["records_adopted"] == 1


def test_an_unconfigured_root_does_not_search_the_home_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No configured root means no move to reconcile, so no walk of ``$HOME``."""
    home = tmp_path / "home"
    _write_record(home / "unrelated-project" / "state", "authz")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv(SPEND_AUTHORITY_ROOT_ENV, raising=False)

    receipt = reconcile_spend_authority_ledger()

    assert receipt["legacy_roots_discovered"] == []
    assert receipt["status"] == "no_legacy_ledger"
