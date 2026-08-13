"""Single-use spend enforcement must work on a hardened host.

The deployed dispatcher runs as a service account whose home is ``/nonexistent``
under a unit with ``ProtectHome=true``. Every paid lane wrote its single-use
consumption record under ``Path.home()``, so the record could never be written
and each paid attempt was refused with a consumption-write blocker *after* its
authority had already validated.

The existing lane tests did not catch it because they monkeypatched the module
constant to a writable tmp_path -- they exercised a location production never
used. These tests drive the real resolution path instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blueprint_pipeline.adp_retained_scene_render_vast import (
    consume_retained_scene_render_paid_attempt_authority_once,
)
from blueprint_pipeline.spend_authority_consumption_root import (
    SPEND_AUTHORITY_ROOT_ENV,
    SpendAuthorityRootError,
    authorizations_root,
    consumption_root,
    spend_authority_root,
)

COMMIT = "0" * 40


def _authority(digest_hex: str = "a" * 64) -> dict[str, object]:
    return {
        "authorization_digest": "sha256:" + digest_hex,
        "bundle_sha256": "sha256:" + "b" * 64,
    }


def test_defaults_to_home_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(SPEND_AUTHORITY_ROOT_ENV, raising=False)
    assert spend_authority_root() == Path.home() / ".blueprint-spend-authority"


def test_configured_root_is_honoured(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(tmp_path / "authority"))
    assert consumption_root() == tmp_path / "authority" / "consumed"
    assert authorizations_root() == tmp_path / "authority" / "authorizations"


def test_relative_root_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # A relative root resolves against the working directory, so the same
    # authorization could be consumed once per directory.
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, "relative/authority")
    with pytest.raises(SpendAuthorityRootError):
        consumption_root()


def test_resolution_is_call_time_not_import_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The original defect survived configuration because the path was bound at
    # import. A process that configures the root during start-up must win.
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(tmp_path / "first"))
    assert consumption_root() == tmp_path / "first" / "consumed"
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(tmp_path / "second"))
    assert consumption_root() == tmp_path / "second" / "consumed"


def test_consumption_succeeds_when_home_is_unwritable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The exact deployed condition: home does not exist, root is configured."""
    monkeypatch.setenv("HOME", "/nonexistent")
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(tmp_path / "authority"))

    outcome = consume_retained_scene_render_paid_attempt_authority_once(
        _authority(), blueprint_commit=COMMIT
    )

    assert outcome["status"] == "consumed", outcome
    records = list((tmp_path / "authority" / "consumed").glob("*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text())
    assert record["maximum_provider_allocations"] == 1
    assert record["blueprint_commit"] == COMMIT


def test_unconfigured_root_with_unwritable_home_still_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without configuration the lane must refuse, never allocate unguarded.

    This is the pre-fix production behaviour, pinned deliberately: an
    unwritable ledger must block the attempt rather than let a paid allocation
    proceed with single-use enforcement silently disabled.
    """
    monkeypatch.delenv(SPEND_AUTHORITY_ROOT_ENV, raising=False)
    monkeypatch.setenv("HOME", "/nonexistent")

    outcome = consume_retained_scene_render_paid_attempt_authority_once(
        _authority("c" * 64), blueprint_commit=COMMIT
    )

    assert outcome["status"] == "blocked"
    # The blocker now names the cause. `attempt_authority_consumption_write_failed`
    # said only that a write failed, which sent the last diagnosis after a
    # spend-authority problem when the fault was the filesystem layout.
    assert outcome["blockers"] == ["spend_authority_consumption_root_unwritable:30"]


def test_a_ledger_directory_left_group_readable_is_tightened_not_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The reconciler created this directory under the process umask (0o755 on
    the deployed host) while every paid lane refuses any group or other bit. So
    on every deployed host each paid attempt was refused, for a state the lane
    owned and could have fixed."""

    root = tmp_path / "authority"
    consumed = root / "consumed"
    consumed.mkdir(parents=True)
    consumed.chmod(0o755)
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(root))

    outcome = consume_retained_scene_render_paid_attempt_authority_once(
        _authority("e" * 64), blueprint_commit=COMMIT
    )

    assert outcome["status"] == "consumed"
    # Tightening enforces the property the check exists for, rather than
    # reporting that it is not met.
    assert consumed.stat().st_mode & 0o077 == 0


def test_a_ledger_directory_owned_by_someone_else_is_still_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Ownership is not something we can make safe by changing a mode."""

    import os

    from blueprint_pipeline.spend_authority_consumption_root import (
        SpendAuthorityRootError,
        prepare_consumption_root,
    )

    root = tmp_path / "authority"
    (root / "consumed").mkdir(parents=True)
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(root))
    foreign_uid = os.getuid() + 12345
    monkeypatch.setattr(os, "getuid", lambda: foreign_uid)

    with pytest.raises(SpendAuthorityRootError, match="not_owned"):
        prepare_consumption_root()


def test_a_symlinked_ledger_directory_is_refused_outright(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from blueprint_pipeline.spend_authority_consumption_root import (
        SpendAuthorityRootError,
        prepare_consumption_root,
    )

    root = tmp_path / "authority"
    root.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir(mode=0o700)
    (root / "consumed").symlink_to(elsewhere)
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(root))

    with pytest.raises(SpendAuthorityRootError, match="is_symlink"):
        prepare_consumption_root()


def test_second_consumption_of_same_authority_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(SPEND_AUTHORITY_ROOT_ENV, str(tmp_path / "authority"))
    authority = _authority("d" * 64)

    first = consume_retained_scene_render_paid_attempt_authority_once(
        authority, blueprint_commit=COMMIT
    )
    second = consume_retained_scene_render_paid_attempt_authority_once(
        authority, blueprint_commit=COMMIT
    )

    assert first["status"] == "consumed"
    assert second["status"] == "blocked"
    assert "attempt_authority_already_consumed" in second["blockers"]
