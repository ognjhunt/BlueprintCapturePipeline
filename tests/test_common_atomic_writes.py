from __future__ import annotations

import json
import stat
import threading
from pathlib import Path

import pytest

from blueprint_pipeline import common


def test_write_json_replace_failure_preserves_previous_complete_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "ledger.json"
    target.write_text('{"generation": "old"}', encoding="utf-8")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(common.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        common.write_json(target, {"generation": "new"})

    assert json.loads(target.read_text(encoding="utf-8")) == {"generation": "old"}
    assert list(tmp_path.glob(".ledger.json.*.tmp")) == []


def test_concurrent_write_json_never_mixes_payloads(tmp_path: Path) -> None:
    target = tmp_path / "ledger.json"
    barrier = threading.Barrier(8)
    failures: list[BaseException] = []

    def writer(index: int) -> None:
        try:
            barrier.wait()
            common.write_json(target, {"writer": index, "values": [index] * 100})
        except BaseException as exc:  # pragma: no cover - assertion reports worker failures
            failures.append(exc)

    threads = [threading.Thread(target=writer, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert failures == []
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["values"] == [payload["writer"]] * 100
    assert list(tmp_path.glob(".ledger.json.*.tmp")) == []


def test_write_text_replaces_symlink_itself_without_overwriting_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    target = tmp_path / "result.txt"
    target.symlink_to(outside)

    common.write_text(target, "inside")

    assert not target.is_symlink()
    assert target.read_text(encoding="utf-8") == "inside"
    assert outside.read_text(encoding="utf-8") == "outside"
    # The replaced link must not donate the pointed-to file's identity.
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_write_json_preserves_permissions_of_the_file_it_replaces(
    tmp_path: Path,
) -> None:
    """Production regression: an atomic rewrite silently narrowed a state file.

    ``tempfile.mkstemp`` stages the replacement at ``0600``, so without
    carrying the previous mode forward every rewrite re-permissions the target
    and can lock out the account that owns it.
    """

    target = tmp_path / "spend_ledger.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o640)

    common.write_json(target, {"revision": 2})

    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert json.loads(target.read_text(encoding="utf-8")) == {"revision": 2}


def test_write_json_never_carries_forward_group_or_world_write(tmp_path: Path) -> None:
    target = tmp_path / "ledger.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o666)

    common.write_json(target, {"revision": 2})

    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_write_json_creates_a_brand_new_file_owner_only(tmp_path: Path) -> None:
    target = tmp_path / "fresh.json"

    common.write_json(target, {"revision": 1})

    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_write_json_restores_previous_owner_when_the_writer_differs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rewrite by another account must hand the file back to its owner.

    Root rewriting a service-account state file is the exact production
    failure: the replacement carried root's ownership and the service could
    never read its own ledger again. Ownership is asserted through the
    recorded ``chown`` because an unprivileged test cannot change owners.
    """

    target = tmp_path / "spend_ledger.json"
    target.write_text('{"total_spend_usd": 16.8586}', encoding="utf-8")
    previous = target.stat()

    monkeypatch.setattr(common.os, "geteuid", lambda: previous.st_uid + 1)
    recorded: list[tuple[int, int]] = []
    real_chown = common.os.chown

    def recording_chown(path: object, uid: int, gid: int) -> None:
        recorded.append((uid, gid))
        real_chown(path, uid, gid)

    monkeypatch.setattr(common.os, "chown", recording_chown)

    common.write_json(target, {"total_spend_usd": 17.0})

    assert recorded == [(previous.st_uid, previous.st_gid)]
    current = target.stat()
    assert (current.st_uid, current.st_gid) == (previous.st_uid, previous.st_gid)


def test_write_json_survives_a_writer_that_may_not_restore_the_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Best-effort ownership restore must never fail an otherwise valid write."""

    target = tmp_path / "ledger.json"
    target.write_text("{}", encoding="utf-8")
    previous = target.stat()

    monkeypatch.setattr(common.os, "geteuid", lambda: previous.st_uid + 1)

    def denied_chown(*_args: object, **_kwargs: object) -> None:
        raise PermissionError("not permitted")

    monkeypatch.setattr(common.os, "chown", denied_chown)

    common.write_json(target, {"revision": 3})

    assert json.loads(target.read_text(encoding="utf-8")) == {"revision": 3}
