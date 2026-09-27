# Covers (for impacted-test selection):
#   src/blueprint_pipeline/control_plane_break_glass.py
#   src/blueprint_pipeline/control_plane_storage_roots.py
"""Every host change made outside the operator door leaves a sealed note.

On 2026-09-26 a deploy ran from a scratch checkout through a transient unit,
and operators and agents changed the control-plane host by hand over SSH.
Nothing recorded any of it.  A break-glass note names who acted, why, what
they did and when; it is sealed by a digest, and the next deploy reports every
note no earlier deploy reported.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from blueprint_pipeline import control_plane_break_glass as break_glass
from blueprint_pipeline.control_plane_storage_roots import classify_path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.secrets_guard import refused_name, scan_bytes  # noqa: E402

NOW = 1_790_424_000  # 2026-09-26T12:00:00Z
COMMIT = "c" * 40
SSH = {"SSH_CONNECTION": "203.0.113.7 50022 10.0.0.2 22", "SUDO_USER": "alice"}


def _record(root: Path, *, now: int = NOW, environ=SSH, **overrides) -> Path:
    fields = {
        "operator": "alice",
        "reason": "restarted the stuck scene progression timer by hand",
        "actions": ["unit-restart"],
        "paths": ["/etc/systemd/system/blueprint-task-evaluation-scene-progression.timer"],
    }
    fields.update(overrides)
    return break_glass.record_note(root=root, now=lambda: now, environ=environ, **fields)


def test_a_recorded_note_verifies_and_names_who_why_what_and_when(tmp_path: Path) -> None:
    root = tmp_path / "cleanup-receipts"
    path = _record(root)

    note = break_glass.verify_note(path)

    assert note["schema_version"] == break_glass.NOTE_SCHEMA
    assert note["operator"] == "alice"
    assert note["reason"] == "restarted the stuck scene progression timer by hand"
    assert note["actions"] == ["unit-restart"]
    assert note["paths"] == [
        "/etc/systemd/system/blueprint-task-evaluation-scene-progression.timer"
    ]
    assert note["ssh_client"] == "203.0.113.7"
    assert note["sudo_user"] == "alice"
    assert isinstance(note["host"], str) and note["host"]
    assert note["created_at_epoch"] == NOW
    assert note["created_at"] == "2026-09-26T12:00:00Z"
    assert note["note_digest"].startswith("sha256:") and len(note["note_digest"]) == 71
    assert path.parent == root
    assert path.name == f"20260926T120000Z-{note['note_digest'][7:19]}.json"
    assert break_glass.note_summary({"name": path.name, **note}) == {
        "name": path.name,
        "digest": note["note_digest"],
        "operator": "alice",
        "reason": "restarted the stuck scene progression timer by hand",
        "created_at": "2026-09-26T12:00:00Z",
    }


def test_a_note_outside_an_ssh_session_records_no_client(tmp_path: Path) -> None:
    note = break_glass.verify_note(_record(tmp_path / "cleanup-receipts", environ={}))

    assert note["ssh_client"] is None
    assert note["sudo_user"] is None


@pytest.mark.parametrize(
    ("tamper", "code"),
    [
        ("reason", "break_glass_note_digest_mismatch"),
        ("digest", "break_glass_note_digest_mismatch"),
        ("added_field", "break_glass_note_fields_invalid"),
        ("renamed", "break_glass_note_name_mismatch"),
        ("not_json", "break_glass_note_not_json"),
    ],
)
def test_an_edited_or_renamed_note_no_longer_verifies(
    tmp_path: Path, tamper: str, code: str
) -> None:
    path = _record(tmp_path / "cleanup-receipts")
    note = json.loads(path.read_text(encoding="utf-8"))
    if tamper == "reason":
        note["reason"] = "nothing happened"
    elif tamper == "digest":
        note["note_digest"] = "sha256:" + "0" * 64
    elif tamper == "added_field":
        note["approved"] = True
    if tamper == "renamed":
        renamed = path.with_name("20260926T120000Z-000000000000.json")
        path.rename(renamed)
        path = renamed
    elif tamper == "not_json":
        path.write_text("{", encoding="utf-8")
    else:
        path.write_text(json.dumps(note), encoding="utf-8")

    with pytest.raises(break_glass.BreakGlassNoteError, match=f"^{code}$"):
        break_glass.verify_note(path)


def test_verify_enforces_a_maximum_age_and_refuses_future_notes(tmp_path: Path) -> None:
    path = _record(tmp_path / "cleanup-receipts")
    day = 24 * 3600

    fresh = break_glass.verify_note(path, max_age_seconds=day, now=lambda: NOW + day)
    assert fresh["created_at_epoch"] == NOW
    with pytest.raises(break_glass.BreakGlassNoteError, match="^break_glass_note_expired$"):
        break_glass.verify_note(path, max_age_seconds=day, now=lambda: NOW + day + 1)
    # A note dated ahead of the clock would stay fresh for longer than a day.
    with pytest.raises(break_glass.BreakGlassNoteError, match="^break_glass_note_from_the_future$"):
        break_glass.verify_note(path, max_age_seconds=day, now=lambda: NOW - 3600)
    with pytest.raises(break_glass.BreakGlassNoteError, match="^break_glass_note_unreadable$"):
        break_glass.verify_note(path.with_name("20260926T120000Z-0123456789ab.json"))


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"reason": "   "}, "break_glass_reason_required"),
        ({"reason": "first line\nsecond line"}, "break_glass_reason_invalid"),
        ({"reason": "x" * 501}, "break_glass_reason_invalid"),
        ({"actions": []}, "break_glass_action_required"),
        ({"actions": ["Deploy From Scratch"]}, "break_glass_action_invalid"),
        ({"actions": "unit-restart"}, "break_glass_action_invalid"),
        ({"operator": ""}, "break_glass_operator_invalid"),
        ({"paths": ["relative/path"]}, "break_glass_path_invalid"),
    ],
)
def test_a_note_without_a_reason_or_an_action_is_refused(
    tmp_path: Path, overrides: dict, code: str
) -> None:
    root = tmp_path / "cleanup-receipts"

    with pytest.raises(break_glass.BreakGlassNoteError, match=f"^{code}$"):
        _record(root, **overrides)

    assert not root.exists()


@pytest.mark.parametrize(
    ("environment", "operator"),
    [({"SUDO_USER": "alice", "USER": "root"}, "alice"), ({"USER": "bob"}, "bob")],
)
def test_record_takes_the_operator_from_sudo_user_else_user(
    tmp_path: Path, monkeypatch, capsys, environment: dict, operator: str
) -> None:
    for name in ("SUDO_USER", "USER", "SSH_CONNECTION"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    root = tmp_path / "cleanup-receipts"

    assert break_glass.main([
        "--root", str(root), "record",
        "--reason", "removed a stale replay cache by hand",
        "--action", "delete-cache", "--action", "delete-cache",
        "--path", "/var/lib/blueprint/pipeline-control-plane/result-artifact-cache",
    ]) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "recorded"
    note = break_glass.verify_note(Path(output["path"]))
    assert Path(output["path"]).parent == root
    assert note["operator"] == operator
    assert note["sudo_user"] == environment.get("SUDO_USER")
    assert note["ssh_client"] is None
    assert note["actions"] == ["delete-cache"]


def test_record_cli_refuses_with_a_typed_code(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("USER", "bob")
    root = tmp_path / "cleanup-receipts"

    assert break_glass.main(
        ["--root", str(root), "record", "--reason", "", "--action", "unit-restart"]
    ) == 2

    assert json.loads(capsys.readouterr().out) == {
        "status": "refused",
        "code": "break_glass_reason_required",
    }
    assert not root.exists()


def test_notes_stay_unreported_until_a_deploy_marks_them(tmp_path: Path, capsys) -> None:
    root = tmp_path / "cleanup-receipts"
    # No directory yet: nothing to report, and reading creates nothing.
    assert break_glass.unreported_notes(root) == []
    assert not root.exists()
    first = _record(root)
    second = _record(
        root,
        now=NOW + 60,
        reason="deployed from a scratch checkout",
        actions=[break_glass.DEPLOY_FROM_UNTRUSTED_SOURCE],
    )

    notes = break_glass.unreported_notes(root)
    assert [note["name"] for note in notes] == [first.name, second.name]
    assert notes[1]["actions"] == [break_glass.DEPLOY_FROM_UNTRUSTED_SOURCE]

    break_glass.mark_reported(root, notes[:1], deploy_commit=COMMIT, now=lambda: NOW + 120)

    assert [note["name"] for note in break_glass.unreported_notes(root)] == [second.name]
    ledger = (root / break_glass.REPORTED_LEDGER).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in ledger] == [
        {
            "schema_version": break_glass.REPORT_SCHEMA,
            "name": first.name,
            "note_digest": notes[0]["note_digest"],
            "deploy_commit": COMMIT,
            "reported_at": "2026-09-26T12:02:00Z",
            "reported_at_epoch": NOW + 120,
        }
    ]
    # Nothing to mark writes nothing; a commit that is not a full sha is refused.
    break_glass.mark_reported(root, [], deploy_commit=COMMIT)
    assert len((root / break_glass.REPORTED_LEDGER).read_text(encoding="utf-8").splitlines()) == 1
    with pytest.raises(
        break_glass.BreakGlassNoteError, match="^break_glass_deploy_commit_invalid$"
    ):
        break_glass.mark_reported(root, notes, deploy_commit="HEAD")

    assert break_glass.main(["--root", str(root), "list"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["unreported"] == [break_glass.note_summary(notes[1])]


def test_a_damaged_note_is_reported_rather_than_hidden(tmp_path: Path) -> None:
    root = tmp_path / "cleanup-receipts"
    path = _record(root)
    path.write_text("{", encoding="utf-8")

    [note] = break_glass.unreported_notes(root)

    assert note == {"name": path.name, "error": "break_glass_note_not_json"}
    assert break_glass.note_summary(note) == {
        "name": path.name,
        "digest": None,
        "error": "break_glass_note_not_json",
    }
    break_glass.mark_reported(root, [note], deploy_commit=COMMIT)
    assert break_glass.unreported_notes(root) == []


def test_a_symlinked_notes_root_is_refused(tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    root = tmp_path / "cleanup-receipts"
    root.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(break_glass.BreakGlassNoteError, match="^break_glass_notes_root_unsafe$"):
        _record(root)
    with pytest.raises(break_glass.BreakGlassNoteError, match="^break_glass_notes_root_unsafe$"):
        break_glass.unreported_notes(root)
    assert list(elsewhere.iterdir()) == []


def test_the_door_can_read_every_note_and_the_ledger(tmp_path: Path) -> None:
    root = tmp_path / "cleanup-receipts"
    previous = os.umask(0o077)
    try:
        path = _record(root)
        break_glass.mark_reported(
            root, break_glass.unreported_notes(root), deploy_commit=COMMIT
        )
    finally:
        os.umask(previous)

    # The door runs as the service account, not root.
    assert stat.S_IMODE(root.stat().st_mode) == 0o755
    assert stat.S_IMODE(path.stat().st_mode) == 0o644
    assert stat.S_IMODE((root / break_glass.REPORTED_LEDGER).stat().st_mode) == 0o644
    assert sorted(entry.name for entry in root.iterdir()) == sorted(
        [path.name, break_glass.REPORTED_LEDGER]
    )


def test_the_door_serves_notes_and_the_ledger_without_refusing_them(tmp_path: Path) -> None:
    root = tmp_path / "cleanup-receipts"
    path = _record(root, actions=[break_glass.DEPLOY_FROM_UNTRUSTED_SOURCE, "unit-restart"])
    break_glass.mark_reported(root, break_glass.unreported_notes(root), deploy_commit=COMMIT)

    # Door reads refuse credential-shaped content, including JSON keys that
    # look like secrets; no note or ledger field may trip that scanner.
    for document in (path, root / break_glass.REPORTED_LEDGER):
        assert scan_bytes(document.read_bytes()) is None
    for name in (path.name, break_glass.REPORTED_LEDGER):
        assert not refused_name(break_glass.DEFAULT_NOTES_ROOT / name)


def test_notes_live_in_a_hot_evidence_root_that_is_never_reclaimed() -> None:
    root = classify_path(str(break_glass.DEFAULT_NOTES_ROOT / "20260926T120000Z-0123456789ab.json"))

    assert root is not None
    assert (root.path, root.storage_class, root.owner) == (
        str(break_glass.DEFAULT_NOTES_ROOT),
        "evidence_hot",
        "root",
    )
