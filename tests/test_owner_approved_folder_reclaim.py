"""Deleting the owner-approved internal folders, against tmp folders and a tmp allowlist."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/owner_approved_folder_reclaim.py
#   config/owner_approved_reclaim_folders.v1.json

from __future__ import annotations

import fcntl
import json
import os
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from blueprint_pipeline import owner_approved_folder_reclaim as reclaim
from blueprint_pipeline.control_plane_storage_roots import STORAGE_ROOTS, StorageRoot, classify_path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "operator-door"))

from operator_door.config import DoorConfig  # noqa: E402
from operator_door.secrets_guard import refused_name  # noqa: E402

PCP = "/var/lib/blueprint/pipeline-control-plane"
FOLDER = "internal-team-evaluation-diagnostic-20261015"
MARKER = b"SYNTHETIC-CONTENT-MARKER "
APPROVAL = {"approved_by": "owner", "approved_on": "2026-11-08", "timezone": "CST",
            "relayed_by": "library-aisle session", "words": ["option 1"], "note": "fixture approval"}


def _allowlist(path: Path, paths: list[str], *, bound: str = "2100-01-01T00:00:00Z") -> None:
    document = {"schema_version": reclaim.ALLOWLIST_SCHEMA, "parent": PCP,
                "folders": [{"path": raw, "refuse_if_modified_after": bound, "approval": APPROVAL} for raw in paths]}
    path.write_text(json.dumps(document), encoding="utf-8")


def _tree(root: Path) -> dict[str, Any]:
    """Every name under ``root`` with its type, size and mtime, never following a link."""

    rows: dict[str, Any] = {}
    for current, names, files in os.walk(root, followlinks=False):
        for name in [*names, *files]:
            path = Path(current) / name
            info = path.lstat()
            rows[str(path.relative_to(root))] = (oct(info.st_mode), info.st_size, info.st_mtime_ns)
    return rows


@pytest.fixture()
def world(tmp_path: Path) -> dict[str, Any]:
    parent = tmp_path / "pipeline-control-plane"
    folder = parent / FOLDER
    (folder / "runs" / "a" / "deep").mkdir(parents=True)
    (folder / "logs").mkdir()
    (folder / "runs" / "a" / "deep" / "episode.bin").write_bytes(MARKER * 4096)
    (folder / "runs" / "a" / "summary.json").write_bytes(MARKER * 3)
    (folder / "logs" / "run.log").write_bytes(MARKER * 10)
    (folder / "top.txt").write_bytes(MARKER)
    os.link(folder / "top.txt", folder / "top-hardlink.txt")
    # Links inside the folder point outside it: they are unlinked, their targets never touched.
    outside = tmp_path / "outside"
    (outside / "keep").mkdir(parents=True)
    (outside / "keep" / "evidence.json").write_bytes(MARKER * 2)
    (outside / "file.txt").write_bytes(MARKER)
    (folder / "runs" / "link-to-dir").symlink_to(outside / "keep")
    (folder / "logs" / "link-to-file").symlink_to(outside / "file.txt")
    sibling = parent / "team-evaluations"
    sibling.mkdir()
    (sibling / "queue.json").write_bytes(MARKER)
    other = parent / "internal-diagnostic-not-approved"
    other.mkdir()
    (other / "x.bin").write_bytes(MARKER)
    (parent / "storage-gc").mkdir()
    settings = reclaim.Settings(allowlist=tmp_path / "allowlist.json", parent=parent,
                                receipts_dir=parent / "storage-gc" / "owner-approved-reclaims")
    _allowlist(settings.allowlist, [f"{PCP}/{FOLDER}"])
    return {"tmp": tmp_path, "parent": parent, "folder": folder, "outside": outside, "sibling": sibling,
            "other": other, "settings": settings}


def _receipts(world: dict[str, Any]) -> list[Path]:
    directory = world["settings"].receipts_dir
    return sorted(directory.iterdir()) if directory.exists() else []


# --- production pins -------------------------------------------------------------------------------

def test_the_production_allowlist_names_only_the_folder_the_owner_approved() -> None:
    document, digest = reclaim.load_allowlist(reclaim.SETTINGS)
    assert reclaim.SETTINGS.allowlist == ROOT / "config" / "owner_approved_reclaim_folders.v1.json"
    assert digest.startswith("sha256:") and document["parent"] == PCP == str(reclaim.CONTROL_PLANE)
    [entry] = document["folders"]
    assert entry["path"] == f"{PCP}/{FOLDER}"
    approval = entry["approval"]
    assert approval["approved_by"] == "owner" and approval["approved_on"] == "2026-11-08"
    assert "option 1" in approval["words"] and '"option 1"' in approval["note"]
    assert "tried to run the delete" in approval["note"]
    # The start of 2026-11-08 CST (UTC-6), after the newest file the owner reviewed (2026-10-17).
    assert entry["refuse_if_modified_after"] == "2026-11-08T06:00:00Z"
    assert reclaim._shape_refusal(entry["path"]) is None and reclaim.protected_storage_root(entry["path"]) is None
    assert classify_path(entry["path"]).storage_class == "container"
    # The door unit's only writable state directory is the folder's parent; the receipts are kept in it,
    # under the storage reclaim reports.
    assert DoorConfig().control_plane_state == PCP == str(reclaim.SETTINGS.parent)
    assert reclaim.RECEIPTS_DIR.parent.parent == reclaim.SETTINGS.parent
    receipts_root = classify_path(str(reclaim.RECEIPTS_DIR))
    assert (receipts_root.path, receipts_root.storage_class) == (f"{PCP}/storage-gc", "evidence_hot")


def test_no_code_service_or_config_names_an_allowlisted_folder() -> None:
    document, _ = reclaim.load_allowlist(reclaim.SETTINGS)
    names = [PurePosixPath(entry["path"]).name for entry in document["folders"]]
    allowlist = reclaim.SETTINGS.allowlist.resolve()
    for top in ("src", "deploy", "config", "configs", "scripts"):
        for path in (ROOT / top).rglob("*"):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts or path.resolve() == allowlist:
                continue
            if path.suffix not in {".py", ".sh", ".json", ".service", ".timer", ".path", ".conf", ".env", ".yaml",
                                   ".yml", ".toml", ".txt", ""}:
                continue
            text = path.read_bytes()
            for name in names:
                assert name.encode() not in text, f"{path} names {name}"


def test_no_name_the_door_reads_is_one_it_refuses() -> None:
    for name in (Path(reclaim.__file__).name, reclaim.SETTINGS.allowlist.name, "door-reclaim-approved-folders.sh",
                 "20261108T190000Z-reclaim-approved-folders-0123abcd.reclaim-approved-folders.json",
                 reclaim.RECEIPT_PREFIX + "20261108T190000Z-0123abcd.json", *reclaim.RECEIPTS_DIR.parts,
                 *PurePosixPath(f"{PCP}/{FOLDER}").parts):
        assert not refused_name(name), name


# --- plan ------------------------------------------------------------------------------------------

def test_a_plan_reports_each_folder_and_writes_nothing(world: dict[str, Any]) -> None:
    before = _tree(world["tmp"])
    result = reclaim.reclaim(settings=world["settings"])

    assert _tree(world["tmp"]) == before and _receipts(world) == []
    assert not world["settings"].receipts_dir.exists()
    assert (result["outcome"], result["code"], result["mode"]) == ("planned", None, "plan")
    [row] = result["folders"]
    assert (row["path"], row["exists"], row["state"], row["refusal"], row["outcome"]) == (
        f"{PCP}/{FOLDER}", True, "present", None, "would_reclaim")
    measured = row["measured"]
    assert measured["file_count"] == 5 and measured["symlink_count"] == 2 and measured["directory_count"] == 5
    assert measured["total_bytes"] == len(MARKER) * (4096 + 3 + 10 + 1 + 1)
    assert measured["multiply_linked_file_count"] == 2 and measured["blocked_directories"] == 0
    assert measured["newest_mtime"].endswith("Z") and measured["newest_file_mtime"].endswith("Z")
    assert result["totals"] == {"folders": 1, "present": 1, "absent": 0, "refused": 0,
                                "total_bytes": measured["total_bytes"], "allocated_bytes": measured["allocated_bytes"],
                                "file_count": 5}
    assert isinstance(result["disk"]["free_bytes"], int) and result["receipt_path"] is None
    assert MARKER.decode() not in json.dumps(result) and result["file_contents_recorded"] is False
    assert result["network_used"] is False and result["compute_started"] is False


def test_an_absent_folder_is_already_reclaimed_in_both_modes(world: dict[str, Any]) -> None:
    _allowlist(world["settings"].allowlist, [f"{PCP}/not-here-any-more"])
    plan = reclaim.reclaim(settings=world["settings"])
    assert plan["outcome"] == "planned"
    assert (plan["folders"][0]["exists"], plan["folders"][0]["outcome"]) == (False, "already_reclaimed")
    before = _tree(world["parent"])
    applied = reclaim.reclaim(apply=True, settings=world["settings"])
    assert applied["outcome"] == "already_reclaimed" and applied["receipt_path"] is None
    assert {key: value for key, value in _tree(world["parent"]).items() if not key.startswith("storage-gc")} == {
        key: value for key, value in before.items() if not key.startswith("storage-gc")}


# --- apply -----------------------------------------------------------------------------------------

def test_apply_deletes_only_the_folder_tree_and_keeps_a_receipt(world: dict[str, Any]) -> None:
    outside = _tree(world["outside"])
    sibling, other = _tree(world["sibling"]), _tree(world["other"])
    result = reclaim.reclaim(apply=True, settings=world["settings"])

    assert (result["outcome"], result["code"]) == ("reclaimed", None)
    assert not os.path.lexists(world["folder"])
    # Link targets outside the folder, the sibling queue and an unlisted folder are untouched.
    assert _tree(world["outside"]) == outside and (world["outside"] / "keep" / "evidence.json").exists()
    assert _tree(world["sibling"]) == sibling and _tree(world["other"]) == other
    [row] = result["folders"]
    assert row["outcome"] == "reclaimed" and row["exists"] is False and row["removed_at"].endswith("Z")
    assert row["removed"] == {"total_bytes": row["measured"]["total_bytes"],
                              "allocated_bytes": row["measured"]["allocated_bytes"], "file_count": 5,
                              "directory_count": 5, "symlink_count": 2, "other_count": 0}
    assert result["removed"]["file_count"] == 5 and result["removed"]["total_bytes"] == row["measured"]["total_bytes"]
    assert result["started_at"].endswith("Z") and result["finished_at"].endswith("Z")
    disk = result["disk"]
    assert disk["freed_bytes"] == disk["free_bytes_after"] - disk["free_bytes_before"]

    [receipt] = _receipts(world)
    assert receipt.name.startswith(reclaim.RECEIPT_PREFIX) and receipt.suffix == ".json"
    assert result["receipt_path"] == str(receipt)
    assert json.loads(receipt.read_text()) == result
    assert oct(receipt.stat().st_mode & 0o777) == oct(0o644)
    assert oct(world["settings"].receipts_dir.stat().st_mode & 0o777) == oct(0o755)
    assert MARKER.decode() not in receipt.read_text()


def test_a_second_apply_changes_nothing(world: dict[str, Any]) -> None:
    first = reclaim.reclaim(apply=True, settings=world["settings"])
    assert first["outcome"] == "reclaimed"
    before = _tree(world["tmp"])
    second = reclaim.reclaim(apply=True, settings=world["settings"])
    assert second["outcome"] == "already_reclaimed" and second["folders"][0]["outcome"] == "already_reclaimed"
    assert _tree(world["tmp"]) == before and len(_receipts(world)) == 1


def test_no_path_can_be_asked_for_and_an_unlisted_folder_is_never_touched(world: dict[str, Any],
                                                                         capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        reclaim.main(["--path", str(world["other"])])
    assert caught.value.code == 2
    capsys.readouterr()
    before = _tree(world["other"])
    reclaim.reclaim(apply=True, settings=world["settings"])
    assert _tree(world["other"]) == before


# --- refusals --------------------------------------------------------------------------------------

def _refused_untouched(world: dict[str, Any], code: str) -> dict[str, Any]:
    before = _tree(world["tmp"])
    for apply in (False, True):
        result = reclaim.reclaim(apply=apply, settings=world["settings"])
        assert (result["outcome"], result["code"]) == ("refused", code), result
        assert result["receipt_path"] is None
    after = _tree(world["tmp"])
    # The apply may make the empty receipts directory; nothing else changes.
    assert {key: value for key, value in after.items() if not key.startswith("pipeline-control-plane/storage-gc")} \
        == {key: value for key, value in before.items() if not key.startswith("pipeline-control-plane/storage-gc")}
    assert _receipts(world) == []
    return result


def test_a_symlinked_folder_is_refused_and_neither_it_nor_its_target_is_touched(world: dict[str, Any]) -> None:
    target = world["tmp"] / "real-diagnostic"
    world["folder"].rename(target)
    world["folder"].symlink_to(target, target_is_directory=True)
    result = _refused_untouched(world, "reclaim_approved_folders_symlink")
    assert result["folders"][0]["refusal"] == "reclaim_approved_folders_symlink"
    assert world["folder"].is_symlink() and (target / "top.txt").exists()


def test_a_symlink_on_the_parent_path_is_refused(world: dict[str, Any]) -> None:
    linked = world["tmp"] / "linked-parent"
    linked.symlink_to(world["parent"], target_is_directory=True)
    settings = reclaim.Settings(allowlist=world["settings"].allowlist, parent=linked,
                                receipts_dir=world["settings"].receipts_dir)
    before = _tree(world["tmp"])
    for apply in (False, True):
        result = reclaim.reclaim(apply=apply, settings=settings)
        assert result["outcome"] == "refused" and result["code"] in {
            "reclaim_approved_folders_parent_unsafe", "reclaim_approved_folders_receipts_unavailable"}
    assert _tree(world["tmp"]) == before


def test_an_evidence_hot_root_is_refused_by_its_class(world: dict[str, Any]) -> None:
    receipts = world["parent"] / "deploy-receipts"
    receipts.mkdir()
    (receipts / "deploy.json").write_bytes(MARKER)
    _allowlist(world["settings"].allowlist, [f"{PCP}/deploy-receipts"])
    result = _refused_untouched(world, "reclaim_approved_folders_protected_storage_root")
    [row] = result["folders"]
    assert row["storage_root"]["path"] == f"{PCP}/deploy-receipts"
    assert row["storage_root"]["storage_class"] == "evidence_hot" and row["measured"] is None
    assert (receipts / "deploy.json").exists()


@pytest.mark.parametrize("root", sorted(root.path for root in STORAGE_ROOTS
                                        if str(PurePosixPath(root.path).parent) == PCP))
def test_every_classified_root_in_the_control_plane_is_refused(root: str) -> None:
    protected = reclaim.protected_storage_root(root)
    assert protected is not None and protected.path == root
    assert any(row.path == root and row.storage_class == protected.storage_class for row in STORAGE_ROOTS)


def test_a_folder_holding_a_classified_root_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    assert reclaim.protected_storage_root(f"{PCP}/holder") is None
    nested = StorageRoot(f"{PCP}/holder/evidence", "evidence_hot", "blueprint", "fixture")
    monkeypatch.setattr(reclaim, "STORAGE_ROOTS", (*STORAGE_ROOTS, nested))
    assert reclaim.protected_storage_root(f"{PCP}/holder") == nested


@pytest.mark.parametrize("raw", [
    "/var/lib/blueprint/task-evaluation-inputs/arena-scratch",
    f"{PCP}/{FOLDER}/runs",
    f"{PCP}/../task-evaluation-inputs",
    f"{PCP}/.{FOLDER}",
    f"{PCP}/{FOLDER}/",
    f"{PCP}//{FOLDER}",
    PCP,
    "/var/lib/blueprint",
    "relative/path",
])
def test_a_path_that_is_not_a_direct_child_of_the_control_plane_is_refused(world: dict[str, Any], raw: str) -> None:
    _allowlist(world["settings"].allowlist, [raw])
    result = _refused_untouched(world, "reclaim_approved_folders_path_outside_parent")
    assert result["folders"][0]["refusal"] == "reclaim_approved_folders_path_outside_parent"


def test_one_refused_folder_stops_the_whole_apply(world: dict[str, Any]) -> None:
    _allowlist(world["settings"].allowlist, [f"{PCP}/{FOLDER}", f"{PCP}/team-evaluations"])
    result = _refused_untouched(world, "reclaim_approved_folders_protected_storage_root")
    assert [row["outcome"] for row in result["folders"]] == ["would_reclaim", "refused"]
    assert (world["folder"] / "top.txt").exists()


def test_a_folder_on_another_filesystem_is_refused(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    real = reclaim._stat

    def other_device(name: str, dir_fd: int) -> os.stat_result:
        info = real(name, dir_fd)
        if name == FOLDER:
            values = list(info)
            values[2] = info.st_dev + 1  # st_dev
            return os.stat_result(values)
        return info

    monkeypatch.setattr(reclaim, "_stat", other_device)
    _refused_untouched(world, "reclaim_approved_folders_mount_point")


def test_a_mount_inside_the_folder_is_refused(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    real = reclaim._stat

    def nested_device(name: str, dir_fd: int) -> os.stat_result:
        info = real(name, dir_fd)
        if name == "deep":
            values = list(info)
            values[2] = info.st_dev + 1
            return os.stat_result(values)
        return info

    monkeypatch.setattr(reclaim, "_stat", nested_device)
    _refused_untouched(world, "reclaim_approved_folders_mount_point_inside")


def test_a_mount_point_in_the_mount_table_is_refused(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reclaim, "_mount_points", lambda: ["/", str(world["folder"] / "runs")])
    _refused_untouched(world, "reclaim_approved_folders_mount_point_inside")


def test_an_unreadable_mount_table_leaves_the_device_check(world: dict[str, Any],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reclaim, "_mount_points", lambda: None)
    result = reclaim.reclaim(settings=world["settings"])
    assert result["outcome"] == "planned" and result["folders"][0]["mount_table_read"] is False


def test_a_file_modified_after_the_approval_is_refused(world: dict[str, Any]) -> None:
    _allowlist(world["settings"].allowlist, [f"{PCP}/{FOLDER}"], bound="2026-11-08T06:00:00Z")
    stamp = reclaim._instant("2026-11-08T07:00:00Z")
    os.utime(world["folder"] / "logs" / "run.log", (stamp, stamp))
    for path in world["folder"].rglob("*"):
        if path.is_file() and not path.is_symlink() and path.name != "run.log":
            os.utime(path, (stamp - 86400 * 30,) * 2)
    for link in (world["folder"] / "runs" / "link-to-dir", world["folder"] / "logs" / "link-to-file"):
        os.utime(link, (stamp - 86400 * 30,) * 2, follow_symlinks=False)
    _refused_untouched(world, "reclaim_approved_folders_modified_after_approval")
    os.utime(world["folder"] / "logs" / "run.log", (stamp - 86400 * 30,) * 2)
    assert reclaim.reclaim(settings=world["settings"])["outcome"] == "planned"


def test_a_directory_this_user_cannot_remove_is_refused_before_anything_goes(
        world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(reclaim, "_removable", lambda name, dir_fd: name != "deep")
    result = _refused_untouched(world, "reclaim_approved_folders_not_removable")
    assert result["folders"][0]["measured"]["blocked_directories"] == 1


def test_a_running_apply_makes_another_refuse_busy(world: dict[str, Any]) -> None:
    world["settings"].receipts_dir.mkdir()
    fd = os.open(world["settings"].receipts_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        result = reclaim.reclaim(apply=True, settings=world["settings"])
    finally:
        os.close(fd)
    assert (result["outcome"], result["code"]) == ("refused", "reclaim_approved_folders_busy")
    assert (world["folder"] / "top.txt").exists()


def test_a_delete_that_stops_part_way_keeps_its_receipt_and_the_next_apply_finishes(
        world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    real_unlink = os.unlink

    def failing_unlink(name: str, *, dir_fd: int | None = None) -> None:
        if name == "summary.json":
            raise PermissionError("fixture")
        real_unlink(name, dir_fd=dir_fd)

    monkeypatch.setattr(reclaim.os, "unlink", failing_unlink)
    result = reclaim.reclaim(apply=True, settings=world["settings"])
    monkeypatch.setattr(reclaim.os, "unlink", real_unlink)

    assert (result["outcome"], result["code"]) == ("refused", "reclaim_approved_folders_delete_incomplete")
    [row] = result["folders"]
    assert row["outcome"] == "incomplete" and row["error_type"] == "PermissionError"
    assert 0 < row["removed"]["file_count"] < 5 and (world["folder"] / "runs" / "a" / "summary.json").exists()
    [receipt] = _receipts(world)
    assert result["receipt_path"] == str(receipt) and json.loads(receipt.read_text()) == result

    finished = reclaim.reclaim(apply=True, settings=world["settings"])
    assert finished["outcome"] == "reclaimed" and not os.path.lexists(world["folder"])
    assert len(_receipts(world)) == 2
    assert (world["outside"] / "keep" / "evidence.json").exists()


@pytest.mark.parametrize("change", ["schema", "extra_key", "duplicate", "no_words", "bad_bound", "parent",
                                    "symlink", "not_json"])
def test_an_invalid_allowlist_refuses_everything(world: dict[str, Any], change: str) -> None:
    path = world["settings"].allowlist
    document = json.loads(path.read_text())
    entry = document["folders"][0]
    if change == "schema":
        document["schema_version"] = "owner_approved_reclaim_folders.v0"
    elif change == "extra_key":
        entry["recursive"] = True
    elif change == "duplicate":
        document["folders"].append(dict(entry))
    elif change == "no_words":
        entry["approval"] = {**APPROVAL, "words": []}
    elif change == "bad_bound":
        entry["refuse_if_modified_after"] = "2026-13-40T00:00:00Z"
    elif change == "parent":
        document["parent"] = "/var/lib/blueprint"
    if change == "symlink":
        real = world["tmp"] / "real-allowlist.json"
        path.rename(real)
        path.symlink_to(real)
    elif change == "not_json":
        path.write_text("{", encoding="utf-8")
    else:
        path.write_text(json.dumps(document), encoding="utf-8")
    result = _refused_untouched(world, "reclaim_approved_folders_allowlist_invalid")
    assert result["folders"] == []


def test_main_prints_the_document_and_exits_by_outcome(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(reclaim, "SETTINGS", world["settings"])
    assert reclaim.main([]) == 0
    assert json.loads(capsys.readouterr().out)["outcome"] == "planned"
    _allowlist(world["settings"].allowlist, [f"{PCP}/deploy-receipts"])
    assert reclaim.main(["--apply"]) == 2
    document = json.loads(capsys.readouterr().out)
    assert document["outcome"] == "refused" and document["code"] in reclaim.REFUSALS
