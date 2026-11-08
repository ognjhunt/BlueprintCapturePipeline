"""The pi0.5 G1 language file restore, against tmp folders and synthetic retained bundles."""

# Covers (for impacted-test selection):
#   src/blueprint_pipeline/native_g1_pi_language_asset_restore.py
#   src/blueprint_pipeline/native_g1_pi_tokenizer_assets.py

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import inspect
import json
import os
import sys
import zipfile
from collections import namedtuple
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import native_g1_pi_language_asset_restore as restore
from blueprint_pipeline import native_g1_pi_tokenizer_assets as assets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "operator-door"))

from operator_door import spool_runner  # noqa: E402
from operator_door.secrets_guard import refused_name, scan_bytes  # noqa: E402

Usage = namedtuple("Usage", "total used free")
CAMPAIGNS = ["g1-" + "1" * 64, "g1-" + "2" * 64, "g1-" + "3" * 64]


def _content(name: str) -> bytes:
    # Synthetic stand-ins with a marker no result may ever contain.
    return (f"SYNTHETIC-CONTENT-MARKER {name} " * 40).encode()


def _inventory_rows() -> list[dict[str, Any]]:
    rows = []
    for name in sorted(assets.EXPECTED_FILES):
        content = _content(name)
        if name in {"tokenizer.json", "tokenizer.model"}:
            identity = {"sha256": hashlib.sha256(content).hexdigest()}
        else:
            identity = {"git_blob_sha1": hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()}
        rows.append({"path": name, "size_bytes": len(content), **identity})
    return rows


def _bundle(path: Path, members: dict[str, bytes] | None = None, *, extra: dict[str, bytes] | None = None) -> Path:
    members = {name: _content(name) for name in assets.EXPECTED_FILES} if members is None else members
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/native_g1_provider_manifest.json", "{}")
        for name, content in sorted(members.items()):
            archive.writestr(restore.ARCHIVE_PREFIX + name, content)
        for name, content in (extra or {}).items():
            archive.writestr(name, content)
    return path


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    inventory = tmp_path / "release" / "configs" / "g1_paligemma_tokenizer_inventory.v1.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(json.dumps({
        "schema_version": "g1_paligemma_tokenizer_inventory.v1",
        "source_repository": assets.SOURCE_REPOSITORY, "source_revision": assets.SOURCE_REVISION,
        "model_license": "gemma", "rights_review_required": True, "files": _inventory_rows(),
    }), encoding="utf-8")
    parent = tmp_path / "inputs" / "g1-development"
    target = parent / "pi-tokenizer-f1eaa13"
    target.mkdir(parents=True)
    for name in sorted(assets.EXPECTED_FILES):
        (target / name).write_bytes(_content(name))
    # The folder as screening found it: two files changed, one gone, one stray file.
    (target / "tokenizer.json").write_bytes(b"stale bytes")
    (target / "special_tokens_map.json").write_bytes(_content("special_tokens_map.json")[:-1] + b"!")
    (target / "added_tokens.json").unlink()
    (target / "notes.txt").write_bytes(b"left by hand")
    work = tmp_path / "campaign-work"
    _bundle(work / CAMPAIGNS[1] / "bundle" / "native_g1_provider_bundle.zip")
    pins = restore.Pins(target_dir=target, source_root=work, inventory_path=inventory,
                        inventory_sha256="sha256:" + hashlib.sha256(inventory.read_bytes()).hexdigest())
    monkeypatch.setattr(restore, "PINS", pins)
    monkeypatch.setenv("BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT", str(tmp_path / "ledger"))
    monkeypatch.setattr(restore, "_disk_usage", lambda _path: Usage(10**15, 0, 10**14))
    return {"pins": pins, "target": target, "parent": parent, "work": work, "inventory": inventory,
            "ledger": tmp_path / "ledger", "tmp": tmp_path}


def _tree(root: Path) -> dict[str, Any]:
    return {str(path.relative_to(root)): (path.read_bytes() if path.is_file() and not path.is_symlink()
                                          else "link" if path.is_symlink() else None)
            for path in sorted(root.rglob("*"))}


def _reservations(ledger: Path) -> list[str]:
    return sorted(path.name for path in ledger.glob("*.json")) if ledger.is_dir() else []


def _verifies(world: dict[str, Any], directory: Path | None = None) -> bool:
    try:
        assets.verify_tokenizer_assets(inventory_path=world["inventory"], asset_dir=directory or world["target"])
    except ValueError:
        return False
    return True


def test_production_pins_are_the_folder_screening_reads_and_the_authority_s_inventory() -> None:
    from blueprint_pipeline import native_g1_beta_bundle, native_g1_provider_bundle
    from blueprint_pipeline.control_plane_storage_roots import STORAGE_ROOTS
    from blueprint_pipeline.native_g1_beta_authority import PI_AUXILIARY
    from blueprint_pipeline.native_g1_beta_tokenizer import INVENTORY, RETAINED_TOKENIZER_DIR

    pins = restore.PINS
    assert pins.target_dir == RETAINED_TOKENIZER_DIR
    assert pins.inventory_path == INVENTORY == ROOT / "configs/g1_paligemma_tokenizer_inventory.v1.json"
    assert pins.inventory_sha256 == PI_AUXILIARY["inventory_file_sha256"]
    assert "sha256:" + hashlib.sha256(INVENTORY.read_bytes()).hexdigest() == pins.inventory_sha256
    assert restore.load_inventory(pins)["files"] == assets._inventory(INVENTORY)["files"]
    assert restore.BUNDLE_NAMES == (native_g1_provider_bundle.BUNDLE_NAME, native_g1_beta_bundle.BUNDLE_NAME)
    assert any(row.path == str(pins.source_root) for row in STORAGE_ROOTS)
    for unit in ("blueprint-native-g1-team-campaign-dispatcher.service",
                 "blueprint-native-g1-team-campaign-settlement.service"):
        text = (ROOT / "deploy/systemd" / unit).read_text(encoding="utf-8")
        assert f"BLUEPRINT_NATIVE_G1_TEAM_CAMPAIGN_WORK_ROOT={pins.source_root}\n" in text
    # The door's own spool runner names the only writable folder; the restore's parent lies inside it (the
    # files sit in the retained folder's ``assets/`` directory, so the parent is that folder).
    writable = Path(spool_runner._G1_PI_LANGUAGE_PARENT)
    assert writable in (pins.parent, *pins.parent.parents)


def test_the_archive_layout_is_the_one_both_bundle_builders_write(world: dict[str, Any]) -> None:
    from blueprint_pipeline import native_g1_beta_bundle
    from blueprint_pipeline.native_g1_provider_bundle import _verify_embedded_pi_tokenizer

    # The provider bundle's own sealed-bundle check accepts a bundle built with this module's prefix.
    staged = world["tmp"] / "staged"
    staged.mkdir()
    for name in assets.EXPECTED_FILES:
        (staged / name).write_bytes(_content(name))
    receipt = assets.verify_tokenizer_assets(inventory_path=world["inventory"], asset_dir=staged)
    bundle = _bundle(world["tmp"] / "check.zip",
                     extra={"configs/" + world["inventory"].name: world["inventory"].read_bytes()})
    with zipfile.ZipFile(bundle) as archive:
        _verify_embedded_pi_tokenizer(archive, {"pi_tokenizer_assets": receipt}, inventory_path=world["inventory"])
    assert f'"{restore.ARCHIVE_PREFIX}" + row["path"]' in inspect.getsource(native_g1_beta_bundle.build_bundle)


def test_no_name_the_door_reads_contains_the_refused_word() -> None:
    for name in (Path(restore.__file__).name, "20261107T231500Z-restore-g1-pi-language-assets-0123abcd.json",
                 "20261107T231500Z-restore-g1-pi-language-assets-0123abcd.g1-pi-language-assets.json",
                 restore.RECEIPT_PREFIX + "20261107T231500Z.json", "door-restore-g1-pi-language-assets.sh"):
        assert not refused_name(name), name
    assert any(refused_name(part) for part in restore.PINS.target_dir.parts), \
        "the folder itself stays unreadable through the door"


def test_a_plan_finds_the_verified_bundle_and_writes_nothing(world: dict[str, Any]) -> None:
    before = _tree(world["tmp"] / "inputs")
    result = restore.restore()

    assert result["outcome"] == "planned" and result["code"] is None and result["mode"] == "plan", result
    assert result["inventory"]["verified"] is True and result["restore_needed"] is True
    target = result["target"]
    assert target["state"] == "invalid" and target["present"] is True
    assert target["verification"]["verified"] is False
    assert {row["path"]: row["state"] for row in target["inventory_files"]} == {
        "added_tokens.json": "absent", "preprocessor_config.json": "matches",
        "special_tokens_map.json": "digest_mismatch", "tokenizer.json": "size_mismatch",
        "tokenizer.model": "matches", "tokenizer_config.json": "matches"}
    listed = {row["path"]: row for row in target["existing"]["entries"]}
    assert set(listed) == {"notes.txt", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json",
                           "tokenizer.model", "tokenizer_config.json"}
    assert listed["tokenizer.json"] == {"path": "tokenizer.json", "type": "file", "size_bytes": 11,
                                        "sha256": "sha256:" + hashlib.sha256(b"stale bytes").hexdigest()}
    bundle = world["work"] / CAMPAIGNS[1] / "bundle" / "native_g1_provider_bundle.zip"
    assert result["source"]["path"] == str(bundle)
    assert result["source"]["sha256"] == "sha256:" + hashlib.sha256(bundle.read_bytes()).hexdigest()
    assert [row["path"] for row in result["source"]["members"]] == sorted(assets.EXPECTED_FILES)
    assert result["source_search"]["checked"] == [{
        "path": str(bundle), "reason": "verified", "size_bytes": bundle.stat().st_size,
        "sha256": result["source"]["sha256"]}]
    assert result["disk"]["sufficient"] is True
    assert result["disk"]["need_bytes"] == sum(len(_content(n)) for n in assets.EXPECTED_FILES) + restore.DISK_MARGIN_BYTES
    assert result["set_aside"]["performed"] is False
    assert Path(result["set_aside"]["would_be"]).name.startswith(".pi-tokenizer-f1eaa13.invalid-")
    assert _tree(world["tmp"] / "inputs") == before and _reservations(world["ledger"]) == []


def test_apply_sets_the_invalid_folder_aside_and_puts_verified_files_in_place(world: dict[str, Any]) -> None:
    original = _tree(world["target"])
    result = restore.restore(apply=True)

    assert result["outcome"] == "restored", result
    target, parent = world["target"], world["parent"]
    assert _verifies(world)
    assert sorted(os.listdir(target)) == sorted(assets.EXPECTED_FILES)
    for name in assets.EXPECTED_FILES:
        assert (target / name).read_bytes() == _content(name)
        assert oct((target / name).stat().st_mode & 0o777) == oct(0o640)
    assert oct(target.stat().st_mode & 0o777) == oct(0o750)
    aside = Path(result["set_aside"]["path"])
    assert result["set_aside"]["performed"] is True and aside.parent == parent
    assert aside.name == ".pi-tokenizer-f1eaa13.invalid-" + result["restored_at"].replace("-", "").replace(":", "")
    assert _tree(aside) == original, "the invalid folder is kept byte for byte"
    assert result["staged"]["verified"] is True and result["verification"]["verified"] is True
    assert [row["path"] for row in result["verification"]["files"]] == sorted(assets.EXPECTED_FILES)
    assert result["written"] == [str(target / name) for name in sorted(assets.EXPECTED_FILES)]
    receipt = Path(result["receipt_path"])
    assert receipt.parent == parent and receipt.name.startswith(restore.RECEIPT_PREFIX)
    assert oct(receipt.stat().st_mode & 0o777) == oct(0o640)
    kept = json.loads(receipt.read_text(encoding="utf-8"))
    assert kept == {**result, "outcome": "restored"}
    assert kept["authorization"]["reference"].count('"') == 4
    assert sorted(path.name for path in parent.iterdir()) == sorted([target.name, aside.name, receipt.name])
    assert _reservations(world["ledger"]) == [], "the disk hold is released"


def test_a_verified_folder_is_a_no_op_in_both_modes(world: dict[str, Any]) -> None:
    assert restore.restore(apply=True)["outcome"] == "restored"
    before = _tree(world["parent"])
    stamps = {path: path.stat().st_mtime_ns for path in world["parent"].rglob("*")}

    plan = restore.restore()
    applied = restore.restore(apply=True)

    assert plan["outcome"] == "planned" and plan["restore_needed"] is False and plan["no_op"] is True
    assert applied["outcome"] == "already_restored" and applied["no_op"] is True
    for result in (plan, applied):
        assert result["target"]["state"] == "verified" and result["source_search"] is None
        assert result["written"] == [] and result["receipt_path"] is None
    assert _tree(world["parent"]) == before
    assert {path: path.stat().st_mtime_ns for path in world["parent"].rglob("*")} == stamps


def test_an_absent_folder_is_restored_with_nothing_set_aside(world: dict[str, Any]) -> None:
    for path in world["target"].iterdir():
        path.unlink()
    world["target"].rmdir()

    result = restore.restore(apply=True)

    assert result["outcome"] == "restored" and result["target"]["state"] == "absent"
    assert result["set_aside"] is None and _verifies(world)
    assert sorted(path.name for path in world["parent"].iterdir()) == sorted(
        [world["target"].name, Path(result["receipt_path"]).name])


def test_the_first_bundle_in_sorted_order_whose_every_file_verifies_is_the_source(world: dict[str, Any]) -> None:
    work = world["work"]
    good = {name: _content(name) for name in assets.EXPECTED_FILES}
    # An older bundle that never embedded the files, a corrupt member, a symlink, a non-zip,
    # a beta bundle after the first verified one, and directories that are not campaigns.
    _bundle(work / CAMPAIGNS[0] / "bundle" / "native_g1_provider_bundle.zip", {})
    _bundle(work / CAMPAIGNS[0] / "bundle" / "native_g1_beta_provider_bundle.zip",
            {**good, "tokenizer.model": good["tokenizer.model"][:-1] + b"?"})
    (work / CAMPAIGNS[1] / "bundle" / "native_g1_provider_bundle.zip").unlink()
    elsewhere = _bundle(world["tmp"] / "elsewhere" / "native_g1_provider_bundle.zip")
    (work / CAMPAIGNS[1] / "bundle" / "native_g1_provider_bundle.zip").symlink_to(elsewhere)
    (work / CAMPAIGNS[1] / "bundle" / "native_g1_beta_provider_bundle.zip").write_bytes(b"not a zip")
    _bundle(work / CAMPAIGNS[2] / "bundle" / "native_g1_provider_bundle.zip")
    _bundle(work / CAMPAIGNS[2] / "bundle" / "native_g1_beta_provider_bundle.zip")
    _bundle(work / ("g1-" + "0" * 63) / "bundle" / "native_g1_provider_bundle.zip")
    _bundle(work / "scratch" / "bundle" / "native_g1_provider_bundle.zip")

    result = restore.restore()

    checked = [(Path(row["path"]).relative_to(work).as_posix(), row["reason"], row.get("member"))
               for row in result["source_search"]["checked"]]
    assert checked == [
        (f"{CAMPAIGNS[0]}/bundle/native_g1_provider_bundle.zip", "bundle_member_missing", "added_tokens.json"),
        (f"{CAMPAIGNS[0]}/bundle/native_g1_beta_provider_bundle.zip", "bundle_member_digest_mismatch",
         "tokenizer.model"),
        (f"{CAMPAIGNS[1]}/bundle/native_g1_provider_bundle.zip", "bundle_symlink_refused", None),
        (f"{CAMPAIGNS[1]}/bundle/native_g1_beta_provider_bundle.zip", "bundle_not_a_zip", None),
        (f"{CAMPAIGNS[2]}/bundle/native_g1_provider_bundle.zip", "verified", None),
    ]
    assert result["outcome"] == "planned"
    assert result["source"]["path"] == str(work / CAMPAIGNS[2] / "bundle" / "native_g1_provider_bundle.zip")
    assert result["source_search"]["campaign_directories"] == 3
    assert set(row["reason"] for row in result["source_search"]["checked"]) <= set(restore.BUNDLE_REASONS)


@pytest.mark.parametrize(("members", "reason"), [
    ("short", "bundle_member_size_mismatch"),
    ("duplicate", "bundle_member_duplicate"),
])
def test_a_bundle_with_a_wrong_or_ambiguous_member_is_not_a_source(world: dict[str, Any], members: str,
                                                                   reason: str) -> None:
    path = world["work"] / CAMPAIGNS[1] / "bundle" / "native_g1_provider_bundle.zip"
    good = {name: _content(name) for name in assets.EXPECTED_FILES}
    if members == "short":
        _bundle(path, {**good, "tokenizer_config.json": b"short"})
    else:
        with pytest.warns(UserWarning, match="Duplicate name"):
            _bundle(path, extra={restore.ARCHIVE_PREFIX + "tokenizer.json": good["tokenizer.json"]})
    before = _tree(world["tmp"] / "inputs")

    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["outcome"] == "refused" and result["code"] == "g1_pi_language_assets_no_verified_source"
        assert [row["reason"] for row in result["source_search"]["checked"]] == [reason]
    assert _tree(world["tmp"] / "inputs") == before and _reservations(world["ledger"]) == []


def test_no_verified_bundle_refuses_by_name_and_touches_nothing(world: dict[str, Any]) -> None:
    (world["work"] / CAMPAIGNS[1] / "bundle" / "native_g1_provider_bundle.zip").unlink()
    _bundle(world["work"] / CAMPAIGNS[0] / "bundle" / "native_g1_provider_bundle.zip", {})
    before = _tree(world["tmp"] / "inputs")

    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["outcome"] == "refused" and result["code"] == "g1_pi_language_assets_no_verified_source"
        assert result["source"] is None and result["staged"] is None and result["set_aside"] is None
        assert [row["reason"] for row in result["source_search"]["checked"]] == ["bundle_member_missing"]
        assert result["target"]["state"] == "invalid"
    assert _tree(world["tmp"] / "inputs") == before


@pytest.mark.parametrize("change", ["missing", "symlink", "file"])
def test_an_unusable_search_root_is_refused(world: dict[str, Any], tmp_path: Path, change: str,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "other-root"
    if change == "symlink":
        root.symlink_to(world["work"], target_is_directory=True)
    elif change == "file":
        root.write_bytes(b"")
    monkeypatch.setattr(restore, "PINS", dataclasses.replace(world["pins"], source_root=root))
    result = restore.restore()
    assert result["outcome"] == "refused" and result["code"] == "g1_pi_language_assets_source_root_unavailable"


@pytest.mark.parametrize("conflict", ["symlink", "file", "parent_symlink"])
def test_a_target_that_is_not_a_real_folder_is_refused_untouched(world: dict[str, Any], tmp_path: Path,
                                                                 conflict: str,
                                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    target = world["target"]
    if conflict == "parent_symlink":
        linked = tmp_path / "linked-inputs"
        linked.symlink_to(world["parent"], target_is_directory=True)
        monkeypatch.setattr(restore, "PINS", dataclasses.replace(world["pins"], target_dir=linked / target.name))
    else:
        moved = tmp_path / "moved"
        target.rename(moved)
        if conflict == "symlink":
            target.symlink_to(moved, target_is_directory=True)
        else:
            target.write_bytes(b"not a folder")
    before = _tree(world["tmp"] / "inputs")

    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["outcome"] == "refused" and result["code"] == "g1_pi_language_assets_target_conflict"
    assert _tree(world["tmp"] / "inputs") == before


def test_a_changed_inventory_is_refused_before_anything_is_read(world: dict[str, Any],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(restore, "PINS", dataclasses.replace(world["pins"], inventory_sha256="sha256:" + "0" * 64))
    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["code"] == "g1_pi_language_assets_inventory_changed"
        assert result["target"]["state"] is None and result["source_search"] is None


def test_too_little_disk_is_refused_before_anything_is_written(world: dict[str, Any],
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(restore, "_disk_usage", lambda _path: Usage(10 * 1024**3, 0, 1024**3))
    before = _tree(world["tmp"] / "inputs")
    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["outcome"] == "refused" and result["code"] == "g1_pi_language_assets_disk_insufficient"
        assert result["disk"]["sufficient"] is False
    assert _tree(world["tmp"] / "inputs") == before and _reservations(world["ledger"]) == []


def test_a_parent_the_service_user_cannot_write_is_refused(world: dict[str, Any],
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(restore, "_parent_writable", lambda _path: False)
    before = _tree(world["tmp"] / "inputs")
    for apply in (False, True):
        result = restore.restore(apply=apply)
        assert result["code"] == "g1_pi_language_assets_target_parent_unwritable"
        assert result["target"]["parent_writable"] is False
    assert _tree(world["tmp"] / "inputs") == before


def test_a_running_restore_makes_another_refuse_busy(world: dict[str, Any]) -> None:
    fd = os.open(world["parent"], os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        result = restore.restore(apply=True)
    finally:
        os.close(fd)
    assert result["code"] == "g1_pi_language_assets_restore_busy"


def test_a_staged_copy_that_does_not_verify_is_removed_and_the_folder_stays(world: dict[str, Any],
                                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    real = restore.verify_tokenizer_assets

    def refuse_staged(*, inventory_path: Path, asset_dir: Path) -> dict[str, Any]:
        if Path(asset_dir).name.endswith(".partial"):
            raise ValueError("g1_pi_tokenizer_file_digest_mismatch:tokenizer.json")
        return real(inventory_path=inventory_path, asset_dir=asset_dir)

    monkeypatch.setattr(restore, "verify_tokenizer_assets", refuse_staged)
    before = _tree(world["tmp"] / "inputs")

    result = restore.restore(apply=True)

    assert result["outcome"] == "refused" and result["code"] == "g1_pi_language_assets_verification_failed"
    assert result["staged"]["verified"] is False and result["set_aside"] is None
    assert result["rolled_back"] == {"complete": True} and result["written"] == []
    assert _tree(world["tmp"] / "inputs") == before, "the staged copy is removed; the folder is untouched"
    assert _reservations(world["ledger"]) == []


def test_a_final_check_that_fails_puts_the_original_folder_back(world: dict[str, Any],
                                                                monkeypatch: pytest.MonkeyPatch) -> None:
    real = restore.verify_tokenizer_assets
    calls: list[str] = []

    def refuse_final(*, inventory_path: Path, asset_dir: Path) -> dict[str, Any]:
        calls.append(Path(asset_dir).name)
        if Path(asset_dir) == world["target"] and len(calls) > 1:
            raise ValueError("g1_pi_tokenizer_file_digest_mismatch:tokenizer.json")
        return real(inventory_path=inventory_path, asset_dir=asset_dir)

    monkeypatch.setattr(restore, "verify_tokenizer_assets", refuse_final)
    before = _tree(world["tmp"] / "inputs")

    result = restore.restore(apply=True)

    assert result["code"] == "g1_pi_language_assets_verification_failed", result
    assert len(calls) == 3 and result["staged"]["verified"] is True
    assert result["rolled_back"] == {"complete": True}
    assert result["set_aside"] == {"performed": False, "returned": True}
    assert _tree(world["tmp"] / "inputs") == before, "the original folder is back under its name"


def test_an_interrupted_copy_leaves_the_folder_as_it_was(world: dict[str, Any],
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    def cut(source: Any, inventory: Any, staged_fd: int, written: list[str]) -> None:
        fd = os.open("tokenizer.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=staged_fd)
        os.close(fd)
        written.append("tokenizer.json")
        raise SystemExit(143)

    monkeypatch.setattr(restore, "_stage", cut)
    before = _tree(world["tmp"] / "inputs")
    with pytest.raises(SystemExit):
        restore.restore(apply=True)
    assert _tree(world["tmp"] / "inputs") == before


def test_a_bundle_changed_between_the_check_and_the_copy_is_refused(world: dict[str, Any],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    real = restore.find_source

    def then_change(pins: Any, inventory: Any, report: dict[str, Any]) -> Any:
        source = real(pins, inventory, report)
        # The copy now reads other bytes for one file than the check read.
        source.members["tokenizer.model"] = source.members["tokenizer.json"]
        return source

    monkeypatch.setattr(restore, "find_source", then_change)
    before = _tree(world["tmp"] / "inputs")
    result = restore.restore(apply=True)
    assert result["code"] == "g1_pi_language_assets_copy_failed"
    assert _tree(world["tmp"] / "inputs") == before


def test_the_result_records_names_and_hashes_never_contents(world: dict[str, Any], tmp_path: Path) -> None:
    plan_out = tmp_path / "plan.json"
    assert restore.main(["--result", str(plan_out)]) == 0
    out = tmp_path / "result.json"
    assert restore.main(["--apply", "--result", str(out)]) == 0
    for path in (plan_out, out, Path(json.loads(out.read_text())["receipt_path"])):
        raw = path.read_bytes()
        assert b"SYNTHETIC-CONTENT-MARKER" not in raw and b"stale bytes" not in raw and b"left by hand" not in raw
        assert scan_bytes(raw) is None, "the door can cat it"
        document = json.loads(raw)
        assert document["file_contents_recorded"] is False and document["network_used"] is False
        assert document["provider_mutation_performed"] is False and document["compute_started"] is False
        assert document["schema_version"] == restore.SCHEMA_VERSION


def test_a_refusal_exits_two_with_its_fixed_code(world: dict[str, Any], capsys: pytest.CaptureFixture[str]) -> None:
    (world["work"] / CAMPAIGNS[1] / "bundle" / "native_g1_provider_bundle.zip").unlink()
    assert restore.main([]) == 2
    document = json.loads(capsys.readouterr().out)
    assert document["outcome"] == "refused" and document["code"] in restore.REFUSALS


def test_the_shared_identity_rule_reads_files_and_archive_members_alike(world: dict[str, Any]) -> None:
    rows = {row["path"]: row for row in assets._inventory(world["inventory"])["files"]}
    bundle = world["work"] / CAMPAIGNS[1] / "bundle" / "native_g1_provider_bundle.zip"
    with zipfile.ZipFile(bundle) as archive:
        for name, row in rows.items():
            info = archive.getinfo(restore.ARCHIVE_PREFIX + name)
            with archive.open(info) as member:
                assert assets.stream_identity(member, row, size=info.file_size) == (
                    row.get("sha256") or row.get("git_blob_sha1"), row["size_bytes"])
    with bundle.open("rb") as stream, pytest.raises(ValueError, match="size_mismatch:tokenizer.json"):
        assets.stream_identity(stream, rows["tokenizer.json"], size=1)
