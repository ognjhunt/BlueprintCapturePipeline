from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from blueprint_pipeline import kitchen_asset_startup_gate as gate
from blueprint_pipeline.common import read_json, write_json

MAIN_USD = gate.DEFAULT_MAIN_USD_RELPATH


def _make_tree(root: Path) -> Path:
    tree = root / "assets"
    main = tree / MAIN_USD
    main.parent.mkdir(parents=True)
    main.write_bytes(b"#usda 1.0\ndef Xform \"KitchenRoom\" {}\n")
    textures = tree / "Collected_KitchenRoom" / "textures"
    textures.mkdir()
    (textures / "wall.png").write_bytes(b"fake-png-bytes")
    (tree / "Collected_KitchenRoom" / "props.usd").write_bytes(b"props-payload")
    return tree


def _write_inventory(path: Path, tree: Path, *, archive: Path | None = None) -> Path:
    inventory = gate.build_asset_inventory(tree, archive_path=archive)
    write_json(path, inventory)
    return path


def _make_archive(archive: Path, tree: Path) -> Path:
    with tarfile.open(archive, "w:gz") as tar:
        for member in sorted(tree.rglob("*")):
            if member.is_file():
                tar.add(member, arcname=member.relative_to(tree).as_posix())
    return archive


def _gate_artifact(out_dir: Path) -> dict:
    return read_json(out_dir / gate.GATE_ARTIFACT_NAME)


# ---------------------------------------------------------------------------
# Inventory contract.
# ---------------------------------------------------------------------------


def test_build_and_load_inventory_round_trip(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    loaded = gate.load_asset_inventory(inventory_path)
    assert loaded["schema_version"] == "kitchen_asset_inventory_checksums.v1"
    assert loaded["main_usd"] == MAIN_USD
    assert loaded["file_count"] == 3
    assert loaded["total_bytes"] == sum(item["bytes"] for item in loaded["files"])
    assert [item["path"] for item in loaded["files"]] == sorted(
        item["path"] for item in loaded["files"]
    )
    assert all(len(item["sha256"]) == 64 for item in loaded["files"])
    assert loaded["archive_sha256"] is None


def test_build_inventory_requires_main_usd(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    (tree / MAIN_USD).unlink()
    with pytest.raises(ValueError, match="main USD missing"):
        gate.build_asset_inventory(tree)


def test_load_inventory_rejects_wrong_schema(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory = gate.build_asset_inventory(tree)
    inventory["schema_version"] = "kitchen_asset_inventory_checksums.v2"
    path = tmp_path / "inventory.json"
    write_json(path, inventory)
    with pytest.raises(ValueError, match="schema_version"):
        gate.load_asset_inventory(path)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda inv: inv.update(file_count=99), "file_count"),
        (lambda inv: inv.update(total_bytes=1), "total_bytes"),
        (lambda inv: inv["files"][0].update(sha256="nothex"), "sha256"),
        (lambda inv: inv["files"][0].update(bytes=-1), "bytes"),
        (lambda inv: inv["files"][0].update(path="../escape.usd"), "path"),
        (lambda inv: inv.update(main_usd="/abs/KitchenRoom.usd"), "main_usd"),
        (lambda inv: inv.update(main_usd="not/in/files.usd"), "main_usd"),
        (lambda inv: inv.update(files=[]), "files"),
        (lambda inv: inv.update(archive_sha256="short"), "archive_sha256"),
    ],
)
def test_load_inventory_rejects_tampered_payloads(tmp_path: Path, mutate, match: str) -> None:
    tree = _make_tree(tmp_path)
    inventory = gate.build_asset_inventory(tree)
    mutate(inventory)
    path = tmp_path / "inventory.json"
    write_json(path, inventory)
    with pytest.raises(ValueError, match=match):
        gate.load_asset_inventory(path)


def test_load_inventory_rejects_duplicate_paths(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory = gate.build_asset_inventory(tree)
    inventory["files"].append(dict(inventory["files"][0]))
    inventory["file_count"] = len(inventory["files"])
    inventory["total_bytes"] += inventory["files"][0]["bytes"]
    path = tmp_path / "inventory.json"
    write_json(path, inventory)
    with pytest.raises(ValueError, match="duplicate"):
        gate.load_asset_inventory(path)


# ---------------------------------------------------------------------------
# Gate: tree input.
# ---------------------------------------------------------------------------


def test_gate_requires_exactly_one_input_and_always_writes_artifact(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)

    out_none = tmp_path / "out_none"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=out_none
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["kitchen_gate_input_invalid"]
    assert _gate_artifact(out_none)["status"] == "blocked"

    out_both = tmp_path / "out_both"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path,
        out_dir=out_both,
        tree_root=tree,
        archive_path=tmp_path / "whatever.tar.gz",
    )
    assert result["blockers"] == ["kitchen_gate_input_invalid"]
    assert (out_both / gate.GATE_ARTIFACT_NAME).is_file()


def test_gate_blocks_on_invalid_inventory(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text("{\"schema_version\": \"bogus\"}", encoding="utf-8")
    out = tmp_path / "out"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=out, tree_root=tree
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["kitchen_inventory_invalid"]
    assert (out / gate.GATE_ARTIFACT_NAME).is_file()


def test_tree_happy_path(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    out = tmp_path / "out"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=out, tree_root=tree
    )
    assert result["status"] == "completed"
    assert result["blockers"] == []
    assert result["verified_file_count"] == 3
    assert result["verified_total_bytes"] == read_json(inventory_path)["total_bytes"]
    assert result["main_usd_present"] is True
    assert result["input_mode"] == "tree"
    boundary = result["claim_boundary"]
    assert boundary["asset_presence_proven_only"] is True
    assert boundary["scene_load_proven"] is False
    assert boundary["placement_proven"] is False
    assert boundary["task_success_proven"] is False
    assert _gate_artifact(out)["status"] == "completed"


def test_tree_missing_file_blocks(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    (tree / "Collected_KitchenRoom" / "props.usd").unlink()
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=tmp_path / "out", tree_root=tree
    )
    assert result["status"] == "blocked"
    assert "kitchen_asset_file_missing" in result["blockers"]
    assert "kitchen_asset_file_count_mismatch" in result["blockers"]
    assert "kitchen_asset_total_bytes_mismatch" in result["blockers"]
    assert result["verified_file_count"] == 2


def test_tree_corrupted_file_same_size_blocks_on_checksum(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    target = tree / "Collected_KitchenRoom" / "props.usd"
    target.write_bytes(b"PROPS-PAYLOAD")  # same length, different content
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=tmp_path / "out", tree_root=tree
    )
    assert result["status"] == "blocked"
    assert "kitchen_asset_checksum_mismatch" in result["blockers"]
    assert "kitchen_asset_total_bytes_mismatch" not in result["blockers"]
    assert "kitchen_asset_file_count_mismatch" not in result["blockers"]


def test_tree_extra_file_blocks(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    (tree / "Collected_KitchenRoom" / "stray.usd").write_bytes(b"stray")
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=tmp_path / "out", tree_root=tree
    )
    assert result["status"] == "blocked"
    assert "kitchen_asset_unexpected_extra_files" in result["blockers"]
    assert "kitchen_asset_file_count_mismatch" in result["blockers"]
    assert "kitchen_asset_total_bytes_mismatch" in result["blockers"]


def test_tree_missing_main_usd_blocks(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    (tree / MAIN_USD).unlink()
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=tmp_path / "out", tree_root=tree
    )
    assert result["status"] == "blocked"
    assert "kitchen_main_usd_missing" in result["blockers"]
    assert result["main_usd_present"] is False


# ---------------------------------------------------------------------------
# Gate: archive input.
# ---------------------------------------------------------------------------


def test_archive_happy_path_extracts_and_verifies(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    archive = _make_archive(tmp_path / "kitchen.tar.gz", tree)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree, archive=archive)
    out = tmp_path / "out"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=out, archive_path=archive
    )
    assert result["status"] == "completed"
    assert result["input_mode"] == "archive"
    assert result["archive_sha256"] == read_json(inventory_path)["archive_sha256"]
    assert result["materialized_tree"] == str(out / "materialized")
    assert (out / "materialized" / MAIN_USD).is_file()
    assert result["progress"], "extraction must record progress"
    assert result["progress"][-1]["files_extracted"] == 3
    assert result["progress"][-1]["bytes_extracted"] == result["verified_total_bytes"]
    disk = result["disk"]
    assert disk["free_bytes_before"] >= disk["required_bytes"]
    assert disk["free_bytes_after"] is not None
    assert result["reuse"] == {"reused": False, "reuse_reason": None}


def test_archive_digest_mismatch_blocks_before_extraction(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    archive = _make_archive(tmp_path / "kitchen.tar.gz", tree)
    inventory = gate.build_asset_inventory(tree, archive_path=archive)
    inventory["archive_sha256"] = "0" * 64
    inventory_path = tmp_path / "inventory.json"
    write_json(inventory_path, inventory)
    out = tmp_path / "out"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=out, archive_path=archive
    )
    assert result["status"] == "blocked"
    assert result["blockers"] == ["kitchen_archive_digest_mismatch"]
    assert not (out / "materialized").exists()


def test_archive_path_traversal_member_rejected_before_extraction(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    archive = tmp_path / "evil.tar"
    with tarfile.open(archive, "w") as tar:
        payload = b"evil"
        info = tarfile.TarInfo(name="../evil.usd")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    out = tmp_path / "out"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=out, archive_path=archive
    )
    assert result["status"] == "blocked"
    assert "kitchen_archive_path_traversal" in result["blockers"]
    assert not (out / "materialized").exists()
    assert not (tmp_path / "evil.usd").exists()


def test_archive_unsafe_symlink_member_rejected(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    archive = tmp_path / "link.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name="Collected_KitchenRoom/link.usd")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside.usd"
        tar.addfile(info)
    out = tmp_path / "out"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=out, archive_path=archive
    )
    assert result["status"] == "blocked"
    assert "kitchen_archive_unsafe_link" in result["blockers"]
    assert not (out / "materialized").exists()


def test_archive_fifo_member_rejected(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    archive = tmp_path / "fifo.tar"
    with tarfile.open(archive, "w") as tar:
        info = tarfile.TarInfo(name="Collected_KitchenRoom/pipe")
        info.type = tarfile.FIFOTYPE
        tar.addfile(info)
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path,
        out_dir=tmp_path / "out",
        archive_path=archive,
    )
    assert result["status"] == "blocked"
    assert "kitchen_archive_unsupported_member_type" in result["blockers"]


def test_verify_archive_safety_reports_digest_and_flags(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    archive = _make_archive(tmp_path / "kitchen.tar.gz", tree)
    safety = gate.verify_archive_safety(archive)
    assert safety["safe"] is True
    assert safety["blockers"] == []
    assert len(safety["archive_sha256"]) == 64
    assert safety["member_count"] == 3


def test_insufficient_disk_blocks_before_extraction(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    archive = _make_archive(tmp_path / "kitchen.tar.gz", tree)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree, archive=archive)
    out = tmp_path / "out"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path,
        out_dir=out,
        archive_path=archive,
        free_disk_probe=lambda path: 10,
    )
    assert result["status"] == "blocked"
    assert "kitchen_asset_insufficient_disk" in result["blockers"]
    assert result["disk"]["free_bytes_before"] == 10
    assert result["disk"]["required_bytes"] > 10
    assert result["progress"] == []
    assert not (out / "materialized").exists()
    assert (out / gate.GATE_ARTIFACT_NAME).is_file()


# ---------------------------------------------------------------------------
# Reuse of a previously extracted tree.
# ---------------------------------------------------------------------------


def _run_first_gate(tmp_path: Path) -> tuple[Path, Path, Path]:
    tree = _make_tree(tmp_path)
    archive = _make_archive(tmp_path / "kitchen.tar.gz", tree)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree, archive=archive)
    out_first = tmp_path / "out_first"
    first = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path, out_dir=out_first, archive_path=archive
    )
    assert first["status"] == "completed"
    return archive, inventory_path, out_first / gate.GATE_ARTIFACT_NAME


def test_reuse_allowed_only_with_passing_prior_manifest_and_same_digest(tmp_path: Path) -> None:
    archive, inventory_path, prior_manifest = _run_first_gate(tmp_path)
    out_second = tmp_path / "out_second"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path,
        out_dir=out_second,
        archive_path=archive,
        reuse_manifest_path=prior_manifest,
    )
    assert result["status"] == "completed"
    assert result["reuse"]["reused"] is True
    assert result["reuse"]["reuse_reason"] == "prior_gate_passed_same_archive_digest"
    assert result["progress"] == []  # no re-extraction
    assert result["materialized_tree"] == read_json(prior_manifest)["materialized_tree"]
    assert not (out_second / "materialized").exists()


def test_reuse_refused_on_prior_digest_mismatch(tmp_path: Path) -> None:
    archive, inventory_path, prior_manifest = _run_first_gate(tmp_path)
    prior = read_json(prior_manifest)
    prior["archive_sha256"] = "0" * 64
    write_json(prior_manifest, prior)
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path,
        out_dir=tmp_path / "out_second",
        archive_path=archive,
        reuse_manifest_path=prior_manifest,
    )
    assert result["status"] == "blocked"
    assert "kitchen_asset_reuse_digest_mismatch" in result["blockers"]
    assert result["reuse"] == {
        "reused": False,
        "reuse_reason": "prior_archive_digest_mismatch",
    }


def test_reuse_falls_back_to_fresh_extraction_when_prior_not_passed(tmp_path: Path) -> None:
    archive, inventory_path, prior_manifest = _run_first_gate(tmp_path)
    prior = read_json(prior_manifest)
    prior["status"] = "blocked"
    write_json(prior_manifest, prior)
    out_second = tmp_path / "out_second"
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path,
        out_dir=out_second,
        archive_path=archive,
        reuse_manifest_path=prior_manifest,
    )
    assert result["status"] == "completed"
    assert result["reuse"]["reused"] is False
    assert result["reuse"]["reuse_reason"] == "prior_manifest_not_passed"
    assert result["progress"], "fresh extraction must record progress"
    assert (out_second / "materialized" / MAIN_USD).is_file()


def test_reuse_falls_back_when_prior_manifest_missing(tmp_path: Path) -> None:
    tree = _make_tree(tmp_path)
    archive = _make_archive(tmp_path / "kitchen.tar.gz", tree)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree, archive=archive)
    result = gate.run_kitchen_asset_startup_gate(
        expected_inventory_path=inventory_path,
        out_dir=tmp_path / "out",
        archive_path=archive,
        reuse_manifest_path=tmp_path / "does_not_exist.json",
    )
    assert result["status"] == "completed"
    assert result["reuse"]["reused"] is False
    assert result["reuse"]["reuse_reason"] == "prior_manifest_unreadable"


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def test_cli_exit_zero_on_pass(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    out = tmp_path / "out"
    exit_code = gate.main(
        [
            "--expected-inventory",
            str(inventory_path),
            "--out-dir",
            str(out),
            "--tree-root",
            str(tree),
        ]
    )
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "completed"
    assert (out / gate.GATE_ARTIFACT_NAME).is_file()


def test_cli_exit_two_on_blocked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tree = _make_tree(tmp_path)
    inventory_path = _write_inventory(tmp_path / "inventory.json", tree)
    (tree / MAIN_USD).unlink()
    out = tmp_path / "out"
    exit_code = gate.main(
        [
            "--expected-inventory",
            str(inventory_path),
            "--out-dir",
            str(out),
            "--tree-root",
            str(tree),
        ]
    )
    assert exit_code == 2
    printed = json.loads(capsys.readouterr().out)
    assert "kitchen_main_usd_missing" in printed["blockers"]
    assert _gate_artifact(out)["status"] == "blocked"
