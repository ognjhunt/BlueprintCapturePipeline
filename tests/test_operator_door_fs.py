"""File access through the door: confined to read roots, never a credential."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/fsview.py

from __future__ import annotations

import io
import json
import os
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.config import DoorConfig  # noqa: E402
from operator_door.fsview import FileView, FsRefused  # noqa: E402


@pytest.fixture()
def tree(tmp_path: Path) -> dict[str, Path]:
    base = tmp_path.resolve()
    root = base / "state"
    outside = base / "outside"
    hidden = root / "provider-secrets"
    for directory in (root / "runs" / "scene-1", outside, hidden):
        directory.mkdir(parents=True)
    (root / "runs" / "scene-1" / "progression.json").write_text('{"stage": "masks"}', encoding="utf-8")
    (root / "runs" / "scene-1" / "big.log").write_bytes(b"x" * 5000)
    (root / "runs" / "scene-1" / "release.env").write_text("A=1\n", encoding="utf-8")
    (root / "runs" / "scene-1" / "leaky.json").write_text(
        '{"refresh_token": "1//0abcdefghijklmnopqrstuvwxyz"}', encoding="utf-8"
    )
    (hidden / "vast_api_key").write_text("nope", encoding="utf-8")
    (outside / "elsewhere.txt").write_text("outside", encoding="utf-8")
    os.symlink(outside / "elsewhere.txt", root / "runs" / "escape.txt")
    os.symlink(hidden / "vast_api_key", root / "runs" / "sneaky.txt")
    os.symlink(root / "runs" / "scene-1", root / "runs" / "latest")
    return {"base": base, "root": root, "outside": outside, "hidden": hidden}


def _view(tree: dict[str, Path], **overrides: object) -> FileView:
    config = DoorConfig(
        read_roots=(str(tree["root"]),),
        hidden_paths=(str(tree["hidden"]),),
        **overrides,  # type: ignore[arg-type]
    )
    return FileView(config)


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("relative/path", "path_not_absolute"),
        ("", "path_not_absolute"),
        ("/etc/passwd", "path_outside_roots"),
        ("/tmp/../etc", "path_outside_roots"),
    ],
)
def test_paths_outside_roots_are_refused(tree: dict[str, Path], raw: str, code: str) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).resolve(raw)
    assert caught.value.code == code


def test_nul_bytes_are_refused(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).resolve(str(tree["root"]) + "/a\x00b")
    assert caught.value.code == "path_invalid"


def test_traversal_out_of_a_root_is_refused(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).resolve(str(tree["root"] / "runs" / ".." / ".." / "outside" / "elsewhere.txt"))
    assert caught.value.code == "path_outside_roots"


def test_symlink_escaping_the_roots_is_refused(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).read_range(str(tree["root"] / "runs" / "escape.txt"))
    assert caught.value.code == "path_symlink_escape"


@pytest.mark.parametrize("name", ["provider-secrets", "provider-secrets/vast_api_key"])
def test_hidden_paths_are_refused(tree: dict[str, Path], name: str) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).resolve(str(tree["root"] / name))
    assert caught.value.code == "path_hidden"


def test_symlink_into_a_hidden_path_is_refused(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).read_range(str(tree["root"] / "runs" / "sneaky.txt"))
    assert caught.value.code == "path_hidden"


def test_symlink_inside_the_roots_is_followed(tree: dict[str, Path]) -> None:
    data, meta = _view(tree).read_range(str(tree["root"] / "runs" / "latest" / "progression.json"))
    assert json.loads(data) == {"stage": "masks"}
    assert meta["size"] == len(data) and meta["eof"] is True


def test_read_range_pages_through_a_file(tree: dict[str, Path]) -> None:
    view = _view(tree, max_read_bytes=2048)
    path = str(tree["root"] / "runs" / "scene-1" / "big.log")
    first, meta = view.read_range(path)
    assert len(first) == 2048 and meta["eof"] is False and meta["size"] == 5000
    last, meta = view.read_range(path, offset=4096)
    assert len(last) == 904 and meta["eof"] is True


def test_negative_offsets_are_refused(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).read_range(str(tree["root"] / "runs" / "scene-1" / "big.log"), offset=-1)
    assert caught.value.code == "range_invalid"


def test_secret_names_are_refused_on_read(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).read_range(str(tree["root"] / "runs" / "scene-1" / "release.env"))
    assert caught.value.code == "secret_name_refused"


def test_secret_content_is_refused_on_read(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).read_range(str(tree["root"] / "runs" / "scene-1" / "leaky.json"))
    assert caught.value.code == "secret_content_refused:json_secret_field"


def test_directories_cannot_be_read_as_files(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).read_range(str(tree["root"] / "runs"))
    assert caught.value.code == "not_regular_file"


def test_listing_marks_secret_names_and_reports_symlinks(tree: dict[str, Path]) -> None:
    listing = _view(tree).list_dir(str(tree["root"] / "runs" / "scene-1"))
    entries = {entry["name"]: entry for entry in listing["entries"]}
    assert entries["progression.json"]["type"] == "file" and entries["progression.json"]["size"] > 0
    assert entries["release.env"] == {"name": "release.env", "type": "file", "refused": "secret_name"}
    assert listing["truncated"] is False
    runs = {entry["name"]: entry for entry in _view(tree).list_dir(str(tree["root"] / "runs"))["entries"]}
    assert runs["latest"]["type"] == "symlink" and runs["latest"]["target"].endswith("scene-1")


def test_listing_marks_hidden_children(tree: dict[str, Path]) -> None:
    entries = {e["name"]: e for e in _view(tree).list_dir(str(tree["root"]))["entries"]}
    assert entries["provider-secrets"] == {"name": "provider-secrets", "type": "dir", "refused": "hidden"}


def test_listing_caps_entries_filters_and_sorts(tree: dict[str, Path]) -> None:
    folder = tree["root"] / "many"
    folder.mkdir()
    for index in range(5):
        path = folder / f"item-{index}.json"
        path.write_text("{}", encoding="utf-8")
        os.utime(path, (1_000_000 + index, 1_000_000 + index))
    view = _view(tree, max_list_entries=3)
    listing = view.list_dir(str(folder), sort="mtime")
    assert [entry["name"] for entry in listing["entries"]] == ["item-4.json", "item-3.json", "item-2.json"]
    assert listing["truncated"] is True
    assert [e["name"] for e in view.list_dir(str(folder), match="item-1*")["entries"]] == ["item-1.json"]


def test_stat_of_a_file_describes_it(tree: dict[str, Path]) -> None:
    listing = _view(tree).list_dir(str(tree["root"] / "runs" / "scene-1" / "progression.json"))
    assert listing["type"] == "file" and listing["entries"] is None and listing["size"] > 0


def test_archive_skips_credentials_and_records_why(tree: dict[str, Path]) -> None:
    sink = io.BytesIO()
    manifest = _view(tree).stream_archive(str(tree["root"] / "runs"), sink.write)
    sink.seek(0)
    with tarfile.open(fileobj=sink, mode="r:gz") as archive:
        names = set(archive.getnames())
        manifest_member = json.load(archive.extractfile(".operator-door-manifest.json"))
    assert "scene-1/progression.json" in names and "scene-1/big.log" in names
    assert "scene-1/release.env" not in names and "scene-1/leaky.json" not in names
    assert not any(name.endswith(("escape.txt", "sneaky.txt", "latest")) for name in names)
    skipped = {item["path"]: item["reason"] for item in manifest["skipped"]}
    assert skipped["scene-1/release.env"] == "secret_name"
    assert skipped["scene-1/leaky.json"] == "secret_content:json_secret_field"
    assert skipped["escape.txt"] == "symlink"
    assert manifest_member == manifest


def test_archive_stops_at_the_byte_cap(tree: dict[str, Path]) -> None:
    sink = io.BytesIO()
    manifest = _view(tree, max_archive_bytes=100).stream_archive(str(tree["root"] / "runs"), sink.write)
    assert manifest["truncated"] is True
    sink.seek(0)
    with tarfile.open(fileobj=sink, mode="r:gz") as archive:
        assert "scene-1/big.log" not in archive.getnames()


def test_archive_of_a_hidden_directory_is_refused(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).stream_archive(str(tree["hidden"]), io.BytesIO().write)
    assert caught.value.code == "path_hidden"
