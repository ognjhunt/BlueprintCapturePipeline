"""File access through the door: confined to read roots, never a credential."""

# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/fsview.py
#   deploy/operator-door/operator_door/secrets_guard.py

from __future__ import annotations

import gzip
import io
import json
import os
import random
import string
import struct
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy" / "operator-door"))

from operator_door.config import DoorConfig  # noqa: E402
from operator_door.fsview import FileView, FsRefused  # noqa: E402
from operator_door.secrets_guard import scan_bytes  # noqa: E402


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


def _etc_view(tree: dict[str, Path]) -> tuple[FileView, Path]:
    etc = tree["root"] / "etc-blueprint"
    etc.mkdir()
    (etc / "task-evaluation-scene-progression.json").write_text('{"scene_root": "/x"}', encoding="utf-8")
    (etc / "c71-execute").write_text("K=1\n", encoding="utf-8")
    (etc / "notes.txt").write_text("hello\n", encoding="utf-8")
    view = FileView(DoorConfig(read_roots=(str(tree["root"]),), hidden_paths=(str(tree["hidden"]),),
                               json_only_roots=(str(etc),)))
    return view, etc


def test_json_only_roots_serve_only_json(tree: dict[str, Path]) -> None:
    view, etc = _etc_view(tree)
    data, _ = view.read_range(str(etc / "task-evaluation-scene-progression.json"))
    assert json.loads(data) == {"scene_root": "/x"}
    for name in ("c71-execute", "notes.txt"):
        with pytest.raises(FsRefused) as caught:
            view.read_range(str(etc / name))
        assert caught.value.code == "json_only_root"
    entries = {e["name"]: e for e in view.list_dir(str(etc))["entries"]}
    assert entries["notes.txt"] == {"name": "notes.txt", "type": "file", "refused": "json_only_root"}


def test_compressed_files_are_scanned_after_decompression(tree: dict[str, Path]) -> None:
    import zipfile

    folder = tree["root"] / "bundles"
    folder.mkdir()
    (folder / "clean.log.gz").write_bytes(gzip.compress(b"stage ok\n" * 1000))
    (folder / "leaky.log.gz").write_bytes(gzip.compress(b"OPENAI_API_KEY=sk-live-abcdefghijklmnopqrstu\n"))
    with zipfile.ZipFile(folder / "runtime.zip", "w") as bundle:
        bundle.writestr("runtime_output/result.json", '{"ok": true}')
    with zipfile.ZipFile(folder / "leaky.zip", "w") as bundle:
        bundle.writestr("config/release.env", "A=1\n")
    (folder / "blob.zst").write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 64)
    view = _view(tree)
    assert view.read_range(str(folder / "clean.log.gz"))[1]["size"] > 0
    assert view.read_range(str(folder / "runtime.zip"))[1]["size"] > 0
    for name, code in (("leaky.log.gz", "secret_content_refused:env_secret_assignment"),
                       ("leaky.zip", "secret_content_refused:zip_member_name"),
                       ("blob.zst", "secret_content_refused:compressed_unscannable:zstd")):
        with pytest.raises(FsRefused) as caught:
            view.read_range(str(folder / name))
        assert caught.value.code == code


def test_git_internals_are_refused(tree: dict[str, Path]) -> None:
    git = tree["root"] / "checkout" / ".git" / "objects"
    git.mkdir(parents=True)
    (git / "pack.idx").write_bytes(b"x")
    with pytest.raises(FsRefused) as caught:
        _view(tree).read_range(str(git / "pack.idx"))
    assert caught.value.code == "secret_name_refused"


def test_archive_refusals_happen_before_streaming(tree: dict[str, Path]) -> None:
    with pytest.raises(FsRefused) as caught:
        _view(tree).prepare_archive(str(tree["root"] / "runs" / "scene-1" / "progression.json"))
    assert caught.value.code == "not_a_directory"


# -- splats: numbers, not text --------------------------------------------------

_SPZ_MAGIC = 0x5053474E
# A real-looking Hugging Face token: hf_ and 34 letters.
_HF_TOKEN = b"hf_" + "".join(random.Random(7).choices(string.ascii_letters, k=34)).encode()
# The bytes the door matched by chance. atlas_depth_splat.encode_spz stores a colour channel c as
# round((c / 255 - 0.5) / SH_C0 * 0.15 * 255 + 127.5), so every channel below 117 is an ASCII digit or
# letter: a dark brown voxel RGB(83, 80, 66) is written "hf_", ten dark grey ones RGB(70, 75, 72) are
# "adb" each, and huggingface_token (hf_ and 30 letters) matches.
_CHANCE_COLOURS = b"\x8b\x8b\x8b" + b"hf_" + b"adb" * 10 + b"\x8b\x8b\x8b"


def _spz(count: int, body: bytes | None = None, *, plant: tuple[tuple[int, bytes], ...] = (),
         version: int = 2, extra: bytes = b"") -> bytes:
    """A gzip-wrapped SPZ v2 splat of SH degree 0; its body is random numbers unless ``body`` is given."""

    raw = bytearray(random.Random(count).randbytes(19 * count) if body is None else body)
    assert len(raw) == 19 * count
    for offset, data in plant:
        raw[offset:offset + len(data)] = data
    return gzip.compress(struct.pack("<IIIBBBB", _SPZ_MAGIC, version, count, 0, 12, 0, 0) + bytes(raw) + extra,
                         mtime=0)


def _depth_splat_spz() -> bytes:
    """Byte for byte what encode_spz writes for 13 voxels at the origin, of which voxel 1 is RGB(83, 80, 66)
    and voxels 2-11 are RGB(70, 75, 72): positions, alphas, colours, scales, rotations."""

    body = bytes(9 * 13) + b"\xe1" * 13 + _CHANCE_COLOURS + b"KK&" * 13 + b"\x80" * 39
    return _spz(13, body)


def _assert_refused(view: FileView, path: Path, code: str) -> None:
    with pytest.raises(FsRefused) as caught:
        view.read_range(str(path))
    assert caught.value.code == code


def test_a_depth_splat_whose_colours_spell_a_token_by_chance_is_served(tree: dict[str, Path]) -> None:
    """The refusal of splat/atlas_depth_splat.spz (secret_content_refused:huggingface_token), reproduced."""

    splat = tree["root"] / "job" / "splat" / "atlas_depth_splat.spz"
    splat.parent.mkdir(parents=True)
    splat.write_bytes(_depth_splat_spz())
    # The text rules match the decompressed body, as they did on the host...
    assert scan_bytes(gzip.decompress(splat.read_bytes())) == "huggingface_token"
    # ...but those bytes are numbers, and the door serves the file and archives it.
    view = _view(tree)
    data, info = view.read_range(str(splat))
    assert data == splat.read_bytes() and info["eof"] is True
    sink = io.BytesIO()
    manifest = view.stream_archive(str(tree["root"] / "job"), sink.write)
    assert manifest["skipped"] == [] and manifest["files_included"] == 1


def test_a_random_splat_with_token_shaped_numbers_is_served(tree: dict[str, Path]) -> None:
    count = 200_000  # 3.8 MB of body: several scan windows
    chance = b"\x8b" + _HF_TOKEN + b"\x8b"
    # Also across the first scan window's end, and exactly where the second window starts (judged by
    # the byte it really follows), and at the body's end.
    plant = ((5_000, chance), ((1 << 20) - 16 - 20, chance), ((1 << 20) - 16 - 4096 - 1, chance),
             (19 * count - len(chance), chance),
             (3 << 20, b"\x9c" + b"sk-" + b"a" * 30 + b"\x9c"), (2_000_000, b"\xa0AKIA" + b"B" * 16 + b"\xa0"))
    path = tree["root"] / "random.spz"
    path.write_bytes(_spz(count, plant=plant))
    assert scan_bytes(gzip.decompress(path.read_bytes())) == "openai_key"
    assert _view(tree).read_range(str(path), 0, 64)[1]["size"] == path.stat().st_size


@pytest.mark.parametrize(
    ("offset", "written", "code"),
    [
        (5_000, b"\x00" + _HF_TOKEN + b"\x00", "huggingface_token"),  # a C string
        (5_000, b'\x8b"' + _HF_TOKEN + b'"\x8b', "huggingface_token"),
        (5_000, b"\x8b " + _HF_TOKEN + b"\n\x8b", "huggingface_token"),
        (5_000, b"\x8b=" + _HF_TOKEN + b",\x8b", "huggingface_token"),
        (5_000, b"\x8b\nHF_TOKEN=" + _HF_TOKEN + b"\n", "env_secret_assignment"),
        (5_000, b'\x8b{"api_key": "abcdefghijklmnop12"}', "json_secret_field"),
        # Across the first scan window's end, and at the body's very end.
        ((1 << 20) - 16 - 20, b"\x00" + _HF_TOKEN + b"\x00", "huggingface_token"),
        (19 * 200_000 - len(_HF_TOKEN) - 1, b"\x00" + _HF_TOKEN, "huggingface_token"),
    ],
)
def test_a_token_written_into_a_splat_as_a_string_is_refused(
    tree: dict[str, Path], offset: int, written: bytes, code: str
) -> None:
    path = tree["root"] / "leaky.spz"
    path.write_bytes(_spz(200_000, plant=((offset, written),)))
    _assert_refused(_view(tree), path, f"secret_content_refused:{code}")


def test_a_splat_of_any_other_length_than_declared_is_scanned_as_text(tree: dict[str, Path]) -> None:
    body = bytes(9 * 13) + b"\xe1" * 13 + _CHANCE_COLOURS + b"KK&" * 13 + b"\x80" * 39
    view = _view(tree)
    for name, blob in (
        ("longer.spz", _spz(13, body, extra=b"\x00")),
        ("appended.spz", _depth_splat_spz() + gzip.compress(b"notes\n", mtime=0)),  # a second gzip member
        ("shorter.spz", gzip.compress(gzip.decompress(_depth_splat_spz())[:-1], mtime=0)),
        ("v3.spz", _spz(13, body, version=3)),  # a layout the pipeline does not write
    ):
        path = tree["root"] / name
        path.write_bytes(blob)
        _assert_refused(view, path, "secret_content_refused:huggingface_token")


def test_other_binary_and_text_files_keep_the_text_scan(tree: dict[str, Path]) -> None:
    folder = tree["root"] / "other"
    folder.mkdir()
    # The same chance bytes, not inside a splat: uncompressed, and in a gzip that is no splat.
    (folder / "colours.bin").write_bytes(b"\x00" * 64 + _CHANCE_COLOURS)
    (folder / "colours.bin.gz").write_bytes(gzip.compress(b"\x00" * 64 + _CHANCE_COLOURS))
    # A pickled string: a length byte before it and an opcode after, as in a gzip-compressed pickle.
    (folder / "args.pkl.gz").write_bytes(gzip.compress(b"\x80\x04\x95\x30\x00\x8c\x25" + _HF_TOKEN + b"\x94."))
    (folder / "notes.txt.gz").write_bytes(gzip.compress(b"login with " + _HF_TOKEN + b"\n"))
    ply = (b"ply\nformat binary_little_endian 1.0\ncomment exported with " + _HF_TOKEN + b"\n"
           b"element vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
           + struct.pack("<6f", 0.5, 1.5, 2.5, 3.5, 4.5, 5.5))
    (folder / "points.ply").write_bytes(ply)
    (folder / "points.ply.gz").write_bytes(gzip.compress(ply))
    view = _view(tree)
    for name in ("colours.bin", "colours.bin.gz", "args.pkl.gz", "notes.txt.gz", "points.ply", "points.ply.gz"):
        _assert_refused(view, folder / name, "secret_content_refused:huggingface_token")
    (folder / "clean.log.gz").write_bytes(gzip.compress(b"stage ok\n" * 1000))
    assert view.read_range(str(folder / "clean.log.gz"))[1]["size"] > 0
