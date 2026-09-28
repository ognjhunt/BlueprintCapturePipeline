# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_output_archive.py
"""ADP-009D/day-28, plan 14 PR 1: only new bytes are archived; landing is atomic and host-linked."""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import remote_cpu_output_archive as archive

ADAPTER = "native-arena-adapter"
SELECT = ("native-arena-adapter/**", "rigid_destination_native_probe_request.v1.json")


def _digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes, mode: int = 0o440) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    os.chmod(path, mode)
    return path


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.lstat(path).st_mode)


def _tree(root: Path, *, reverse: bool = False) -> Path:
    files = [
        (f"{ADAPTER}/result.json", b'{"adapter":"result"}\n', 0o440),
        (f"{ADAPTER}/runtime/receipt.json", b'{"runtime":"receipt"}\n', 0o440),
        (f"{ADAPTER}/runtime/duplicate.json", b'{"adapter":"result"}\n', 0o440),
        ("rigid_destination_native_probe_request.v1.json", b'{"probe":1}\n', 0o440),
        ("configured-scene/scene.usda", b"#usda 1.0\n" * 64, 0o440),
        ("native-task-arena-bundle.zip", b"PK-not-landed" * 32, 0o440),
    ]
    for relative, data, mode in reversed(files) if reverse else files:
        _write(root / relative, data, mode)
    for directory in (root / ADAPTER / "runtime", root / ADAPTER, root / "configured-scene"):
        os.chmod(directory, 0o750)
    os.chmod(root, 0o750)
    return root


def _sealed(root: Path, host_known: dict | None = None) -> tuple[dict, bytes]:
    index = archive.index_tree(root, host_known=host_known or {})
    stream = io.BytesIO()
    written = archive.write_blobs_tar(root, index, stream)
    data = stream.getvalue()
    assert written == {"digest": _digest(data), "size_bytes": len(data)}
    return index, data


class RangeReader:
    def __init__(self, data: bytes, *, fail_after: int | None = None) -> None:
        self.data, self.fail_after, self.calls = data, fail_after, []

    def __call__(self, offset: int, length: int) -> io.BytesIO:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise OSError("range read interrupted")
        self.calls.append((offset, length))
        return io.BytesIO(self.data[offset:offset + length])


def _land(index: dict, reader, destination: Path, **changes) -> dict:
    arguments = {
        "index": index,
        "reader": reader,
        "host_sources": {},
        "destination_root": destination,
        "selectors": SELECT,
        "member_store": None,
    }
    arguments.update(changes)
    return archive.land_subset(**arguments)


def _reason(call) -> str:
    with pytest.raises(archive.RemoteCpuArchiveError) as caught:
        call()
    return str(caught.value)


def _files(root: Path) -> dict[str, tuple[str, int]]:
    return {
        path.relative_to(root).as_posix(): (_digest(path.read_bytes()), _mode(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_archive_is_deterministic_across_two_passes(tmp_path: Path) -> None:
    first_root = _tree(tmp_path / "first")
    first_index, first_tar = _sealed(first_root)
    for path in first_root.rglob("*"):
        os.utime(path, (1_000_000 + len(path.name), 2_000_000 + len(path.name)), follow_symlinks=False)
    again_index, again_tar = _sealed(first_root)
    second_index, second_tar = _sealed(_tree(tmp_path / "second", reverse=True))

    assert first_index == again_index == second_index
    assert first_tar == again_tar == second_tar
    assert archive.blobs_tar_size(first_index) == len(first_tar) == first_index["archive_size_bytes"]
    assert first_index["index_digest"].startswith("sha256:")
    assert archive.index_bytes(first_index) == (
        json.dumps(first_index, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    assert [entry["path"] for entry in first_index["entries"]] == sorted(entry["path"] for entry in first_index["entries"])
    assert first_index["paths_total"] == 6
    assert first_index["bytes_total"] == sum(len((first_root / e["path"]).read_bytes()) for e in first_index["entries"])
    assert first_index["root_mode"] == "0750"
    assert {row["path"]: row["mode"] for row in first_index["directories"]} == {
        ADAPTER: "0750", f"{ADAPTER}/runtime": "0750", "configured-scene": "0750",
    }

    blobs = [row["blob"] for row in first_index["blobs"]]
    assert blobs == sorted(blobs) and len(blobs) == 5
    with tarfile.open(fileobj=io.BytesIO(first_tar)) as members:
        infos = members.getmembers()
        assert [info.name for info in infos] == [f"sha256/{blob[7:]}" for blob in blobs]
        assert {(info.mtime, info.uid, info.gid, info.uname, info.gname, info.mode) for info in infos} == {
            (0, 0, 0, "", "", 0o444)
        }
        for info, row in zip(infos, first_index["blobs"]):
            assert info.offset_data == row["offset"] and info.size == row["size_bytes"]
            assert _digest(members.extractfile(info).read()) == row["blob"]

    os.chmod(first_root / ADAPTER / "result.json", 0o4640)
    assert _reason(lambda: archive.index_tree(first_root, host_known={})) == (
        "remote_cpu_archive_mode_invalid:native-arena-adapter/result.json"
    )
    os.chmod(first_root / ADAPTER / "result.json", 0o440)
    (first_root / ADAPTER / "link.json").symlink_to("result.json")
    assert _reason(lambda: archive.index_tree(first_root, host_known={})).startswith(
        "remote_cpu_archive_symlink_refused:native-arena-adapter/link.json"
    )


def test_host_known_blobs_are_indexed_by_origin_and_never_archived(tmp_path: Path) -> None:
    root = _tree(tmp_path / "worker")
    scene = b"INPUT-SCENE-BYTES" * 100
    member = b"RUNTIME-MEMBER-BYTES" * 50
    _write(root / ADAPTER / "scene.usd", scene)
    _write(root / "configured-scene" / "copy.usd", scene, 0o640)
    _write(root / ADAPTER / "runtime" / "member.bin", member)
    zip_digest = "sha256:" + "2" * 64
    host_known = {
        _digest(scene): {"input": _digest(scene)},
        _digest(member): {"input_member": {"input": zip_digest, "member": "members/member.bin"}},
    }

    index, data = _sealed(root, host_known)

    origins = {entry["path"]: entry["origin"] for entry in index["entries"]}
    assert origins[f"{ADAPTER}/scene.usd"] == {"input": _digest(scene)}
    assert origins["configured-scene/copy.usd"] == {"input": _digest(scene)}
    assert origins[f"{ADAPTER}/runtime/member.bin"] == {
        "input_member": {"input": zip_digest, "member": "members/member.bin"}
    }
    assert origins[f"{ADAPTER}/result.json"] == "archive"
    archived = {row["blob"] for row in index["blobs"]}
    assert _digest(scene) not in archived and _digest(member) not in archived
    assert index["host_known"] == {"count": 3, "bytes": 2 * len(scene) + len(member)}
    assert b"INPUT-SCENE-BYTES" not in data and b"RUNTIME-MEMBER-BYTES" not in data
    assert b'{"adapter":"result"}' in data
    assert data.count(b'{"adapter":"result"}') == 1

    for bad in (
        {_digest(scene): {"input": "sha256:" + "3" * 64}},
        {_digest(member): {"input_member": {"input": zip_digest, "member": "../escape.bin"}}},
        {_digest(member): {"input": _digest(member), "extra": True}},
        {"https://b2.example.test/k?X-Amz-Signature=SECRETSIG": {"input": _digest(scene)}},
    ):
        refusal = _reason(lambda: archive.index_tree(root, host_known=bad))
        assert refusal.startswith("remote_cpu_archive_host_known_invalid") and "SECRETSIG" not in refusal


def test_stream_verifier_refuses_bad_framing_unindexed_blobs_and_digest_mismatch(tmp_path: Path) -> None:
    root = _tree(tmp_path / "worker")
    index, data = _sealed(root)

    verdict = archive.verify_blobs_stream(io.BytesIO(data), index, expected_digest=_digest(data))
    assert verdict == {"digest": _digest(data), "size_bytes": len(data), "blob_count": 5}

    def refused(payload: bytes, *, expected: str | None = None, against: dict | None = None) -> str:
        return _reason(lambda: archive.verify_blobs_stream(
            io.BytesIO(payload), against or index, expected_digest=expected or _digest(payload)
        ))

    first = index["blobs"][0]
    header = bytearray(data)
    header[first["offset"] - 512 + 108] ^= 0x01
    assert refused(bytes(header)).startswith("remote_cpu_archive_framing_invalid")
    body = bytearray(data)
    body[first["offset"]] ^= 0x01
    assert refused(bytes(body)) == f"remote_cpu_archive_blob_digest_mismatch:{first['blob']}"
    padded = bytearray(data)
    padded[first["offset"] + first["size_bytes"]] = 0x01
    assert refused(bytes(padded)).startswith("remote_cpu_archive_framing_invalid")
    assert refused(data, expected="sha256:" + "0" * 64) == "remote_cpu_archive_digest_mismatch"
    assert refused(data[:-700]) == "remote_cpu_archive_truncated"
    assert refused(data + b"\0" * 512) == "remote_cpu_archive_trailing_bytes"

    _write(root / ADAPTER / "extra.json", b'{"unindexed":true}\n')
    extra_index, extra_data = _sealed(root)
    assert refused(extra_data).startswith("remote_cpu_archive_unindexed_blob:sha256/")
    missing = sorted(
        {row["blob"] for row in extra_index["blobs"]} - {row["blob"] for row in index["blobs"]}
    )[0]
    assert refused(data, against=extra_index).startswith(
        ("remote_cpu_archive_blob_missing:", "remote_cpu_archive_unindexed_blob:")
    )
    assert missing.startswith("sha256:")
    unsealed = dict(index, paths_total=index["paths_total"] + 1)
    assert refused(data, against=unsealed).startswith("remote_cpu_archive_index_invalid")
    assert refused(data, against=dict(index, root_mode="0\ud800")) == "remote_cpu_archive_index_invalid:not_json"


def test_landing_assembles_in_a_temporary_directory_and_renames_once_complete(tmp_path: Path, monkeypatch) -> None:
    index, data = _sealed(_tree(tmp_path / "worker"))
    parent = tmp_path / "compiled-episodes"
    parent.mkdir()
    destination = parent / "prep-1"

    assert _reason(lambda: _land(index, RangeReader(data, fail_after=1), destination)).startswith(
        "remote_cpu_landing_source_unavailable"
    )
    assert not destination.exists()
    [landing] = [path for path in parent.iterdir()]
    assert landing.name.startswith(".prep-1.landing-") and landing.is_dir()
    # result.json was read; duplicate.json shares its (blob, mode) and is a link, not a read.
    assert sorted(_files(landing)) == [f"{ADAPTER}/result.json", f"{ADAPTER}/runtime/duplicate.json"]

    renamed = archive._rename_no_replace
    observed = {}

    def watch(source: Path, target: Path) -> None:
        observed["destination_existed"] = target.exists()
        observed["files_before_rename"] = sorted(_files(source))
        renamed(source, target)

    monkeypatch.setattr(archive, "_rename_no_replace", watch)
    reader = RangeReader(data)
    summary = _land(index, reader, destination)

    assert observed["destination_existed"] is False
    assert observed["files_before_rename"] == sorted(_files(destination))
    assert summary["state"] == "landed" and summary["resumed_paths"] == 2
    assert sorted(path.name for path in parent.iterdir()) == ["prep-1"]
    assert sorted(_files(destination)) == [
        f"{ADAPTER}/result.json",
        f"{ADAPTER}/runtime/duplicate.json",
        f"{ADAPTER}/runtime/receipt.json",
        "rigid_destination_native_probe_request.v1.json",
    ]
    assert summary["paths"] == 4
    assert summary["bytes"] == sum(len(path.read_bytes()) for path in destination.rglob("*") if path.is_file())
    assert _land(index, RangeReader(data), destination)["state"] == "already_landed"
    assert _reason(lambda: _land(index, RangeReader(data), destination, selectors=["configured-scene/*.usda"])) == (
        "remote_cpu_landing_selector_invalid"
    )


def test_landing_resume_accepts_identical_files_and_refuses_different_ones(tmp_path: Path) -> None:
    index, data = _sealed(_tree(tmp_path / "worker"))

    def interrupted(parent: Path) -> tuple[Path, Path]:
        parent.mkdir()
        destination = parent / "prep-1"
        with pytest.raises(archive.RemoteCpuArchiveError):
            _land(index, RangeReader(data, fail_after=2), destination)
        [landing] = list(parent.iterdir())
        return destination, landing

    destination, _ = interrupted(tmp_path / "resume")
    summary = _land(index, RangeReader(data), destination)
    assert summary["state"] == "landed" and summary["resumed_paths"] == 3

    destination, landing = interrupted(tmp_path / "tampered")
    landed = sorted(path for path in landing.rglob("*") if path.is_file())[0]
    keep = tmp_path / "old-inode"
    os.link(landed, keep)
    landed.unlink()
    _write(landed, b"not the indexed bytes\n")
    relative = landed.relative_to(landing).as_posix()
    assert _reason(lambda: _land(index, RangeReader(data), destination)) == f"remote_cpu_landing_conflict:{relative}"
    assert not destination.exists()

    complete = tmp_path / "complete"
    complete.mkdir()
    final = complete / "prep-1"
    _land(index, RangeReader(data), final)
    assert _land(index, RangeReader(data), final)["state"] == "already_landed"
    probe = final / "rigid_destination_native_probe_request.v1.json"
    keep_final = tmp_path / "old-final-inode"
    os.link(probe, keep_final)
    probe.unlink()
    _write(probe, b'{"probe":2}\n')
    assert _reason(lambda: _land(index, RangeReader(data), final)) == (
        "remote_cpu_landing_conflict:rigid_destination_native_probe_request.v1.json"
    )
    os.unlink(probe)
    os.link(keep_final, probe)
    _write(final / ADAPTER / "stray.json", b"{}\n")
    assert _reason(lambda: _land(index, RangeReader(data), final)) == (
        "remote_cpu_landing_conflict:native-arena-adapter/stray.json"
    )


def test_landing_refuses_a_planted_directory_symlink_before_writing_through_it(tmp_path: Path) -> None:
    index, data = _sealed(_tree(tmp_path / "worker"))
    parent = tmp_path / "compiled-episodes"
    parent.mkdir()
    destination = parent / "prep-1"
    with pytest.raises(archive.RemoteCpuArchiveError):
        _land(index, RangeReader(data, fail_after=0), destination)
    [landing] = list(parent.iterdir())
    outside = tmp_path / "outside"
    outside.mkdir()
    for planted in (ADAPTER, f"{ADAPTER}/runtime"):
        shutil.rmtree(landing / ADAPTER, ignore_errors=True)
        if planted != ADAPTER:
            (landing / ADAPTER).mkdir(mode=0o700)
        target = landing / planted
        target.symlink_to(outside, target_is_directory=True)
        assert _reason(lambda: _land(index, RangeReader(data), destination)) == (
            f"remote_cpu_landing_conflict:{planted}"
        )
        assert list(outside.iterdir()) == [] and not destination.exists()
        target.unlink()


def test_landing_resolves_input_and_input_member_origins_from_host_copies(tmp_path: Path) -> None:
    scene, member, cached = b"SCENE" * 400, b"MEMBER" * 300, b"CACHED" * 200
    host = tmp_path / "prepared-references" / "prep-1"
    host_scene = _write(host / "scene.usd", scene)
    runtime_zip = host / "runtime.zip"
    with zipfile.ZipFile(runtime_zip, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("members/member.bin", member)
        bundle.writestr("members/cached.bin", cached)
    os.chmod(runtime_zip, 0o440)
    store = tmp_path / "compiled-episodes" / "content-addressed" / "adapter-members" / "sha256"
    store_file = _write(store / hashlib.sha256(cached).hexdigest(), cached)
    zip_digest = _digest(runtime_zip.read_bytes())

    root = _tree(tmp_path / "worker")
    _write(root / ADAPTER / "scene.usd", scene)
    _write(root / ADAPTER / "runtime" / "member.bin", member)
    _write(root / ADAPTER / "runtime" / "cached.bin", cached)
    host_known = {
        _digest(scene): {"input": _digest(scene)},
        _digest(member): {"input_member": {"input": zip_digest, "member": "members/member.bin"}},
        _digest(cached): {"input_member": {"input": zip_digest, "member": "members/cached.bin"}},
    }
    index, data = _sealed(root, host_known)
    reader = RangeReader(data)
    destination = tmp_path / "compiled-episodes" / "prep-1"
    host_sources = {_digest(scene): host_scene, zip_digest: runtime_zip}

    summary = _land(index, reader, destination, host_sources=host_sources, member_store=store)

    assert summary["state"] == "landed"
    assert os.stat(destination / ADAPTER / "scene.usd").st_ino == os.stat(host_scene).st_ino
    assert os.stat(destination / ADAPTER / "runtime" / "cached.bin").st_ino == os.stat(store_file).st_ino
    extracted = destination / ADAPTER / "runtime" / "member.bin"
    assert extracted.read_bytes() == member and _mode(extracted) == 0o440
    assert os.stat(extracted).st_ino not in {os.stat(runtime_zip).st_ino, os.stat(store_file).st_ino}
    archived = {(row["offset"], row["size_bytes"]) for row in index["blobs"]}
    assert set(reader.calls) <= archived
    assert {length for _, length in reader.calls}.isdisjoint({len(scene), len(member), len(cached)})
    assert not (destination / "configured-scene").exists()
    assert _files(destination)[f"{ADAPTER}/scene.usd"] == (_digest(scene), 0o440)

    corrupt = tmp_path / "corrupt" / "scene.usd"
    _write(corrupt, b"SCENF" * 400)
    other = tmp_path / "other"
    other.mkdir()
    assert _reason(lambda: _land(
        index, RangeReader(data), other / "prep-1", host_sources={**host_sources, _digest(scene): corrupt},
        member_store=store,
    )) == f"remote_cpu_landing_host_source_invalid:{_digest(scene)}"
    assert _reason(lambda: _land(
        index, RangeReader(data), tmp_path / "missing" / "prep-1", host_sources={}, member_store=store,
    )).startswith("remote_cpu_landing_destination_parent_invalid")


def test_landing_preserves_modes_and_shares_one_inode_per_blob_and_mode(tmp_path: Path) -> None:
    root = tmp_path / "worker"
    same = b"shared bytes\n"
    _write(root / "x" / "a.bin", same, 0o440)
    _write(root / "x" / "b.bin", same, 0o440)
    _write(root / "x" / "y" / "c.bin", same, 0o640)
    _write(root / "x" / "y" / "d.bin", b"other\n", 0o400)
    host_bytes = b"HOST INPUT BYTES\n" * 10
    _write(root / "x" / "host.bin", host_bytes, 0o640)
    os.chmod(root / "x" / "y", 0o700)
    os.chmod(root / "x", 0o750)
    os.chmod(root, 0o750)
    host_copy = _write(tmp_path / "prepared-references" / "host.bin", host_bytes, 0o440)
    index, data = _sealed(root, {_digest(host_bytes): {"input": _digest(host_bytes)}})
    parent = tmp_path / "compiled-episodes"
    parent.mkdir()
    destination = parent / "prep-1"

    _land(index, RangeReader(data), destination, selectors=["**"],
          host_sources={_digest(host_bytes): host_copy})

    inode = {name: os.stat(destination / "x" / name).st_ino for name in ("a.bin", "b.bin", "host.bin")}
    assert inode["a.bin"] == inode["b.bin"]
    assert os.stat(destination / "x" / "y" / "c.bin").st_ino != inode["a.bin"]
    assert _files(destination) == {
        "x/a.bin": (_digest(same), 0o440),
        "x/b.bin": (_digest(same), 0o440),
        "x/host.bin": (_digest(host_bytes), 0o640),
        "x/y/c.bin": (_digest(same), 0o640),
        "x/y/d.bin": (_digest(b"other\n"), 0o400),
    }
    assert (_mode(destination), _mode(destination / "x"), _mode(destination / "x" / "y")) == (0o750, 0o750, 0o700)
    assert inode["host.bin"] != os.stat(host_copy).st_ino
    assert _mode(host_copy) == 0o440 and os.stat(host_copy).st_nlink == 1
