"""One service-account reader proves every published byte without a process per file."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import blueprint_pipeline.service_account_reader as reader_module
from blueprint_pipeline.service_account_reader import ServiceAccountReader
from scripts import provision_task_evaluation_scene_configuration_release as provisioner


def _reader() -> ServiceAccountReader:
    # Tests run the reader as the current user; production prefixes sudo -u.
    return ServiceAccountReader("blueprint", command_prefix=())


def test_one_reader_process_returns_every_exact_byte(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spawned: list[list[str]] = []
    original = subprocess.Popen

    def counting_popen(argv, *args, **kwargs):
        spawned.append(list(argv))
        return original(argv, *args, **kwargs)

    monkeypatch.setattr(reader_module.subprocess, "Popen", counting_popen)
    payloads = {
        tmp_path / "empty.bin": b"",
        tmp_path / "small.txt": b"renderer source\n",
        tmp_path / "large.bin": bytes(range(256)) * 12_289,  # just over 3 MiB
        tmp_path / "name with spaces and ünïcode.json": b'{"ok": true}',
    }
    for path, body in payloads.items():
        path.write_bytes(body)

    with _reader() as read:
        for path, body in payloads.items():
            assert read(path) == body
        assert read(tmp_path / "small.txt") == payloads[tmp_path / "small.txt"]

    assert len(spawned) == 1


def test_an_unreadable_path_fails_like_the_per_file_reader(tmp_path: Path) -> None:
    present = tmp_path / "present.txt"
    present.write_bytes(b"still served after a refusal")

    with _reader() as read:
        with pytest.raises(subprocess.CalledProcessError):
            read(tmp_path / "missing.txt")
        assert read(present) == b"still served after a refusal"


def test_production_reader_runs_as_the_service_account_with_an_isolated_interpreter() -> None:
    argv = ServiceAccountReader.command("blueprint", python="/venv/bin/python")

    assert argv[:5] == ["sudo", "-n", "-u", "blueprint", "--"]
    assert argv[5:8] == ["/venv/bin/python", "-I", "-S"]


def test_closed_reader_refuses_further_reads(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_bytes(b"x")
    read = _reader()
    read.close()

    with pytest.raises(ValueError, match="service_account_reader_closed"):
        read(path)


def test_provisioner_publishes_with_one_reader_and_closes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_provision(**kwargs):
        read = kwargs["readback"]
        observed["reader"] = read
        observed["actor"] = kwargs["readback_actor"]
        return {"status": "ready"}

    monkeypatch.setattr(provisioner, "provision_scene_configuration_release", fake_provision)
    monkeypatch.setattr(
        provisioner,
        "service_account_readback",
        lambda user: ServiceAccountReader(user, command_prefix=()),
    )
    argv = ["provision"]
    for name in (
        "repository-root", "source-commit", "node-executable", "browser-root",
        "browser-executable", "node-modules-root", "artifixer-root", "content-agents-root",
        "text-to-cad-root", "multi-agent-cad-root",
    ):
        argv += [f"--{name}", str(tmp_path)]
    argv += ["--readback-user", "blueprint"]
    monkeypatch.setattr("sys.argv", argv)

    assert provisioner.main() == 0

    reader = observed["reader"]
    assert isinstance(reader, ServiceAccountReader)
    assert observed["actor"] == "service-account:blueprint"
    assert reader.closed


def test_a_reader_that_cannot_start_reports_why(tmp_path: Path) -> None:
    """A refused sudo must surface its own message, as the per-file dd did."""
    path = tmp_path / "file.txt"
    path.write_bytes(b"x")
    refusing = ("sh", "-c", "echo 'sudo: a password is required' >&2; exit 1", "sh")
    read = ServiceAccountReader("blueprint", command_prefix=refusing)

    with pytest.raises(subprocess.CalledProcessError) as raised:
        read(path)

    assert b"a password is required" in raised.value.stderr
    assert read.closed


def test_a_file_larger_than_one_chunk_streams_exact_bytes_and_digest(tmp_path: Path) -> None:
    """2026-10-13 ARTISAN deploy: whole-file reads of a 1.19 GB library failed the readback."""
    import hashlib

    body = bytes(range(256)) * (reader_module.CHUNK_BYTES // 256 * 2 + 4099)  # more than two chunks
    path = tmp_path / "libtorch_like.so"
    path.write_bytes(body)
    with _reader() as read:
        assert read.digest(path) == (len(body), "sha256:" + hashlib.sha256(body).hexdigest())
        assert read(path) == body
        assert read.digest(tmp_path / "libtorch_like.so")[0] == len(body)


def test_digest_of_an_unreadable_path_fails_and_the_reader_keeps_serving(tmp_path: Path) -> None:
    present = tmp_path / "present.bin"
    present.write_bytes(b"served after a refused digest")
    with _reader() as read:
        with pytest.raises(subprocess.CalledProcessError):
            read.digest(tmp_path / "missing.bin")
        assert read.digest(present)[0] == len(b"served after a refused digest")


def test_identity_opens_as_the_account_without_reading_and_names_the_inode(tmp_path: Path) -> None:
    """An inode an earlier readback read in full is opened at its new path, not read again."""
    import os

    from blueprint_pipeline.immutable_file_digest import sealed_identity

    path = tmp_path / "libtorch_like.so"
    path.write_bytes(b"\0" * (reader_module.CHUNK_BYTES + 1))
    os.link(path, tmp_path / "next-release.so")
    with _reader() as read:
        assert read.identity(tmp_path / "next-release.so") == sealed_identity(os.stat(path))
        with pytest.raises(subprocess.CalledProcessError):
            read.identity(tmp_path / "missing.so")
        assert read.digest(path)[0] == reader_module.CHUNK_BYTES + 1
        assert read.identity(path) == sealed_identity(os.stat(path))
