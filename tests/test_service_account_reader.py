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
    argv += ["--readback-user", "blueprint", "--website-mapanything-template", str(tmp_path / "none")]
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
