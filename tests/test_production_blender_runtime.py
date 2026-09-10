"""Portable Blender admission tests use tiny fake archives and no network/processes."""
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from blueprint_pipeline import production_blender_runtime as runtime


@pytest.fixture(autouse=True)
def linux_and_disk(monkeypatch):
    monkeypatch.setattr(runtime.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=20 * 1024**3))


def archive(tmp_path, monkeypatch, extras=()):
    path = tmp_path / "fake.tar.xz"
    with tarfile.open(path, "w:xz") as bundle:
        info = tarfile.TarInfo(runtime.EXECUTABLE_RELATIVE)
        body = b"#!/bin/sh\n# hermetic fixture; never executed\n"
        info.size, info.mode = len(body), 0o755
        bundle.addfile(info, io.BytesIO(body))
        for name, target, kind in extras:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.SYMTYPE or kind == tarfile.LNKTYPE:
                info.linkname = target
                bundle.addfile(info)
            else:
                info.size = 1
                bundle.addfile(info, io.BytesIO(b"x"))
    monkeypatch.setattr(runtime, "ARCHIVE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest())
    return lambda url, destination: shutil.copyfile(path, destination)


def runner(args, **kwargs):
    assert args[1:] == ["--background", "--factory-startup", "--version"]
    assert kwargs["timeout"] == 30
    return subprocess.CompletedProcess(args, 0, stdout="Blender 5.2.1\n  build hash: fake\n", stderr="")


def test_explicit_install_and_validation_receipt(tmp_path, monkeypatch):
    downloader = archive(tmp_path, monkeypatch)
    root = tmp_path / "installed"
    result = runtime.install_runtime(root, downloader=downloader, runner=runner)
    assert result["version"] == "5.2.1"
    assert Path(result["executable"]).is_file()
    receipt = json.loads((root / runtime.RECEIPT_NAME).read_text())
    assert receipt["archive_sha256"] == runtime.ARCHIVE_SHA256
    assert receipt["executable_sha256"] == hashlib.sha256(Path(result["executable"]).read_bytes()).hexdigest()
    assert runtime.validate_runtime(root, runner=runner) == result
    def no_download(*_):
        pytest.fail("existing installation must not redownload")
    assert runtime.install_runtime(root, downloader=no_download, runner=runner) == result


def test_mutated_binary_is_never_executed_or_repaired(tmp_path, monkeypatch):
    downloader = archive(tmp_path, monkeypatch)
    root = tmp_path / "installed"
    result = runtime.install_runtime(root, downloader=downloader, runner=runner)
    Path(result["executable"]).write_text("mutated")
    def forbidden(*_, **__):
        pytest.fail("must refuse before downloading or running mutated binary")
    with pytest.raises(runtime.BlenderRuntimeError, match="digest_mismatch"):
        runtime.install_runtime(root, downloader=forbidden, runner=forbidden)


def test_wrong_archive_digest_stops_before_extract_or_execute(tmp_path, monkeypatch):
    downloader = archive(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, "ARCHIVE_SHA256", "0" * 64)
    root = tmp_path / "installed"
    with pytest.raises(runtime.BlenderRuntimeError, match="archive_digest_mismatch"):
        runtime.install_runtime(root, downloader=downloader, runner=runner)
    assert not root.exists()
    assert not list(tmp_path.glob(".blender-install-*"))


@pytest.mark.parametrize("name,target,kind", [
    ("../escaped", "", tarfile.REGTYPE),
    ("/tmp/escaped", "", tarfile.REGTYPE),
    ("unsafe-link", "../escaped", tarfile.SYMTYPE),
    ("unsafe-link", "/tmp/escaped", tarfile.SYMTYPE),
    ("hardlink", runtime.EXECUTABLE_RELATIVE, tarfile.LNKTYPE),
    ("device", "", tarfile.CHRTYPE),
])
def test_archive_escape_and_special_members_refused(tmp_path, monkeypatch, name, target, kind):
    downloader = archive(tmp_path, monkeypatch, [(name, target, kind)])
    root = tmp_path / "installed"
    with pytest.raises(runtime.BlenderRuntimeError, match="archive_"):
        runtime.install_runtime(root, downloader=downloader, runner=runner)
    assert not root.exists()
    assert not (tmp_path / "escaped").exists()


def test_internal_relative_symlink_is_preserved(tmp_path, monkeypatch):
    folder = str(Path(runtime.EXECUTABLE_RELATIVE).parent)
    downloader = archive(tmp_path, monkeypatch, [(f"{folder}/blender-link", "blender", tarfile.SYMTYPE)])
    root = tmp_path / "installed"
    runtime.install_runtime(root, downloader=downloader, runner=runner)
    assert (root / folder / "blender-link").resolve() == root / runtime.EXECUTABLE_RELATIVE


def test_disk_admission_precedes_download(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.shutil, "disk_usage", lambda _: SimpleNamespace(free=100))
    def forbidden(*_):
        pytest.fail("no download before disk admission")
    with pytest.raises(runtime.BlenderRuntimeError, match="disk_headroom"):
        runtime.install_runtime(tmp_path / "installed", downloader=forbidden, runner=runner)


def test_archive_and_expansion_size_bounds(tmp_path, monkeypatch):
    downloader = archive(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, "MAX_EXTRACTED_BYTES", 1)
    with pytest.raises(runtime.BlenderRuntimeError, match="extracted_size_limit"):
        runtime.install_runtime(tmp_path / "installed", downloader=downloader, runner=runner)
    monkeypatch.setattr(runtime, "MAX_ARCHIVE_BYTES", 1)
    with pytest.raises(runtime.BlenderRuntimeError, match="download_size_or_path_invalid"):
        runtime.install_runtime(tmp_path / "installed", downloader=downloader, runner=runner)


def test_version_mismatch_leaves_no_installation(tmp_path, monkeypatch):
    downloader = archive(tmp_path, monkeypatch)
    def wrong_runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="Blender 4.0.0\n", stderr="")
    with pytest.raises(runtime.BlenderRuntimeError, match="version_mismatch"):
        runtime.install_runtime(tmp_path / "installed", downloader=downloader, runner=wrong_runner)
    assert not (tmp_path / "installed").exists()
    retained = list(tmp_path.glob(".blender-failed-*"))
    assert len(retained) == 1
    assert (retained[0] / "runtime" / runtime.EXECUTABLE_RELATIVE).is_file()
    assert hashlib.sha256((retained[0] / runtime.ARCHIVE_NAME).read_bytes()).hexdigest() == runtime.ARCHIVE_SHA256
    assert "version_mismatch" in json.loads((retained[0] / "failure.json").read_text())["error"]


def test_validate_only_cli_and_linux_x64_requirement(tmp_path, monkeypatch, capsys):
    def forbidden(*_, **__):
        pytest.fail("CLI validation must not install")
    monkeypatch.setattr(runtime, "install_runtime", forbidden)
    with pytest.raises(SystemExit) as exc:
        runtime.main(["--root", str(tmp_path / "absent")])
    assert exc.value.code == 1
    assert "runtime_receipt_missing" in capsys.readouterr().err
    monkeypatch.setattr(runtime.platform, "machine", lambda: "aarch64")
    with pytest.raises(runtime.BlenderRuntimeError, match="requires_linux_x64"):
        runtime.validate_runtime(tmp_path)


def test_changed_receipt_archive_identity_refused(tmp_path, monkeypatch):
    root = tmp_path / "installed"
    runtime.install_runtime(root, downloader=archive(tmp_path, monkeypatch), runner=runner)
    path = root / runtime.RECEIPT_NAME
    data = json.loads(path.read_text())
    data["archive_sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(runtime.BlenderRuntimeError, match="receipt_identity_mismatch"):
        runtime.validate_runtime(root, runner=runner)


def test_download_enforces_stream_size_and_elapsed_time(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        headers = {}
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **k: Response(b"12345"))
    monkeypatch.setattr(runtime, "MAX_ARCHIVE_BYTES", 4)
    with pytest.raises(runtime.BlenderRuntimeError, match="download_size_limit"):
        runtime._download_archive("https://fixture.invalid", tmp_path / "size.xz")
    ticks = iter([0, runtime.DOWNLOAD_TIMEOUT_SECONDS + 1])
    monkeypatch.setattr(runtime.time, "monotonic", lambda: next(ticks))
    with pytest.raises(runtime.BlenderRuntimeError, match="download_time_limit"):
        runtime._download_archive("https://fixture.invalid", tmp_path / "time.xz")


def test_download_identifies_installer_to_official_server(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        headers = {"Content-Length": "3"}
    def urlopen(request, *, timeout):
        assert request.full_url == runtime.ARCHIVE_URL
        assert request.get_header("User-agent").startswith("BlueprintCapturePipeline/")
        assert timeout == 30
        return Response(b"abc")
    monkeypatch.setattr(runtime.urllib.request, "urlopen", urlopen)
    target = tmp_path / "archive.xz"
    runtime._download_archive(runtime.ARCHIVE_URL, target)
    assert target.read_bytes() == b"abc"


def test_version_failure_retains_missing_library_diagnostic():
    def failed(args, **kwargs):
        return subprocess.CompletedProcess(args, 127, stdout="", stderr="libXrender.so.1: cannot open shared object file")
    with pytest.raises(runtime.BlenderRuntimeError, match="libXrender.so.1"):
        runtime._version(Path("/fake/blender"), failed)
