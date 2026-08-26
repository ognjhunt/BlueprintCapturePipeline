from __future__ import annotations

import hashlib
import io
import subprocess
import tarfile
from pathlib import Path

from scripts import bootstrap_task_evaluation_splat_render_prerequisites as subject


def _node_archive() -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:xz") as archive:
        for relative, payload, mode in (
            ("bin/node", b"linux-node", 0o755),
            ("lib/node_modules/npm/bin/npm-cli.js", b"npm-cli", 0o644),
        ):
            info = tarfile.TarInfo(
                f"node-v{subject.NODE_VERSION}-linux-x64/{relative}"
            )
            info.size = len(payload)
            info.mode = mode
            archive.addfile(info, io.BytesIO(payload))
    return stream.getvalue()


def test_bootstrap_publishes_pinned_linux_prerequisites(
    tmp_path: Path, monkeypatch
) -> None:
    repository = tmp_path / "repository/tools/splat_render"
    repository.mkdir(parents=True)
    (repository / "package.json").write_text("{}\n", encoding="utf-8")
    (repository / "package-lock.json").write_text("{}\n", encoding="utf-8")
    archive = _node_archive()
    monkeypatch.setattr(
        subject,
        "NODE_ARCHIVE_SHA256",
        "sha256:" + hashlib.sha256(archive).hexdigest(),
    )

    def runner(argv, *, cwd, capture_output, text, check, timeout, env=None):
        del capture_output, text, check, timeout
        cwd = Path(cwd)
        if "ci" in argv:
            for package in (
                "@sparkjsdev/spark",
                "playwright",
                "playwright-core",
                "three",
            ):
                marker = cwd / "node_modules" / package / "index.js"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.write_text(package, encoding="utf-8")
            cli = cwd / "node_modules/playwright/cli.js"
            cli.parent.mkdir(parents=True, exist_ok=True)
            cli.write_text("// cli\n", encoding="utf-8")
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert env is not None
        cache = Path(env["PLAYWRIGHT_BROWSERS_PATH"])
        browser = cache / "chromium-test/chrome-linux/chrome"
        browser.parent.mkdir(parents=True, exist_ok=True)
        browser.write_bytes(b"chromium")
        browser.chmod(0o755)
        if "-e" in argv:
            return subprocess.CompletedProcess(argv, 0, str(browser), "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    output = tmp_path / "prerequisites"

    def readback(path: Path) -> bytes:
        publication_root = next(
            parent for parent in path.parents if parent.parent == output.parent
        )
        assert publication_root.stat().st_mode & 0o001
        assert path.stat().st_mode & 0o004
        return path.read_bytes()

    manifest = subject.bootstrap_splat_render_prerequisites(
        repository_root=tmp_path / "repository",
        output_root=output,
        readback=readback,
        readback_actor="service-account:blueprint",
        downloader=lambda _url: archive,
        runner=runner,
    )

    assert manifest["status"] == "published_full_byte_readback_passed"
    assert manifest["full_byte_service_account_readback_passed"] is True
    assert (output / manifest["entrypoints"]["node"]).is_file()
    assert (output / manifest["entrypoints"]["browser"]).is_file()
    assert (output / "node_modules/playwright/index.js").is_file()
    assert output.stat().st_mode & 0o222 == 0
    assert not any(path.is_symlink() for path in output.rglob("*"))
    reopened = subject.validate_splat_render_prerequisites(
        root=output,
        repository_root=tmp_path / "repository",
    )
    assert reopened["manifest"]["prerequisite_digest"] == manifest[
        "prerequisite_digest"
    ]
    repeated = subject.bootstrap_splat_render_prerequisites(
        repository_root=tmp_path / "repository",
        output_root=output,
        readback=readback,
        readback_actor="service-account:blueprint",
        downloader=lambda _url: (_ for _ in ()).throw(AssertionError("downloaded twice")),
        runner=runner,
    )
    assert repeated == manifest
