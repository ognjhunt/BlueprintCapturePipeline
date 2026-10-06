"""Offline portable-release and independent scheduler contracts; no deployment."""
import configparser
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from tools.daily_research.runner import Refusal, configuration
from tools.daily_research.standalone import (
    BLUEPRINT_RUNTIME_FILES,
    FILES,
    PREFIX,
    RUNTIME_PREFIX,
    build,
)

SOURCE = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for name in FILES:
        target = root / PREFIX / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / PREFIX / name, target)
    for name in BLUEPRINT_RUNTIME_FILES:
        target = root / "src/blueprint_pipeline" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / "src/blueprint_pipeline" / name, target)
    # Even tracked unrelated/private inputs must not enter the portable release.
    (root / ".env").write_text("SYNTHETIC_PRIVATE_INPUT=do-not-package\n")
    (root / PREFIX / "crm.json").write_text('{"private": "synthetic"}')
    (root / "unrelated_gpu.py").write_text("raise RuntimeError('unrelated')\n")
    git(root, "init", "--quiet")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Offline fixture")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "Synthetic package fixture")
    return root, git(root, "rev-parse", "HEAD")


def test_exact_commit_bundle_is_deterministic_and_preserves_hashes(repository, tmp_path):
    root, revision = repository
    first = build(revision, tmp_path / "first", root)
    second = build(revision, tmp_path / "second", root)
    assert first == second
    archive = tmp_path / "first" / first["archive"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == first["sha256"]
    assert archive.stat().st_size == first["bytes"]
    assert archive.stat().st_mode & 0o777 == 0o600
    with tarfile.open(archive) as package:
        assert set(package.getnames()) == ({PREFIX + name for name in FILES}
            | {RUNTIME_PREFIX + name for name in BLUEPRINT_RUNTIME_FILES}
            | {"manifest.json"})
        manifest = json.load(package.extractfile("manifest.json"))
        assert manifest["source_commit"] == revision
        assert manifest["activation_performed"] is False
        assert manifest["credential_binding_verified"] is False
        assert manifest["persistent_host_verified"] is False
        for name, expected in manifest["files"].items():
            assert hashlib.sha256(package.extractfile(name).read()).hexdigest() == expected
        assert all(entry.isfile() and entry.mode == 0o644 for entry in package)
    # Export reads pinned blobs, even when the development worktree is dirty.
    (root / PREFIX / "runner.py").write_text("unreviewed worktree change")
    assert build(revision, tmp_path / "third", root) == first


@pytest.mark.parametrize("revision", ["main", "HEAD", "a" * 7, "a" * 39, "../main"])
def test_moving_or_invalid_revision_refused(repository, tmp_path, revision):
    root, _ = repository
    with pytest.raises(ValueError, match="immutable"):
        build(revision, tmp_path / "output", root)
    assert not (tmp_path / "output").exists()


def test_existing_destination_refused_without_clobber(repository, tmp_path):
    root, revision = repository
    target = tmp_path / "output"
    target.mkdir()
    (target / "receipt.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        build(revision, target, root)
    assert (target / "receipt.json").read_text() == "preserve"


def test_git_replacement_cannot_substitute_reviewed_commit(repository, tmp_path):
    root, reviewed = repository
    expected = build(reviewed, tmp_path / "before", root)
    (root / PREFIX / "runner.py").write_text("substituted commit bytes")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "Synthetic substitute")
    replacement = git(root, "rev-parse", "HEAD")
    git(root, "replace", reviewed, replacement)
    assert git(root, "show", reviewed + ":" + PREFIX + "runner.py") == "substituted commit bytes"
    assert build(reviewed, tmp_path / "after", root) == expected


def test_symlink_source_refused(repository, tmp_path):
    root, _ = repository
    source = root / PREFIX / "runner.py"
    source.unlink()
    source.symlink_to("../../.env")
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "Synthetic unsafe source")
    with pytest.raises(ValueError, match="regular source"):
        build(git(root, "rev-parse", "HEAD"), tmp_path / "output", root)
    assert not (tmp_path / "output").exists()


def test_enabled_export_refused(repository, tmp_path):
    root, _ = repository
    path = root / PREFIX / "standalone.config.example.json"
    config = json.loads(path.read_text())
    config["enabled"] = True
    path.write_text(json.dumps(config))
    git(root, "add", ".")
    git(root, "commit", "--quiet", "-m", "Synthetic unsafe config")
    with pytest.raises(ValueError, match="disabled"):
        build(git(root, "rev-parse", "HEAD"), tmp_path / "output", root)


def test_export_runs_isolated_without_sdk_or_pipeline(repository, tmp_path):
    root, revision = repository
    receipt = build(revision, tmp_path / "output", root)
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    with tarfile.open(tmp_path / "output" / receipt["archive"]) as package:
        package.extractall(isolated, filter="data")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from tools.daily_research.runner import main; "
        "assert not any(m.startswith(('openai', 'blueprint_pipeline', 'torch')) for m in sys.modules); "
        "raise SystemExit(main(['--config', sys.argv[2], '--state-dir', sys.argv[3], 'status']))"
    )
    run = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(isolated),
                          str(isolated / PREFIX / "standalone.config.example.json"),
                          str(tmp_path / "state")], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == []


@pytest.mark.slow
def test_optional_findall_import_uses_only_exported_stdlib_closure(repository, tmp_path):
    root, revision = repository
    receipt = build(revision, tmp_path / "output", root)
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    with tarfile.open(tmp_path / "output" / receipt["archive"]) as package:
        package.extractall(isolated, filter="data")
    code = (
        "import json, pathlib, socket, sys; sys.path.insert(0, sys.argv[1]); "
        "socket.create_connection = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network')); "
        "from tools.daily_research import findall; findall.runtime(); "
        "assert not any(m.startswith(('openai', 'torch')) for m in sys.modules); "
        "assert len([n for n in sys.modules if n == 'blueprint_pipeline' or n.startswith('blueprint_pipeline.')]) ==7; "
        "assert all(pathlib.Path(m.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve()) "
        "for n, m in sys.modules.items() if n.startswith('blueprint_pipeline')); "
        "print(json.dumps(findall.runtime_status()))"
    )
    run = subprocess.run([sys.executable, "-I", "-S", "-c", code, str(isolated)],
                         capture_output=True, text=True, check=True,
                         env={"PYTHONDONTWRITEBYTECODE": "1"}, timeout=30)
    status = json.loads(run.stdout)
    assert status["module_imported"] is True
    assert status["credential_binding_present"] is False
    assert status["callable_handler_installed"] is False
    assert status["production_binding_verified"] is False


def test_disabled_v3_configuration_requires_real_cutover():
    value = json.loads((SOURCE / PREFIX / "standalone.config.example.json").read_text())
    assert configuration(value)["enabled"] is False
    assert value["research_contract_version"] == 3
    assert "knowledge_filters" not in value
    value["enabled"] = True
    with pytest.raises(Refusal, match="scheduler_cutover_not_approved"):
        configuration(value)


def test_units_have_independent_runtime_and_no_application_hold_dependency():
    service = (SOURCE / PREFIX / "systemd/blueprint-researcher-daily.service").read_text()
    timer = (SOURCE / PREFIX / "systemd/blueprint-researcher-daily.timer").read_text()
    parsed = configparser.ConfigParser(interpolation=None)
    parsed.read_string(service)
    assert parsed["Service"]["WorkingDirectory"] == "/opt/blueprint/researcher/current"
    assert parsed["Service"]["ReadWritePaths"] == "/var/lib/blueprint/researcher"
    assert parsed["Service"]["KillSignal"] == "SIGTERM"
    assert parsed["Service"]["TimeoutStopSec"] == "90s"
    assert parsed["Service"]["ProtectSystem"] == "strict"
    assert "Restart=" not in service
    parsed.read_string(timer)
    assert parsed["Timer"]["OnCalendar"] == "*-*-* 07:00:00 America/Chicago"
    assert parsed["Timer"]["Persistent"] == "true"
    assert parsed["Timer"]["AccuracySec"] == "1s"
    assert parsed["Timer"]["RandomizedDelaySec"] == "0"
    assert all(word not in (service + timer).lower()
               for word in ("dot", "codex", "operator-door", "paperclip", "task-evaluation-control-plane"))


@pytest.mark.parametrize(("base", "next_utc"), [
    ("2026-10-31 23:00:00 UTC", "2026-11-01 13:00:00 UTC"),
    ("2027-03-13 23:00:00 UTC", "2027-03-14 12:00:00 UTC"),
    ("2027-01-02 23:00:00 UTC", "2027-01-03 13:00:00 UTC"),
    ("2027-06-02 23:00:00 UTC", "2027-06-03 12:00:00 UTC"),
])
def test_actual_systemd_calendar_preserves_seven_am_across_dst(base, next_utc):
    # This is a native parser contract, not a mock of our own date arithmetic.
    run = subprocess.run(["systemd-analyze", "calendar", "--base-time=" + base,
                          "*-*-* 07:00:00 America/Chicago"], check=True,
                         capture_output=True, text=True,
                         env={**os.environ, "TZ": "UTC", "LC_ALL": "C"})
    assert next_utc in run.stdout


def test_release_packages_the_screen_admission_module():
    # render imports screen_admission for the scheduler step, and site-screen.py runs its owner commands.
    for name in ("screen_admission.py", "site_screen.py", "operators/site-screen.py"):
        assert name in FILES and (SOURCE / PREFIX / name).is_file()


def test_release_packages_the_site_universe_module_and_owner_command():
    # runner imports site_universe, so the isolated export test above also proves the module is packaged.
    for name in ("site_universe.py", "operators/site-universe-backlog.py"):
        assert name in FILES and (SOURCE / PREFIX / name).is_file()
