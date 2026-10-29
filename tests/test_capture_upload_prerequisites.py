"""Host prerequisites must not depend on the literal default capture path."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from blueprint_pipeline import capture_upload_prerequisites as prerequisites
from blueprint_pipeline.capture_upload_runtime import CaptureUploadRuntimeError
from blueprint_pipeline.control_plane_storage_roots import classify_path


@pytest.fixture
def host(tmp_path, monkeypatch):
    root = tmp_path / "host"
    (root / "var/lib/blueprint").mkdir(parents=True)
    scanner = tmp_path / "clamscan"
    updater = tmp_path / "freshclam"
    definitions = tmp_path / "definitions"
    definitions.mkdir()
    monkeypatch.setattr(prerequisites, "SYSTEM_SCANNER", scanner)
    monkeypatch.setattr(prerequisites, "SYSTEM_UPDATER", updater)
    monkeypatch.setattr(prerequisites, "DEFINITION_ROOT", definitions)
    def create_as_test_owner(parent_fd, name, *, mode, owner_ids):
        assert owner_ids == (os.getuid(), os.getgid())
        try:
            os.mkdir(name, mode=mode, dir_fd=parent_fd)
        except FileExistsError:
            return False
        return True
    monkeypatch.setattr(prerequisites, "_mkdir_as_owner", create_as_test_owner)
    return SimpleNamespace(root=root, scanner=scanner, updater=updater, definitions=definitions,
                           ids=(os.getuid(), os.getgid()), store="/var/lib/blueprint/capture-intake")


def _installed(host, *, definitions=True):
    for path in (host.scanner, host.updater):
        path.write_bytes(b"fixture executable; never run")
        path.chmod(0o700)
    if definitions:
        for name in ("main.cvd", "daily.cld"):
            (host.definitions / name).write_bytes(b"fixture vendor definitions")


def _configuration(host):
    return {"store_root": host.store, "scanner_argv": [str(host.scanner), "--no-summary"]}


def test_evidence_directory_swap_cannot_change_or_write_through_a_symlink(tmp_path, monkeypatch):
    store = tmp_path / "store"
    victim = tmp_path / "victim"
    store.mkdir()
    victim.mkdir(mode=0o750)
    before = victim.stat()
    real_mkdir = os.mkdir
    ownership_changes = []
    real_fchown = os.fchown

    def swap_after_mkdir(name, mode=0o777, *, dir_fd=None):
        real_mkdir(name, mode=mode, dir_fd=dir_fd)
        if name == "runtime-provisioning":
            os.rmdir(name, dir_fd=dir_fd)
            os.symlink(victim, name, dir_fd=dir_fd)

    def record_fchown(descriptor, *ids):
        ownership_changes.append(os.fstat(descriptor).st_ino)
        return real_fchown(descriptor, *ids)

    monkeypatch.setattr(prerequisites.os, "mkdir", swap_after_mkdir)
    monkeypatch.setattr(prerequisites.os, "fchown", record_fchown)
    monkeypatch.setattr(prerequisites.os, "chown", lambda *a, **kw: pytest.fail("pathname chown"))
    with pytest.raises(CaptureUploadRuntimeError, match="evidence_root_invalid"):
        prerequisites._record_process(store, {"phase": "fixture", "exit_code": 0},
                                      (os.getuid(), os.getgid()))
    after = victim.stat()
    assert (after.st_uid, after.st_gid, after.st_mode) == (before.st_uid, before.st_gid, before.st_mode)
    assert before.st_ino not in ownership_changes
    assert list(victim.iterdir()) == []


def test_evidence_store_ancestor_symlink_is_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "store").mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(actual, target_is_directory=True)
    with pytest.raises(CaptureUploadRuntimeError, match="evidence_root_invalid"):
        prerequisites._record_process(alias / "store", {"phase": "fixture", "exit_code": 0},
                                      (os.getuid(), os.getgid()))
    assert not (actual / "store/runtime-provisioning").exists()


@pytest.mark.parametrize("evidence", [True, False])
def test_directory_replacement_never_changes_the_replacement_permissions(tmp_path, monkeypatch, evidence):
    parent = tmp_path / "var/lib/blueprint"
    parent.mkdir(parents=True)
    victim = parent / "victim"
    victim.mkdir(mode=0o750 if evidence else 0o700)
    before = victim.stat()
    name_to_replace = "runtime-provisioning" if evidence else "capture-intake"
    real_mkdir = os.mkdir

    def swap_after_mkdir(name, mode=0o777, *, dir_fd=None):
        real_mkdir(name, mode=mode, dir_fd=dir_fd)
        if name == name_to_replace:
            os.rmdir(name, dir_fd=dir_fd)
            os.rename(victim, name, dst_dir_fd=dir_fd)

    monkeypatch.setattr(prerequisites.os, "mkdir", swap_after_mkdir)
    for operation in ("chown", "fchown", "chmod", "fchmod"):
        monkeypatch.setattr(prerequisites.os, operation,
                            lambda *a, **kw: pytest.fail("privileged directory mutation"))
    with pytest.raises(CaptureUploadRuntimeError, match="evidence_root_invalid|permissions_unsafe"):
        if evidence:
            prerequisites._record_process(parent, {"phase": "fixture", "exit_code": 0},
                                          (os.getuid(), os.getgid()))
        else:
            prerequisites.ensure_store("/var/lib/blueprint/capture-intake",
                                      owner_ids=(os.getuid(), os.getgid()), root_prefix=tmp_path)
    replacement = parent / name_to_replace
    after = replacement.stat()
    assert (after.st_uid, after.st_gid, after.st_mode) == (before.st_uid, before.st_gid, before.st_mode)
    assert list(replacement.iterdir()) == []


@pytest.mark.parametrize("parent_uid,parent_mode", [(995, 0o40750), (0, 0o40777), (0, 0o40775)])
def test_mutable_parent_creation_runs_with_only_the_service_identity(monkeypatch, parent_uid, parent_mode):
    monkeypatch.setattr(prerequisites.os, "geteuid", lambda: 0)
    monkeypatch.setattr(prerequisites.os, "fstat", lambda _: SimpleNamespace(st_uid=parent_uid, st_mode=parent_mode))
    calls = []
    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout=b"created\n")
    monkeypatch.setattr(prerequisites.subprocess, "run", run)
    assert prerequisites._mkdir_as_owner(17, "capture-intake", mode=0o750, owner_ids=(995, 986))
    argv, options = calls[0]
    assert argv[1:4] == ["-I", "-S", "-c"]
    assert argv[-3:] == ["17", "capture-intake", "0o750"]
    assert (options["user"], options["group"], options["extra_groups"]) == (995, 986, [])
    assert options["pass_fds"] == (17,)
    assert not any("KEY" in key or "TOKEN" in key for key in options["env"])


def test_root_owned_parent_creates_only_the_new_service_directory(tmp_path, monkeypatch):
    service_ids = (60000, 60000)
    parent = tmp_path / "root-owned-parent"
    parent.mkdir(mode=0o755)
    descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
    real_fstat, real_mkdir, real_stat = os.fstat, os.mkdir, os.stat
    parent_before = real_stat(parent)
    ownership = {parent_before.st_ino: (0, 0)}
    configured = []

    def ownership_metadata(value):
        if value.st_ino in ownership:
            fields = list(value)
            fields[4:6] = ownership[value.st_ino]
            return os.stat_result(fields)
        return value

    def create(name, mode=0o777, *, dir_fd=None):
        real_mkdir(name, mode=mode, dir_fd=dir_fd)
        ownership[real_stat(name, dir_fd=dir_fd, follow_symlinks=False).st_ino] = (0, 0)

    def configure(fd, uid, gid):
        info = real_fstat(fd)
        assert info.st_ino != parent_before.st_ino
        assert ownership[info.st_ino] == (0, 0)
        assert (uid, gid) == service_ids
        configured.append(info.st_ino)
        ownership[info.st_ino] = service_ids

    monkeypatch.setattr(prerequisites.os, "geteuid", lambda: 0)
    monkeypatch.setattr(prerequisites.os, "getegid", lambda: 0)
    monkeypatch.setattr(prerequisites.os, "fstat", lambda fd: ownership_metadata(real_fstat(fd)))
    monkeypatch.setattr(prerequisites.os, "stat", lambda *a, **kw: ownership_metadata(real_stat(*a, **kw)))
    monkeypatch.setattr(prerequisites.os, "mkdir", create)
    monkeypatch.setattr(prerequisites.os, "fchown", configure)
    monkeypatch.setattr(prerequisites.os, "chown", lambda *a, **kw: pytest.fail("pathname chown"))
    monkeypatch.setattr(prerequisites.subprocess, "run", lambda *a, **kw:
                        pytest.fail("service identity cannot write the root-owned 0755 parent"))
    try:
        assert prerequisites._mkdir_as_owner(descriptor, "capture-intake", mode=0o750,
                                             owner_ids=service_ids)
        child = parent / "capture-intake"
        child_before = child.stat()
        assert (child_before.st_uid, child_before.st_gid, child_before.st_mode & 0o777) == (*service_ids, 0o750)
        assert configured == [child_before.st_ino]
        assert not prerequisites._mkdir_as_owner(descriptor, "capture-intake", mode=0o750,
                                                 owner_ids=service_ids)
        assert child.stat() == child_before
        assert configured == [child_before.st_ino]
        assert parent.stat().st_mode & 0o777 == 0o755
        parent_after = real_stat(parent)
        assert (parent_after.st_ino, parent_after.st_uid, parent_after.st_gid, parent_after.st_mode) == (
            parent_before.st_ino, parent_before.st_uid, parent_before.st_gid, parent_before.st_mode)
    finally:
        os.close(descriptor)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="Actual Linux root/service identity boundary")
def test_actual_root_parent_and_service_owned_descendants():
    prefix = [] if os.geteuid() == 0 else ["sudo", "-n", "--"]
    if prefix:
        try:
            available = subprocess.run([*prefix, "/usr/bin/true"], capture_output=True, timeout=5)
        except FileNotFoundError:
            available = None
        if available is None or available.returncode:
            if os.environ.get("GITHUB_ACTIONS") == "true":
                pytest.fail("Linux qualification requires the root/service permission rehearsal")
            pytest.skip("Noninteractive fixture-only root permission unavailable")
    source = Path(prerequisites.__file__).resolve().parents[1]
    program = '''
import json, os, pathlib, shutil, sys, tempfile
sys.path.insert(0, sys.argv[1])
from blueprint_pipeline.capture_upload_prerequisites import ensure_store, _record_process
root = pathlib.Path(tempfile.mkdtemp(prefix="blueprint-capture-root-fixture-"))
try:
    root.chmod(0o755)
    parent = root / "var/lib/blueprint"
    parent.mkdir(parents=True)
    before = parent.stat()
    ids = (60000, 60000)
    result = ensure_store("/var/lib/blueprint/capture-intake/nested", owner_ids=ids, root_prefix=root)
    store = parent / "capture-intake/nested"
    for directory in (store.parent, store):
        info = directory.stat()
        assert (info.st_uid, info.st_gid, info.st_mode & 0o777) == (*ids, 0o750)
    diagnostic = _record_process(store, {"phase": "fixture", "exit_code": 0}, ids)
    evidence = store / diagnostic["evidence_relative_path"]
    info = evidence.stat()
    assert (info.st_uid, info.st_gid, info.st_mode & 0o777) == (*ids, 0o600)
    assert json.loads(evidence.read_text())["phase"] == "fixture"
    again = ensure_store("/var/lib/blueprint/capture-intake/nested", owner_ids=ids, root_prefix=root)
    assert again["created"] == []
    after = parent.stat()
    assert (after.st_uid, after.st_gid, after.st_mode) == (before.st_uid, before.st_gid, before.st_mode)
    print(json.dumps({"status": "passed", "root_parent": "0755", "created": result["created"],
                      "service_uid": ids[0], "service_gid": ids[1]}))
finally:
    shutil.rmtree(root)
'''
    result = subprocess.run([*prefix, sys.executable, "-I", "-c", program, str(source)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] == "passed"


def test_existing_environment_override_provisions_the_effective_store(host):
    unit = ('[Service]\nReadWritePaths=/var/lib/blueprint/capture-intakes\n'
            'Environment=PIPELINE_CAPTURE_INTAKE_STORE_ROOT=/var/lib/blueprint/capture-intakes\n'
            'Environment=\'PIPELINE_CAPTURE_MALWARE_SCANNER_ARGV_JSON=["/usr/bin/clamscan"]\'\n'
            'EnvironmentFile=/etc/blueprint/fixture.env\n')
    configuration = prerequisites.load_configuration(unit, read_file=lambda _: (
        "IGNORED_SECRET='unbalanced\nPIPELINE_CAPTURE_INTAKE_STORE_ROOT=" + host.store + "\n"))
    assert configuration["store_root"] == host.store
    result = prerequisites.ensure_store(configuration["store_root"], owner_ids=host.ids, root_prefix=host.root)
    actual = host.root / host.store.lstrip("/")
    assert actual.is_dir()
    assert actual.stat().st_mode & 0o777 == 0o750
    assert not (host.root / "var/lib/blueprint/capture-intakes").exists()
    assert result["existing_permissions_changed"] is False
    assert "IGNORED_SECRET" not in json.dumps(configuration)


@pytest.mark.parametrize("path", ["", "relative/path", "/tmp/captures", "/var/lib/blueprint",
                                  "/var/lib/blueprint/../escape"])
def test_store_outside_the_existing_service_state_tree_is_not_created(host, path):
    with pytest.raises(CaptureUploadRuntimeError, match="not_provisionable"):
        prerequisites.ensure_store(path, owner_ids=host.ids, root_prefix=host.root)
    assert not list((host.root / "var/lib/blueprint").iterdir())


def test_existing_store_permissions_and_aliased_ancestors_are_never_repaired(host, tmp_path):
    store = host.root / host.store.lstrip("/")
    store.mkdir(parents=True)
    store.chmod(0o777)
    with pytest.raises(CaptureUploadRuntimeError, match="permissions_unsafe"):
        prerequisites.ensure_store(host.store, owner_ids=host.ids, root_prefix=host.root)
    assert store.stat().st_mode & 0o777 == 0o777
    other = tmp_path / "other"
    other.mkdir()
    (store.parent / "alias").symlink_to(other, target_is_directory=True)
    with pytest.raises(CaptureUploadRuntimeError, match="provision_failed"):
        prerequisites.ensure_store("/var/lib/blueprint/alias/child", owner_ids=host.ids, root_prefix=host.root)
    assert not list(other.iterdir())


def test_working_system_scanner_and_definitions_do_not_trigger_package_operations(host, monkeypatch):
    _installed(host)
    monkeypatch.setattr(prerequisites.subprocess, "run", lambda *a, **k: pytest.fail("unexpected host command"))
    result = prerequisites.provision_runtime(_configuration(host), owner_ids=host.ids, root_prefix=host.root)
    assert result["status"] == "provisioned_controls_required"
    assert result["prerequisite_processes"] == []
    assert result["source_footage_processed"] is False


def test_custom_scanner_is_preserved_and_does_not_install_clamav(host, tmp_path, monkeypatch):
    custom = tmp_path / "custom-scanner"
    custom.write_text("fixture")
    custom.chmod(0o700)
    monkeypatch.setattr(prerequisites.subprocess, "run", lambda *a, **k: pytest.fail("custom scanner replaced"))
    result = prerequisites.provision_runtime({"store_root": host.store, "scanner_argv": [str(custom)]},
                                             owner_ids=host.ids, root_prefix=host.root,
                                             install_system_scanner=True)
    assert result["scanner_program"] == "custom-scanner"
    assert result["status"] == "provisioned_controls_required"


def test_missing_scanner_is_installed_once_and_full_process_receipts_are_retained(host, monkeypatch):
    monkeypatch.setattr(prerequisites.os, "geteuid", lambda: 0)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        assert kwargs["shell"] is False
        assert not any("KEY" in key or "TOKEN" in key for key in kwargs["env"])
        if "install" in argv:
            assert argv[-2:] == ["clamav", "clamav-freshclam"]
            _installed(host)
        return SimpleNamespace(returncode=0, stdout=b"retained installer stdout\n", stderr=b"")

    monkeypatch.setattr(prerequisites.subprocess, "run", run)
    result = prerequisites.provision_runtime(_configuration(host), owner_ids=host.ids,
        root_prefix=host.root, install_system_scanner=True)
    assert len(calls) == 2
    assert [v["phase"] for v in result["prerequisite_processes"]] == ["apt_update", "apt_install"]
    store = host.root / host.store.lstrip("/")
    for row in result["prerequisite_processes"]:
        path = store / row["evidence_relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["evidence_sha256"]
        assert path.stat().st_mode & 0o777 == 0o600
        assert base64.b64decode(json.loads(path.read_text())["stdout_base64"]) == b"retained installer stdout\n"


def test_updater_already_running_is_waited_for_without_stopping_or_restarting_it(host, monkeypatch):
    _installed(host, definitions=False)
    monkeypatch.setattr(prerequisites.os, "geteuid", lambda: 0)
    calls = []
    monkeypatch.setattr(prerequisites.subprocess, "run", lambda argv, **kw:
        (calls.append(argv) or SimpleNamespace(returncode=0, stdout=b"active\n", stderr=b"")))
    monkeypatch.setattr(prerequisites.time, "sleep", lambda _: _installed(host))
    result = prerequisites.provision_runtime(_configuration(host), owner_ids=host.ids,
        root_prefix=host.root, install_system_scanner=True)
    assert calls == [["/usr/bin/systemctl", "is-active", "clamav-freshclam.service"]]
    assert result["status"] == "provisioned_controls_required"


def test_package_failure_retains_output_and_does_not_claim_scanner_readiness(host, monkeypatch):
    monkeypatch.setattr(prerequisites.os, "geteuid", lambda: 0)
    monkeypatch.setattr(prerequisites.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=100, stdout=b"full failed output", stderr=b"fixture install failure"))
    with pytest.raises(CaptureUploadRuntimeError, match="prerequisite_failed:apt_update") as failed:
        prerequisites.provision_runtime(_configuration(host), owner_ids=host.ids,
            root_prefix=host.root, install_system_scanner=True)
    diagnostic = failed.value.diagnostic
    assert diagnostic["exit_code"] == 100
    record = host.root / host.store.lstrip("/") / diagnostic["evidence_relative_path"]
    assert base64.b64decode(json.loads(record.read_text())["stderr_base64"]) == b"fixture install failure"


def test_definition_wait_failure_is_retained_and_never_kills_the_existing_updater(host, monkeypatch):
    _installed(host, definitions=False)
    monkeypatch.setattr(prerequisites.os, "geteuid", lambda: 0)
    calls = []
    monkeypatch.setattr(prerequisites.subprocess, "run", lambda argv, **kw:
        (calls.append(argv) or SimpleNamespace(returncode=0, stdout=b"active\n", stderr=b"")))
    with pytest.raises(CaptureUploadRuntimeError, match="definitions_not_ready") as failure:
        prerequisites.provision_runtime(_configuration(host), owner_ids=host.ids,
            root_prefix=host.root, install_system_scanner=True, definition_wait_seconds=0)
    assert failure.value.diagnostic["phase"] == "freshclam_status"
    assert len(calls) == 2
    assert not any("stop" in call or "restart" in call for call in calls)


@pytest.mark.parametrize("root", ["capture-intake", "capture-intakes"])
def test_effective_and_default_capture_stores_are_protected_evidence(root):
    assert classify_path("/var/lib/blueprint/" + root + "/session/raw/source.mp4").storage_class == "evidence_hot"


@pytest.mark.parametrize("valid", [False, True])
def test_host_entrypoint_requires_complete_service_user_controls(tmp_path, monkeypatch, capsys, valid):
    unit = tmp_path / "intake.service"
    unit.write_text("fixture")
    configuration = {"store_root": str(tmp_path), "scanner_argv": ["/usr/bin/clamscan"]}
    monkeypatch.setattr(prerequisites.sys, "argv", ["capture-prerequisites", "--unit-file", str(unit)])
    monkeypatch.setattr(prerequisites.pwd, "getpwnam", lambda _: SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid()))
    monkeypatch.setattr(prerequisites, "load_configuration", lambda _: configuration)
    monkeypatch.setattr(prerequisites, "provision_runtime", lambda *a, **k: {"status": "provisioned_controls_required"})

    def run(argv, **kwargs):
        assert argv[:4] == ["/usr/sbin/runuser", "--user", "blueprint", "--"]
        assert json.loads(kwargs["input"]) == configuration
        assert not any("TOKEN" in key or "KEY" in key for key in kwargs["env"])
        value = {"status": "ready", "provider_mutation_performed": False,
                 "source_footage_processed": False, "robot_episode_executed": False}
        if valid:
            value["controls"] = {"clean": {"exit_code": 0}, "known_test_signature": {"exit_code": 1}}
        else:
            value["untrusted_extra"] = "private-canary"
        return SimpleNamespace(returncode=0, stdout=json.dumps(value).encode(), stderr=b"")

    monkeypatch.setattr(prerequisites.subprocess, "run", run)
    assert prerequisites.main() == (0 if valid else 2)
    printed = capsys.readouterr().out
    assert "private-canary" not in printed
    assert json.loads(printed)["status"] == ("ready" if valid else "blocked")
