# Covers (for impacted-test selection):
#   scripts/install_scene_retirement_runtime.py
#   src/blueprint_pipeline/runtime_source_identity.py
"""Disposable Linux permission proof for the installed source-identity reader.

The actual installer source helpers publish a tiny local Git fixture, using an
explicit fixture admission inventory at the cryptographic boundary. A real
root-to-nobody child imports the unchanged leaf from its published generation
and calls its default __file__ reader. This proves those filesystem permissions
and refusal boundaries, not production attestation, service or engine acceptance.
Mac skips remain unmet native proof; no Mac sudo, ACL or owner substitution runs.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pwd
import select
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_NAME = "src/blueprint_pipeline/runtime_source_identity.py"
CASES = (
    "admitted", "module_bytes", "module_unreadable", "module_writable", "module_symlink",
    "claim_bytes", "claim_unreadable", "claim_writable", "claim_symlink",
    "manifest_unreadable", "manifest_writable", "writable_ancestor",
    "legacy_generation", "unadmitted_checkout",
)
UNKNOWN = {"schema_version": "runtime_source_identity.v1", "state": "unproven",
           "reason": "runtime_source_identity_unproven", "source_commit": None}


def _require_disposable_root():
    assert sys.platform == "linux" and os.environ.get("BLUEPRINT_DISPOSABLE_LINUX_TEST") == "1"
    assert os.geteuid() == os.getuid() == 0


def _git(checkout, *args):
    result = subprocess.run(
        ["/usr/bin/git", "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgSign=false",
         "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid", "-C", str(checkout), *args],
        capture_output=True, check=True, timeout=10,
        env=os.environ | {"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_TERMINAL_PROMPT": "0"},
    )
    return result.stdout


def _installed_fixture(root):
    spec = importlib.util.spec_from_file_location("identity_native_installer", ROOT / "scripts/install_scene_retirement_runtime.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    assert installer._OWNER == 0
    installer._RUNTIME_ROOT, installer._BOOT_ROOT = root / "runtime", root / "bootstrap"
    checkout = root / "checkout"
    files = {
        "src/blueprint_pipeline/__init__.py": (ROOT / "src/blueprint_pipeline/__init__.py").read_bytes(),
        MODULE_NAME: (ROOT / MODULE_NAME).read_bytes(),
        "scripts/scene_retirement_continuous_bootstrap.py": b"# development-only source fixture\n",
        "deploy/systemd/fixture.service": b"[Service]\nType=oneshot\n",
        "uv.lock": b"version = 1\n",
    }
    for name, body in files.items():
        target = checkout / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
        target.write_bytes(body)
    _git(checkout, "init", "--quiet")
    _git(checkout, "add", ".")
    _git(checkout, "commit", "--quiet", "-m", "development-only source fixture")
    commit = _git(checkout, "rev-parse", "HEAD").decode().strip()
    inventory = []
    for item in _git(checkout, "ls-tree", "-rz", commit).split(b"\0"):
        if item:
            head, raw_name = item.split(b"\t", 1)
            mode, kind, oid = head.decode().split()
            name = raw_name.decode()
            assert kind == "blob"
            body = files[name]
            inventory.append({"path": name, "git_blob_oid": oid, "mode": mode,
                              "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
    # This is explicit fixture authority, not a production signature or waiver.
    manifest = {"sources": [{"repository": "ognjhunt/BlueprintCapturePipeline",
                              "commit": commit, "files": sorted(inventory, key=lambda row: row["path"])}]}
    deadline = time.monotonic() + 60
    protected = installer._signed_release(checkout, commit, deadline, manifest)
    admitted = installer._runtime_source_release(protected, commit, manifest, deadline, checkout=checkout)
    rows, sources = {}, {}
    for name in ("src/blueprint_pipeline", "scripts", "deploy/systemd"):
        installer._tree(protected / name, Path(name), rows, sources, deadline)
    generation, digest = installer._generation("source", rows, sources, deadline, admitted)
    legacy, _ = installer._generation("source", rows, sources, deadline)
    claim = installer._sdk_root() / "release-inputs" / (commit + ".claim.json")
    record = installer._RUNTIME_ROOT / "manifests" / ("source-" + digest + ".json")
    # Assert actual helper output modes. Never chmod a published positive to pass.
    for target in (generation / MODULE_NAME, claim, record):
        assert target.stat().st_uid == 0 and stat.S_IMODE(target.stat().st_mode) == 0o644
        for parent in target.parents:
            assert parent.stat().st_uid == 0 and stat.S_IMODE(parent.stat().st_mode) == 0o755
            if parent == root:
                break
    return dict(runtime=installer._RUNTIME_ROOT, bootstrap=installer._BOOT_ROOT,
                module=generation / MODULE_NAME, claim=claim, manifest=record,
                generation=generation, legacy=legacy / MODULE_NAME,
                checkout=checkout / MODULE_NAME, commit=commit)


def _receive(fd, timeout=10):
    assert select.select([fd], [], [], timeout)[0], "native source reader child timed out"
    raw = os.read(fd, 16384)
    assert raw and len(raw) < 16384
    return json.loads(raw)


def _child(info, case, foreign, command, result):
    try:
        os.setgroups([])
        os.setresgid(foreign.pw_gid, foreign.pw_gid, foreign.pw_gid)
        os.setresuid(foreign.pw_uid, foreign.pw_uid, foreign.pw_uid)
        assert os.getuid() == os.geteuid() == foreign.pw_uid != 0 and os.getgroups() == []
        selected = info["legacy"] if case == "legacy_generation" else info["checkout"] if case == "unadmitted_checkout" else info["module"]
        spec = importlib.util.spec_from_file_location("actual_installed_identity_reader", selected)
        reader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reader)
        assert Path(reader.__file__) == selected and reader._OWNER == 0
        reader._RUNTIME_ROOT, reader._BOOT_ROOT = info["runtime"], info["bootstrap"]
        os.environ["BLUEPRINT_RUNNING_COMMIT"] = "f" * 40
        os.environ["EXPECTED_PRODUCTION_COMMIT"] = "e" * 40
        os.write(result, json.dumps({"status": "loaded", "uid": os.geteuid()}).encode())
        assert os.read(command, 1) == b"1"
        # A loaded leaf must read only the installed evidence, without Git,
        # network, credentials or a privileged helper process.
        import socket

        def forbidden(*args, **kwargs):
            raise AssertionError("identity reader attempted external I/O")

        socket.socket.connect = forbidden
        subprocess.Popen = forbidden
        proof = reader.running_source_identity()
        if case == "admitted":
            assert proof["state"] == "proven" and proof["source_commit"] == info["commit"]
            assert proof["basis"] == "protected_admitted_source_snapshot"
        else:
            assert proof == UNKNOWN
        os.write(result, json.dumps({"status": "passed", "uid": os.geteuid(), "state": proof["state"]}).encode())
        os._exit(0)
    except BaseException:
        os.write(result, b'{"status":"failed","code":"native_source_identity_child_failed"}')
        os._exit(1)


def _mutate(info, case):
    if case in {"admitted", "legacy_generation", "unadmitted_checkout"}:
        return lambda: None
    kind = case.split("_", 1)[0]
    target = info["generation"] if case == "writable_ancestor" else info[kind]
    if case.endswith("_bytes"):
        before = target.read_bytes()
        target.write_bytes(before + b"\n# changed after admission\n")
        return lambda: target.write_bytes(before)
    if case.endswith("_symlink"):
        renamed = target.with_name(target.name + ".fixture-original")
        target.rename(renamed)
        target.symlink_to(renamed.name)

        def restore():
            target.unlink()
            renamed.rename(target)

        return restore
    original = stat.S_IMODE(target.stat().st_mode)
    target.chmod(0o775 if case == "writable_ancestor" else 0o600 if case.endswith("_unreadable") else 0o664)
    return lambda: target.chmod(original)


def _observe_case(info, case, foreign):
    command_read, command_write = os.pipe()
    result_read, result_write = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(command_write)
        os.close(result_read)
        _child(info, case, foreign, command_read, result_write)
    os.close(command_read)
    os.close(result_write)
    def restore():
        pass

    reaped = False
    try:
        loaded = _receive(result_read)
        assert loaded == {"status": "loaded", "uid": foreign.pw_uid}
        restore = _mutate(info, case)
        os.write(command_write, b"1")
        proof = _receive(result_read)
        assert proof == {"status": "passed", "uid": foreign.pw_uid,
                         "state": "proven" if case == "admitted" else "unproven"}, case
        _, status = os.waitpid(pid, 0)
        reaped = True
        assert os.waitstatus_to_exitcode(status) == 0
        return proof["state"]
    finally:
        if not reaped:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            os.waitpid(pid, 0)
        restore()
        os.close(command_write)
        os.close(result_read)


def _root_fixture():
    _require_disposable_root()
    foreign = pwd.getpwnam("nobody")
    assert foreign.pw_uid != 0
    previous_umask = os.umask(0o022)
    root = None
    try:
        root = Path(tempfile.mkdtemp(prefix="blueprint-source-identity-", dir="/var/lib"))
        root.chmod(0o755)  # Only this new, empty disposable fixture parent.
        info = _installed_fixture(root)
        states = {case: _observe_case(info, case, foreign) for case in CASES}
        return {"status": "passed", "foreign_uid": foreign.pw_uid,
                "claim_mode": "0644", "ancestor_mode": "0755", "reader_owner": 0,
                "states": states, "claim_scope": "development_only_permission_read_proof"}
    finally:
        if root is not None:
            shutil.rmtree(root)
        os.umask(previous_umask)


@pytest.mark.parametrize("flag", [None, "0"])
def test_root_driver_refuses_before_fixture_or_privilege_operations(monkeypatch, flag):
    if flag is None:
        monkeypatch.delenv("BLUEPRINT_DISPOSABLE_LINUX_TEST", raising=False)
    else:
        monkeypatch.setenv("BLUEPRINT_DISPOSABLE_LINUX_TEST", flag)

    def forbidden(*args, **kwargs):
        pytest.fail("non-disposable driver reached a fixture or privilege operation")

    monkeypatch.setattr(tempfile, "mkdtemp", forbidden)
    monkeypatch.setattr(os, "fork", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    with pytest.raises(AssertionError):
        _root_fixture()


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux" or os.environ.get("BLUEPRINT_DISPOSABLE_LINUX_TEST") != "1",
                    reason="mandatory disposable Linux root/different-UID identity proof; Mac skip is unmet")
def test_actual_root_published_identity_is_readable_and_refuses_unsafe_evidence():
    command = [sys.executable, str(Path(__file__).resolve()), "--source-identity-root-fixture"]
    if os.geteuid() != 0:
        command = ["sudo", "-n", "env", "BLUEPRINT_DISPOSABLE_LINUX_TEST=1", "PYTHONDONTWRITEBYTECODE=1", *command]
    done = subprocess.run(command, capture_output=True, text=True, timeout=90,
                          env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["foreign_uid"] != 0 and result["reader_owner"] == 0
    assert result["claim_mode"] == "0644" and result["ancestor_mode"] == "0755"
    assert result["states"] == {case: "proven" if case == "admitted" else "unproven" for case in CASES}
    assert result["status"] == "passed" and result["claim_scope"] == "development_only_permission_read_proof"


if __name__ == "__main__" and sys.argv[1:] == ["--source-identity-root-fixture"]:
    print(json.dumps(_root_fixture(), sort_keys=True))
