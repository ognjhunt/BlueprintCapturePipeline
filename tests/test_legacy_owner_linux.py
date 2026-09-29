# Covers (for impacted-test selection):
#   deploy/operator-door/operator_door/spool_runner.py
#   src/blueprint_pipeline/control_plane_lane_legacy_owner.py
"""Actual disposable Linux rights for the read-only legacy census door unit.

Mac skips are not acceptance. CI requires this exact selector to pass with a real
systemd transient unit, foreign UID writer, private root store and /proc reader.
"""

import fcntl
import json
import os
import pwd
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


def _root_fixture() -> dict:
    assert sys.platform == "linux" and os.geteuid() == 0
    assert Path("/proc/1/exe").resolve() == Path("/usr/lib/systemd/systemd")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deploy/operator-door"))
    from operator_door.config import DoorConfig
    from operator_door.spool_runner import _legacy_owner_properties
    from blueprint_pipeline.control_plane_lane_legacy_owner import snapshot_generation

    foreign = pwd.getpwnam("nobody")
    root = Path(tempfile.mkdtemp(prefix="blueprint-adp-legacy-door-", dir="/var/lib"))
    root.chmod(0o755)
    child = None
    lock = None
    try:
        work = root / "work" / "lanes"
        target = work / "diagnostics" / "old"
        target.mkdir(parents=True)
        target.chmod(0o700)
        os.chown(target, foreign.pw_uid, foreign.pw_gid)
        payload = target / "one.log"
        payload.write_bytes(b"foreign writer bytes")
        payload.chmod(0o600)
        os.chown(payload, foreign.pw_uid, foreign.pw_gid)
        inputs = root / "inputs" / "lanes"
        inputs.mkdir(parents=True)
        pins = root / "pins"
        pins.mkdir()
        release = root / "release"
        release.mkdir()
        private = root / "private"
        private.mkdir(mode=0o700)
        policy, environment = private / "policy.json", private / "environment"
        policy.write_bytes(b"SECRET_CANARY_POLICY")
        environment.write_bytes(b"SECRET_CANARY_ENV")
        policy.chmod(0o600)
        environment.chmod(0o600)
        hidden = private / "hidden-provider-secrets"
        hidden.mkdir(mode=0o700)
        (hidden / "secret").write_bytes(b"SECRET_CANARY_HIDDEN")
        state = root / "state"
        registry = state / "requests" / "legacy-owner-registrations"
        registry.mkdir(parents=True, mode=0o700)
        registry.chmod(0o700)
        lock_path = registry / ".legacy-owner.lock"
        lock_path.write_bytes(b"")
        lock_path.chmod(0o600)
        lock = os.open(lock_path, os.O_RDONLY)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        private_read, private_write = os.pipe()
        private_child = os.fork()
        if private_child == 0:
            os.close(private_read)
            os.setgid(foreign.pw_gid)
            os.setuid(foreign.pw_uid)
            denied = 0
            for protected in (policy, environment, lock_path):
                try:
                    protected.open("rb").close()
                except PermissionError:
                    denied += 1
            os.write(private_write, str(denied).encode())
            os._exit(0)
        os.close(private_write)
        private_denials = int(os.read(private_read, 10))
        os.close(private_read)
        _, private_status = os.waitpid(private_child, 0)
        assert os.waitstatus_to_exitcode(private_status) == 0 and private_denials == 3
        results = state / "requests" / "results"
        results.mkdir(mode=0o755)
        config = DoorConfig(state_root=str(state), lane_owner_policy_file=str(policy),
                            experiment_gc_environment_file=str(environment),
                            lane_scratch_work_root=str(work), lane_scratch_inputs_root=str(inputs),
                            hidden_paths=DoorConfig().hidden_paths + (str(hidden),))
        first = snapshot_generation(target, allowed_roots=(root / "work",))

        child = subprocess.Popen(
            [sys.executable, "-c",
             "import pathlib,time,sys; f=open(sys.argv[1],'rb'); "
             "pathlib.Path(sys.argv[2]).write_text('ready'); time.sleep(30)",
             str(payload), str(target / "ready")],
            preexec_fn=lambda: (os.setgid(foreign.pw_gid), os.setuid(foreign.pw_uid)),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        ready = target / "ready"
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.05)
        assert ready.exists() and child.poll() is None
        # Install the exact source scanner under /var/lib: the transient has
        # ProtectHome=yes and cannot import from the checkout in /Users.
        package = root / "installed" / "blueprint_pipeline"
        package.mkdir(parents=True)
        sources = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"
        for name in ("__init__", "control_plane_lane_scratch_census",
                     "control_plane_disk_usage", "control_plane_lane_scratch",
                     "control_plane_storage_pins", "control_plane_storage_roots",
                     "decision_evidence_contracts", "control_plane_lane_legacy_owner",
                     "control_plane_lane_legacy_owner_door", "control_plane_lane_owner_consents",
                     "control_plane_lane_scratch_decisions"):
            shutil.copyfile(sources / f"{name}.py", package / f"{name}.py")
        process_root = root / "fixture-proc"
        process_root.mkdir()
        (process_root / str(child.pid)).symlink_to(Path("/proc") / str(child.pid))
        probe = root / "probe.py"
        probe.write_text(
            "import ctypes,errno,fcntl,json,os,sys,time\n"
            "from pathlib import Path\n"
            "payload,registry,lock,environment,hidden,results,pid,installed,process_root,pins,release=sys.argv[1:]\n"
            "pid=int(pid)\n"
            "stage='start'\n"
            "def failed(cls,value,tb):\n"
            " Path(results).write_text(json.dumps({'status':'failed','code':stage}))\n"
            "sys.excepthook=failed\n"
            "stage='foreign_payload'\n"
            "assert Path(payload).read_bytes()==b'foreign writer bytes'\n"
            "stage='fixed_environment'\n"
            "assert b'SECRET_CANARY' in Path(environment).read_bytes()\n"
            "stage='hidden_secret'\n"
            "try: Path(hidden,'secret').read_bytes()\n"
            "except OSError: pass\n"
            "else: raise AssertionError('hidden provider secret readable')\n"
            "stage='foreign_proc_cmdline'\n"
            "assert Path('/proc/%d/cmdline'%pid).read_bytes()\n"
            "stage='foreign_proc_environ'\n"
            "Path('/proc/%d/environ'%pid).read_bytes()\n"
            "stage='foreign_proc_fd'\n"
            "fd_denied=False\n"
            "try: [os.readlink('/proc/%d/fd/%s'%(pid,n)) for n in os.listdir('/proc/%d/fd'%pid)]\n"
            "except OSError: fd_denied=True\n"
            "assert fd_denied\n"
            "stage='exact_process_scanner'\n"
            "sys.path.insert(0,installed)\n"
            "from blueprint_pipeline.control_plane_lane_scratch_census import _process_references,build_census\n"
            "errors=[]\n"
            "found=_process_references([Path(payload).parent],Path(process_root),errors,time.monotonic()+10)\n"
            "assert errors==['process_inventory_unreadable'] and Path(payload).parent in found\n"
            "stage='bounded_unknown_census'\n"
            "census=build_census(work_root=Path(payload).parents[3],inputs_root=Path(payload).parents[4]/'inputs',process_root=Path(process_root),pins_root=Path(pins),queue_roots=[],release_link=Path(release),active_run_roots=[],max_seconds=10)\n"
            "assert census['status']=='incomplete' and census['scan_errors']==['process_inventory_unreadable']\n"
            "assert census['candidate_count']==len(census['rows'])==1 and census['rows'][0]['references']==['process']\n"
            "stage='bounded_unknown_public_report'\n"
            "from blueprint_pipeline.control_plane_lane_legacy_owner_door import _public_report\n"
            "public=_public_report(census|{'observed_owner_count':0})\n"
            "assert public['status']=='reference_incomplete' and len(public['rows'])==1\n"
            "assert public['rows'][0]['classification']=='unclassified' and public['rows'][0]['gc_eligible'] is False\n"
            "stage='foreign_memory'\n"
            "for mode in ('rb','r+b'):\n"
            " try: open('/proc/%d/mem'%pid,mode).close()\n"
            " except PermissionError: pass\n"
            " else: raise AssertionError('foreign process memory readable')\n"
            "stage='ptrace'\n"
            "libc=ctypes.CDLL(None,use_errno=True)\n"
            "assert libc.ptrace(0x7fffffff,pid,0,0)==-1 and ctypes.get_errno()==errno.EPERM\n"
            "assert libc.process_vm_readv(pid,None,0,None,0,0)==-1 and ctypes.get_errno()==errno.EPERM\n"
            "assert libc.process_vm_writev(pid,None,0,None,0,0)==-1 and ctypes.get_errno()==errno.EPERM\n"
            "stage='lock'\n"
            "locked=False\n"
            "with open(lock,'rb') as handle:\n"
            " try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
            " except BlockingIOError: locked=True\n"
            "assert locked\n"
            "stage='read_only'\n"
            "denied=[]\n"
            "for path in (payload,registry+'/forbidden'):\n"
            " try: open(path,'wb').write(b'wrong')\n"
            " except OSError: denied.append(True)\n"
            "assert len(denied)==2\n"
            "Path(results).write_text(json.dumps({'status':'passed','lock_blocked':locked,'cross_uid_process_seen':True,'process_fd_unknown_kept':True,'bounded_report_seen':True,'read_only':True,'ptrace_denied':True,'hidden_secret_denied':True,'foreign_memory_denied':True}))\n"
        )
        probe.chmod(0o644)
        receipt = results / "probe.json"
        properties = _legacy_owner_properties(config, {"kind": "legacy-owner-census"})
        argv = ["systemd-run", "--unit=blueprint-adp-legacy-door-probe-" + root.name[-8:],
                "--wait", "--collect", "--service-type=exec", "--property=RuntimeMaxSec=60s",
                *("--property=" + item for item in properties), "--",
                "/usr/bin/python3", str(probe), str(payload), str(registry), str(lock_path),
                str(environment), str(hidden), str(receipt), str(child.pid),
                str(package.parent), str(process_root), str(pins), str(release)]
        run = subprocess.run(argv, capture_output=True, text=True, timeout=75)
        if run.returncode:
            detail = json.loads(receipt.read_bytes()) if receipt.exists() else {}
            code = detail.get("code", "unit_before_report")
            assert code in {"start", "foreign_payload", "fixed_environment", "hidden_secret",
                            "foreign_proc_cmdline", "foreign_proc_environ", "foreign_proc_fd",
                            "exact_process_scanner", "bounded_unknown_census",
                            "bounded_unknown_public_report", "foreign_memory",
                            "ptrace", "lock", "read_only", "unit_before_report"}
            raise AssertionError(f"disposable unit failed at {code}; exit={run.returncode}")
        result = json.loads(receipt.read_bytes())
        assert result["status"] == "passed"
        assert "SECRET_CANARY" not in json.dumps(result) + run.stdout + run.stderr

        replacement = work / "replacement"
        replacement.mkdir()
        replacement.chmod(0o700)
        os.chown(replacement, foreign.pw_uid, foreign.pw_gid)
        target.rename(work / "old-original")
        replacement.rename(target)
        assert snapshot_generation(target, allowed_roots=(root / "work",)) != first
        return result | {"foreign_uid": foreign.pw_uid, "private_denials": private_denials,
                         "target_generation_revoked": True}
    finally:
        if child is not None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        if lock is not None:
            os.close(lock)
        shutil.rmtree(root)


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux" or
                    os.environ.get("BLUEPRINT_DISPOSABLE_LINUX_TEST") != "1",
                    reason="mandatory disposable hosted Linux systemd/UID proof; Mac skip is unmet")
def test_actual_legacy_owner_door_privilege_and_revocation():
    command = [sys.executable, str(Path(__file__).resolve()), "--root-fixture"]
    if os.geteuid() != 0:
        command = ["sudo", "-n", "env", "BLUEPRINT_DISPOSABLE_LINUX_TEST=1",
                   "PYTHONDONTWRITEBYTECODE=1", *command]
    done = subprocess.run(command, capture_output=True, text=True, timeout=90,
                          cwd=Path(__file__).parents[1],
                          env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stdout + done.stderr
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result == dict(status="passed", foreign_uid=result["foreign_uid"],
                          private_denials=3, lock_blocked=True,
                          cross_uid_process_seen=True, read_only=True,
                          process_fd_unknown_kept=True, bounded_report_seen=True,
                          ptrace_denied=True, hidden_secret_denied=True,
                          foreign_memory_denied=True,
                          target_generation_revoked=True)
    assert result["foreign_uid"] != 0


if __name__ == "__main__" and sys.argv[1:] == ["--root-fixture"]:
    print(json.dumps(_root_fixture(), sort_keys=True))
