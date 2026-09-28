"""Real kernel denial tests; exercised in a default, unprivileged Docker container."""

import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import shutil
import time

import pytest

from blueprint_pipeline.asset_authoring_sandbox import SandboxedAssetRunner

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(platform.system() != "Linux", reason="Linux kernel boundary"),
]


@pytest.fixture(params=[None, 1])
def isolated(tmp_path, monkeypatch, request):
    if request.param is not None:
        import functools
        from blueprint_pipeline import asset_landlock

        monkeypatch.setattr(
            asset_landlock,
            "run_with_landlock",
            functools.partial(asset_landlock.run_with_landlock, abi_limit=request.param),
        )
    readable = tmp_path / "readable"
    writable = tmp_path / "writable"
    readable.mkdir()
    writable.mkdir()
    (readable / "input").write_text("allowed")
    secret = tmp_path / "secret"
    secret.write_text("must-not-leak")
    secret.chmod(0o644)  # DAC permits it; the kernel sandbox must still deny it.
    runner = SandboxedAssetRunner(
        read_roots=[readable, Path(sys.prefix)],
        write_root=writable,
        executable_roots=[Path("/usr")],
    )
    runner.backend = "landlock_seccomp"
    return runner, readable, writable, secret


def execute(runner, code, **kwargs):
    result = runner(["/usr/bin/python3", "-c", code], timeout=10, **kwargs)
    assert result.returncode == 0, result.stderr
    return result


def test_allowed_output_and_no_capabilities_or_credentials(isolated, monkeypatch):
    runner, readable, writable, _ = isolated
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret-must-not-pass")
    result = execute(
        runner,
        f"""import os,json,pathlib
assert pathlib.Path({str(readable / "input")!r}).read_text() == 'allowed'
pathlib.Path({str(writable / "output")!r}).write_text('completed')
s = dict(line.split(':',1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert all(int(s[k].strip(),16)==0 for k in ['CapEff','CapPrm','CapInh','CapAmb'])
assert s['NoNewPrivs'].strip() == '1'
assert 'OPENAI_API_KEY' not in os.environ
print('passed')
""",
    )
    assert "passed" in result.stdout
    assert (writable / "output").read_text() == "completed"


def test_denies_outside_read_write_metadata_and_aliases(isolated):
    runner, readable, writable, secret = isolated
    before = secret.stat()
    execute(
        runner,
        f"""import os,pathlib
secret={str(secret)!r}; readonly={str(readable / "input")!r}; writable={str(writable)!r}
operations = [lambda: open(secret).read(), lambda: open(secret,'w'),
 lambda: open(readonly,'w'), lambda: os.truncate(secret,0),
 lambda: os.open(readonly,os.O_RDONLY|os.O_TRUNC),
 lambda: os.chmod(secret,0), lambda: os.utime(secret,(0,0)),
 lambda: os.setxattr(secret,'user.test',b'value'),
 lambda: os.rename(secret,writable+'/stolen'),
 lambda: os.link(readonly,writable+'/alias')]
for i, operation in enumerate(operations):
 try: operation()
 except OSError: continue
 raise AssertionError('sandbox allowed denied operation '+str(i))
os.symlink(secret,writable+'/link')
try: open(writable+'/link').read()
except OSError: pass
else: raise AssertionError('symlink escape')
""",
    )
    assert secret.read_text() == "must-not-leak"
    assert secret.stat().st_mode == before.st_mode
    assert secret.stat().st_mtime_ns == before.st_mtime_ns


def test_denies_network_and_parent_signal_but_allows_private_ipc(isolated):
    runner, _, _, _ = isolated
    execute(
        runner,
        f"""import os,socket,fcntl
pair=socket.socketpair(); pair[0].send(b'ok'); assert pair[1].recv(2)==b'ok'
try: socket.socketpair(type=socket.SOCK_DGRAM)
except OSError: pass
else: raise AssertionError('datagram pair permits external sendto')
for family in [socket.AF_INET,socket.AF_INET6,socket.AF_UNIX]:
 try: socket.socket(family,socket.SOCK_STREAM)
 except OSError: pass
 else: raise AssertionError('socket allowed')
for operation in [lambda: os.kill({os.getpid()},0), lambda: fcntl.fcntl(1,fcntl.F_SETOWN,{os.getpid()})]:
 try: operation()
 except OSError: pass
 else: raise AssertionError('parent signal allowed')
os.kill(os.getpid(),0)
assert os.getppid() <= 0
for operation in [lambda: os.getpgid({os.getpid()}), lambda: os.getsid({os.getpid()}),
                  lambda: os.sched_getaffinity({os.getpid()})]:
 try: operation()
 except OSError: pass
 else: raise AssertionError('parent process query allowed')
""",
    )


def test_exec_child_inherits_filesystem_restrictions(isolated):
    runner, _, _, secret = isolated
    child = f"import pathlib; pathlib.Path({str(secret)!r}).read_text()"
    result = execute(
        runner,
        f"""import subprocess
r=subprocess.run(['/usr/bin/python3','-c',{child!r}],capture_output=True,text=True)
assert r.returncode != 0 and 'PermissionError' in r.stderr
print('inherited')
""",
    )
    assert "inherited" in result.stdout


def test_timeout_kills_descendants(isolated):
    runner, _, writable, _ = isolated
    with pytest.raises(subprocess.TimeoutExpired):
        runner(
            [
                "/usr/bin/python3",
                "-c",
                f"""import subprocess,time,pathlib
p=subprocess.Popen(['/usr/bin/python3','-c','import time; time.sleep(60)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
pathlib.Path({str(writable / "child.pid")!r}).write_text(str(p.pid))
time.sleep(60)
""",
            ],
            timeout=2,
        )
    child = int((writable / "child.pid").read_text())
    status = Path(f"/proc/{child}/status")
    # SIGKILL delivery and reparenting/reaping are asynchronous; verify that
    # the child stops within a bounded scheduling interval, not the same tick.
    def stopped():
        try:
            return "\nState:\tZ" in status.read_text()
        except FileNotFoundError:
            return True
    deadline = time.monotonic() + 1
    while not stopped() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stopped()


def test_real_preflight_selects_a_kernel_backend(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    runner = SandboxedAssetRunner(read_roots=[Path("/usr")], write_root=output)
    runner.preflight()
    receipt = json.loads((tmp_path / "output.sandbox_preflight.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["unsandboxed_execution_allowed"] is False
    assert receipt["attempts"][-1]["returncode"] == 0


def test_changed_root_identity_refuses_before_candidate(isolated, monkeypatch):
    from blueprint_pipeline import asset_landlock

    runner, readable, writable, _ = isolated
    identity = asset_landlock._identity

    def changed(path):
        row = identity(path)
        if Path(path) == readable:
            row["inode"] += 1
        return row

    monkeypatch.setattr(asset_landlock, "_identity", changed)
    result = runner(
        ["/usr/bin/python3", "-c", f"open({str(writable / 'must-not-exist')!r},'w').close()"]
    )
    assert result.returncode != 0 and "root_identity_changed" in result.stderr
    assert not (writable / "must-not-exist").exists()


def test_uncaptured_output_is_forwarded_without_inheriting_parent_file_fds(isolated, capsys):
    runner, _, _, _ = isolated
    result = runner(
        ["/usr/bin/python3", "-c", "import os; print('forwarded'); assert not os.isatty(1)"],
        capture_output=False,
    )
    assert result.returncode == 0 and result.stdout is None
    assert "forwarded" in capsys.readouterr().out


@pytest.mark.skipif(os.geteuid() != 0, reason='reproduces provider container root')
def test_foreign_owned_python_tree_is_readable_without_restoring_capabilities(tmp_path):
    from blueprint_pipeline.asset_runtime_permissions import prepare_runtime_code_access
    vendor = tmp_path / 'vendor'
    runtime = vendor / 'kit' / 'python'
    runtime.mkdir(parents=True)
    code = runtime / 'stdlib.py'
    code.write_text('trusted public code')
    executable = runtime / 'python-helper'
    executable.write_text('#!/usr/bin/python3\nprint("private-executable-ready")\n')
    secret = vendor / 'private'
    secret.write_text('not admitted')
    (runtime / 'outside-link').symlink_to(secret)
    for p in [vendor, vendor / 'kit', runtime, code, executable]:
        os.chown(p, 1234, 1234)
        p.chmod(0o700 if p.is_dir() else 0o600)
    executable.chmod(0o700)
    secret_mode = secret.stat().st_mode
    writable = tmp_path / 'output'
    writable.mkdir()
    runner = SandboxedAssetRunner(read_roots=[runtime, Path('/usr')], write_root=writable)
    runner.backend = 'landlock_seccomp'
    failed = runner(['/usr/bin/true'])
    assert failed.returncode == 126 and 'Permission denied' in failed.stderr
    report = prepare_runtime_code_access([runtime])
    assert report['changed_count'] == 5
    assert secret.stat().st_mode == secret_mode
    assert code.stat().st_uid == 1234 and code.read_text() == 'trusted public code'
    execute(runner, f'''import pathlib
assert pathlib.Path({str(code)!r}).read_text() == 'trusted public code'
s=dict(line.split(':',1) for line in pathlib.Path('/proc/self/status').read_text().splitlines() if ':' in line)
assert all(int(s[k].strip(),16)==0 for k in ['CapEff','CapPrm','CapInh','CapAmb'])
try: pathlib.Path({str(runtime / 'outside-link')!r}).read_text()
except PermissionError: pass
else: raise AssertionError('external symlink admitted')
''')
    result = runner([str(executable)])
    assert result.returncode == 0 and 'private-executable-ready' in result.stdout


def test_shared_library_loader_is_preserved_only_for_sandboxed_program(tmp_path):
    compiler = shutil.which('cc')
    if compiler is None:
        pytest.skip('ELF linker fixture requires a C compiler')
    runtime = tmp_path / 'vendor'
    library = runtime / 'lib'
    library.mkdir(parents=True)
    source = runtime / 'library.c'
    source.write_text('int vendor_value(void) { return 17; }\n')
    main = runtime / 'main.c'
    main.write_text('extern int vendor_value(void); int main(void) { return vendor_value() == 17 ? 0 : 1; }\n')
    executable = runtime / 'program'
    subprocess.run([compiler, '-shared', '-fPIC', str(source), '-o', str(library/'libvendor.so')], check=True)
    subprocess.run([compiler, str(main), '-L'+str(library), '-lvendor', '-o', str(executable)], check=True)
    # If the library path leaks into the trusted launcher, its policy library
    # load will fail before restrictions can be installed.
    (library/'libseccomp.so.2').write_text('must never load in trusted launcher')
    output = tmp_path/'output'
    output.mkdir()
    runner = SandboxedAssetRunner(read_roots=[runtime, Path('/usr')], write_root=output)
    runner.backend = 'landlock_seccomp'
    failed = runner([str(executable)])
    assert failed.returncode == 127 and 'libvendor.so' in failed.stderr
    runner = SandboxedAssetRunner(read_roots=[runtime, Path('/usr')], write_root=output,
        library_environment={'LD_LIBRARY_PATH': str(library)}, library_executables=[executable])
    runner.backend = 'landlock_seccomp'
    assert runner([str(executable)]).returncode == 0
    execute(runner, "import os; assert 'LD_LIBRARY_PATH' not in os.environ")
