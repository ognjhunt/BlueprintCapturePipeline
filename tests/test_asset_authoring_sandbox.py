from types import SimpleNamespace
import pytest

from blueprint_pipeline import asset_authoring_sandbox as sandbox


def test_linux_deep_mount_ancestors_are_empty_traversable_namespace_dirs(tmp_path, monkeypatch):
    read_root = tmp_path / 'private-runtime' / 'version' / 'bin'
    write_root = tmp_path / 'private-output' / 'attempt'
    read_root.mkdir(parents=True)
    write_root.mkdir(parents=True)
    monkeypatch.setattr(sandbox.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(sandbox.shutil, 'which', lambda name: '/usr/bin/bwrap')
    monkeypatch.setattr(sandbox.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(sandbox.pwd, 'getpwnam', lambda name: SimpleNamespace(pw_uid=995, pw_gid=995))
    monkeypatch.setattr(sandbox.os, 'chown', lambda *args: None)
    commands = []
    monkeypatch.setattr(sandbox.subprocess, 'run', lambda command, **kwargs:
        commands.append(command) or SimpleNamespace(returncode=0))
    runner = sandbox.SandboxedAssetRunner(read_roots=[read_root], write_root=write_root)
    runner(['/usr/bin/true'])
    command = commands[0]
    first_bind = command.index('--ro-bind')
    for parent in (read_root.parent, write_root.parent):
        index = command.index(str(parent))
        assert command[index-3:index] == ['--perms', '0755', '--dir']
        assert index < first_bind
    assert command[command.index('--tmpfs')-2:command.index('--tmpfs')+2] == ['--perms', '0755', '--tmpfs', '/tmp']
    bound = [command[index+1] for index, item in enumerate(command) if item == '--ro-bind']
    assert str(read_root) in bound
    assert str(read_root.parent) not in bound
    assert '--no-new-privs' in command and '--unshare-net' in command
    assert '--chmod' not in command


def test_unavailable_kernel_backends_never_run_candidate(tmp_path, monkeypatch):
    import pytest
    from blueprint_pipeline import asset_landlock
    monkeypatch.setattr(sandbox.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(sandbox.shutil, 'which', lambda name: None)
    monkeypatch.setattr(asset_landlock, 'run_with_landlock', lambda *args, **kwargs:
        SimpleNamespace(returncode=126, stderr='kernel unavailable', stdout=''))
    runner = sandbox.SandboxedAssetRunner(read_roots=[], write_root=tmp_path)
    with pytest.raises(sandbox.AssetSandboxError, match='asset_sandbox_unavailable'):
        runner.preflight()


def test_repeated_preflight_preserves_backend_and_original_attempts(tmp_path, monkeypatch):
    import json
    monkeypatch.setattr(sandbox.platform, 'system', lambda: 'Linux')
    monkeypatch.setattr(sandbox.shutil, 'which', lambda name: '/usr/bin/bwrap')
    calls = []
    def invoke(self, *args, **kwargs):
        calls.append(self.backend)
        return SimpleNamespace(returncode=0 if self.backend else 1, stderr='namespace denied')
    monkeypatch.setattr(sandbox.SandboxedAssetRunner, '__call__', invoke)
    output = tmp_path / 'output'
    output.mkdir()
    runner = sandbox.SandboxedAssetRunner(read_roots=[], write_root=output)
    runner.preflight()
    receipt = (tmp_path / 'output.sandbox_preflight.json').read_bytes()
    runner.preflight()
    assert calls == [None, 'landlock_seccomp']
    assert runner.backend == 'landlock_seccomp'
    assert (tmp_path / 'output.sandbox_preflight.json').read_bytes() == receipt
    assert [r['backend'] for r in json.loads(receipt)['attempts']] == ['namespace', 'landlock_seccomp']


def test_home_override_cannot_escape_attempt(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(sandbox.shutil, 'which', lambda name: '/usr/bin/bwrap')
    output = tmp_path / 'output'
    output.mkdir()
    runner = sandbox.SandboxedAssetRunner(read_roots=[], write_root=output)
    with pytest.raises(sandbox.AssetSandboxError, match='home_outside_attempt'):
        runner(['/usr/bin/true'], env={'HOME': str(tmp_path)})


@pytest.mark.parametrize('system', ['Darwin', 'Linux'])
def test_only_admitted_library_paths_are_inherited_by_target(tmp_path, monkeypatch, system):
    library = tmp_path / 'lib'
    library.mkdir()
    output = tmp_path / 'output'
    output.mkdir()
    monkeypatch.setattr(sandbox.platform, 'system', lambda: system)
    monkeypatch.setattr(sandbox.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(sandbox.shutil, 'which', lambda name: '/usr/bin/sandbox-exec')
    seen = []
    monkeypatch.setattr(sandbox.subprocess, 'run', lambda *a, **kw:
        seen.append((a[0], kw['env'])) or SimpleNamespace(returncode=0))
    runner = sandbox.SandboxedAssetRunner(read_roots=[library], write_root=output,
        library_environment={'LD_LIBRARY_PATH': str(library)}, library_executables=['/usr/bin/true'])
    runner(['/usr/bin/true'])
    runner(['/usr/bin/true'], env={'OPENAI_API_KEY': 'must-not-inherit'})
    for command, environment in seen:
        assert 'LD_LIBRARY_PATH' not in environment and 'OPENAI_API_KEY' not in environment
        inner = command.index('/usr/bin/env')
        assert command[inner:] == ['/usr/bin/env', '--', 'LD_LIBRARY_PATH='+str(library), '/usr/bin/true']
    with pytest.raises(sandbox.AssetSandboxError, match='loader_path_not_admitted'):
        runner(['/usr/bin/true'], env={'LD_LIBRARY_PATH': str(tmp_path)})
    with pytest.raises(sandbox.AssetSandboxError, match='library_environment_invalid'):
        sandbox.SandboxedAssetRunner(read_roots=[library], write_root=output,
            library_environment={'OPENAI_API_KEY': 'must-not-inherit'})
