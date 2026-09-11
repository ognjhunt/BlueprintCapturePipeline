from types import SimpleNamespace

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
