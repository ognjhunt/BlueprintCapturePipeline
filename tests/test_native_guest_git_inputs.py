# Covers (for impacted-test selection):
#   scripts/native_linux_guest_git_inputs.py
"""Exact committed input transport excludes local credentials and dirty bytes."""
import subprocess
import time
import hashlib
import os
import sys
from types import SimpleNamespace

import pytest

from scripts import native_linux_guest_git_inputs as inputs

linux_restore = pytest.mark.skipif(sys.platform != 'linux',
    reason='Linux /proc descriptor-bound Git restore requires actual Linux; portable runs do not prove it')


@pytest.fixture
def source(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Disposable fixture')
    git('config', 'user.email', 'fixture@example.invalid')
    (root / 'payload.txt').write_text('committed original\n')
    git('add', 'payload.txt')
    git('commit', '-qm', 'tiny input')
    commit = git('rev-parse', 'HEAD')
    git('config', 'http.extraheader', 'AUTHORIZATION: fixture-value-must-not-transfer')
    (root / 'payload.txt').write_text('dirty original remains\n')
    (root / 'untracked').write_text('preserve me\n')
    return root, commit, git


@linux_restore
def test_roundtrip_uses_git_objects_and_preserves_source_dirty_work_without_credentials(source, tmp_path):
    root, commit, git = source
    before = git('status', '--porcelain')
    pack = tmp_path / 'input.pack'
    manifest = inputs.write_git_input(root, commit, pack, deadline_monotonic=time.monotonic() + 10)
    restored = tmp_path / 'restored'
    inputs.restore_git_input(pack, manifest, restored, deadline_monotonic=time.monotonic() + 10)
    assert (restored / 'payload.txt').read_text() == 'committed original\n'
    assert not (restored / 'untracked').exists()
    assert 'fixture-value' not in (restored / '.git/config').read_text()
    assert subprocess.check_output(['git', '-C', str(restored), 'rev-parse', 'HEAD'], text=True).strip() == commit
    assert git('status', '--porcelain') == before
    assert (root / 'payload.txt').read_text() == 'dirty original remains\n'


@pytest.mark.parametrize('change', ['commit', 'pack', 'tree', 'destination', 'deadline'])
@linux_restore
def test_unbound_inputs_or_existing_destination_cannot_be_restored(source, tmp_path, change):
    root, commit, _ = source
    pack = tmp_path / 'input.pack'
    manifest = inputs.write_git_input(root, commit, pack, deadline_monotonic=time.monotonic() + 10)
    target = tmp_path / 'target'
    deadline = time.monotonic() + 10
    if change == 'commit':
        manifest['commit'] = 'a' * 40
    elif change == 'tree':
        manifest['tree'] = 'b' * 40
    elif change == 'pack':
        pack.write_bytes(pack.read_bytes() + b'changed')
    elif change == 'destination':
        target.mkdir()
        (target / 'preserve').write_bytes(b'original')
    else:
        deadline = time.monotonic() - 1
    with pytest.raises(inputs.GitInputError):
        inputs.restore_git_input(pack, manifest, target, deadline_monotonic=deadline)
    if change == 'destination':
        assert (target / 'preserve').read_bytes() == b'original'


def test_export_refuses_nonselected_commit_before_pack_output(source, tmp_path):
    root, _, _ = source
    output = tmp_path / 'output.pack'
    with pytest.raises(inputs.GitInputError):
        inputs.write_git_input(root, 'a' * 40, output, deadline_monotonic=time.monotonic() + 10)
    assert not output.exists()


@pytest.mark.parametrize('field', ['working_tree_bytes', 'working_tree_files'])
@linux_restore
def test_forged_transfer_footprint_does_not_admit_larger_git_tree(source, tmp_path, field):
    root, commit, _ = source
    pack = tmp_path / 'input.pack'
    manifest = inputs.write_git_input(root, commit, pack, deadline_monotonic=time.monotonic() + 10)
    manifest[field] = 0
    with pytest.raises(inputs.GitInputError, match='tree_footprint'):
        inputs.restore_git_input(pack, manifest, tmp_path / 'restored',
                                 deadline_monotonic=time.monotonic() + 10)


@linux_restore
def test_actual_shallow_contracts_style_input_keeps_exact_boundary_without_remote_config(source, tmp_path):
    root, _, git = source
    (root / 'second.txt').write_text('second committed file\n')
    git('add', 'second.txt')
    git('commit', '-qm', 'second tiny input')
    commit = git('rev-parse', 'HEAD')
    shallow = tmp_path / 'shallow'
    subprocess.run(['git', '-c', 'core.hooksPath=/dev/null', 'clone', '-q', '--depth=1',
                    root.as_uri(), str(shallow)], check=True, env=inputs._environment() | {'GIT_ALLOW_PROTOCOL': 'file'})
    pack = tmp_path / 'shallow.pack'
    manifest = inputs.write_git_input(shallow, commit, pack, deadline_monotonic=time.monotonic() + 10)
    assert manifest['shallow'] == [commit]
    restored = tmp_path / 'restored'
    inputs.restore_git_input(pack, manifest, restored, deadline_monotonic=time.monotonic() + 10)
    assert (restored / 'second.txt').read_text() == 'second committed file\n'
    assert (restored / '.git/shallow').read_text() == commit + '\n'
    assert not subprocess.check_output(['git', '-C', str(restored), 'remote'], text=True).strip()


@linux_restore
def test_verified_pack_is_not_reopened_after_replacement(source, tmp_path, monkeypatch):
    root, commit, git = source
    pack = tmp_path / 'input.pack'
    manifest = inputs.write_git_input(root, commit, pack, deadline_monotonic=time.monotonic() + 10)
    original_bytes = pack.read_bytes()
    (root / 'other.txt').write_text('a different committed object\n')
    git('add', 'other.txt')
    git('commit', '-qm', 'additional object set')
    replacement = tmp_path / 'replacement.pack'
    inputs.write_git_input(root, git('rev-parse', 'HEAD'), replacement,
                           deadline_monotonic=time.monotonic() + 10)
    observed = []
    original = inputs._git
    def git_command(root, arguments, deadline, **kwargs):
        result = original(root, arguments, deadline, **kwargs)
        if arguments[0] == 'init':
            pack.rename(tmp_path / 'original.pack')
            pack.write_bytes(replacement.read_bytes())
        if arguments[0] == 'index-pack':
            stream = kwargs['stdin']
            stream.seek(0)
            observed.append(hashlib.sha256(stream.read()).hexdigest())
        return result
    monkeypatch.setattr(inputs, '_git', git_command)
    with pytest.raises(inputs.GitInputError, match='pack'):
        inputs.restore_git_input(pack, manifest, tmp_path / 'restored',
                                 deadline_monotonic=time.monotonic() + 10)
    assert not observed or observed == [hashlib.sha256(original_bytes).hexdigest()]


def test_nonregular_pack_is_acquired_without_blocking_on_fifo(tmp_path, monkeypatch):
    fifo = tmp_path / 'fifo'
    os.mkfifo(fifo)
    original = inputs.os.open
    def open_file(path, flags, *args, **kwargs):
        if path == fifo:
            assert flags & os.O_NONBLOCK, 'FIFO acquisition can otherwise block past its deadline'
        return original(path, flags, *args, **kwargs)
    monkeypatch.setattr(inputs.os, 'open', open_file)
    with pytest.raises(inputs.GitInputError, match='pack'):
        inputs._pack_hash(fifo, time.monotonic() + 1)


@linux_restore
def test_restore_uses_owned_directory_after_ancestor_replacement(source, tmp_path, monkeypatch):
    root, commit, _ = source
    pack = tmp_path / 'input.pack'
    manifest = inputs.write_git_input(root, commit, pack, deadline_monotonic=time.monotonic() + 10)
    parent = tmp_path / 'parent'
    parent.mkdir()
    destination = parent / 'restored'
    foreign = tmp_path / 'foreign'
    foreign.mkdir()
    (foreign / 'restored').mkdir()
    (foreign / 'restored' / 'preserve').write_bytes(b'foreign-original')
    original = inputs._git
    def git_command(root, arguments, deadline, **kwargs):
        if arguments[0] == 'init':
            parent.rename(tmp_path / 'owned-parent')
            parent.symlink_to(foreign, target_is_directory=True)
        return original(root, arguments, deadline, **kwargs)
    monkeypatch.setattr(inputs, '_git', git_command)
    with pytest.raises(inputs.GitInputError, match='destination'):
        inputs.restore_git_input(pack, manifest, destination, deadline_monotonic=time.monotonic() + 10)
    assert list((foreign / 'restored').iterdir()) == [foreign / 'restored' / 'preserve']


@linux_restore
def test_shallow_metadata_never_overwrites_an_existing_leaf(source, tmp_path, monkeypatch):
    root, commit, _ = source
    shallow = tmp_path / 'shallow'
    subprocess.run(['git', 'clone', '-q', '--depth=1', root.as_uri(), str(shallow)],
                   check=True, env=inputs._environment() | {'GIT_ALLOW_PROTOCOL': 'file'})
    pack = tmp_path / 'input.pack'
    manifest = inputs.write_git_input(shallow, commit, pack, deadline_monotonic=time.monotonic() + 10)
    destination = tmp_path / 'restored'
    original = inputs._git
    def git_command(root, arguments, deadline, **kwargs):
        result = original(root, arguments, deadline, **kwargs)
        if arguments[0] == 'init':
            (destination / '.git/shallow').write_bytes(b'foreign-original')
        return result
    monkeypatch.setattr(inputs, '_git', git_command)
    with pytest.raises(inputs.GitInputError, match='shallow'):
        inputs.restore_git_input(pack, manifest, destination, deadline_monotonic=time.monotonic() + 10)
    assert (destination / '.git/shallow').read_bytes() == b'foreign-original'


def test_export_does_not_start_pack_process_after_setup_consumes_deadline(source, tmp_path, monkeypatch):
    root, commit, _ = source
    observed = [time.monotonic()]
    deadline = observed[0] + 10
    monkeypatch.setattr(inputs.time, 'monotonic', lambda: observed[0])
    original = inputs.selectors.DefaultSelector
    class SlowSetup(original):
        def __enter__(self):
            result = super().__enter__()
            observed[0] = deadline + 1
            return result
    monkeypatch.setattr(inputs, 'selectors', SimpleNamespace(DefaultSelector=SlowSetup, EVENT_READ=inputs.selectors.EVENT_READ))
    launches = []
    popen = inputs.subprocess.Popen
    def record(command, **kwargs):
        if 'pack-objects' in command:
            launches.append(command)
        return popen(command, **kwargs)
    monkeypatch.setattr(inputs.subprocess, 'Popen', record)
    with pytest.raises(inputs.GitInputError, match='deadline'):
        inputs.write_git_input(root, commit, tmp_path / 'input.pack', deadline_monotonic=deadline)
    assert launches == []


@pytest.mark.parametrize('operation', ['export', 'restore'])
def test_last_git_selection_guard_cannot_return_after_deadline(source, tmp_path, monkeypatch, operation):
    if operation == 'restore' and sys.platform != 'linux':
        pytest.skip('Linux /proc descriptor-bound Git restore requires actual Linux')
    root, commit, _ = source
    pack = tmp_path / 'input.pack'
    manifest = inputs.write_git_input(root, commit, pack, deadline_monotonic=time.monotonic() + 10)
    observed = [time.monotonic()]
    deadline = observed[0] + 10
    monkeypatch.setattr(inputs.time, 'monotonic', lambda: observed[0])
    original = inputs._git
    head_calls = [0]
    def git_command(root, arguments, original_deadline, **kwargs):
        result = original(root, arguments, original_deadline, **kwargs)
        if arguments == ['rev-parse', 'HEAD']:
            head_calls[0] += 1
            if operation == 'restore' or head_calls[0] == 2:
                observed[0] = deadline + 1
        return result
    monkeypatch.setattr(inputs, '_git', git_command)
    with pytest.raises(inputs.GitInputError, match='deadline'):
        if operation == 'export':
            inputs.write_git_input(root, commit, tmp_path / 'second.pack', deadline_monotonic=deadline)
        else:
            inputs.restore_git_input(pack, manifest, tmp_path / 'restored', deadline_monotonic=deadline)


def test_export_pack_cannot_change_after_final_selection_guard(source, tmp_path, monkeypatch):
    root, commit, _ = source
    pack = tmp_path / 'input.pack'
    original = inputs._git
    calls = [0]
    def git_command(root, arguments, deadline, **kwargs):
        result = original(root, arguments, deadline, **kwargs)
        if arguments == ['rev-parse', 'HEAD']:
            calls[0] += 1
            if calls[0] == 2:
                raw = pack.read_bytes()
                pack.rename(tmp_path / 'original.pack')
                pack.write_bytes(raw)
        return result
    monkeypatch.setattr(inputs, '_git', git_command)
    with pytest.raises(inputs.GitInputError, match='pack'):
        inputs.write_git_input(root, commit, pack, deadline_monotonic=time.monotonic() + 10)
