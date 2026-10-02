# Covers (for impacted-test selection):
#   scripts/native_linux_guest_git_inputs.py
"""Exact committed input transport excludes local credentials and dirty bytes."""
import subprocess
import time

import pytest

from scripts import native_linux_guest_git_inputs as inputs


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
def test_forged_transfer_footprint_does_not_admit_larger_git_tree(source, tmp_path, field):
    root, commit, _ = source
    pack = tmp_path / 'input.pack'
    manifest = inputs.write_git_input(root, commit, pack, deadline_monotonic=time.monotonic() + 10)
    manifest[field] = 0
    with pytest.raises(inputs.GitInputError, match='tree_footprint'):
        inputs.restore_git_input(pack, manifest, tmp_path / 'restored',
                                 deadline_monotonic=time.monotonic() + 10)


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
